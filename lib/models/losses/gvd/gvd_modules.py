import torch
import torch.nn as nn


class NoiseAdapter(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            self.feat = nn.Sequential(
                Bottleneck(channels, channels, reduction=8),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            self.feat = nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
            )
        self.pred = nn.Linear(channels, 2)

    def forward(self, x):
        x = self.feat(x).flatten(1)
        x = self.pred(x).softmax(1)[:, 0]
        return x

    
class DiffusionModel(nn.Module):
    def __init__(self, channels_in, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.time_embedding = nn.Embedding(1280, channels_in)

        if kernel_size == 3:
            self.pred = nn.Sequential(
                Bottleneck(channels_in, channels_in),
                Bottleneck(channels_in, channels_in),
                nn.Conv2d(channels_in, channels_in, 1),
                nn.BatchNorm2d(channels_in)
            )
        else:
            self.pred = nn.Sequential(
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1),
                nn.BatchNorm2d(channels_in),
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1)
            )

    def forward(self, noisy_image, t):
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy_image
        feat = feat + self.time_embedding(t)[..., None, None]
        ret = self.pred(feat)
        return ret


class AutoEncoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, 1, padding=0),
            nn.BatchNorm2d(latent_channels)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels, 1, padding=0),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return hidden, out

    def forward_encoder(self, x):
        return self.encoder(x)
    
    def forward_decoder(self, x):
        return self.decoder(x)
    

class DDIMPipeline:
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, model, scheduler, noise_adapter=None, solver='ddim', classifier_tail=None):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter = noise_adapter
        self._iter = 0
        self.solver = solver
        self.classifier_tail=classifier_tail

    def __call__(
            self,
            batch_size,
            device,
            dtype,
            shape,
            feat,
            generator = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            proj = None,
            t_start=None,
            guidance_features=None,
            guidance_scale=1.0,
            guidance_stop_t = -1,
            targets=None
    ):

        # Sample gaussian noise to begin loop
        image_shape = (batch_size, *shape)

        if t_start is not None:
            # 1) create x_t_start
            noise = torch.randn_like(feat, generator=generator)
            t_start_vec = torch.full((batch_size,), int(t_start), device=device, dtype=torch.long)
            image = self.scheduler.add_noise(feat, noise, t_start_vec)

            # 2) set scheduler steps (safe) then override actual timesteps we will use
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            # self.scheduler.timesteps = self.scheduler.timesteps[
            #     self.scheduler.timesteps <= t_start
            # ]
        
            self.scheduler.timesteps = torch.linspace(
                t_start, 0, steps=num_inference_steps, device=device
            ).round().long()
            
            # print("self.scheduler.timesteps",self.scheduler.timesteps)
            # 3) DDIM denoise from t_start -> 0 in S steps
            #record the score at the intial timestep
            score0=None
            for t in self.scheduler.timesteps:
                noise_pred = self.model(image, t)
                if score0 is None:
                    score0 = self.ddim_score_from_noise_pred(noise_pred, t, self.scheduler)
                image = self.scheduler.step(
                    noise_pred, t, image, eta=eta,
                    use_clipped_model_output=True,
                    generator=generator,
                    guided_features=guidance_features,
                    guidance_scale=guidance_scale,
                    guidance_stop_t = guidance_stop_t,
                    targets=targets
                )["prev_sample"]
            
            self._iter += 1
            return image, score0

        elif self.noise_adapter is not None:
            timesteps = self.noise_adapter(feat)
            noise = torch.randn(image_shape, device=device, dtype=dtype)
            image = self.scheduler.add_noise_diff2(feat, noise, timesteps)
        else:
            image = feat

        # set step values
        self.scheduler.set_timesteps(num_inference_steps*2)
        for t in self.scheduler.timesteps[len(self.scheduler.timesteps)//2:]: #take the second half of the diffusion time steps
            noise_pred = self.model(image, t.to(device))
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                noise_pred, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample'] 
        self._iter += 1        
        return image

    @torch.no_grad()
    def ddim_score_from_noise_pred(self, noise_pred, t, scheduler):
        """
        noise_pred: (B,C,H,W) = epsilon_theta(x_t, t)
        t:          int or (B,) long tensor (timesteps)
        scheduler:  must provide scheduler.alphas_cumprod (shape [T])
        returns:    (B,C,H,W) score estimate s_theta(x_t,t)
        """
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=noise_pred.device, dtype=torch.long)

        if t.ndim == 0:
            t = t[None]  # (1,)
        if t.numel() == 1:
            # broadcast to batch
            t = t.expand(noise_pred.shape[0])

        alpha_bar = scheduler.alphas_cumprod.to(noise_pred.device)[t]  # (B,)
        sigma = torch.sqrt(1.0 - alpha_bar).view(-1, 1, 1, 1)          # (B,1,1,1)
        score = -noise_pred / (sigma + 1e-8)
        return score

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x
