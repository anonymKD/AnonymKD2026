import torch
from torch import nn
import torch.nn.functional as F
from .gvd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline
from .scheduling_ddim import DDIMScheduler


class LatentDiff(nn.Module):
    def __init__(
            self,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels
        
        # diffusion model - predict noise
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.pipeline = DDIMPipeline(self.model, self.scheduler)
        # self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))

    def forward(self, teacher_feat, sampling=False, t_start=999, query_feat=None, guidance_scale=0.5, guidance_stop_t=100, targets=None, eta=0.0,eta_seed=0):
        if not sampling:
            # use autoencoder on teacher feature
            if self.use_ae:
                hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
                rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
                teacher_feat = hidden_t_feat.detach()
            else:
                rec_loss = None

            # train diffusion model
            ddim_loss = self.ddim_loss(teacher_feat)
            return ddim_loss, rec_loss   
    
        else: #sampling mode
            if t_start==0:
                return teacher_feat, teacher_feat, teacher_feat, teacher_feat # return input teacher features without inference

            if query_feat != None : #think query features as student features in the final stages
                if self.use_ae:
                    query_feat = self.ae.forward_encoder(query_feat)
                score, noise, noise_pred = self.ddim_score_from_noise_pred(query_feat, t_start, bring_to_same_noise=True)
                return query_feat, score, noise, noise_pred
            
            if self.use_ae:
                hidden_t_feat = self.ae.forward_encoder(teacher_feat)
                teacher_feat = hidden_t_feat.detach()

            #generate different generators to increase ddim diveristy(ddpm style)
            step_seed = int(eta_seed)  # or timestep + offset
            local_gen = torch.Generator(device=teacher_feat.device)
            local_gen.manual_seed(step_seed)

            # denoise given feature and geenrate a new
            refined_feat, score = self.pipeline(
                batch_size=teacher_feat.shape[0],
                device=teacher_feat.device,
                dtype=teacher_feat.dtype,
                shape=teacher_feat.shape[1:],
                feat=teacher_feat,
                num_inference_steps=self.diffusion_inference_steps,
                t_start=t_start,
                guidance_features=teacher_feat,
                guidance_scale=guidance_scale,
                guidance_stop_t = guidance_stop_t,
                targets=targets,
                generator=local_gen,
                eta=eta
            )
            
            refined_feat_final=refined_feat
            if self.use_ae:
                refined_feat_final = self.ae.forward_decoder(refined_feat)

            return refined_feat, refined_feat_final, teacher_feat, score #raw_diff_samples, decoded_diff_samples, intial_encoded_samples, score at t_start


    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    def ddim_score_from_noise_pred(self, image, t, bring_to_same_noise=True):
        """
        image
        t:          int or (B,) long tensor (timesteps)
        returns:    (B,C,H,W) score estimate s_theta(x_t,t)
        """
        bs = image.shape[0]
        timesteps=torch.full((bs,), t, device=image.device, dtype=torch.long)
        noise = torch.randn(image.shape, device=image.device) #modify this to reuse the same noise per batch (or per step) for stability, it can make the gradient very noisy
        if bring_to_same_noise:
            image = self.scheduler.add_noise(image, noise, timesteps)

        noise_pred = self.model(image, timesteps) #noise_pred: (B,C,H,W) = epsilon_theta(x_t, t)
        alpha_bar = self.scheduler.alphas_cumprod.to(noise_pred.device)[timesteps]  # (B,)
        sigma = torch.sqrt(1.0 - alpha_bar).view(-1, 1, 1, 1)          # (B,1,1,1)
        score = -noise_pred / (sigma + 1e-8)
        return score, noise, noise_pred

    def set_guidance_parameters(self, feature_guidance=False, classifier_guidance=False, classifier_tail=None):
        self.scheduler.feature_guidance=feature_guidance
        self.scheduler.classifier_guidance=classifier_guidance
        if classifier_guidance:
            self.scheduler.classifier_tail=classifier_tail
            self.scheduler.ae=self.ae
            self.scheduler.use_ae=self.use_ae


class DiffKD(nn.Module):
    def __init__(
            self,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels
        
        # transform student feature to the same dimension as teacher
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        # diffusion model - predict noise
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))

    def forward(self, student_feat, teacher_feat):
        # project student feature to the same dimension as teacher feature
        student_feat = self.trans(student_feat)

        # use autoencoder on teacher feature
        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        # denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
        refined_feat = self.proj(refined_feat)
        
        # train diffusion model
        ddim_loss = self.ddim_loss(teacher_feat)
        return refined_feat, teacher_feat, ddim_loss, rec_loss

    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss
    

