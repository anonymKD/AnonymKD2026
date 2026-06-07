import math
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from .kl_div import KLDivergence
from .dist_kd import DIST
from .gvd import DiffKD
from lib.utils.layer_details import DIFF_MODULES, TailToLogits


import logging
logger = logging.getLogger()


KD_MODULES = {
    'cifar_MobileNetV2': dict(modules=['conv2', 'classifier'], channels=[1280, 100]),
    'cifar_ResNet50': dict(modules=['layer4', 'linear'], channels=[2048, 100]),
    'cifar_MobileNetV2': dict(modules=['blocks.6', 'classifier'], channels=[160, 100]),
    'cifar_ShuffleV1': dict(modules=['layer2', 'linear'], channels=[480, 100]),
    'cifar_ShuffleV2': dict(modules=['layer2', 'linear'], channels=[232, 100]),
    'cifar_resnet32x4': dict(modules=['layer3', 'fc'], channels=[256, 100]),
    'cifar_resnet8x4': dict(modules=['layer3', 'fc'], channels=[256, 100]),
    'cifar_wrn_40_1': dict(modules=['relu', 'fc'], channels=[64, 100]),
    'cifar_wrn_40_2': dict(modules=['relu', 'fc'], channels=[128, 100]),
    'cifar_resnet56': dict(modules=['layer3', 'fc'], channels=[64, 100]),
    'cifar_resnet20': dict(modules=['layer3', 'fc'], channels=[64, 10]),
    'cifar_resnet110': dict(modules=['layer3', 'fc'], channels=[64, 10]),
    'tv_resnet50': dict(modules=['layer4', 'fc'], channels=[2048, 1000]),
    'tv_resnet34': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'tv_resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'tv_mobilenet_v2': dict(modules=['features.18', 'classifier'], channels=[1280, 1000]),
    'nas_model': dict(modules=['features.conv_out', 'classifier'], channels=[1280, 1000]),  # mbv2
    'timm_tf_efficientnet_b0': dict(modules=['conv_head', 'classifier'], channels=[1280, 1000]),
    'mobilenet_v1': dict(modules=['model.13', 'fc'], channels=[1024, 1000]),
    'timm_swin_large_patch4_window7_224': dict(modules=['norm', 'head'], channels=[1536, 1000]),
    'timm_swin_tiny_patch4_window7_224': dict(modules=['norm', 'head'], channels=[768, 1000]),

    #tinyimagenet
    'tiny_tv_resnet50': dict(modules=['layer4', 'fc'], channels=[2048, 200]),
    'tiny_tv_resnet34': dict(modules=['layer4', 'fc'], channels=[512, 200]),
    'tiny_tv_resnet18': dict(modules=['layer4', 'fc'], channels=[512, 200]),
    'tiny_mobilenet_v1': dict(modules=['feat', 'fc'], channels=[1024, 200]),

    #ts models
    'ts_lstm_100_3': dict(modules=['seq_feat'], channels=[100]), #all hidden layers
    'ts_lstm_32_2': dict(modules=['seq_feat'], channels=[32]), #all hidden layers
    'ts_lstm': dict(modules=['seq_feat'], channels=[32]), #all hidden layers
    'ts_lstm_8_1': dict(modules=['seq_feat'], channels=[8]), #all hidden layers
    'ts_resnet_64_3': dict(modules=['feat'], channels=[128]), 
    'ts_resnet_8_3': dict(modules=['feat'], channels=[16]), 
    'ts_inception_32_6': dict(modules=['feat'], channels=[32]), 
    'ts_inception_16_3': dict(modules=['feat'], channels=[16]), 
    'ts_inception_8_3': dict(modules=['feat'], channels=[8])
}

from collections import OrderedDict
from typing import Optional, Callable, Tuple

def randomize(x):
    x_noise = x.clone()
    noise = torch.randn_like(x_noise)
    x_noise = 0.9 * x_noise + 0.1 * noise
    return x_noise

def gaussian_randomize(x, sigma=0.1):
    return x + sigma * torch.randn_like(x)

def mixup_randomize(features, alpha=0.4):
    # features: [B, D] or [B, C, H, W]
    
    batch_size = features.size(0)
    
    # sample lambda from Beta distribution
    lam = torch.distributions.Beta(alpha, alpha).sample().to(features.device)
    
    # shuffle indices
    index = torch.randperm(batch_size).to(features.device)
    
    # mix features
    mixed = lam * features + (1 - lam) * features[index]
    
    return mixed

def mixup_randomize_with_targets(features, targets, num_classes=100, alpha=0.4):
    batch_size = features.size(0)
    device = features.device

    lam = torch.distributions.Beta(alpha, alpha).sample().to(device)
    index = torch.randperm(batch_size, device=device)

    mixed_features = lam * features + (1 - lam) * features[index]

    if targets.dim() == 1:
        targets = F.one_hot(targets, num_classes=num_classes).float()
    else:
        targets = targets.float()

    mixed_targets = lam * targets + (1 - lam) * targets[index]

    return mixed_features, mixed_targets

class DiffCacheSingleT_LRU:
    """
    Single-timestep cache with LRU eviction.
    Key: dataset idx (int)
    Value: CPU fp16 tensor [C,H,W] by default
    """

    def __init__(self, max_items=200_000, dtype=torch.float32, store_on_cpu=True):
        self.max_items = int(max_items)
        self.dtype = dtype
        self.store_on_cpu = store_on_cpu

        self.current_t: Optional[int] = None
        self._data: "OrderedDict[int, torch.Tensor]" = OrderedDict()

    def reset_for_t(self, t: int):
        t = int(t)
        if self.current_t != t:
            self.current_t = t
            self._data.clear()

    def _evict_if_needed(self):
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

    @torch.no_grad()
    def get_or_compute(
        self,
        *,
        t: int,
        idx: torch.Tensor,              # [B] (cpu or gpu)
        teacher_feat: torch.Tensor,      # [B,C,H,W] on GPU
        gen_prior: Callable,             # callable(teacher_feat_subset, sampling=True, t_start=t) -> ...
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns diff_samples [B,C,H,W] on `device`.
        Computes only missing items, inserts into LRU, and assembles output in batch order.
        """
        t = int(t)
        self.reset_for_t(t)

        idx_cpu = idx.detach().to("cpu")
        idx_list = [int(i) for i in idx_cpu.tolist()]
        B = len(idx_list)

        out_list = [None] * B
        miss_pos = []
        miss_idx = []
        miss_feats = []

        # one pass: try cache, collect misses
        for j, k in enumerate(idx_list):
            v = self._data.get(k, None)
            if v is None:
                miss_pos.append(j)
                miss_idx.append(k)
                miss_feats.append(teacher_feat[j:j+1])   # keep batch dim
            else:
                self._data.move_to_end(k)               # LRU touch
                out_list[j] = v                          # CPU tensor [C,H,W]

        # compute only misses
        if len(miss_pos) > 0:
            miss_feats = torch.cat(miss_feats, dim=0)    # [Bm,C,H,W] on GPU
            _raw, diff_miss, _latent, _score = gen_prior(miss_feats, sampling=True, t_start=t)

            # store to cache (CPU fp16 by default)
            store = diff_miss.detach()
            if self.store_on_cpu:
                store = store.to("cpu", non_blocking=True).to(self.dtype)
            else:
                store = store.to(self.dtype)

            # insert into cache + fill output slots
            for m, k in enumerate(miss_idx):
                self._data[k] = store[m]
                self._data.move_to_end(k)
                out_list[miss_pos[m]] = store[m]

            self._evict_if_needed()

        # assemble [B,C,H,W] on CPU then move to GPU
        out = torch.stack(out_list, dim=0)               # CPU
        return out.to(device, non_blocking=True)

    
class KDLoss():
    '''
    kd loss wrapper.
    '''

    def __init__(
        self,
        student,
        teacher,
        student_name,
        teacher_name,
        ori_loss,
        kd_method='kdt4',
        ori_loss_weight=1.0,
        kd_loss_weight=1.0,
        kd_loss_kwargs={},
        generative_prior_kwargs={},
        total_epochs=200
    ):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight

        self._teacher_out = None
        self._student_out = None

        self.generative_prior_kwargs=generative_prior_kwargs
        self.total_epochs=float(total_epochs)
        

        # init kd loss
        # module keys for distillation. '': output logits
        teacher_modules = ['',]
        student_modules = ['',]
        if kd_method == 'kd':
            self.kd_loss = KLDivergence(tau=4)
        elif kd_method == 'dist':
            self.kd_loss = DIST(beta=1, gamma=1, tau=1)
        elif kd_method.startswith('dist_t'):
            tau = float(kd_method[6:])
            self.kd_loss = DIST(beta=1, gamma=1, tau=tau)
        elif kd_method.startswith('kdt'):
            tau = float(kd_method[3:])
            self.kd_loss = KLDivergence(tau)
        elif kd_method == 'diffkd':
            # get configs
            ae_channels = kd_loss_kwargs.get('ae_channels', 1024)
            use_ae = kd_loss_kwargs.get('use_ae', True)
            tau = kd_loss_kwargs.get('tau', 1)

            print(kd_loss_kwargs)
            kernel_sizes = [3, 1]  # distillation on feature and logits
            student_modules = KD_MODULES[student_name]['modules'][:1]
            student_channels = KD_MODULES[student_name]['channels'][:1]
            teacher_modules = KD_MODULES[teacher_name]['modules'][:1]
            teacher_channels = KD_MODULES[teacher_name]['channels'][:1]
            self.diff = nn.ModuleDict()
            self.kd_loss = nn.ModuleDict()
            for tm, tc, sc, ks in zip(teacher_modules, teacher_channels, student_channels, kernel_sizes):
                self.diff[tm] = DiffKD(sc, tc, kernel_size=ks, use_ae=(ks!=1) and use_ae, ae_channels=ae_channels)
                self.kd_loss[tm] = nn.MSELoss() if ks != 1 else KLDivergence(tau=tau)
            self.diff.cuda()
            # add diff module to student for optimization
            self.student._diff = self.diff
        elif kd_method == 'mse':
            # distillation on feature
            student_modules = KD_MODULES[student_name]['modules'][:1]
            student_channels = KD_MODULES[student_name]['channels'][:1]
            teacher_modules = KD_MODULES[teacher_name]['modules'][:1]
            teacher_channels = KD_MODULES[teacher_name]['channels'][:1]
            self.kd_loss = nn.MSELoss()
            if student_channels[0] != teacher_channels[0]:
                self.align = nn.Conv2d(student_channels[0], teacher_channels[0], 1)
                self.align.cuda()
                # add align module to student for optimization
                self.student._align = self.align
        elif kd_method == 'TeKAP': 
            self.augmented_t = kd_loss_kwargs.get('augmented_t', 3)
            student_modules = KD_MODULES[student_name]['modules'][:1]
            student_channels = KD_MODULES[student_name]['channels'][:1]
            teacher_modules = KD_MODULES[teacher_name]['modules'][:1]
            teacher_channels = KD_MODULES[teacher_name]['channels'][:1]
            self.kd_loss = nn.MSELoss()
            if student_channels[0] != teacher_channels[0]:
                self.align = nn.Conv2d(student_channels[0], teacher_channels[0], 1)
                self.align.cuda()
                # add align module to student for optimization
                self.student._align = self.align
        elif kd_method == 'GVD' : #Generative Vicinity Distillation
            student_modules = KD_MODULES[student_name]['modules'][:1]
            student_channels = KD_MODULES[student_name]['channels'][:1]
            teacher_modules = KD_MODULES[teacher_name]['modules'][:1]
            teacher_channels = KD_MODULES[teacher_name]['channels'][:1]

            self.start_t = generative_prior_kwargs.get('uncertain_teacher_diffusion_steps', [50])[0]
            self.end_t = generative_prior_kwargs.get('uncertain_teacher_diffusion_steps', [30])[-1]
            self.start_g = generative_prior_kwargs.get('guidance_scale', [0.3])[0]
            self.end_g = generative_prior_kwargs.get('guidance_scale', [0.5])[-1]
            self.gen_prior = generative_prior_kwargs.get('gen_prior').cuda()

            self.add_logits_loss = bool(generative_prior_kwargs.get("add_logits_loss", False))
            self.is_reverse_sched = bool(generative_prior_kwargs.get("is_reverse_sched", False))

            self.teacher_tail= TailToLogits(teacher, tm=teacher_modules[0]).cuda().eval()
            for p in self.teacher_tail.parameters():
                p.requires_grad_(False)
            feature_guidance = bool(generative_prior_kwargs.get("feature_guidance", False))
            classifier_guidance = bool(generative_prior_kwargs.get("classifier_guidance", False))
            if hasattr(self.gen_prior, "module"):
                self.gen_prior.module.set_guidance_parameters(feature_guidance=feature_guidance, classifier_guidance=classifier_guidance, classifier_tail=self.teacher_tail) #DDP wrapper
            else:
                self.gen_prior.set_guidance_parameters(feature_guidance=feature_guidance, classifier_guidance=classifier_guidance, classifier_tail=self.teacher_tail)

            self.kd_loss = nn.MSELoss()
            self.kd_loss_logits = KLDivergence(tau=4)
            if student_channels[0] != teacher_channels[0]:
                self.align = nn.Conv2d(student_channels[0], teacher_channels[0], 1)
                self.align.cuda()
                # add align module to student for optimization
                self.student._align = self.align
        else:
            raise RuntimeError(f'KD method {kd_method} not found.')

        # register forward hook
        # dicts that store distillation outputs of student and teacher
        self._teacher_out = {}
        self._student_out = {}

        for student_module, teacher_module in zip(student_modules, teacher_modules):
            self._register_forward_hook(student, student_module, teacher=False)
            self._register_forward_hook(teacher, teacher_module, teacher=True)
        self.student_modules = student_modules
        self.teacher_modules = teacher_modules

        teacher.eval()
        self._iter = 0

    def __call__(self, x, targets, ep=0, idx=None):
        with torch.no_grad():
            t_logits = self.teacher(x)

        # compute ori loss of student
        logits = self.student(x)
        ori_loss = self.ori_loss(logits, targets)

        kd_loss = 0
        kd_loss_=0

        for tm, sm in zip(self.teacher_modules, self.student_modules):

            # transform student feature
            if self.kd_method == 'diffkd':
                self._student_out[sm], self._teacher_out[tm], diff_loss, ae_loss = \
                    self.diff[tm](self._reshape_BCHW(self._student_out[sm]), self._reshape_BCHW(self._teacher_out[tm]))
            if hasattr(self, 'align'):
                self._student_out[sm] = self.align(self._student_out[sm])

            # if spatial dimensions of teacher if greater than that of student make them equal
            if self._student_out[sm].shape[-2:] != self._teacher_out[tm].shape[-2:]:
                self._teacher_out[tm] = F.adaptive_avg_pool2d(
                    self._teacher_out[tm],
                    self._student_out[sm].shape[-2:]
                )

            # compute kd loss
            if isinstance(self.kd_loss, nn.ModuleDict):
                kd_loss_ = self.kd_loss[tm](self._student_out[sm], self._teacher_out[tm])
            else:
                if self.kd_method == 'GVD':
                    if self.is_reverse_sched:
                        grad_t = self.start_t - self.cosine_t_schedule(ep) 
                    else:
                        grad_t = self.cosine_t_schedule(ep) 
                    
                    grad_g = self.cosine_t_schedule(ep, start_t=self.start_g, end_t=self.end_g)

                    _raw_diff_samples, diff_samples, _latent_samples, _score = self.gen_prior(self._reshape_BCHW(self._teacher_out[tm]), sampling=True, t_start=grad_t, guidance_scale=grad_g, guidance_stop_t=0, targets=targets) 
                    kd_loss_ = self.kd_loss(self._student_out[sm], diff_samples)

                    #added logit loss condionally
                    if self.add_logits_loss:
                        tail = TailToLogits(self.teacher, tm).cuda().eval()
                        logits_diff = tail(diff_samples)
                        kd_loss_ +=  self.kd_loss_logits(logits, logits_diff)

                elif self.kd_method == 'TeKAP':
                    kd_loss_ = self.kd_loss(self._student_out[sm], self._teacher_out[tm])

                    for _ in range(self.augmented_t):
                        augmented_feature = randomize(self._teacher_out[tm])
                        kd_loss_ += self.kd_loss(self._student_out[sm], augmented_feature)

                    kd_loss_ /= (self.augmented_t + 1)
                else:
                    kd_loss_ = self.kd_loss(self._student_out[sm], self._teacher_out[tm])

            if self.kd_method == 'diffkd':
                # add additional losses in DiffKD
                if ae_loss is not None:
                    kd_loss += diff_loss + ae_loss
                    if self._iter % 50 == 0:
                        logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f} Diff loss: {diff_loss.item():.4f} AE loss: {ae_loss.item():.4f}')
                else:
                    kd_loss += diff_loss
                    if self._iter % 50 == 0:
                        logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f} Diff loss: {diff_loss.item():.4f}')
            else:
                if self._iter % 50 == 0:
                    logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f}')
            kd_loss += kd_loss_

        self._teacher_out = {}
        self._student_out = {}

        self._iter += 1
        return ori_loss * self.ori_loss_weight + kd_loss * self.kd_loss_weight, kd_loss

    def _register_forward_hook(self, model, name, teacher=False):
        if name == '':
            # use the output of model
            model.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))
        else:
            module = None
            for k, m in model.named_modules():
                if k == name:
                    module = m
                    break
            module.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))

    def _forward_hook(self, module, input, output, name, teacher=False):
        if teacher:
            self._teacher_out[name] = output[0] if isinstance(output, (tuple, list)) else output
        else:
            self._student_out[name] = output[0] if isinstance(output, (tuple, list)) else output


    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x
    
    def cosine_t_schedule(self, ep, decay_portion=0.99, start_t=None, end_t=None):
        start_t = self.start_t if start_t is None else start_t
        end_t = self.end_t if end_t is None else end_t

        denom = max(self.total_epochs - 1, 1)
        frac = min(ep / denom, 1.0)

        if frac >= decay_portion:
            return int(end_t)

        decay_frac = frac / max(decay_portion, 1e-12)

        cos_term = 0.5 * (1.0 + math.cos(math.pi * decay_frac))
        t_float = end_t + (start_t - end_t) * cos_term

        t = int(math.floor(t_float + 1e-9))
        lo, hi = sorted([start_t, end_t])

        return max(lo, min(t, hi))



class LatentDiffTrainLoss():
    '''
    latent diffsuion training loss wrapper.
    '''
    def __init__(
        self,
        latentDiffNetwork,
        teacher,
        teacher_name,
        generative_prior_kwargs,
        log_interval=10
    ):  
        diff_loss_weight = generative_prior_kwargs.get('diff_loss_weight', 1.0)
        ae_loss_weight = generative_prior_kwargs.get('ae_loss_weight', 1.0)

        self.diff = latentDiffNetwork.cuda()
        self.teacher=teacher
        self.diff_loss_weight = diff_loss_weight
        self.ae_loss_weight = ae_loss_weight
        self._teacher_out = None
        self.teacher_name=teacher_name
        self.log_interval= log_interval

        teacher_modules = DIFF_MODULES[teacher_name]['modules']
     
        # register forward hook
        # dicts that store outputs of the teacher
        self._teacher_out = {}
        for teacher_module in teacher_modules:
            self._register_forward_hook(teacher, teacher_module)
        self.teacher_modules = teacher_modules

        teacher.eval()
        self._iter = 0

    def __call__(self, x, targets):
        with torch.no_grad():
            t_logits = self.teacher(x)

        total_loss=0.0
        for tm in self.teacher_modules:
            diff_loss, ae_loss = self.diff(self._reshape_BCHW(self._teacher_out[tm]))

            if ae_loss is not None:
                total_loss += diff_loss + ae_loss
                if self._iter % self.log_interval == 0:
                    logger.info(f'Teacher Name ({self.teacher_name}) Layer name: ({tm}) total loss: {total_loss.item():.4f} Diff loss: {diff_loss.item():.4f} AE loss: {ae_loss.item():.4f}')
            else:
                total_loss += diff_loss
                if self._iter % self.log_interval == 0:
                    logger.info(f'Teacher Name ({self.teacher_name}) Layer name: ({tm}) total loss: {total_loss.item():.4f} Diff loss: {diff_loss.item():.4f}')    

        self._teacher_out = {}

        self._iter += 1
        return total_loss, t_logits, diff_loss

    def _register_forward_hook(self, model, name):
        if name == '':
            # use the output of model
            model.register_forward_hook(partial(self._forward_hook, name=name))
        else:
            module = None
            for k, m in model.named_modules():
                if k == name:
                    module = m
                    break
            module.register_forward_hook(partial(self._forward_hook, name=name))

    def _forward_hook(self, module, input, output, name):
        # for other layers: if (tensor, aux, ...) take the first tensor-like output
        out = output[0] if isinstance(output, (tuple, list)) else output

        self._teacher_out[name] = out

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x
    


class LatentDiffSampleLoss():
    '''
    latent diffsuion training loss wrapper.
    '''
    def __init__(
        self,
        latentDiffNetwork,
        teacher,
        teacher_name,
        generative_prior_kwargs,
        log_interval=10,
        reconstruction_loss = nn.MSELoss()
    ):  
        self.teacher=teacher.cuda()
        self.teacher_name=teacher_name

        # which teacher modules to hook
        teacher_modules = DIFF_MODULES[teacher_name]["modules"]
        self.teacher_modules = teacher_modules

        # storage for hooked outputs
        self._teacher_out = {}

        # register hooks
        for layer_name in teacher_modules:
            self._register_forward_hook(self.teacher, layer_name)

        teacher.eval()
        self._iter = 0
        self.log_interval= log_interval

        self.t_start = int(generative_prior_kwargs.get("t_start", 300))
        self.reconstruction_loss=reconstruction_loss
        
        self.teacher_tail= TailToLogits(teacher, tm=teacher_modules[0]).cuda().eval()
        for p in self.teacher_tail.parameters():
            p.requires_grad_(False)
        self.diff = latentDiffNetwork.cuda()
        feature_guidance = bool(generative_prior_kwargs.get("feature_guidance", False))
        classifier_guidance = bool(generative_prior_kwargs.get("classifier_guidance", False))
        self.guidance_scale =generative_prior_kwargs.get('guidance_scale', [0.5])[-1]
        if hasattr(self.diff, "module"):
            self.diff.module.set_guidance_parameters(feature_guidance=feature_guidance, classifier_guidance=classifier_guidance, classifier_tail=self.teacher_tail) #DDP wrapper
        else:
            self.diff.set_guidance_parameters(feature_guidance=feature_guidance, classifier_guidance=classifier_guidance, classifier_tail=self.teacher_tail)

    def __call__(self, x, targets=None):
        # forward teacher and collect features
        self._teacher_out = {}  # clear before forward to avoid stale data
        with torch.no_grad():
            t_logits_init = self.teacher(x)

        diff_samples_all = []
        latent_samples_all = []
        teacher_features_all = []
        reconst_loss=0.0
     
        for tm in self.teacher_modules:
            stop_t=0
            raw_diff_samples, _diff_samples, _latent_samples, _score = self.diff(self._reshape_BCHW(self._teacher_out[tm]), sampling=True, t_start=self.t_start, guidance_scale=self.guidance_scale,  guidance_stop_t=stop_t,targets=targets, eta=0.0,eta_seed=0)
            reconst_loss += self.reconstruction_loss(self._teacher_out[tm], _diff_samples)
        
        # aggregate (supports tensor or list/tuple of tensors)
            self.append_all(diff_samples_all, _diff_samples)
            self.append_all(latent_samples_all, _latent_samples)
            self.append_all(teacher_features_all, self._teacher_out[tm])
       

        #intialize a tail model to test the quality of the diffsuion generated samples (for the first feature extraction layer)
        with torch.no_grad():
            t_logits_reconstructed = self.teacher_tail(_diff_samples)

        # clear to free refs
        self._teacher_out = {}
        self._iter += 1
        print("reconst_loss", reconst_loss)
        return diff_samples_all, latent_samples_all, teacher_features_all, t_logits_init, t_logits_reconstructed, reconst_loss

    def _register_forward_hook(self, model, name):
        if name == '':
            # use the output of model
            model.register_forward_hook(partial(self._forward_hook, name=name))
        else:
            module = None
            for k, m in model.named_modules():
                if k == name:
                    module = m
                    break
            module.register_forward_hook(partial(self._forward_hook, name=name))

    def _forward_hook(self, module, input, output, name):
        # for other layers: if (tensor, aux, ...) take the first tensor-like output
        out = output[0] if isinstance(output, (tuple, list)) else output

        self._teacher_out[name] = out

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x
    
    def append_all(self, container, data):
        if isinstance(data, (list, tuple)):
            container.extend(data)
        else:
            container.append(data)



############################

