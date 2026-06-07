import torch
import torch.nn as nn
from typing import List, Union
from torch.nn.parallel import DistributedDataParallel as DDP

DIFF_MODULES = {
    'cifar_MobileNetV2': dict(modules=['conv2'], channels=[1280]),
    'cifar_ResNet50': dict(modules=['layer4'], channels=[2048]),
    'cifar_ShuffleV1': dict(modules=['layer2'], channels=[480]),
    'cifar_ShuffleV2': dict(modules=['layer2'], channels=[232]),
    'cifar_resnet32x4': dict(modules=['layer3'], channels=[256]),
    'cifar_resnet8x4': dict(modules=['layer3'], channels=[256]),
    'cifar_wrn_40_1': dict(modules=['relu'], channels=[64]),
    'cifar_wrn_40_2': dict(modules=['relu'], channels=[128]),
    'cifar_resnet56': dict(modules=['layer3'], channels=[64]),
    'cifar_resnet20': dict(modules=['layer3'], channels=[64]),
    'cifar_resnet110': dict(modules=['layer3'], channels=[64]),
    'tv_resnet50': dict(modules=['layer4'], channels=[2048]),
    'tv_resnet34': dict(modules=['layer4'], channels=[512]),
    'tv_resnet18': dict(modules=['layer4'], channels=[512]),
    'resnet18': dict(modules=['layer4'], channels=[512]),
    'tv_mobilenet_v2': dict(modules=['features.18'], channels=[1280]),
    'nas_model': dict(modules=['features.conv_out'], channels=[1280]),  # mbv2
    'timm_tf_efficientnet_b0': dict(modules=['conv_head'], channels=[1280]),
    'mobilenet_v1': dict(modules=['model.13', 'fc'], channels=[1024]),
    'timm_swin_large_patch4_window7_224': dict(modules=['norm'], channels=[1536]),
    'timm_swin_tiny_patch4_window7_224': dict(modules=['norm'], channels=[768]),

    #tinyimagenet
    'tiny_tv_resnet50': dict(modules=['layer4'], channels=[2048]),
    'tiny_tv_resnet34': dict(modules=['layer4'], channels=[512]),
    'tiny_tv_resnet18': dict(modules=['layer4'], channels=[512]),
    'tiny_mobilenet_v1': dict(modules=['feat'], channels=[1024]),
    
    #ts models
    'ts_lstm_100_3': dict(modules=['seq_feat'], channels=[100]), #all hidden layers
    'ts_lstm_32_2': dict(modules=['seq_feat'], channels=[32]), #all hidden layers
    'ts_lstm_8_1': dict(modules=['seq_feat'], channels=[8]), #all hidden layers
    'ts_lstm': dict(modules=['seq_feat'], channels=[100]), #all hidden layers
    'ts_resnet_64_3': dict(modules=['feat'], channels=[128]), 
    'ts_resnet_8_3': dict(modules=['feat'], channels=[16]), 
    'ts_inception_32_6': dict(modules=['feat'], channels=[32]), 
    'ts_inception_16_3': dict(modules=['feat'], channels=[16]), 
    'ts_inception_8_3': dict(modules=['feat'], channels=[8]), 
    # 'ts_lstm_100_3': dict(modules=['feat'], channels=[100]), #final hidden layer
}

def _unwrap_ddp(m: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped by DDP."""
    return m.module if isinstance(m, DDP) else m

import torch
import torch.nn as nn
from typing import List, Union


class TailToLogits(nn.Module):
    """
    Maps activation at feature tap `tm` -> final logits.
    Reuses teacher modules (no copying; shared weights).
    Works with DDP-wrapped models.

    Supports:
    - CNN taps like layer3 / layer4 / features.18
    - old RNN tap 'rnn'
    - new LSTM taps:
        * 'seq_feat' : [B, H, T, 1]
        * 'feat'     : [B, H, 1, 1]
    """

    def __init__(self, teacher: nn.Module, tm: str):
        super().__init__()
        self.teacher = teacher
        self.base = _unwrap_ddp(teacher)
        self.tm = tm
        self.top_tm = _get_top_name(tm)

        # ensure tm top exists
        _ = self.base.get_submodule(self.top_tm)

        ops: List[Union[nn.Module, str]] = []

        # ---- New LSTM special cases ----
        # safest: delegate back to model's own tail helpers
        if self.top_tm == "seq_feat":
            self.ops = ["LSTM_SEQ_TO_LOGITS"]
            return

        if self.top_tm == "feat":
            self.ops = ["LSTM_FEAT_TO_LOGITS"]
            return

        # ---- Old RNN special case: tm='rnn' ----
        if self.top_tm == "rnn":
            ops.append("RNN_LAST_STEP")
            started = False
            for name, child in self.base.named_children():
                if name == self.top_tm:
                    started = True
                    continue
                if not started:
                    continue
                ops.append(child)  # include fc
            self.ops = ops
            return

        # ---- If tm is nested like layer3.0 or features.18, run remaining inside that Sequential ----
        top_mod = self.base.get_submodule(self.top_tm)
        subpath = _split_qualname(tm)[1:]  # everything after the top module name
        if subpath:
            ops.extend(_slice_sequential_after(top_mod, subpath))

        # ---- Then run remaining TOP-LEVEL children after top_tm (INCLUDING classifier) ----
        started = False
        for name, child in self.base.named_children():
            if name == self.top_tm:
                started = True
                continue
            if not started:
                continue
            ops.append(child)

        self.ops = ops

    def forward(self, feat):
        x = feat

        for op in self.ops:
            # ---- New LSTM hook-friendly taps ----
            if op == "LSTM_SEQ_TO_LOGITS":
                if not hasattr(self.base, "logits_from_seq_feat"):
                    raise AttributeError(
                        f"Model {type(self.base).__name__} does not implement logits_from_seq_feat"
                    )
                return self.base.logits_from_seq_feat(x)

            if op == "LSTM_FEAT_TO_LOGITS":
                if not hasattr(self.base, "logits_from_feat"):
                    raise AttributeError(
                        f"Model {type(self.base).__name__} does not implement logits_from_feat"
                    )
                return self.base.logits_from_feat(x)

            # ---- Old RNN special case ----
            if op == "RNN_LAST_STEP":
                if isinstance(x, (tuple, list)):
                    x = x[0]
                # assume batch_first [B, T, H]
                if x.dim() == 3:
                    x = x[:, -1, :]
                continue

            if isinstance(op, nn.Linear) and x.dim() > 2:
                # WRN special handling (tap = relu)
                if self.tm == "relu" and hasattr(self.base, "logits_from_relu"):
                    return self.base.logits_from_relu(x)

                # fallback (other models)
                x = torch.flatten(x, 1)

            # apply module
            x = op(x)

            # auto flatten when entering classifier parts
            if isinstance(op, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)) and x.dim() == 4:
                x = torch.flatten(x, 1)

        return x

# class TailToLogits(nn.Module):
    # """
    # Maps activation at feature tap `tm` -> final logits.
    # Reuses teacher modules (no copying; shared weights).
    # Works with DDP-wrapped models.
    # """

    # def __init__(self, teacher: nn.Module, tm: str):
    #     super().__init__()
    #     self.teacher = teacher          # keep original (could be DDP)
    #     self.base = _unwrap_ddp(teacher)  # use for module introspection
    #     self.tm = tm
    #     self.top_tm = _get_top_name(tm)

    #     # ensure tm top exists (on the *base* model, not DDP wrapper)
    #     _ = self.base.get_submodule(self.top_tm)

    #     ops: List[Union[nn.Module, str]] = []

    #     # ---- RNN special case: tm='rnn' ----
    #     if self.top_tm == "rnn":
    #         # hook output may be (output, (h,c)) or output
    #         # then we take last time-step and run remaining children INCLUDING fc
    #         ops.append("RNN_LAST_STEP")
    #         started = False
    #         for name, child in self.base.named_children():
    #             if name == self.top_tm:
    #                 started = True
    #                 continue
    #             if not started:
    #                 continue
    #             ops.append(child)  # include fc
    #         self.ops = ops
    #         return

    #     # ---- If tm is nested like layer3.0 or features.18, run remaining inside that Sequential ----
    #     top_mod = self.base.get_submodule(self.top_tm)
    #     subpath = _split_qualname(tm)[1:]  # everything after the top module name
    #     if subpath:
    #         ops.extend(_slice_sequential_after(top_mod, subpath))

    #     # ---- Then run remaining TOP-LEVEL children after top_tm (INCLUDING classifier) ----
    #     started = False
    #     for name, child in self.base.named_children():
    #         if name == self.top_tm:
    #             started = True
    #             continue
    #         if not started:
    #             continue
    #         ops.append(child)

    #     self.ops = ops

    # def forward(self, feat):
    #     x = feat
    #     for op in self.ops:
    #         if op == "RNN_LAST_STEP":
        #         if isinstance(x, (tuple, list)):
        #             x = x[0]
        #         # assume batch_first [B,T,H]
        #         if x.dim() == 3:
        #             x = x[:, -1, :]
        #         continue

        #     # apply module
        #     x = op(x)

        #     # auto flatten when entering classifier parts
        #     if isinstance(op, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)) and x.dim() == 4:
        #         x = torch.flatten(x, 1)
        #     if isinstance(op, nn.Linear) and x.dim() > 2:
        #         x = torch.flatten(x, 1)

        # return x



def _get_top_name(qualname: str) -> str:
    return qualname.split(".")[0] if qualname else qualname

def _split_qualname(qualname: str) -> List[str]:
    return qualname.split(".") if qualname else []

def _is_int(s: str) -> bool:
    try:
        int(s); return True
    except Exception:
        return False

def _slice_sequential_after(mod: nn.Module, subpath: List[str]) -> List[nn.Module]:
    """
    Supports tm like 'features.18' or 'model.13' when the top module is nn.Sequential.
    Returns modules AFTER that index.
    """
    if not subpath:
        return []
    if isinstance(mod, nn.Sequential) and _is_int(subpath[0]):
        idx = int(subpath[0])
        return list(mod.children())[idx + 1 :]
    raise ValueError(
        f"Unsupported nested tm='{'.'.join(subpath)}' inside {type(mod)}. "
        f"Supported: <sequential>.<int> like features.18 or model.13. "
        f"For tm like features.conv_out, use tm='features' for the tail (see note)."
    )

# class TailToLogitsbk(nn.Module):
#     """
#     Maps activation at feature tap `tm` -> final logits.
#     Reuses teacher modules (no copying; shared weights).
#     """

#     def __init__(self, teacher: nn.Module, tm: str):
#         super().__init__()
#         self.teacher = teacher
#         self.tm = tm
#         self.top_tm = _get_top_name(tm)

#         # ensure tm top exists
#         _ = self.teacher.get_submodule(self.top_tm)

#         ops: List[Union[nn.Module, str]] = []

#         # ---- RNN special case: tm='rnn' ----
#         if self.top_tm == "rnn":
#             # hook output may be (output, (h,c)) or output
#             # then we take last time-step and run remaining children INCLUDING fc
#             ops.append("RNN_LAST_STEP")
#             started = False
#             for name, child in self.teacher.named_children():
#                 if name == self.top_tm:
#                     started = True
#                     continue
#                 if not started:
#                     continue
#                 ops.append(child)  # include fc
#             self.ops = ops
#             return

#         # ---- If tm is nested like features.18 or model.13, run remaining inside that Sequential ----
#         top_mod = self.teacher.get_submodule(self.top_tm)
#         subpath = _split_qualname(tm)[1:]
#         if subpath:
#             ops.extend(_slice_sequential_after(top_mod, subpath))

#         # ---- Then run remaining TOP-LEVEL children after top_tm (INCLUDING classifier) ----
#         started = False
#         for name, child in self.teacher.named_children():
#             if name == self.top_tm:
#                 started = True
#                 continue
#             if not started:
#                 continue
#             ops.append(child)

#         self.ops = ops

#     def forward(self, feat):
#         x = feat
#         for op in self.ops:
#             if op == "RNN_LAST_STEP":
#                 if isinstance(x, (tuple, list)):
#                     x = x[0]
#                 # assume batch_first [B,T,H]
#                 if x.dim() == 3:
#                     x = x[:, -1, :]
#                 continue

#             # apply module
#             x = op(x)

#             # auto flatten when entering Linear / classifier
#             if isinstance(op, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)) and x.dim() == 4:
#                 x = torch.flatten(x, 1)
#             if isinstance(op, nn.Linear) and x.dim() > 2:
#                 x = torch.flatten(x, 1)

#         return x
