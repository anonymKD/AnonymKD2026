import yaml
import torch
from torch import nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional, cast
import sys
import re

__all__ = ['ts_lstm', 'ts_lstm_32_2', 'ts_lstm_100_3', 'ts_lstm_8_1', 'ts_resnet_64_3', 'ts_resnet_16_3', 'ts_resnet_8_3'
           , 'ts_resnet_16_2', 'ts_resnet_8_2', 'ts_inception_32_6', 'ts_inception_16_3', 'ts_inception_8_3' ]

##################LSTM implementation
class LSTMClassifier(nn.Module):
    """
    LSTM-based time-series classifier with hook-friendly feature taps.

    Exposed feature taps:
    - seq_feat : [B, H, T, 1]   full sequence feature for diffusion/KD
    - feat     : [B, H, 1, 1]   final-step feature for diffusion/KD

    Tail utilities:
    - logits_from_seq_feat(seq_feat)
    - logits_from_feat(feat)
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True
        )

        # identity taps for forward hooks
        self.seq_feat = nn.Identity()
        self.feat = nn.Identity()

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, get_ha=False):
        """
        x: [B, T, input_dim]
        returns logits: [B, num_classes]
        """
        h0, c0 = self.init_hidden(x)

        # out_t: [B, T, H]
        # hn:    [num_layers, B, H]
        # cn:    [num_layers, B, H]
        out_t, (hn, cn) = self.rnn(x, (h0, c0))

        # final vector actually used by classifier
        last_vec = out_t[:, -1, :]   # [B, H]

        # hook-friendly full sequence feature
        # [B, T, H] -> [B, H, T, 1]
        seq_feat = out_t.transpose(1, 2).unsqueeze(-1)
        seq_feat = self.seq_feat(seq_feat)

        # hook-friendly final feature
        # [B, H] -> [B, H, 1, 1]
        feat = last_vec.unsqueeze(-1).unsqueeze(-1)
        feat = self.feat(feat)

        logits = self.fc(last_vec)

        if get_ha:
            return logits, out_t, hn, cn
        return logits

    def logits_from_seq_feat(self, seq_feat):
        """
        seq_feat: [B, H, T, 1]
        Reconstruct logits using the same classifier tail logic as forward().
        """
        if seq_feat.dim() != 4:
            raise ValueError(f"seq_feat must have shape [B, H, T, 1], got {seq_feat.shape}")

        out_t = seq_feat.squeeze(-1).transpose(1, 2)   # [B, T, H]
        last_vec = out_t[:, -1, :]                     # [B, H]
        logits = self.fc(last_vec)
        return logits

    def logits_from_feat(self, feat):
        """
        feat: [B, H, 1, 1]
        Reconstruct logits using the same classifier tail logic as forward().
        """
        if feat.dim() != 4:
            raise ValueError(f"feat must have shape [B, H, 1, 1], got {feat.shape}")

        last_vec = feat.squeeze(-1).squeeze(-1)        # [B, H]
        logits = self.fc(last_vec)
        return logits

    def init_hidden(self, x):
        """
        x: [B, T, input_dim]
        returns h0, c0 each of shape [num_layers, B, H]
        """
        device = x.device
        batch_size = x.size(0)

        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device)
        return h0, c0
    

def ts_lstm(n_chans, num_classes, kwargs) -> LSTMClassifier:
    return LSTMClassifier(input_dim=n_chans, hidden_dim=kwargs['hidden_size'], layer_dim=kwargs['num_layers'], output_dim=num_classes)

def ts_lstm_32_2(n_chans, num_classes, _kwargs) -> LSTMClassifier:
    return LSTMClassifier(input_dim=n_chans, hidden_dim=32, layer_dim=2, output_dim=num_classes)

def ts_lstm_8_1(n_chans, num_classes, _kwargs) -> LSTMClassifier:
    return LSTMClassifier(input_dim=n_chans, hidden_dim=8, layer_dim=1, output_dim=num_classes)

def ts_lstm_100_3(n_chans, num_classes, _kwargs) -> LSTMClassifier:
    return LSTMClassifier(input_dim=n_chans, hidden_dim=100, layer_dim=3, output_dim=num_classes)


################# Resnet implementation
class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)

class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)
    
class ResNetBlock2(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels]
        kernel_sizes = [5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x) 
       
class ResNetBaseline(nn.Module):
    """
    ResNet baseline with hook-friendly feature tap.

    Exposed feature taps:
    - feat : [B, C, T]     final convolutional feature map (x3)

    Tail utilities:
    - logits_from_feat(feat)
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.block1 = ResNetBlock(in_channels=in_channels, out_channels=mid_channels)
        self.block2 = ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2)
        self.block3 = ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2)

        # hook-friendly tap
        self.feat = nn.Identity()

        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False, chan_last=True) -> torch.Tensor:
        """
        x: [B, in_channels, T]
        returns logits: [B, num_classes]
        """
        if chan_last:
            x = x.transpose(1, 2)   # [B, T, C] -> [B, C, T]

        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)          # [B, C, T]

        # pass through identity tap so hooks can attach here
        feat = x3.unsqueeze(-1)           # [B, C, T, 1]
        feat = self.feat(feat)

        logits = self.logits_from_feat(feat)

        if get_ha:
            return logits, x1, x2, x3
        return logits

    def logits_from_feat(self, feat):
        if feat.dim() != 4:
            raise ValueError(f"feat must have shape [B, C, T, 1], got {feat.shape}")

        x = feat.squeeze(-1)      # [B, C, T]
        x = x.mean(dim=-1)        # [B, C]
        logits = self.final(x)
        return logits

class ResNetBaseline2(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.block1 = ResNetBlock2(in_channels=in_channels, out_channels=mid_channels)
        self.block2 = ResNetBlock2(in_channels=mid_channels, out_channels=mid_channels * 2)

        # hook-friendly tap
        self.feat = nn.Identity()

        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False, chan_last=True) -> torch.Tensor:  # type: ignore
        """
        x: [B, in_channels, T]
        returns logits: [B, num_classes]
        """
        if chan_last:
            x = x.transpose(1, 2)   # [B, T, C] -> [B, C, T]
    
        x1=self.block1(x)
        x2=self.block2(x1)

        # pass through identity tap so hooks can attach here
        feat = x2.unsqueeze(-1)           # [B, C, T, 1]
        feat = self.feat(feat)

        logits = self.logits_from_feat(feat)
        
        if get_ha:
            return logits,x1, x2
        return logits
    
    def logits_from_feat(self, feat):
        if feat.dim() != 4:
            raise ValueError(f"feat must have shape [B, C, T, 1], got {feat.shape}")

        x = feat.squeeze(-1)      # [B, C, T]
        x = x.mean(dim=-1)        # [B, C]
        logits = self.final(x)
        return logits
    
def ts_resnet_64_3(n_chans, num_classes, _kwargs) -> ResNetBaseline:
    return ResNetBaseline(in_channels=n_chans, mid_channels=64, num_pred_classes=num_classes)

def ts_resnet_16_3(n_chans, num_classes, _kwargs) -> ResNetBaseline:
    return ResNetBaseline(in_channels=n_chans, mid_channels=16, num_pred_classes=num_classes)

def ts_resnet_8_3(n_chans, num_classes, _kwargs) -> ResNetBaseline:
    return ResNetBaseline(in_channels=n_chans, mid_channels=8, num_pred_classes=num_classes)

def ts_resnet_16_2(n_chans, num_classes, _kwargs) -> ResNetBaseline2:
    return ResNetBaseline2(in_channels=n_chans, mid_channels=16, num_pred_classes=num_classes)

def ts_resnet_8_2(n_chans, num_classes, _kwargs) -> ResNetBaseline2:
    return ResNetBaseline2(in_channels=n_chans, mid_channels=8, num_pred_classes=num_classes)



#################### InceptionTime
class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 residual: bool, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41) -> None:
        assert kernel_size > 3, "Kernel size must be strictly greater than 3"
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels,
                                                kernel_size=1, bias=False)
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_size_s[i], stride=stride, bias=False)
            for i in range(len(kernel_size_s))
        ])

        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])

    # def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
    #     org_x = x
    #     if self.use_bottleneck:
    #         x = self.bottleneck(x)
    #     x = self.conv_layers(x)

    #     if self.use_residual:
    #         x = x + self.residual(org_x)
    #     return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_x = x

        if self.use_bottleneck:
            x = self.bottleneck(x)

        x = self.conv_layers(x)
        x = self.batchnorm(x)

        if self.use_residual:
            x = x + self.residual(org_x)

        x = self.relu(x)
        return x

class InceptionModel(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    in_channels:
        The number of input channels (i.e. input.shape[-1])
    out_channels:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_blocks: int, in_channels: int, out_channels: Union[List[int], int],
                 bottleneck_channels: Union[List[int], int], kernel_sizes: Union[List[int], int],
                 use_residuals: Union[List[bool], bool, str] = 'default',
                 num_pred_classes: int = 1
                 ) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'num_blocks': num_blocks,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'num_pred_classes': num_pred_classes
        }

        channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
                                                                          num_blocks))
        bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
                                                                     num_blocks))
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks)
        )

        self.blocks = nn.Sequential(*[
            InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                           residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                           kernel_size=kernel_sizes[i]) for i in range(num_blocks)
        ])

        # hook-friendly tap
        self.feat = nn.Identity()

        self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes)

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor, get_ha=False, chan_last=True) -> torch.Tensor:
        if chan_last:
            x = x.transpose(1, 2)   # [B, T, C] -> [B, C, T]

        feat = self.blocks(x)               # [B, C_out, T]
        feat = feat.unsqueeze(-1)           # [B, C_out, T, 1]
        feat = self.feat(feat)

        logits = self.logits_from_feat(feat)

        if get_ha:
            return logits, feat
        return logits

    def logits_from_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 4:
            raise ValueError(f"feat must have shape [B, C, T, 1], got {feat.shape}")

        x = feat.squeeze(-1)      # [B, C, T]
        x = x.mean(dim=-1)        # [B, C]
        logits = self.linear(x)
        return logits

   
def ts_inception_32_6(n_chans, num_classes, _kwargs) -> InceptionModel:
    return InceptionModel(num_blocks=6, in_channels=n_chans, out_channels=32, bottleneck_channels=32, kernel_sizes=[9, 19, 39, 9, 19, 39], use_residuals=True, num_pred_classes=num_classes)

def ts_inception_16_3(n_chans, num_classes, _kwargs) -> InceptionModel:
    return InceptionModel(num_blocks=3, in_channels=n_chans, out_channels=16, bottleneck_channels=16, kernel_sizes=[9, 19, 39], use_residuals=True, num_pred_classes=num_classes)

def ts_inception_8_3(n_chans, num_classes, _kwargs) -> InceptionModel:
    return InceptionModel(num_blocks=3, in_channels=n_chans, out_channels=8, bottleneck_channels=8, kernel_sizes=[9, 19, 39], use_residuals=True, num_pred_classes=num_classes)



class FCNBaseline(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        #added for intermediate feature extarction
        self.layer1 = ConvBlock(in_channels, 128, 8, 1)
        self.layer2 = ConvBlock(128, 256, 5, 1)
        self.layer3 =  ConvBlock(256, 128, 3, 1)

        # self.linear1 = nn.linear(128,128)# to get linear embeddings
        self.final = nn.Linear(128, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False) -> torch.Tensor:  # type: ignore
        # x = self.layers(x)
        x1= self.layer1(x)
        x2= self.layer2(x1)
        x3= self.layer3(x2)
        # x4= self.linear1(x3)
        linear = self.final(x3.mean(dim=-1))

        if get_ha:
            return x1, x2, x3, x3.mean(dim=-1),linear
            
        return linear
        
class FCNBaselineSmall(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int =8, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        #added for intermediate feature extarction
        self.layer1 = ConvBlock(in_channels, mid_channels, 8, 1)
        self.layer2 = ConvBlock(mid_channels, mid_channels*2, 5, 1)
        self.layer3 =  ConvBlock(mid_channels*2, mid_channels, 3, 1)
        self.final = nn.Linear(mid_channels, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False) -> torch.Tensor:  # type: ignore
        # x = self.layers(x)
        x1= self.layer1(x)
        x2= self.layer2(x1)
        x3= self.layer3(x2)
        linear = self.final(x3.mean(dim=-1))

        if get_ha:
            return linear, x1, x2, x3
            
        return linear

class FCNBaselineSmall2(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int =8, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        #added for intermediate feature extarction
        self.layer1 = ConvBlock(in_channels, mid_channels, 5, 1)
        self.layer2 = ConvBlock(mid_channels, mid_channels*2, 3, 1)
        # self.layer3 =  ConvBlock(mid_channels*2, mid_channels, 3, 1)
        self.final = nn.Linear(mid_channels*2, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False) -> torch.Tensor:  # type: ignore
        # x = self.layers(x)
        x1= self.layer1(x)
        x2= self.layer2(x1)
        # x3= self.layer3(x2)
        linear = self.final(x2.mean(dim=-1))

        if get_ha:
            return linear, x1, x2, 0
            
        return linear


class LSTMClassifierTeacherUnrolled(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn1 = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.rnn2 = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        self.rnn3 = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x, get_ha=False):
        h0, c0 = self.init_hidden(x)
        out_t1, (hn1, cn1) = self.rnn1(x, (h0, c0))
        out_t2, (hn2, cn2) = self.rnn2(out_t1, (hn1, cn1))
        out_t3, (hn3, cn3) = self.rnn3(out_t2, (hn2, cn2))
        out = self.fc(out_t3[:, -1, :])
        if get_ha:
            return out, out_t1, out_t2, out_t3
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]



class FCNBaseline_orig(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels, 128, 8, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        ])
        self.final = nn.Linear(128, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return self.final(x.mean(dim=-1))