"""
EAT: End-to-End Audio Transformer Model
Code:
https://github.com/Alibaba-MIIL/AudioClassfication/blob/main/modules/soundnet.py

Gazneli, Avi, et al.
"End-to-end audio strikes back: Boosting augmentations
towards an efficient audio classification network."
arXiv preprint arXiv:2204.11479 (2022).
"""
from typing import List
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1dTF(nn.Module):
    """
    Residual block for the EAT model.

    Args:
        dim (int): Input dimension.
        dilation (int, optional): Dilation factor. Defaults to 1.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
    """

    def __init__(self, dim, dilation=1, kernel_size=3):
        super().__init__()
        self.block_t = nn.Sequential(
            nn.ReflectionPad1d(dilation * (kernel_size // 2)),
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                stride=1,
                bias=False,
                dilation=dilation,
                groups=dim,
            ),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True),
        )
        self.block_f = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True),
        )
        self.shortcut = nn.Conv1d(dim, dim, 1, 1)

    def forward(self, x):
        return self.shortcut(x) + self.block_f(x) + self.block_t(x)


class TransformerAggregate(nn.Module):
    """
    Applies a Transformer encoder to a sequence of embeddings optionally followed by a
    linear layer to project to a classification output.

    Args:
        clip_length (int): Length of the input clip.
        embed_dim (int, optional): Embedding dimension. Defaults to 64.
        n_layers (int, optional): Number of layers in the Transformer encoder.
            Defaults to 6.
        nhead (int, optional): Number of heads in the Transformer encoder.
            Defaults to 6.
        n_classes (int, optional): Number of classes in the classification task.
            Defaults to None. If None, no linear layer is applied.
        dim_feedforward (int, optional): Dimension of the feedforward layer in the
            Transformer encoder. Defaults to 512.
    """

    def __init__(
        self,
        clip_length=None,
        embed_dim=64,
        n_layers=6,
        nhead=6,
        n_classes=None,
        dim_feedforward=512,
    ):
        super(TransformerAggregate, self).__init__()
        self.num_tokens = 1
        drop_rate = 0.1
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            activation="gelu",
            dim_feedforward=dim_feedforward,
            dropout=drop_rate,
        )
        self.transformer_enc = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, clip_length + self.num_tokens, embed_dim)
        )
        if n_classes is not None:
            self.fc = nn.Linear(embed_dim, n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(m.weight, 1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.orthogonal_(m.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)
        if hasattr(self, "fc"):
            pred = self.fc(o[0])
        else:
            pred = o[0]
        return pred


class AntiAliasDownsample(nn.Module):
    """
    Implements a downsample layer with a triangular anti-aliasing filter.
    From: https://arxiv.org/pdf/1904.11486.pdf

    Args:
        filt_size (int): Filter size
        stride (int): Stride
        channels (int): Number of channels
    """

    def __init__(self, filt_size=3, stride=2, channels=None):
        super(AntiAliasDownsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        ha = torch.arange(1, filt_size // 2 + 1 + 1, 1)
        a = torch.cat(
            (
                ha,
                ha.flip(
                    dims=[
                        -1,
                    ]
                )[1:],
            )
        ).float()
        a = a / a.sum()
        filt = a[None, :]
        self.register_buffer("filt", filt[None, :, :].repeat((self.channels, 1, 1)))

    def forward(self, x):
        x_pad = F.pad(x, (self.filt_size // 2, self.filt_size // 2), "reflect")
        y = F.conv1d(x_pad, self.filt, stride=self.stride, padding=0, groups=x.shape[1])
        return y


class Down(nn.Module):
    """
    Downsample block

    Args:
        channels (int): Number of input channels
        d (int): Downsampling factor
        k (int): Filter size
    """

    def __init__(self, channels, d=2, k=3):
        super().__init__()
        kk = d + 1
        self.down = nn.Sequential(
            nn.ReflectionPad1d(kk // 2),
            nn.Conv1d(channels, channels * 2, kernel_size=kk, stride=1, bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.LeakyReLU(0.2, True),
            AntiAliasDownsample(channels=channels * 2, stride=d, filt_size=k),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class EAT(nn.Module):
    """
    EAT: End-to-End Audio Transformer

    Args:
        input_length (int): Length of the input audio clips
        nf (int): Number of filters in the first layer
        embed_dim (int): Embedding dimension output by the transformer
        n_layers (int): Number of transformer layers
        nhead (int): Number of attention heads
        dim_feedforward (int): Dimension of the feedforward layer in the transformer
        factors (list): Downsampling factors for each ResBlock
        n_classes (int): Number of classes in the dataset
    """

    def __init__(
        self,
        input_length: int,
        nf: int = 16,
        embed_dim: int = 128,
        n_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 512,
        factors: List[int] = [4, 4, 4, 4],
        n_classes: Optional[int] = None,
    ):
        super().__init__()
        self.down_factors = factors

        # Calculate the clip length from the input length and the factors
        clip_length = (input_length // (np.prod(factors) * 4)) + 1
        self.clip_length = clip_length

        # Model requires a specific minimum input length based on the downsampling
        # factors and the padding applied during the last ResBlock
        min_length = EAT.minimum_input_length(self.down_factors)
        if input_length < min_length:
            raise ValueError(
                f"Input length must be at least {min_length} for this "
                f"model. Received {input_length}"
            )

        model = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(1, nf, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, True),
        ]
        self.start = nn.Sequential(*model)
        model = []
        for i, f in enumerate(self.down_factors):
            model += [Down(channels=nf, d=f, k=f * 2 + 1)]
            nf *= 2
            if i % 2 == 0:
                model += [ResBlock1dTF(dim=nf, dilation=1, kernel_size=15)]
        self.down = nn.Sequential(*model)

        factors = [2, 2]
        model = []
        for _, f in enumerate(factors):
            for i in range(1):
                for j in range(3):
                    model += [ResBlock1dTF(dim=nf, dilation=3**j, kernel_size=15)]
            model += [Down(channels=nf, d=f, k=f * 2 + 1)]
            nf *= 2
        self.down2 = nn.Sequential(*model)
        self.project = nn.Conv1d(nf, embed_dim, 1)

        self.tf = TransformerAggregate(
            embed_dim=embed_dim,
            clip_length=clip_length,
            n_layers=n_layers,
            nhead=nhead,
            n_classes=n_classes,
            dim_feedforward=dim_feedforward,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.start(x)
        x = self.down(x)
        x = self.down2(x)
        x = self.project(x)
        pred = self.tf(x)
        return pred

    @staticmethod
    def minimum_input_length(factors=List[int]):
        """
        The model requires a minimum input length based on the downsampling factors
        and padding based on the last ResBlock which has a kernel size of 15 and
        a dilation of 3^2. B/c the padding used is reflection padding, the minimum
        input length required is the downsampling factos (based on the factors times 2)
        multiplied by the maximum padding.
        """
        maximum_pad = 9 * (15 // 2)
        maximum_down = np.prod(factors) * 2
        min_length = (maximum_pad + 1) * maximum_down + 1
        return min_length
