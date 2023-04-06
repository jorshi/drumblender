from typing import List
from typing import Optional
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

# | export
""" Module to compute the dwt of a signal.

    Args:
        input_size: signal length
        wavelet: string denoting the type of wavelet to use
        max_level: maximum dwt level
"""
class WaveletTransform(torch.nn.Module):
    def __init__(self, input_size: int, wavelet="db2", max_level=None):
        super().__init__()
        self.input_size = input_size
        self.num_levels = int(np.round(np.log2(input_size)))
        if max_level is not None:
            self.num_levels = min(self.num_levels, max_level)

        self.wavelet = pywt.Wavelet(wavelet)
        self.filter_bank = self.wavelet.filter_bank
        self.max_level = max_level

        # Fixed lowpass filter
        self.dec_lo = torch.nn.Conv1d(
            1,
            1,
            kernel_size=len(self.filter_bank[0]),
            padding=len(self.filter_bank[0]) // 2,
            bias=False,
        )
        self.dec_lo.weight.data = torch.tensor(self.filter_bank[0]).view(1, 1, -1)
        self.dec_lo.weight.requires_grad = False

        # Fixed highpass filter
        self.dec_hi = torch.nn.Conv1d(
            1,
            1,
            kernel_size=len(self.filter_bank[0]),
            padding=len(self.filter_bank[0]) // 2,
            bias=False,
        )
        self.dec_hi.weight.data = torch.tensor(self.filter_bank[1]).view(1, 1, -1)
        self.dec_hi.weight.requires_grad = False

    def forward(self, x):
        # Add a single channel dimension
        x = rearrange(x, "b t -> b () t")

        multi_dimensional = []
        for level in range(self.num_levels):
            # Apply the filters
            y_lo = self.dec_lo(x)
            y_hi = self.dec_hi(x)

            # Downsample by 2
            x = y_lo[:, :, ::2]
            y_hi = y_hi[:, :, ::2]

            # Remove the channel dimension from highpass
            y_hi = rearrange(y_hi, "b () t -> b t")
            multi_dimensional.append(y_hi)

        multi_dimensional.append(rearrange(x, "b () t -> b t"))
        return multi_dimensional
    


""" Computes an inverse dwt from a multi dimensional dwt 
    and a set of predicted filters

    Args:    
        x: list of tensors with format [nb,len]
        filters: list of dictionaries with format
            filter[i]['rec_lo'] : [nb,kw]
            filter[i]['rec_hi'] : [nb,kw]
"""
def InverseWaveletTransformFunctional(x, filters):

    # Group batches as channels and use grouped 1d conv
    prev_lo = rearrange(x[-1], "b t -> 1 b t")
    assert len(x) -1 == len(filters)
    for i in range(len(filters) - 1, -1, -1):
        y_lo = torch.zeros(1, x[i].shape[0], x[i].shape[1] * 2)
        y_hi = torch.zeros(1, x[i].shape[0], x[i].shape[1] * 2)

        # Upsample by 2
        y_lo[:, :, ::2] = prev_lo
        y_hi[:, :, ::2] = x[i]

        rec_lo = rearrange(filters[i]['rec_lo'], "b k -> b 1 k")

        # Apply the filters
        y_lo = F.conv1d(input=y_lo,
                        weight=rec_lo,
                        bias=None,
                        stride=1,
                        groups=rec_lo.shape[-3],
                        padding=rec_lo.shape[-1]//2-1)


        rec_hi = rearrange(filters[i]['rec_hi'], "b k -> b 1 k")

        y_hi = F.conv1d(input=y_hi,
                        weight=rec_hi,
                        bias=None,
                        stride=1,
                        groups=rec_hi.shape[-3],
                        padding=rec_hi.shape[-1]//2-1)

        prev_lo = y_lo + y_hi

    y = rearrange(prev_lo, "1 b t -> b t")
    return y[:, :-1]