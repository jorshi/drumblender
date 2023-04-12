import numpy as np
import pywt
import torch
import torch.nn.functional as F
from einops import rearrange


class WaveletTransform(torch.nn.Module):
    """Module to compute the dwt of a signal.

    Args:
        input_size: signal length
        wavelet: string denoting the type of wavelet to use
        max_level: maximum dwt level
    """

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


def idwt_functional(x, filters):
    """Computes an inverse dwt from a multi dimensional dwt
    and a set of predicted filters
    Args:
        x: list of dwt tensors with format [nb,len]
        filters: tensor of filters with format [nb,levels,type,kw]
    """
    # Group batches as channels and use grouped 1d conv
    prev_lo = rearrange(x[-1], "b t -> 1 b t")

    filters = torch.split(filters, 1, dim=1)

    assert len(x) - 1 == len(filters)
    for i in range(len(filters) - 1, -1, -1):
        y_lo = torch.zeros(1, x[i].shape[0], x[i].shape[1] * 2)
        y_hi = torch.zeros(1, x[i].shape[0], x[i].shape[1] * 2)

        # Upsample by 2
        y_lo[:, :, ::2] = prev_lo
        y_hi[:, :, ::2] = x[i]

        # This leaves a dimension shape of [nb 1 kw] adequate for conv1d
        rec_lo = filters[i][:, :, 0, :]

        # Apply the filters
        y_lo = F.conv1d(
            input=y_lo,
            weight=rec_lo,
            bias=None,
            stride=1,
            groups=rec_lo.shape[-3],
            padding=rec_lo.shape[-1] // 2 - 1,
        )

        # This leaves a dimension shape of [nb 1 kw] adequate for conv1d
        rec_hi = filters[i][:, :, 1, :]

        y_hi = F.conv1d(
            input=y_hi,
            weight=rec_hi,
            bias=None,
            stride=1,
            groups=rec_hi.shape[-3],
            padding=rec_hi.shape[-1] // 2 - 1,
        )

        prev_lo = y_lo + y_hi

    y = rearrange(prev_lo, "1 b t -> b t")
    return y[:, :-1]


class WaveletConv(torch.nn.Module):
    """Computes a transform on the Wavelet Space

    Args:
    num_samples: Lenght of the audio excerpt
    wavelet: Type of wavelet
    max_level: adds a cap on levels.
        Otherwise, this is computed as
            num_levels = int(np.round(np.log2(window_size)))
    """

    def __init__(self, num_samples: int, wavelet="db5", max_level=None):
        super().__init__()
        self.dwt = WaveletTransform(num_samples, wavelet=wavelet, max_level=max_level)

    def forward(self, x, idwt_filters):
        """Args:
        x: audio window. format: [nb,samples]
        filters: tensor of filters with format [nb,levels,type,kw]
        """
        y = self.dwt(x)
        return idwt_functional(y, idwt_filters)


class WaveletConvOLA(torch.nn.Module):
    """
    Splits an input signal into overlapping chunks and applies
    a wavelet transform to each chunk.
    """

    def __init__(self, window_size: int = 512, num_windows: int = 8, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.hop_size = window_size // 2
        self.num_windows = num_windows
        self.padding = self.hop_size
        self.input_length = (
            (self.hop_size * (self.num_windows - 1))
            + self.window_size
            - self.padding * 2
        )

        self.wavelets = torch.nn.ModuleList()
        for _ in range(self.num_windows):
            self.wavelets.append(WaveletConv(self.window_size, **kwargs))

    @property
    def num_samples(self):
        return self.input_length

    def get_frames(self, x):
        x = rearrange(x, "b n -> b 1 1 n")
        x_unfold = torch.nn.functional.unfold(
            x,
            kernel_size=(1, self.window_size),
            padding=(0, self.padding),
            stride=self.hop_size,
        )
        return x_unfold

    def overlap_add(self, x):
        # Apply windowing
        window = torch.hann_window(self.window_size, periodic=True)
        window = rearrange(window, "n -> 1 n 1")
        x = x * window

        x = torch.nn.functional.fold(
            x,
            output_size=(1, self.input_length),
            kernel_size=(1, self.window_size),
            padding=(0, self.padding),
            stride=self.hop_size,
        )
        x = rearrange(x, "b 1 1 n -> b n")
        return x

    def forward(self, x: torch.tensor, idwt_filters):
        """Reconstructs x using dwt and idwt

        Args:
            x: audio in format [nb,samples]
            idwt_filters: predicted idwt filters
                format: [nb,windows,levels,type,kw]
        """
        # Split in windows
        idwt_filters = torch.split(idwt_filters, 1, dim=1)
        assert len(idwt_filters) == self.num_windows

        x_unfold = self.get_frames(x)  # Unfolds the signal

        # Apply wavelet transform to each window
        x_split = torch.split(
            x_unfold, 1, dim=2
        )  # Splits into list (each entry is a view)
        x_cat = []
        for i in range(self.num_windows):
            # Squeeze window dimension
            filter = rearrange(idwt_filters[i], "b 1 l t k -> b l t k")
            xin = x_split[i].squeeze(-1)
            x_cat.append(self.wavelets[i](xin, filter).unsqueeze(-1))

        x = torch.cat(x_cat, dim=-1)
        x = self.overlap_add(x)

        return x
