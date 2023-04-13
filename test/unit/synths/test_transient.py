import numpy as np
import pywt
import torch
from einops import repeat

from drumblender.synths import idwt_functional
from drumblender.synths import WaveletConvOLA
from drumblender.synths import WaveletTransform


def test_wavelet_reconstruction():
    batch_size = 16
    input_size = 4096
    sample_rate = 1024
    dwt = WaveletTransform(input_size=input_size, wavelet="db3", max_level=None)

    # Synthesize cosine waves of varying frequency
    batched_audio = torch.ones(batch_size, input_size) / sample_rate
    batched_audio = torch.cumsum(batched_audio, 1)
    for i in range(batch_size):
        batched_audio[i, :] = torch.cos(2 * torch.pi * batched_audio[i, :] * (i + 1))

    y = dwt(batched_audio)

    # Create the filters for reconstruction.
    # For this test, we use the same filters for different batches by repeating them.
    # With the DNN, each batch has its own set of filters and inputs

    wavelet = pywt.Wavelet("db3")
    wavelet_filters = wavelet.filter_bank
    num_levels = dwt.num_levels

    rec_lo = torch.tensor(wavelet_filters[2]).unsqueeze(0)
    rec_hi = torch.tensor(wavelet_filters[3]).unsqueeze(0)
    filter = torch.concat((rec_lo, rec_hi), dim=0)

    # Repeat filter pair by batch, num_windows, num_levels,
    # type of filter, and filter kernel.
    filters = repeat(filter, "t f -> b l t f", b=batch_size, l=num_levels)

    y_hat = idwt_functional(y, filters)

    assert torch.mean(torch.abs(y_hat - batched_audio)) < 0.05


def test_wavenet_conv_ola_yields_correct_sizes():
    test_signal_size = 4096
    window_size = 1024
    num_windows = 8
    batch_size = 16
    num_levels = int(np.round(np.log2(window_size)))

    wavelet = pywt.Wavelet("db5")
    wavelet_filters = wavelet.filter_bank

    rec_lo = torch.tensor(wavelet_filters[2]).unsqueeze(0)
    rec_hi = torch.tensor(wavelet_filters[3]).unsqueeze(0)
    filter = torch.concat((rec_lo, rec_hi), dim=0)

    # Copy filters: batch, num_windows, num_levels, type of filter, filter kernel
    idwt_filters = repeat(
        filter, "t f -> b w l t f", b=batch_size, w=num_windows, l=num_levels
    )

    wavelet_ola = WaveletConvOLA(window_size, num_windows, wavelet="db5")

    test_signal = torch.rand(batch_size, test_signal_size)

    reconstruction = wavelet_ola(test_signal, idwt_filters)

    assert reconstruction.shape[0] == batch_size
    assert reconstruction.shape[1] == wavelet_ola.num_samples
