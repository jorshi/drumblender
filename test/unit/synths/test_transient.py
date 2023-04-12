import pywt
import torch
from einops import repeat

from percussionsynth.synths import idwt_functional
from percussionsynth.synths import WaveletTransform


def test_wavelet_reconstruction():
    batch_size = 16
    input_size = 4096
    sample_rate = 1024
    dwt = WaveletTransform(input_size=input_size, wavelet="db3", max_level=None)

    # Synthesize some cosine waves of different frequencies
    batched_audio = torch.ones(batch_size, input_size) / sample_rate
    batched_audio = torch.cumsum(batched_audio, 1)
    for i in range(batch_size):
        batched_audio[i, :] = torch.cos(2 * torch.pi * batched_audio[i, :] * (i + 1))

    y = dwt(batched_audio)

    # Now, create the inverse filters and reconstruct

    # Create the filters as a list of tensors.
    # For this test, we use the same filters for different batches by repeating them.
    # With the DNN, each batch has its own set of filters and inputs
    wavelet = pywt.Wavelet("db3")
    wavelet_filters = wavelet.filter_bank
    num_levels = dwt.num_levels
    filters = []
    filt_pair = {}
    for i in range(num_levels):
        filt_pair["rec_lo"] = repeat(
            torch.tensor(wavelet_filters[2]), "f -> b f", b=batch_size
        )
        filt_pair["rec_hi"] = repeat(
            torch.tensor(wavelet_filters[3]), "f -> b f", b=batch_size
        )
        filters.append(filt_pair.copy())

    y_hat = idwt_functional(y, filters)

    assert torch.mean(torch.abs(y_hat - batched_audio)) < 0.05
