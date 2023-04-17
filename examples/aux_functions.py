from typing import List
from typing import Tuple

import numpy as np
import torch
from scipy.io import wavfile

"""
aux_functions.py
Auxiliary functions for running the example notebooks.
"""


def load_audio_torch(path: str, normalize: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Load audio from a wave file. Only supports 16bit audio.
    """
    sr, audio = wavfile.read(path)
    assert audio.dtype == np.int16, (
        "Only 16bit audio currently supported: %r" % audio.dtype
    )

    # Convert to float in range [-1, 1] and convert to mono
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    elif audio.ndim > 2:
        raise ValueError("Audio has more than 2 channels")

    # Normalize
    if normalize:
        audio /= np.max(np.abs(audio))

    # Convert to tensor
    audio = torch.tensor(np.array([audio]))

    return audio, sr


def stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 128,
    hop_size: int = 64,
    n_fft: int = None,
    decibel: bool = False,
    complex: bool = False,
) -> torch.FloatType:
    """
    Compute the loss between the magnitude spectrum
    of a predicted and target signal
    """
    w = torch.hann_window(window_size)
    n_fft = n_fft if n_fft is not None else window_size
    pred_stft = torch.stft(
        pred,
        n_fft,
        hop_length=hop_size,
        win_length=window_size,
        window=w,
        return_complex=False,
        normalized=True,
        pad_mode="constant",
    )
    target_stft = torch.stft(
        target,
        n_fft,
        hop_length=hop_size,
        win_length=window_size,
        window=w,
        return_complex=False,
        normalized=True,
        pad_mode="constant",
    )

    if complex:
        real_err = torch.mean((pred_stft[..., 0] - target_stft[..., 0]).pow(2))
        imag_err = torch.mean((pred_stft[..., 1] - target_stft[..., 1]).pow(2))
        return real_err + imag_err

    pred_mag = torch.sqrt(
        torch.clamp(pred_stft[..., 0].pow(2) + pred_stft[..., 1].pow(2), min=1e-8)
    )
    target_mag = torch.sqrt(
        torch.clamp(pred_stft[..., 0].pow(2) + target_stft[..., 1].pow(2), min=1e-8)
    )

    if decibel:
        pred_mag = 20 * torch.log10(pred_mag)
        target_mag = 20 * torch.log10(target_mag)

    return torch.mean(torch.abs(pred_mag - target_mag))


def safe_log(x, eps=1e-7):
    eps = torch.tensor(eps)
    return torch.log(x + eps)


# | export
def multiscale_stft(signal: torch.Tensor, scales: List, overlap: float):
    """
    Computes a multiscale STFT from a batched signal and a list of scales.
    Returns the corresponding list of STFTs, each being a torch.Tensor.
    """
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


# | export


def ddsp_msfft_loss(a1, a2, scales=[4096, 2048, 1024, 512, 256, 128], overlap=0.75):
    """
    DDSP Original MS FFT loss with lin + log spectra analysis.

    Some remarks:
        The stfts have to be normalized otherwise
        the netowrk weights different excerpts to different importance.

        We compute the mean of the L1 difference between normalized
        magnitude spectrograms so that the magnitude of the loss do
        not change with the window size.
    """
    if len(a1.size()) == 3:
        a1 = a1.squeeze(-1)
    if len(a2.size()) == 3:
        a2 = a2.squeeze(-1)
    ori_stft = multiscale_stft(
        a1,
        scales,
        overlap,
    )
    rec_stft = multiscale_stft(
        a2,
        scales,
        overlap,
    )

    loss = 0
    for s_x, s_y in zip(ori_stft, rec_stft):
        lin_loss = (s_x - s_y).abs().mean()
        log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss

    return loss
