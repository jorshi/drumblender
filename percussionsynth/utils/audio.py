"""
Audio utility functions
"""
from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import torch
import torchaudio
from einops import rearrange
from einops import repeat


def preprocess_audio_file(
    input_file: Path,
    output_file: Path,
    sample_rate: int,
    num_samples: Optional[int] = None,
    mono: bool = True,
    remove_start_silence: bool = True,
    silence_threshold_db: float = -60.0,
):
    """
    Preprocess an audio file.

    Args:
        input_file: Path to the input audio file.
        output_file: Path to the output audio file.
        sample_rate: Sample rate of the output audio file.
        num_samples: Number of samples to use from the input audio file.
            Defaults to None. If included, the audio file will be truncated or
            padded with zeros to the desired length.
        mono: Whether to convert the audio to mono. Defaults to True.
    """
    waveform, orig_freq = torchaudio.load(input_file)
    assert waveform.ndim == 2, "Expecting a 2D tensor, channels x samples"

    # Convert to mono if necessary
    if mono:
        num_channels = waveform.shape[0]
        if num_channels > 1:
            waveform = waveform[:1, :]

        # Should be a mono signal now
        assert waveform.shape[0] == 1, "Expecting a mono signal"

    # Resample the waveform to the desired sample rate
    if orig_freq != sample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=sample_rate
        )(waveform)

    # Remove silent samples from the beginning of the waveform
    waveform = cut_start_silence(waveform, threshold_db=silence_threshold_db)

    # Truncate or pad the waveform to the desired number of samples
    if num_samples is not None and waveform.shape[1] != num_samples:
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]
        else:
            num_pad = num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, num_pad))

    torchaudio.save(output_file, waveform, sample_rate)


def generate_sine_wave(
    frequency, num_samples, sample_rate, stereo=False
) -> torch.Tensor:
    """Generate a sine wave.

    Args:
        frequency: Frequency of the sine wave in Hz
        num_samples: Duration of the sine wave in samples
        sample_rate: Sample rate of the sine wave
        stereo: Whether to generate a stereo signal. Defaults to False.

    Returns:
        A 2D tensor of shape (channels, num_samples)
    """
    n = torch.arange(num_samples)
    x = torch.sin(frequency * 2 * torch.pi * n / sample_rate)
    x = repeat(x, "n -> c n", c=2 if stereo else 1)
    return x


def modal_synth(
    freqs: torch.Tensor,
    amps: torch.Tensor,
    sample_rate: int,
    num_samples: int,
) -> torch.Tensor:
    """
    Synthesizes a modal signal from a set of frequencies, phases, and amplitudes.

    Args:
        freqs: A 3D tensor of frequencies in Hz of shape (batch_size, num_modes,
            num_frames)
        amps: A 3D tensor of amplitudes of shape (batch_size, num_modes, num_frames)
        sample_rate: Sample rate of the output signal
        num_samples: Number of samples in the output signal
    """
    (batch_size, num_modes, num_frames) = freqs.shape
    assert freqs.shape == amps.shape

    # Convert frequencies to angular frequencies
    w = 2 * torch.pi * freqs / sample_rate

    # Interpolate the frequencies and amplitudes
    w = torch.nn.functional.interpolate(w, size=num_samples, mode="linear")
    a = torch.nn.functional.interpolate(amps, size=num_samples, mode="linear")

    a = rearrange(a, "b m n -> (b m) n")
    w = rearrange(w, "b m n -> (b m) n")
    phase_env = torch.cumsum(w, dim=1)

    # Generate the modal signal
    y = a * torch.sin(phase_env)
    y = rearrange(y, "(b m) n -> b m n", b=batch_size, m=num_modes)

    # Sum the modes
    y = torch.sum(y, dim=1)

    return y


def first_non_silent_sample(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -60.0,
) -> Union[int, None]:
    """
    Returns the index of the first non-silent sample in a waveform, determined as the
    first frame greater than the threshold.
    Implementation based on https://essentia.upf.edu/reference/std_StartStopCut.html

    Args:
        x: A 1D tensor of shape (num_samples,)
        frame_size: frame size in samples used for power calculations
        hop_size: hop size in samples used to split the signal into frames
        threshold_db: threshold in dB below which a frame is considered silent

    Returns:
        The index of the first non-silent sample in the waveform or None if the entire
        waveform is silent.
    """

    assert x.ndim == 1, "Expecting a 1D tensor"
    x = torch.split(x, frame_size)
    thrshold_power = np.power(10.0, threshold_db / 10.0)

    for i, frame in enumerate(x):
        power = torch.inner(frame, frame) / frame.shape[-1]
        if power > thrshold_power:
            return i * hop_size

    return None


def cut_start_silence(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -60.0,
) -> torch.Tensor:
    """
    Removes silent samples from the beginning of a waveform.

    Args:
        x: A 2D tensor of shape (channels, num_samples)
        frame_size: frame size in samples used for power calculations
        hop_size: hop size in samples used to split the signal into frames
        threshold_db: threshold in dB below which a frame is considered silent

    Returns:
        The waveform with silent samples removed from the beginning.

    Raises:
        ValueError: If the entire waveform is below the threshold dB
    """
    assert x.ndim == 2, "Expecting a 2D tensor with shape (channels, num_samples)"

    start_samples = []
    for channel in x:
        start_sample = first_non_silent_sample(
            channel, frame_size=frame_size, hop_size=hop_size, threshold_db=threshold_db
        )
        if start_sample is not None:
            start_samples.append(start_sample)

    if len(start_samples) == 0:
        raise ValueError(f"Entire wavfile below threshold level {threshold_db}dB")

    return x[:, min(start_samples) :]
