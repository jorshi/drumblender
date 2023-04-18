import pytest
import torch

from drumblender.synths import modal_synth
from drumblender.synths import ModalSynthFreqs


@pytest.fixture
def modal_synth_freqs():
    return ModalSynthFreqs(
        window_size=512,
    )


def test_modal_synth_produces_correct_output_size_three_inputs(modal_synth_freqs):
    hop_size = 256
    batch_size = 5
    frame_length = 120
    num_modes = 24

    amp_env = torch.rand(batch_size, num_modes, frame_length)
    freq_env = torch.rand(batch_size, num_modes, frame_length)
    phases = torch.rand(batch_size, num_modes)

    y = modal_synth_freqs((amp_env, freq_env, phases))
    print(y.shape)
    assert y.shape == (batch_size, hop_size * (frame_length - 1))


def test_modal_synth_produces_correct_output_size_two_inputs(modal_synth_freqs):
    hop_size = 256
    batch_size = 4
    frame_length = 690
    num_modes = 3

    amp_env = torch.rand(batch_size, num_modes, frame_length)
    freq_env = torch.rand(batch_size, num_modes, frame_length)

    y = modal_synth_freqs((amp_env, freq_env))
    assert y.shape == (batch_size, hop_size * (frame_length - 1))


def test_modal_synth_produces_test_tone(modal_synth_freqs):
    hop_size = 256
    batch_size = 16
    frame_length = 690
    num_modes = 2
    freq_tone = 1500
    sr = 48000

    n = hop_size * (frame_length - 1)

    amp_env = torch.zeros(batch_size, num_modes, frame_length)
    # Set an amplitude of 1 in a single mode
    amp_env[:, 0, :] = 1
    freq_env = torch.ones(batch_size, num_modes, frame_length) * freq_tone
    freq_env = 2 * torch.pi * freq_env / sr
    # Add an initial phase of zero.
    phase = torch.zeros(batch_size, num_modes)

    y = modal_synth_freqs((amp_env, freq_env, phase))

    # Manually synthesize a test tone
    t = torch.ones(batch_size, n) / sr
    t = torch.cumsum(t, dim=1)
    y_ref = torch.cos(2 * torch.pi * freq_tone * t)

    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-4)


def test_modal_synth(tmp_path):
    num_modes = 3
    num_frames = 24
    sample_rate = 16000
    num_samples = 16000

    # Create modal frequencies, amplitudes, and phases
    modal_freqs = torch.ones(1, num_modes, num_frames)
    modal_freqs[:, 0, :] = 440
    modal_freqs[:, 1, :] = 880
    modal_freqs[:, 2, :] = 1320

    # Convert to angular frequencies
    modal_freqs = 2 * torch.pi * modal_freqs / sample_rate

    modal_amps = torch.ones(1, num_modes, num_frames)
    modal_amps[:, 0, :] = 0.5 * torch.linspace(1.0, 0.0, num_frames)
    modal_amps[:, 1, :] = 0.3 * torch.linspace(0.0, 1.0, num_frames)
    modal_amps[:, 2, :] = 0.2

    audio = modal_synth(modal_freqs, modal_amps, num_samples)

    assert audio.shape == (1, num_samples)
