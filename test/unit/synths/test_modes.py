import torch

from drumblender.synths import modal_synth


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
