import pytest
import torch

from drumblender.synths import ModalSynthFreqs


@pytest.fixture
def modal_synth():
    return ModalSynthFreqs(
        window_size=512,
    )


def test_modal_synth_produces_correct_output_size_three_inputs(modal_synth):
    hop_size = 256
    batch_size = 5
    frame_length = 120
    num_modes = 24

    amp_env = torch.rand(batch_size, num_modes, frame_length)
    freq_env = torch.rand(batch_size, num_modes, frame_length)
    phases = torch.rand(batch_size, num_modes)

    y = modal_synth((amp_env, freq_env, phases))
    print(y.shape)
    assert y.shape == (batch_size, hop_size * (frame_length - 1))


def test_modal_synth_produces_correct_output_size_two_inputs(modal_synth):
    hop_size = 256
    batch_size = 4
    frame_length = 690
    num_modes = 3

    amp_env = torch.rand(batch_size, num_modes, frame_length)
    freq_env = torch.rand(batch_size, num_modes, frame_length)

    y = modal_synth((amp_env, freq_env))
    assert y.shape == (batch_size, hop_size * (frame_length - 1))


def test_modal_synth_produces_test_tone(modal_synth):
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

    y = modal_synth((amp_env, freq_env, phase))

    # Manually synthesize a test tone
    t = torch.ones(batch_size, n) / sr
    t = torch.cumsum(t, dim=1)
    y_ref = torch.cos(2 * torch.pi * freq_tone * t)

    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-4)
