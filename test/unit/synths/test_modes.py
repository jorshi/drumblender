import pytest
import torch
from einops import repeat

from percussionsynth.synths import ModalSynthFreqs


@pytest.fixture
def modal_synth():
    return ModalSynthFreqs(
        window_size=512,
    )


def test_modal_synth_produces_correct_output_size_three_inputs(modal_synth):
    hop_size = 256
    batch_size = 16
    frame_length = 500
    num_modes = 120

    amp_env = torch.rand(batch_size, num_modes,frame_length)
    freq_env = torch.rand(batch_size, num_modes,frame_length)
    phases = torch.rand(batch_size, num_modes)

    y = modal_synth((amp_env,freq_env,phases))
    assert y.shape == (batch_size,hop_size*(frame_length-1))

def test_modal_synth_produces_correct_output_size_two_inputs(modal_synth):
    hop_size = 256
    batch_size = 16
    frame_length = 500
    num_modes = 120

    amp_env = torch.rand(batch_size, num_modes,frame_length)
    freq_env = torch.rand(batch_size, num_modes,frame_length)

    y = modal_synth((amp_env,freq_env))
    assert y.shape == (batch_size,hop_size*(frame_length-1))

def test_modal_synth_produces_test_tone(modal_synth):
    hop_size = 256
    batch_size = 16
    frame_length = 500
    num_modes = 120
    freq_tone = 1000
    sr = 48000

    amp_env = torch.zeros(batch_size, num_modes,frame_length)
    amp_env[:,0,:] = 1
    freq_env = torch.ones(batch_size, num_modes,frame_length) * freq_tone
    freq_env = 2 * torch.pi * freq_env / sr

    y = modal_synth((amp_env,freq_env))

    t_end = hop_size*(frame_length-1)/sr
    t = torch.linspace(0,t_end,hop_size*(frame_length-1))
    y_ref = torch.cos(2*torch.pi*freq_tone*t)
    y_ref = repeat(y_ref,"s -> b s",b=batch_size)
    
    assert torch.allclose(y,y_ref) == True