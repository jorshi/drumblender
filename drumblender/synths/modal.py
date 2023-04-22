from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModalSynthFreqs(nn.Module):
    def __init__(
        self,
        window_size: int = 1024,  # Window size OLA amplitude envelope
    ):
        """Overlap-add modal synthesis with given frequencies and amplitudes"""
        super().__init__()
        # Amplitude envelope as complex OLA
        self.window_size = window_size
        self.hop_size = self.window_size // 2
        self.padding = self.window_size // 2

    # Create an amplitude envelope
    def amp_envelope(self, amp_env):
        batch_size, num_modes, num_frames = amp_env.size()
        n = (num_frames - 1) * self.hop_size

        amp = torch.ones(batch_size, num_modes, n)

        # Create a complex OLA window for amplitude
        amp = rearrange(amp, "b m n -> (b m) 1 1 n")

        amp_unfold = torch.nn.functional.unfold(
            amp,
            kernel_size=(1, self.window_size),
            padding=(0, self.padding),
            stride=self.hop_size,
        )
        # Window each frame
        window = torch.hann_window(self.window_size)
        window = rearrange(window, "n -> 1 n 1")
        amp_unfold = amp_unfold * window

        # Apply the learned amplitude for each window
        amp_env = rearrange(amp_env, "b m n -> (b m) 1 n")
        # amp_env = torch.square(amp_env)
        amp_unfold = amp_unfold * amp_env

        # Stitch the windows back together
        amp = torch.nn.functional.fold(
            amp_unfold,
            output_size=(1, n),
            kernel_size=(1, self.window_size),
            padding=(0, self.padding),
            stride=self.hop_size,
        )

        amp = rearrange(amp, "(b m) 1 1 n -> b m n", m=num_modes)
        return amp

    def get_f0(self, freq_env):
        """
        Interpolates the predicted frequency envelope
        """
        batch_size, num_modes, num_frames = freq_env.size()
        n = (num_frames - 1) * self.hop_size

        freqs = F.interpolate(freq_env, size=n, mode="linear")
        freqs = rearrange(freqs, "b m n -> (b m) n")
        return freqs

    def forward(self, x: Tuple[torch.Tensor, ...]):
        """
        x: Tuple with predicted amps and frequency envelope in angular
           frequency. Phases are optional.
        amp_env : [nb,num_modes,num_frames]
        freq_env: [nb,num_modes,num_frames]
        phase: [nb,num_modes]
        NOTES:  1. The frequency envelope is constructed with
                   freq_env = 2 * np.pi * freqs / sr
                2. Overall amplitude is not processed here. It should be
                   implemented before on the mode decoder.
        """
        if len(x) == 2:
            amp_env, freq_env = x
            phase = torch.zeros(amp_env.size()[0], amp_env.size()[1])
        elif len(x) == 3:
            amp_env, freq_env, phase = x

        # Rearrange the time-varying frequency for each mode
        f0_env = self.get_f0(freq_env)

        # Enforce non-aliasing frequencies
        f0_env = torch.clamp(f0_env, 0, torch.pi)

        # For optimizer: Normalize between 0 to pi.
        phase = phase % (2 * torch.pi)
        phase = rearrange(phase, "b m -> (b m) 1")
        phase_env = torch.cumsum(f0_env, dim=1) + phase
        y = torch.cos(phase_env)

        # Apply amplitude envelope
        amp_env = self.amp_envelope(amp_env)
        y = y * rearrange(amp_env, "b m n -> (b m) n")

        # Sum the modes
        num_modes = amp_env.size()[1]
        y = rearrange(y, "(b m) n -> b m n", m=num_modes)
        y = torch.sum(y, dim=1)

        return y


class ModalSynth(torch.nn.Module):
    """
    Modal synthesis with given frequencies, amplitudes, and optional phase
    Users linear interpolation to generate the frequency envelope.
    """

    def forward(self, params: torch.Tensor, num_samples: int):
        """
        params: [nb,num_params,num_modes,num_frames], expected parameters
                are: frequency, amplitude, and phase (optional)
        num_samples: number of samples to generate
        """
        assert params.ndim == 4, "Expected 4D tensor"
        assert params.size()[1] in [2, 3], "Expected 2 or 3 parameters"

        params = torch.chunk(params, params.size()[1], dim=1)
        params = [p.squeeze(1) for p in params]

        # Pass the phase if it was given, otherwise None
        phase = None
        if len(params) == 3:
            phase = params[2]

        y = modal_synth(params[0], params[1], num_samples, phase)
        y = rearrange(y, "b n -> b 1 n")
        return y


def modal_synth(
    freqs: torch.Tensor,
    amps: torch.Tensor,
    num_samples: int,
    phase: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Synthesizes a modal signal from a set of frequencies, phases, and amplitudes.

    Args:
        freqs: A 3D tensor of frequencies in angular frequency of shape
            (batch_size, num_modes, num_frames)
        amps: A 3D tensor of amplitudes of shape (batch_size, num_modes, num_frames)
        sample_rate: Sample rate of the output signal
        num_samples: Number of samples in the output signal
    """
    (batch_size, num_modes, num_frames) = freqs.shape
    assert freqs.shape == amps.shape

    # Interpolate the frequencies and amplitudes
    w = torch.nn.functional.interpolate(freqs, size=num_samples, mode="linear")
    a = torch.nn.functional.interpolate(amps, size=num_samples, mode="linear")

    a = rearrange(a, "b m n -> (b m) n")
    w = rearrange(w, "b m n -> (b m) n")
    phase_env = torch.cumsum(w, dim=1)
    if phase is not None:
        phase = rearrange(phase, "b m n -> (b m) n")
        phase_env = phase_env + phase[..., 0, None]

    # Generate the modal signal
    y = a * torch.sin(phase_env)
    y = rearrange(y, "(b m) n -> b m n", b=batch_size, m=num_modes)

    # Sum the modes
    y = torch.sum(y, dim=1)

    return y
