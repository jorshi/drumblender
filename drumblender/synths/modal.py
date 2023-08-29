from typing import Optional

import torch
from einops import rearrange


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
