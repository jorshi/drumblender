from typing import Tuple
from typing import Optional
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalSynthFreqs(nn.Module):

    def __init__(
        self,
        window_size: int = 1024,  # Window size OLA amplitude envelope
    ):
        """Overlap-add modal synthesis with given frequencies and amplitudes"""

        # Amplitude envelope as complex OLA
        self.window_size = window_size
        self.hop_size = self.window_size // 2
        self.padding = self.window_size // 2

    # Create an amplitude envelope
    def amp_envelope(self,amp_env):

        batch_size,num_modes,num_frames = amp_env.size()
        n = (num_frames-1)*self.hop_size

        amp = torch.ones(batch_size, num_modes, n)
        
        # Create a complex OLA window for amplitude
        amp = rearrange(amp, "b m n -> (b m) 1 1 n")

        amp_unfold = torch.nn.functional.unfold(
            amp, 
            kernel_size=(1, self.window_size),
            padding=(0, self.padding),
            stride=self.hop_size
        )
        # Window each frame
        window = torch.hann_window(self.window_size)
        window = rearrange(window, "n -> 1 n 1")
        amp_unfold = amp_unfold * window

        # Apply the learned amplitude for each window
        amp_env = rearrange(amp_env, "b m n -> (b m) 1 n")
        #amp_env = torch.square(amp_env)
        amp_unfold = amp_unfold * amp_env

        # Stitch the windows back together
        amp = torch.nn.functional.fold(
            amp_unfold,
            output_size=(1, self.n),
            kernel_size=(1, self.window_size),
            padding=(0, self.padding),
            stride=self.hop_size
        )

        amp = rearrange(amp, "(b m) 1 1 n -> b m n", m=self.num_modes)
        return amp

    def get_f0(self,freqs):
        freqs = rearrange(freqs, "b m n -> (b m) n")
        return freqs

    """
        x: Tuple with predicted amps and frequency envelope in angular frequency. Phases are optional
        amp_env : [nb,num_modes,num_frames]
        freq_env: [nb,num_modes,num_frames]
        phase: [nb,num_modes]
        NOTES:  1. The frequency envelope is constructed with 
                    freq_env = 2 * np.pi * freqs / sr
                2. Overall amp is not processed here. It should be implemented before on the mode decoder.
    """
    def forward(self, x : Tuple[torch.Tensor,torch.Tensor,...]):

        if len(x) == 2:
            amp_env, freq_env = x
            phase = torch.zeros(amps.size()[0],amps.size()[1])
        elif len(x) == 3:
            amps, freqs, phase = x

        # Rearrange the time-varying frequency for each mode
        f0_env =  self.get_f0(freq_env)
    
        # Enforce non-aliasing frequencies
        f0_env = torch.clamp(f0_env, 0, torch.pi)

        # For optimizer: Normalize between 0 to pi.
        phase = torch.sigmoid(phase) * torch.pi 
        phase = rearrange(phase, "b m -> (b m) 1")
        phase_env = torch.cumsum(f0_env, dim=1) + phase      

        y = torch.cos(phase_env)

        # Apply amplitude envelope
        amp_env = self.amp_envelope(amp_env)
        y = y * rearrange(amp_env, "b m n -> (b m) n")

        # Sum the modes
        y = rearrange(y, "(b m) n -> b m n", m=self.num_modes)
        y = torch.sum(y, dim=1)

        # Enforce [-1, 1] -- also adds some nice squash
        y = torch.tanh(y)

        return y