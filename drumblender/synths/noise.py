import torch
import torch.nn as nn
from einops import rearrange


class NoiseGenerator(nn.Module):
    """Overlap-and-Add noise generator.

    This is the DDSP implementation of the Noise Generator,
    using rectangular filters on linear frequency domain, with
    overlap-and-add using Hann windows.

    Args:
      window_size(int): Size of overlap and add window
    """

    def __init__(
        self,
        window_size: int = 1024,  # Overlap-and-add window size.
    ):
        super().__init__()
        # Noise amplitude decay, required by optimizer

        # Amplitude envelope as complex OLA
        self.window_size = window_size
        self.hop_size = self.window_size // 2
        self.padding = self.window_size // 2

    def amp_to_impulse_response(self, amp: torch.Tensor, target_size: int):
        """
        Creates the time evolving impulse response of a FIR filter
        defined in freq domain.

        Args:
            amp (torch.Tensor): Time-evolving frequency bands of
                shape [nb, frame_len, num_bands]
            target_size: Size of the window

        How it works:
            Treat time evolving amplitude bands as frequency responses
            and compute the IR. Each amplitude band in frequency domain
            is a very narrow filter. We breadth the bandwidth of the filter
            by multiplying the IR with a Hann window. Then we convolve the IR
            with the noise source.

        Returns:
            Sequence of IR filters of size target_size.
                Shape: [nb, frame_len, target_size]
        """
        # Cast tensor as if it was complex adding a zeroed imaginary part
        # with torch stack.
        amp = torch.stack([amp, torch.zeros_like(amp)], -1)
        amp = torch.view_as_complex(amp)

        # Compute the impulse response of the filters.
        # This gives a zero-centered response.
        amp = torch.fft.irfft(amp)
        filter_size = amp.shape[-1]

        # Smooth amp response with Hann window.
        # First: roll response to apply window.
        amp = torch.roll(amp, filter_size // 2, -1)
        win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

        amp = amp * win

        # Second: Zero pad impulse response to the right.
        # Then roll back to center in zero again.
        amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
        amp = torch.roll(amp, -filter_size // 2, -1)

        return amp

    def overlap_and_add(self, noise_unfolded):
        """
        Overlaps and adds a filtered noise sequence.

        Args:
            noise_unfolded (torch.Tensor): format [nb,frame_len,window_length]

        Returns:
            noise signals with format [b,n]
        """
        # Window each frame
        window = torch.hann_window(self.window_size)
        noise_unfolded = noise_unfolded * window

        # Compute final sample size
        n = (noise_unfolded.size()[1] - 1) * self.hop_size
        # Stitch the windows back together.
        # Expects tensors in format [batch, C*kernel_size, L ].
        # C = 1 for our signal.
        noise_unfolded = rearrange(noise_unfolded, "b l k -> b k l")
        noise = torch.nn.functional.fold(
            noise_unfolded,
            output_size=(1, n),
            kernel_size=(1, self.window_size),
            padding=(0, self.padding),
            stride=self.hop_size,
        )
        noise = rearrange(noise, "b 1 1 n -> b n")
        return noise

    def fft_convolve(self, signal: torch.Tensor, kernel: torch.Tensor):
        # Zero pad signal to the right with signal.shape[-1] elements.
        signal = nn.functional.pad(signal, (0, signal.shape[-1]))
        # Zero pad kernel to the left with kernel.shape[-1] elements.
        kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

        # Convolve in freq domain and get back to time domain.
        output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
        output = output[..., output.shape[-1] // 2 :]

        return output

    def forward(self, noise_bands: torch.Tensor):
        """
        Args:
            noise_bands (torch.Tensor): A tensor with predicted filtered
                noise coefficients [nb, frame_len, num_bands]
        """
        # Create a sequence of IRs according to input.
        impulse = self.amp_to_impulse_response(noise_bands, self.window_size)

        # Random uniform noise with range [-1,1]
        noise = (
            torch.rand(
                impulse.shape[0],
                impulse.shape[1],
                self.window_size,
            ).to(impulse)
            * 2
            - 1
        )

        noise = self.fft_convolve(noise, impulse).contiguous()
        noise = self.overlap_and_add(noise)

        return noise
