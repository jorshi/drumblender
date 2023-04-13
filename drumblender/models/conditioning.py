import torch
from einops import rearrange
from einops import repeat
from torch import nn


class GaussianConditioning(nn.Module):
    """Implements a Gaussian noise conditioning layer.

    Args:
        output_channels (int): The number of output channels.
    """

    def __init__(self, output_channels: int):
        super().__init__()
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *shape, channels, length = x.shape
        return torch.randn(
            *shape, self.output_channels, length, device=x.device, dtype=x.dtype
        )


class RandomFourierFeatureConditioning(nn.Module):
    """Implements a random Fourier feature conditioning layer.

    Args:
        output_channels (int): The number of output channels.
        freq_scale (float): The frequency scale of the random Fourier features.
        trainable (bool): Whether the random Fourier features are trainable.
    """

    def __init__(
        self, output_channels: int, freq_scale: float, trainable: bool = False
    ):
        super().__init__()
        self.output_channels = output_channels
        self.freq_scale = freq_scale
        self.freqs = nn.Linear(1, output_channels, bias=True)

        if not trainable:
            self.freqs.weight.requires_grad = False
            self.freqs.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        n = torch.linspace(-1.0, 1.0, length, device=x.device, dtype=x.dtype)[..., None]
        phase = self.freqs(n * self.freq_scale)
        signal = torch.sin(phase)
        signal = rearrange(signal, "l c -> () c l")
        signal = repeat(signal, "1 c l -> b c l", b=batch)
        return signal
