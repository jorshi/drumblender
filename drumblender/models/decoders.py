"""
Decoders for the Drum Blender model.
"""
from typing import List

import torch
from einops import rearrange


class ResidualBlock(torch.nn.Module):
    """
    A residual block for a soundstream inspired decoder.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The kernel size of the convolutional layers.
        dilations (int): The dilation of the convolutional layers.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilations: int
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilations
        self.net = torch.nn.Sequential(
            *[
                torch.nn.ConstantPad1d((padding, 0), 0),
                torch.nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilations,
                    padding=0,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv1d(out_channels, out_channels, 1, dilation=1, padding=0),
            ]
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.net(x) + x


class DecoderBlock(torch.nn.Module):
    """
    Upsampling decoder block for SoundStream inspired decoder.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The kernel size of the convolutional layers in the residual
            blocks.
        stride (int): The stride of the upsampling convolutional layer (kernel size is
            2 * stride).
        dilations (List[int]): The dilations of the convolutional layers in the residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilations: List[int],
    ):
        super().__init__()
        net = [torch.nn.ConvTranspose1d(in_channels, out_channels, 2 * stride, stride)]
        net.append(torch.nn.ReLU())
        for dilation in dilations:
            net.append(ResidualBlock(out_channels, out_channels, kernel_size, dilation))
        self.net = torch.nn.Sequential(*net)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.net(x)


class SoundStreamDecoder(torch.nn.Module):
    """
    SoundStream inspired decoder -- uses only three decoder blocks instead of the
    original four -- this is because we're using this to map from a latent space
    to time-varying parameters for a noise synthesizer so we don't reduce the
    channels by as much.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strides: List[int],
        transpose_output: bool = False,
    ) -> None:
        super().__init__()

        # Input layer
        self.net = [torch.nn.Conv1d(in_channels, 256, 1)]

        assert len(strides) == 3, "Supports only 3 decoder blocks."
        self.net.append(DecoderBlock(256, 128, 7, strides[0], [1, 3, 9]))
        self.net.append(DecoderBlock(128, 64, 7, strides[1], [1, 3, 9]))
        self.net.append(DecoderBlock(64, 64, 7, strides[2], [1, 3, 9]))

        # Output layer
        self.net.append(torch.nn.Conv1d(64, out_channels, 1))
        self.net = torch.nn.Sequential(*self.net)
        self.transpose_output = transpose_output

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.net(x)
        if self.transpose_output:
            x = rearrange(x, "b c t -> b t c")

        return x
