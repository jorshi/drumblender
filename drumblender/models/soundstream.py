from typing import Optional

import torch
from einops import rearrange
from torch import nn

from drumblender.models.components import FiLM
from drumblender.models.components import Pad


class _SoundStreamResidualUnit(nn.Module):
    def __init__(
        self,
        width: int,
        dilation: int,
        kernel_size: int = 7,
        causal: bool = False,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
    ):
        super().__init__()

        self.net = nn.Sequential(
            Pad(kernel_size, dilation, causal=causal),
            nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=0),
            nn.ELU(),
            nn.Conv1d(width, width, 1),
        )

        self.final_activation = nn.ELU()

        if film_conditioning:
            self.film = FiLM(
                film_embedding_size,
                width,
                film_batch_norm,
            )
        else:
            self.film = None

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = self.net(x)
        if self.film is not None:
            y = self.film(y, film_embedding)
        return x + self.final_activation(y)


class _SoundStreamEncoderBlock(nn.Module):
    def __init__(
        self,
        width: int,
        stride: int,
        kernel_size: int = 7,
        causal: bool = False,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
    ):
        super().__init__()

        self.net = nn.ModuleList(
            [
                _SoundStreamResidualUnit(
                    width // 2,
                    1,
                    kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                ),
                _SoundStreamResidualUnit(
                    width // 2,
                    3,
                    kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                ),
                _SoundStreamResidualUnit(
                    width // 2,
                    9,
                    kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                ),
            ]
        )
        self.output = nn.Sequential(
            Pad(2 * stride, 1, causal=causal),
            nn.Conv1d(width // 2, width, 2 * stride, stride=stride, padding=0),
            nn.ELU(),
        )

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.net:
            x = layer(x, film_embedding)
        return self.output(x)


class SoundStreamEncoder(nn.Module):

    """Convolutional waveform encoder from SoundStream model, without vector
    quantization.

    Args:
        input_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        output_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size. Defaults to 7.
        strides (tuple[int, ...], optional): Strides. Defaults to (2, 2, 4, 4).
        causal (bool, optional): Whether to use causal padding. Defaults to False.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        kernel_size: int = 7,
        strides: tuple[int, ...] = (2, 2, 4, 4),
        causal: bool = False,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
        transpose_output: bool = False,
    ):
        super().__init__()

        self.input = nn.Sequential(
            Pad(kernel_size, 1, causal=causal),
            nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=0),
        )

        encoder_blocks = []
        for stride in strides:
            hidden_channels *= 2
            encoder_blocks.append(
                _SoundStreamEncoderBlock(
                    hidden_channels,
                    stride,
                    kernel_size=kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                )
            )
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        self.output = nn.Sequential(
            Pad(3, 1, causal=causal),
            nn.Conv1d(hidden_channels, output_channels, 3, padding=0),
        )
        self.transpose_output = transpose_output

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.input(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, film_embedding)

        if self.transpose_output:
            x = self.output(x)
            x = rearrange(x, "b c t -> b t c")
            return x

        return self.output(x)
