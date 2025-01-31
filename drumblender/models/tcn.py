from typing import Literal
from typing import Optional

import torch
from torch import nn

from drumblender.models.components import FiLM
from drumblender.models.components import GatedActivation
from drumblender.models.components import Pad
from drumblender.models.components import TFiLM

__all__ = ["TCN"]


def _get_activation(activation: str):
    if activation == "gated":
        return GatedActivation()
    return getattr(nn, activation)()


class _DilatedResidualBlock(nn.Module):
    """Temporal convolutional network internal block

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Dilation factor.
        causal (bool, optional): Whether to use causal padding. Defaults to True.
        norm (Literal["batch", "instance", None], optional): Normalization type.
        activation (str, optional): Activation function in `torch.nn` or "gated".
            Defaults to "GELU".
        film_conditioning (bool, optional): Whether to use FiLM conditioning. Defaults
            to False.
        film_embedding_size (int, optional): Size of the FiLM embedding. Defaults to
            None.
        film_batch_norm (bool, optional): Whether to use batch normalization in FiLM.
            Defaults to True.
        use_temporal_film (bool, optional): Whether to use TFiLM conditioning. Defaults
            to False.
        temporal_film_block_size (int, optional): TFiLM block size. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        causal: bool = True,
        norm: Literal["batch", "instance", None] = None,
        activation: str = "GELU",
        film_conditioning: bool = False,
        film_embedding_size: Optional[int] = None,
        film_batch_norm: bool = True,
        use_temporal_film: bool = False,
        temporal_film_block_size: Optional[int] = None,
    ):
        super().__init__()

        if film_conditioning and (
            film_embedding_size is None
            or not isinstance(film_embedding_size, int)
            or film_embedding_size < 1
        ):
            raise ValueError(
                "FiLM conditioning requires a valid embedding size (int >= 1)."
            )

        if use_temporal_film and (
            temporal_film_block_size is None
            or not isinstance(temporal_film_block_size, int)
            or temporal_film_block_size < 1
        ):
            raise ValueError(
                "TFiLM conditioning requires a valid block size (int >= 1)."
            )

        net = []

        pre_activation_channels = (
            out_channels * 2 if activation == "gated" else out_channels
        )

        if norm is not None:
            if norm not in ("batch", "instance"):
                raise ValueError("Invalid norm type (must be batch or instance)")
            _Norm = nn.BatchNorm1d if norm == "batch" else nn.InstanceNorm1d
            net.append(_Norm(in_channels))

        net.extend(
            [
                Pad(kernel_size, dilation, causal=causal),
                nn.Conv1d(
                    in_channels,
                    pre_activation_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=0,
                ),
            ]
        )
        self.net = nn.Sequential(*net)

        self.film = (
            FiLM(film_embedding_size, pre_activation_channels, film_batch_norm)
            if film_conditioning
            else None
        )

        self.activation = _get_activation(activation)

        self.tfilm = (
            TFiLM(out_channels, temporal_film_block_size) if use_temporal_film else None
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None):
        activations = self.net(x)
        if self.film is not None:
            activations = self.film(activations, film_embedding)
        y = self.activation(activations)

        if self.tfilm is not None:
            y = self.tfilm(y)

        return y + self.residual(x)


class TCN(nn.Module):
    """Temporal convolutional network

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        dilation_base (int, optional): Base of the dilation factor. Defaults to 2.
        num_layers (int, optional): Number of layers. Defaults to 8.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
        causal (bool, optional): Whether to use causal padding. Defaults to True.
        norm (Literal["batch", "instance", None], optional): Normalization type.
        activation (str, optional): Activation function in `torch.nn` or "gated".
            Defaults to "GELU".
        film_conditioning (bool, optional): Whether to use FiLM conditioning. Defaults
            to False.
        film_embedding_size (int, optional): FiLM embedding size. Defaults to None.
        film_batch_norm (bool, optional): Whether to use batch normalization in FiLM.
            Defaults to True.
        use_temporal_film (bool, optional): Whether to use TFiLM conditioning. Defaults
            to False.
        temporal_film_block_size (int, optional): TFiLM block size. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dilation_base: int = 2,
        dilation_blocks: Optional[int] = None,
        num_layers: int = 8,
        kernel_size: int = 3,
        causal: bool = True,
        norm: Literal["batch", "instance", None] = None,
        activation: str = "GELU",
        film_conditioning: bool = False,
        film_embedding_size: Optional[int] = None,
        film_batch_norm: bool = True,
        use_temporal_film: bool = False,
        temporal_film_block_size: Optional[int] = None,
    ):
        super().__init__()

        self.in_projection = nn.Conv1d(in_channels, hidden_channels, 1)
        self.out_projection = nn.Conv1d(hidden_channels, out_channels, 1)

        net = []
        dilation_blocks = dilation_blocks if dilation_blocks is not None else num_layers
        for n in range(num_layers):
            dilation = dilation_base ** (n % dilation_blocks)
            net.append(
                _DilatedResidualBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    causal=causal,
                    norm=norm,
                    activation=activation,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_temporal_film=use_temporal_film,
                    temporal_film_block_size=temporal_film_block_size,
                )
            )

        self.net = nn.ModuleList(net)

    def forward(self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None):
        x = self.in_projection(x)
        for layer in self.net:
            x = layer(x, film_embedding)
        x = self.out_projection(x)

        return x
