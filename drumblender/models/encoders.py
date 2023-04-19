"""
Encoders for the synthesis models
"""
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from einops import repeat


class DummyParameterEncoder(torch.nn.Module):
    """
    Dummy encoder returns a set of learnable parameters
    """

    def __init__(self, param_shape: Union[Tuple, torch.Size]):
        super().__init__()
        self.params = torch.nn.Parameter(torch.rand(param_shape))

    def forward(self, x: torch.tensor, params: Optional[torch.tensor] = None):
        return self.params


class ModalAmpParameters(DummyParameterEncoder):
    """
    Modal amp parameters encoder -- multiplies the amplitude of the modal
    parameters by a set of learnable parameters. Static per mode amplitude modulation
    """

    def __init__(self, num_modes: int):
        super().__init__(torch.Size([num_modes]))

    def forward(
        self, params: Optional[torch.tensor], embedding: Optional[torch.tensor] = None
    ):
        assert params.ndim == 4
        batch_size, num_params, num_modes, num_steps = params.shape
        assert num_params == 3

        amp_mod = repeat(self.params, "m -> b m 1", b=batch_size)
        amp_mod = amp_mod[:, :num_modes]

        params = torch.chunk(params, 3, dim=1)
        params = [p.squeeze(1) for p in params]

        params[1] = params[1] * amp_mod

        return torch.stack(params, dim=1)


class NoiseParameters(torch.nn.Module):
    def __init__(self, param_shape: Union[Tuple, torch.Size], scale: float = 0.002):
        super().__init__()
        p = torch.randn(param_shape) * scale
        self.params = torch.nn.Parameter(p)

    def forward(self, x: torch.tensor, params: Optional[torch.tensor] = None):
        return self.params


class AutoEncoder(torch.nn.Module):
    """
    Vanilla autoencoder for the synthesis parameters

    Args:
        encoder: Encoder module
        decoder: Decoder module
        latent_size: Size of the latent space
        return_latent: Whether to return the latent space
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_size: int,
        return_latent: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size
        self.return_latent = return_latent

    def forward(
        self, x: torch.tensor, params: Optional[torch.tensor] = None
    ) -> Union[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        z = self.encoder(x)
        x = self.decoder(z)

        assert z.shape[-1] == self.latent_size, "Latent size mismatch"
        if self.return_latent:
            return x, z

        return x
