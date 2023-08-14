"""
Encoders for the synthesis models
"""
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from einops import repeat


class VariationalEncoder(torch.nn.Module):
    """Variational Encoder wrapper for torch modules implementing Encoders.

    Assumes: Encoder Module parameterizes a Multimodal Gaussian distribution.
      Furthermore, data prior P(Z) is Gaussian with zero mean and unit std.

      Under these assumptions, implements the reparametrization trick and
      computes the KL Divergence.

    Features "warmed up" flag for stopping gradient propagation after
      representation learning stage. (could be useful).

    Extracted from RAVE: https://github.com/acids-ircam/RAVE/

    """

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("warmed_up", torch.tensor(0))

    def reparametrize(self, z: torch.tensor):
        """Sample from a parameterized gaussian given as an input.
        Args:
          z (torch.tensor): A batch of inputs where the parameterized Gaussian
            is at dim=1.

        Returns:
          A tuple containing the sampled vector (with dim=1 halved),
          and the kl divergence.
        """
        mean, scale = z.chunk(2, 1)
        std = torch.nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return z, kl

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.warmed_up:
            z = z.detach()
        return z


class DummyParameterEncoder(torch.nn.Module):
    """
    Dummy encoder returns a set of learnable parameters
    """

    def __init__(self, param_shape: Union[Tuple, torch.Size], ones: bool = False):
        super().__init__()
        val = torch.rand(param_shape)
        if ones:
            val = torch.ones(param_shape)
        self.params = torch.nn.Parameter(val)

    def forward(self, x: torch.tensor, params: Optional[torch.tensor] = None):
        return (self.params, None)


class ModalAmpParameters(DummyParameterEncoder):
    """
    Modal amp parameters encoder -- multiplies the amplitude of the modal
    parameters by a set of learnable parameters. Static per mode amplitude modulation
    """

    def __init__(self, num_modes: int, **kwargs):
        super().__init__(torch.Size([num_modes]), **kwargs)

    def forward(
        self,
        embedding: Optional[torch.tensor] = None,
        params: Optional[torch.tensor] = None,
    ):
        assert params.ndim == 4
        batch_size, num_params, num_modes, num_steps = params.shape
        assert num_params == 3

        amp_mod = repeat(self.params, "m -> b m 1", b=batch_size)
        amp_mod = amp_mod[:, :num_modes]

        params = torch.chunk(params, 3, dim=1)
        params = [p.squeeze(1) for p in params]

        params[1] = params[1] * amp_mod

        return (torch.stack(params, dim=1), None)


class NoiseParameters(torch.nn.Module):
    def __init__(self, param_shape: Union[Tuple, torch.Size], scale: float = 0.002):
        super().__init__()
        p = torch.randn(param_shape) * scale
        self.params = torch.nn.Parameter(p)

    def forward(self, x: torch.tensor, params: Optional[torch.tensor] = None):
        return (self.params, None)


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
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, ...]:
        z = self.encoder(x)
        x = self.decoder(z)

        assert z.shape[-1] == self.latent_size, "Latent size mismatch"
        return x, z


class ModalAutoEncoder(torch.nn.Module):
    def __init__(self, autoencoder: torch.nn.Module, mod_dim: int) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.mod_dim = mod_dim

    def forward(self, x: torch.tensor, params: torch.tensor) -> torch.tensor:
        assert params.ndim == 4
        batch_size, num_params, num_modes, num_steps = params.shape
        assert num_params == 3

        params = torch.chunk(params, 3, dim=1)
        params = [p.squeeze(1) for p in params]

        x, z = self.autoencoder(x)

        # Modulate the amplitude of the modal parameters
        params[self.mod_dim] = params[self.mod_dim] * x[..., None]
        params = torch.stack(params, dim=1)

        return params, z
