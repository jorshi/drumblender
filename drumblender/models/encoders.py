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
    def __init__(self, param_shape: Union[Tuple, torch.Size]):
        super().__init__()
        p = torch.randn(param_shape) * 0.002
        self.params = torch.nn.Parameter(p)

    def forward(self, x: torch.tensor, params: Optional[torch.tensor] = None):
        return self.params
