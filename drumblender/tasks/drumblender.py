"""
LightningModule for drum synthesis
"""
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange


class DrumBlender(pl.LightningModule):
    """
    LightningModule for kick synthesis from a modal frequency input

    # TODO: Alot of these are currently optional to help with testing and devlopment,
    # but they should be required in the future

    Args:
        modal_synth (nn.Module): Synthesis model takes modal parameters and generates
            audio
        loss_fn (Union[Callable, nn.Module]): Loss function to use for training
        noise_synth (Optional[nn.Module]): Receives noise parameters and generates
            noise audio signal
        transient_synth (Optional[nn.Module]): Receives audio plus transient parameters
            and generates transient audio signal
        modal_autoencoder (Optional[nn.Module]): Receives main embedding and
            generates modal parameters
        noise_autoencoder (Optional[nn.Module]): Receives main embedding and
            generates noise parameters
        transient_autoencoder (Optional[nn.Module]): Receives main embedding and
            generates transient parameters
        encoder (Optional[nn.Module]): Receives audio and generates main embedding
        float32_matmul_precision(Literal["medium", "high", "highest", None]): Sets
            the precision of float32 matmul operations.
    """

    def __init__(
        self,
        modal_synth: nn.Module,
        loss_fn: Union[Callable, nn.Module],
        noise_synth: Optional[nn.Module] = None,
        transient_synth: Optional[nn.Module] = None,
        modal_autoencoder: Optional[nn.Module] = None,
        noise_autoencoder: Optional[nn.Module] = None,
        transient_autoencoder: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        float32_matmul_precision: Literal["medium", "high", "highest", None] = None,
    ):
        super().__init__()

        self.modal_synth = modal_synth
        self.loss_fn = loss_fn
        self.noise_synth = noise_synth
        self.transient_synth = transient_synth
        self.modal_autoencoder = modal_autoencoder
        self.noise_autoencoder = noise_autoencoder
        self.transient_autoencoder = transient_autoencoder
        self.encoder = encoder

        if float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(float32_matmul_precision)

    def forward(
        self,
        original: torch.Tensor,
        params: torch.Tensor,
    ):
        # Main embeddings
        embedding = None
        if self.encoder is not None:
            embedding = self.encoder(original)

        # Modal parameter autoencoder
        modal_params = params
        if self.modal_autoencoder is not None:
            modal_params, _ = self.modal_autoencoder(embedding, params)

        noise_params = None
        if self.noise_autoencoder is not None:
            noise_params, _ = self.noise_autoencoder(embedding)

        transient_params = None
        if self.transient_autoencoder is not None:
            transient_params, _ = self.transient_autoencoder(embedding)

        # Synthesis
        y_hat = self.modal_synth(modal_params, original.shape[-1])

        if self.noise_synth is not None:
            assert noise_params is not None, "Noise params must be provided"
            noise = self.noise_synth(noise_params, original.shape[-1])
            noise = rearrange(noise, "b n -> b () n")
            y_hat = y_hat + noise

        if self.transient_synth is not None:
            y_hat = self.transient_synth(y_hat, transient_params)

        return y_hat

    def _do_step(self, batch: Tuple[torch.Tensor, ...]):
        if len(batch) == 2:
            original, params = batch
        else:
            raise ValueError("Expected batch to be a tuple of length 2")

        y_hat = self(original, params)
        loss = self.loss_fn(y_hat, original)
        return loss, y_hat

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, _ = self._do_step(batch)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, _ = self._do_step(batch)
        self.log("validation/loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, y_hat = self._do_step(batch)
        self.log("test/loss", loss)
        return loss
