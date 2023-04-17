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


class DrumBlender(pl.LightningModule):
    """
    LightningModule for kick synthesis from a modal frequency input

    Args:
        model(nn.Module): A PyTorch model, which will be trained.
        loss_fn(Callable): A loss function
        conditioning_model(nn.Module): A PyTorch model, which will be used to
            condition the synthesis model. If None, the modal input will be used.
        embedding_model(nn.Module): A PyTorch model, which will be used to create
            an embedding from the original audio or pre-computed features.
            This embedding is then passed into the synthesis model as the
            FiLM conditioning. If None, raw features (if available) will be used.
        save_test_audio(bool): If True, the audio from the test batch will be saved
        save_audio_sr(int): The sample rate to use when saving audio
        batch_ids_to_save(Tuple[int, ...]): The batch ids to save audio for. If None,
            all batches will be saved.
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
        self.modal_autoencoder = modal_autoencoder

        self.noise_synth = noise_synth
        self.noise_autoencoder = noise_autoencoder

        self.transient_synth = transient_synth
        self.transient_autoencoder = transient_autoencoder

        self.encoder = encoder

        self.loss_fn = loss_fn

        if float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(float32_matmul_precision)

    def forward(
        self,
        original: torch.Tensor,
        params: torch.Tensor,
    ):
        # Main embedding
        embedding = None
        if self.encoder is not None:
            embedding = self.encoder(original)

        # Autoencoder
        modal_params = self.modal_autoencoder(params, embedding)

        noise_params = self.noise_autoencoder(embedding)
        transient_params = self.transient_autoencoder(embedding)

        # Synthesis
        modes = self.modal_synth(modal_params)

        modes_transients = self.transient_synth(modes, transient_params)

        noise = self.noise_synth(noise_params)

        y_hat = modes_transients + noise
        return y_hat

    def _do_step(self, batch: Tuple[torch.Tensor, ...]):
        if len(batch) == 2:
            original, params = batch
        else:
            raise ValueError("Expected batch to be a tuple of length 2 or 3")

        y_hat = self(original, params)

        # y_hat = self(original, modal, features)
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
