"""
LightningModule for kick synthesis from a modal frequency input
"""
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn


class KickSynth(pl.LightningModule):
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
        model: nn.Module,
        loss_fn: Union[Callable, nn.Module],
        conditioning_model: Optional[nn.Module] = None,
        embedding_model: Optional[nn.Module] = None,
        float32_matmul_precision: Literal["medium", "high", "highest", None] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.conditioning_model = conditioning_model
        self.embedding_model = embedding_model

        if float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(float32_matmul_precision)

    def forward(
        self,
        original: torch.Tensor,
        modal: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ):
        if self.conditioning_model is not None:
            conditioning = self.conditioning_model(original)
        else:
            conditioning = modal

        # If using an embedding model, then pass either the pre-computed features
        # or the original audio into the embedding model. Otherwise, pass the features
        # directly into the synthesis model.
        if self.embedding_model is not None:
            if features is not None:
                embedding = self.embedding_model(features)
            else:
                embedding = self.embedding_model(original)
        else:
            embedding = features

        return self.model(conditioning, embedding)

    def _do_step(self, batch: Tuple[torch.Tensor, ...]):
        if len(batch) == 2:
            original, modal = batch
            features = None
        elif len(batch) == 3:
            original, modal, features = batch
        else:
            raise ValueError("Expected batch to be a tuple of length 2 or 3")

        y_hat = self(original, modal, features)
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
