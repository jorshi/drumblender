import subprocess
from functools import partial
from functools import reduce
from pathlib import Path
from types import MethodType
from typing import Any
from typing import Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning.loggers import WandbLogger
from wandb import Audio


class LogAudioCallback(Callback):
    """Log audio samples to Weights & Biases."""

    model: pl.LightningModule
    stored_forward: MethodType

    def __init__(
        self,
        on_train: bool,
        on_val: bool,
        on_test: bool,
        save_audio_sr: int = 48000,
        n_batches: int = 1,
        log_on_epoch_end: bool = False,
    ):
        self.on_train = on_train
        self.on_val = on_val
        self.on_test = on_test

        self.save_audio_sr = save_audio_sr
        self.n_batches = n_batches

        self.saved_targets: dict[str, list[Any]] = dict(train=[], val=[], test=[])
        self.saved_reconstructions: dict[str, list[Any]] = dict(
            train=[], val=[], test=[]
        )

        self.log_on_epoch_end = log_on_epoch_end

    # Store a local reference to the model on setup
    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        self.model = pl_module

    def _wrap_forward(self, split: str) -> None:
        self.stored_forward = self.model.forward

        # Wrap the model's forward method to save the audio
        def wrapped_forward(self, *args, callback=None, split=None, **kwargs):
            output = callback.stored_forward(*args, **kwargs)
            callback._save_batch(output, split, "reconstruction")
            return output

        wrapped_forward = partial(
            MethodType(wrapped_forward, self.model),
            callback=self,
            split=split,
        )

        self.model.forward = wrapped_forward  # type: ignore

    def _unwrap_forward(self) -> None:
        self.model.forward = self.stored_forward  # type: ignore

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.on_train and batch_idx < self.n_batches:
            self._wrap_forward("train")
            self._save_batch(batch[0], "train", "target")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.on_train and batch_idx < self.n_batches:
            self._unwrap_forward()
        elif (
            self.on_train and batch_idx == self.n_batches and not self.log_on_epoch_end
        ):
            self._log_audio("train")
            self._clear_saved_batches("train")

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.on_train and self.log_on_epoch_end:
            self._log_audio("train")
        self._clear_saved_batches("train")

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.on_val and batch_idx < self.n_batches:
            self._wrap_forward("val")
            self._save_batch(batch[0], "val", "target")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.on_val and batch_idx < self.n_batches:
            self._unwrap_forward()
        elif self.on_val and batch_idx == self.n_batches and not self.log_on_epoch_end:
            self._log_audio("val")
            self._clear_saved_batches("val")

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.on_val and self.log_on_epoch_end:
            self._log_audio("val")
        self._clear_saved_batches("val")

    def on_test_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.on_test and batch_idx < self.n_batches:
            self._wrap_forward("test")
            self._save_batch(batch[0], "test", "target")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.on_test and batch_idx < self.n_batches:
            self._unwrap_forward()
        elif self.on_test and batch_idx == self.n_batches and not self.log_on_epoch_end:
            self._log_audio("test")
            self._clear_saved_batches("test")

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if self.on_test and self.log_on_epoch_end:
            self._log_audio("test")
        self._clear_saved_batches("test")

    def _save_batch(
        self, batch: Any, split: str, type: Literal["target", "reconstruction"]
    ) -> None:
        batch = batch.detach().cpu()
        if type == "target":
            self.saved_targets[split].append(batch)
        elif type == "reconstruction":
            self.saved_reconstructions[split].append(batch)
        else:
            raise ValueError(f"Unknown type {type}")

    def _clear_saved_batches(self, split: str) -> None:
        self.saved_targets[split] = []
        self.saved_reconstructions[split] = []

    def _log_audio(self, split: str) -> None:
        if (
            len(self.saved_targets[split]) == 0
            or len(self.saved_reconstructions[split]) == 0
        ):
            return

        targets = torch.cat(self.saved_targets[split], dim=0)
        reconstructions = torch.cat(self.saved_reconstructions[split], dim=0)

        signals = reduce(
            lambda x, y: x + y, zip(targets, reconstructions)  # type: ignore
        )
        audio_signal = torch.hstack(signals).squeeze().cpu().numpy()

        audio = Audio(
            audio_signal,
            caption=f"{split}/audio",
            sample_rate=self.save_audio_sr,
        )
        if self.model.logger is not None:
            self.model.logger.experiment.log({f"{split}/audio": audio})


class CleanWandbCacheCallback(pl.Callback):
    def __init__(self, every_n_epochs: int = 1, max_size_in_gb: float = 1.0):
        self.every_n_epochs = every_n_epochs
        self.gb_str = f"{max_size_in_gb}GB"

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            subprocess.Popen(["wandb", "artifact", "cache", "cleanup", self.gb_str])


class SaveConfigCallbackWanb(SaveConfigCallback):
    """
    Custom callback to move the config file saved by LightningCLI to the
    experiment directory created by WandbLogger. This has a few benefits:
    1. The config file is saved in the same directory as the other files created
         by wandb, so it's easier to find.
    2. The config file is uploaded to wandb and can be viewed in the UI.
    3. Subsequent runs won't be blocked by the config file already existing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        super().setup(trainer, pl_module, stage)
        if isinstance(trainer.logger, WandbLogger):
            config = Path(trainer.log_dir).joinpath("config.yaml")
            assert config.exists()
            experiment_dir = Path(trainer.logger.experiment.dir)

            # If this is the first time using wandb logging on this machine,
            # the experiment directory won't exist yet.
            if not experiment_dir.exists():
                experiment_dir.mkdir(parents=True)

            config.rename(experiment_dir.joinpath("model-config.yaml"))
