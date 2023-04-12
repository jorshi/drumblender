"""LightningModule representing a regression task.
"""
from typing import Callable
from typing import Union

import pytorch_lightning as pl
from torch import nn


class Regression(pl.LightningModule):
    """Regression task implemented as a LightningModule.

    Args:
      model(nn.Module): A PyTorch model, which will be trained.
      loss_fn(Callable): A loss function
      optimizer(torch.optim.Optimizer): A PyTorch optimizer

    Returns:

    """

    def __init__(self, model: nn.Module, loss_fn: Union[Callable, nn.Module]):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        """

        Args:
          x:

        Returns:

        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """

        Args:
          batch:
          batch_idx:

        Returns:

        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """

        Args:
          batch:
          batch_idx:

        Returns:

        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """

        Args:
          batch:
          batch_idx:

        Returns:

        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test/loss", loss)
        return loss
