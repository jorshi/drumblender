"""
Helpers for loss functions.
"""
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import torch


class FirstOrderDifferenceLoss(torch.nn.Module):
    """
    A loss function that calculates the first-order difference
    of the input and target tensors and then calculates the L1
    loss between the two. This essentially applies a high-pass
    filter to the signal before calculating the loss, which may
    potentially be useful for emphasizing transient components.

    Args:
        reduction (str): The reduction method to use, passed
            into the L1 loss. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_diff = torch.diff(pred)
        target_diff = torch.diff(target)
        return self.loss(pred_diff, target_diff)


class WeightedLoss(torch.nn.Module):
    """
    A loss function that combines and sums weightings of multiple loss functions.

    Args:
        losses: A list of loss functions.
        weights: A list of weights for each loss function. Defaults to None, which
            results in equal weighting of all loss functions.
    """

    def __init__(
        self,
        loss_fns: List[Union[Callable, torch.nn.Module]],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.loss_fns = loss_fns
        if weights is None:
            weights = [1.0] * len(loss_fns)
        else:
            assert len(loss_fns) == len(
                weights
            ), "Number of losses and weights must match."
        self.weights = weights

    def forward(self, *args, **kwargs):
        losses = [
            weight * loss_fn(*args, **kwargs)
            for loss_fn, weight in zip(self.loss_fns, self.weights)
        ]

        return sum(losses)
