"""
Tests for the `kick2kick.loss` module.
"""
import jsonargparse
import pytest
import torch

import kick2kick.loss as loss


def test_first_order_difference_loss():
    loss_fn = loss.FirstOrderDifferenceLoss()
    pred = torch.ones(1, 1, 100)
    target = torch.ones(1, 1, 100)
    assert loss_fn(pred, target) == 0.0


def test_weighted_loss_forwards():
    loss_fn = loss.WeightedLoss(
        [torch.nn.L1Loss(), torch.nn.L1Loss()], weights=[2.0, 1.0]
    )
    pred = torch.ones(1, 1, 100)
    target = torch.zeros(1, 1, 100)
    assert loss_fn(pred, target) == 3.0


def test_weighted_loss_forwards_no_weights():
    loss_fn = loss.WeightedLoss(
        [torch.nn.L1Loss(), torch.nn.L1Loss()],
    )
    pred = torch.ones(1, 1, 100)
    target = torch.zeros(1, 1, 100)

    # Default weights are 1.0
    assert loss_fn(pred, target) == 2.0


def test_weighted_loss_different_weights():
    with pytest.raises(AssertionError):
        loss.WeightedLoss([torch.nn.L1Loss()], weights=[2.0, 1.0])


def test_weighted_loss_with_jsonargparse_config(monkeypatch):
    # Monkeypatch the torch.nn.L1Loss and torch.nn.MSELoss classes to return a constant
    # value so that we can test the weighted loss.
    monkeypatch.setattr(torch.nn.L1Loss, "forward", lambda self, x, y: 1.0)
    monkeypatch.setattr(torch.nn.MSELoss, "forward", lambda self, x, y: 20.0)
    expected_loss = 4.0

    config = (
        "loss:\n"
        "  class_path: kick2kick.loss.WeightedLoss\n"
        "  init_args:\n"
        "    loss_fns: \n"
        "    - class_path: torch.nn.L1Loss\n"
        "      init_args:\n"
        "        reduction: mean\n"
        "    - class_path: torch.nn.MSELoss\n"
        "      init_args:\n"
        "        reduction: sum\n"
        "    weights: [2.0, 0.1]"
    )

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--loss", type=torch.nn.Module)
    args = parser.parse_string(config)
    objs = parser.instantiate_classes(args)

    shape = 13, 4, 9, 2
    a = torch.testing.make_tensor(*shape, dtype=torch.float32, device="cpu")
    b = torch.testing.make_tensor(*shape, dtype=torch.float32, device="cpu")

    actual_loss = objs.loss(a, b)
    assert actual_loss == expected_loss
