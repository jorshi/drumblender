import pytest
import torch

from percussionsynth.models import MLP


@pytest.fixture
def mlp():
    return MLP(
        input_size=2,
        hidden_size=10,
        output_size=1,
        hidden_layers=2,
    )


def test_mlp_correctly_forwards_input(mlp):
    x = torch.rand(2)
    y = mlp(x)
    assert y.shape == (1,)


def test_mlp_correctly_forwards_batch(mlp):
    x = torch.rand(2, 2)
    y = mlp(x)
    assert y.shape == (2, 1)
