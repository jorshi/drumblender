"""
Tests for the EAT model. Full testing of each individual component is probably
overkill at this point. These are some tests to make sure the model is producing
the expected output given some input audio.
"""
import pytest
import torch

from drumblender.models import EAT


@pytest.mark.parametrize("embed_dim", [32, 64])
def test_eat_embedding_dim(embed_dim):
    channels = 1
    batch_size = 2
    seq_len = 48000

    fake_audio = torch.testing.make_tensor(
        (batch_size, channels, seq_len),
        dtype=torch.float32,
        device="cpu",
        low=-1.0,
        high=1.0,
    )

    eat = EAT(input_length=seq_len, embed_dim=embed_dim)
    embedding = eat(fake_audio)
    assert embedding.shape == (batch_size, embed_dim)


@pytest.mark.parametrize("n_classes", [2, 16])
def test_eat_project_to_classes(n_classes):
    channels = 1
    batch_size = 2
    seq_len = 48000
    embed_dim = 128

    fake_audio = torch.testing.make_tensor(
        (batch_size, channels, seq_len),
        dtype=torch.float32,
        device="cpu",
        low=-1.0,
        high=1.0,
    )

    eat = EAT(input_length=seq_len, embed_dim=embed_dim, n_classes=n_classes)
    embedding = eat(fake_audio)
    assert embedding.shape == (batch_size, n_classes)


def test_eat_short_input_fails():
    factors = [4, 4, 4, 4]
    min_length = EAT.minimum_input_length(factors)

    with pytest.raises(ValueError):
        EAT(input_length=min_length - 1)


@pytest.mark.parametrize(
    "factors",
    [
        (
            4,
            4,
            4,
            4,
        ),
        (
            2,
            2,
            2,
        ),
    ],
)
def test_eat_min_length_input_succeeds(factors):
    min_length = EAT.minimum_input_length(factors)
    eat = EAT(input_length=min_length, factors=factors)

    channels = 1
    batch_size = 2
    embed_dim = 128

    fake_audio = torch.testing.make_tensor(
        (batch_size, channels, min_length),
        dtype=torch.float32,
        device="cpu",
        low=-1.0,
        high=1.0,
    )

    embedding = eat(fake_audio)
    assert embedding.shape == (batch_size, embed_dim)
