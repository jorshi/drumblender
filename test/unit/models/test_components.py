import torch

from drumblender.models.components import FiLM
from drumblender.models.components import GatedActivation
from drumblender.models.components import TFiLM


def test_film_correctly_forwards_input():
    batch_size = 11
    in_channels = 13
    seq_len = 31
    film_embedding_size = 7

    film = FiLM(film_embedding_size, in_channels)
    x = torch.testing.make_tensor(
        batch_size, in_channels, seq_len, device="cpu", dtype=torch.float32
    )
    film_embedding = torch.testing.make_tensor(
        batch_size,
        film_embedding_size,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )

    y = film(x, film_embedding)
    assert y.shape == (batch_size, in_channels, seq_len)

    (dy_dx,) = torch.autograd.grad(y.sum().square(), film_embedding)
    assert (dy_dx.abs() > 0.0).all()


def test_film_can_toggle_batch_norm(mocker):
    spy_batch_norm_init = mocker.spy(torch.nn.BatchNorm1d, "__init__")
    spy_batch_norm_forward = mocker.spy(torch.nn.BatchNorm1d, "forward")

    batch_size = 7
    in_channels = 13
    seq_len = 37
    film_embedding_size = 5

    x = torch.testing.make_tensor(
        batch_size, in_channels, seq_len, device="cpu", dtype=torch.float32
    )
    film_embedding = torch.testing.make_tensor(
        batch_size, film_embedding_size, device="cpu", dtype=torch.float32
    )

    # Test with batch norm
    film = FiLM(film_embedding_size, in_channels, use_batch_norm=True)
    film(x, film_embedding)
    assert spy_batch_norm_init.call_count == 1
    assert spy_batch_norm_forward.call_count == 1

    # Test without batch norm
    film = FiLM(film_embedding_size, in_channels, use_batch_norm=False)
    film(x, film_embedding)
    assert spy_batch_norm_init.call_count == 1
    assert spy_batch_norm_forward.call_count == 1


def test_gated_activation_correctly_forwards_input():
    batch_size = 11
    out_channels = 17
    seq_len = 23

    ga = GatedActivation()
    x = torch.testing.make_tensor(
        batch_size, out_channels * 2, seq_len, device="cpu", dtype=torch.float32
    )

    y = ga(x)
    assert y.shape == (batch_size, out_channels, seq_len)


def test_gated_activation_gates_input():
    batch_size = 11
    out_channels = 17
    seq_len = 23

    ga = GatedActivation()
    x_1 = torch.testing.make_tensor(
        batch_size, out_channels, seq_len, device="cpu", dtype=torch.float32
    )
    x_2 = torch.testing.make_tensor(
        batch_size,
        out_channels,
        seq_len,
        device="cpu",
        dtype=torch.float32,
        low=-1e8,
        high=-1e8,
    )
    x = torch.cat([x_1, x_2], dim=1)

    y = ga(x)
    assert y.shape == (batch_size, out_channels, seq_len)

    assert y.abs().sum() == 0.0


def test_tfilm_correctly_forwards_input():
    batch_size = 3
    channels = 11
    block_size = 16
    seq_len = block_size * 10

    tfilm = TFiLM(
        channels=channels,
        block_size=block_size,
    )

    x = torch.testing.make_tensor(
        batch_size, channels, seq_len, device="cpu", dtype=torch.float32
    )

    y = tfilm(x)
    assert y.shape == (batch_size, channels, seq_len)
