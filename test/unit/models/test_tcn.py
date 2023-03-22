import pytest
import torch

from percussionsynth.models import KickTCN
from percussionsynth.models import TCN
from percussionsynth.models.components import Pad


def test_pad_correctly_applies_causal_padding():
    batch_size, in_channels, seq_len = 1, 1, 10
    expected_seq_len = 14

    kernel_size = 3
    dilation = 2

    expected_padding = 4

    pad = Pad(kernel_size=kernel_size, dilation=dilation, causal=True)
    x = torch.testing.make_tensor(
        (batch_size, in_channels, seq_len),
        device="cpu",
        dtype=torch.float32,
        low=1.0,
        high=10.0,
    )
    padded_x = pad(x)

    assert padded_x.shape == (batch_size, in_channels, expected_seq_len)

    # Check that causal padding is applied correctly
    assert padded_x[0, 0, 0:expected_padding].abs().sum() == 0.0


def test_pad_correctly_applies_non_causal_padding():
    batch_size, in_channels, seq_len = 1, 1, 15
    expected_seq_len = 21

    kernel_size = 3
    dilation = 3

    expected_padding = 6

    pad = Pad(kernel_size=kernel_size, dilation=dilation, causal=False)
    x = torch.testing.make_tensor(
        (batch_size, in_channels, seq_len), device="cpu", dtype=torch.float32
    )
    padded_x = pad(x)

    assert padded_x.shape == (1, 1, expected_seq_len)

    # check that non-causal padding is applied correctly
    assert padded_x[0, 0, 0 : expected_padding // 2].abs().sum() == 0.0
    assert padded_x[0, 0, -expected_padding // 2 :].abs().sum() == 0.0


def test_tcn_correctly_forwards_input():
    in_channels = 23
    out_channels = 17
    batch_size = 11
    seq_len = 19

    tcn = TCN(
        in_channels=in_channels,
        hidden_channels=10,
        out_channels=out_channels,
        dilation_base=2,
        num_layers=8,
        kernel_size=3,
        causal=True,
    )
    expected_shape = (batch_size, out_channels, seq_len)

    x = torch.rand(batch_size, in_channels, seq_len)
    y = tcn(x)
    assert y.shape == expected_shape


def test_causal_tcn_does_not_have_non_causal_gradients():
    tcn = TCN(
        in_channels=1,
        hidden_channels=1,
        out_channels=1,
        dilation_base=1,
        num_layers=1,
        kernel_size=3,
        causal=True,
        activation="ELU",  # smooth activation to ensure ReLU isn't hiding nonzero grad
        norm=None,  # batch norm is not causal
    )
    x = torch.testing.make_tensor(
        1, 1, 32, device="cpu", dtype=torch.float32, requires_grad=True
    )
    x.grad = None

    y = tcn(x)
    (dy0_dx,) = torch.autograd.grad(y[0, 0, 0], x)

    assert dy0_dx[0, 0, 1:].abs().sum() == 0.0


def test_tcn_allows_film_conditioning():
    batch_size = 11
    in_channels = 13
    out_channels = 17
    seq_len = 31
    film_embedding_size = 7

    tcn = TCN(
        in_channels=in_channels,
        hidden_channels=10,
        out_channels=out_channels,
        dilation_base=2,
        num_layers=8,
        kernel_size=3,
        causal=True,
        film_conditioning=True,
        film_embedding_size=film_embedding_size,
    )
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

    y = tcn(x, film_embedding)
    assert y.shape == (batch_size, out_channels, seq_len)

    (dy_dx,) = torch.autograd.grad(y.sum().square(), film_embedding)
    assert (dy_dx.abs() > 0.0).all()


def test_tcn_uses_gated_activation(mocker):
    # TODO: switch to dependency injection of activations to avoid monkeypatching?
    import percussionsynth.models.tcn

    spy = mocker.spy(percussionsynth.models.tcn.GatedActivation, "__call__")

    batch_size = 13
    in_channels = 17
    out_channels = 19
    seq_len = 41

    tcn = TCN(
        in_channels=in_channels,
        hidden_channels=10,
        out_channels=out_channels,
        dilation_base=2,
        num_layers=8,
        kernel_size=3,
        causal=True,
        activation="gated",
    )

    x = torch.testing.make_tensor(
        batch_size, in_channels, seq_len, device="cpu", dtype=torch.float32
    )
    y = tcn(x)

    assert y.shape == (batch_size, out_channels, seq_len)
    assert spy.call_count == 8


@pytest.mark.parametrize(
    "norm,cls_string", [("batch", "BatchNorm1d"), ("instance", "InstanceNorm1d")]
)
def test_tcn_can_use_norm_layers(mocker, norm, cls_string):
    import percussionsynth.models.tcn

    spy = mocker.spy(percussionsynth.models.tcn.nn, cls_string)

    batch_size = 11
    in_channels = 3
    hidden_channels = 9
    hidden_layers = 5
    out_channels = 5
    seq_len = 7

    tcn = TCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dilation_base=2,
        num_layers=hidden_layers,
        kernel_size=3,
        causal=True,
        activation="ReLU",
        norm=norm,
    )

    spy.assert_has_calls([mocker.call(hidden_channels)] * hidden_layers)
    mocker.stop(spy)

    spy = mocker.spy(getattr(percussionsynth.models.tcn.nn, cls_string), "__call__")

    x = torch.testing.make_tensor(
        batch_size, in_channels, seq_len, device="cpu", dtype=torch.float32
    )
    _ = tcn(x)

    assert spy.call_count == hidden_layers


def test_tcn_throws_error_if_film_conditioning_requested_without_embedding_size():
    with pytest.raises(ValueError):
        TCN(
            in_channels=1,
            hidden_channels=1,
            out_channels=1,
            dilation_base=2,
            num_layers=1,
            kernel_size=3,
            causal=True,
            film_conditioning=True,
        )


def test_kick_tcn_use_deterministic_noise_false(mocker):
    batch_size = 4
    channels = 1
    seq_len = 16

    net = KickTCN(
        transient=TCN(in_channels=1, hidden_channels=1, out_channels=1),
        sustain=TCN(in_channels=1, hidden_channels=1, out_channels=1),
        fusion=TCN(in_channels=2, hidden_channels=1, out_channels=1),
        transient_length=16,
        use_deterministic_noise=False,
    )
    assert net.use_deterministic_noise is False
    assert hasattr(net, "noise") is False

    # Make sure this forwards correctly with no noise buffer and that
    # torech.randn_like is called
    spy = mocker.spy(torch, "randn_like")
    x = torch.testing.make_tensor(
        batch_size, channels, seq_len, device="cpu", dtype=torch.float32
    )
    y = net(x)
    assert y.shape == (batch_size, channels, seq_len)
    spy.assert_called_once()


def test_can_use_tfilm_in_tcn(mocker):
    import percussionsynth.models.tcn

    spy = mocker.spy(percussionsynth.models.tcn, "TFiLM")

    batch_size = 11
    in_channels = 3
    hidden_channels = 9
    hidden_layers = 5
    out_channels = 5
    block_size = 7
    seq_len = block_size * 9

    tcn = TCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dilation_base=2,
        num_layers=hidden_layers,
        kernel_size=3,
        causal=True,
        activation="ReLU",
        use_temporal_film=True,
        temporal_film_block_size=block_size,
    )

    spy.assert_has_calls([mocker.call(hidden_channels, block_size)] * hidden_layers)
    mocker.stop(spy)

    spy = mocker.spy(percussionsynth.models.tcn.TFiLM, "__call__")
    x = torch.testing.make_tensor(
        batch_size, in_channels, seq_len, device="cpu", dtype=torch.float32
    )
    _ = tcn(x)
    assert spy.call_count == hidden_layers


def test_tcn_throws_if_tfilm_requested_without_block_size():
    with pytest.raises(ValueError):
        TCN(
            in_channels=1,
            hidden_channels=1,
            out_channels=1,
            dilation_base=2,
            num_layers=1,
            kernel_size=3,
            causal=True,
            use_temporal_film=True,
        )


def test_kick_tcn_correctly_forwards_input(mocker):
    import percussionsynth.models.tcn

    batch_size = 4
    channels = 1
    seq_len = 16

    spy = mocker.spy(percussionsynth.models.tcn.TCN, "__call__")

    net = KickTCN(
        transient=TCN(in_channels=1, hidden_channels=1, out_channels=1),
        sustain=TCN(in_channels=1, hidden_channels=1, out_channels=1),
        fusion=TCN(in_channels=2, hidden_channels=1, out_channels=1),
        transient_length=16,
    )

    x = torch.testing.make_tensor(
        batch_size, channels, seq_len, device="cpu", dtype=torch.float32
    )
    y = net(x)

    assert y.shape == (batch_size, channels, seq_len)
    assert spy.call_count == 3


def test_kick_tcn_use_deterministic_noise_true(mocker):
    net = KickTCN(
        transient=TCN(in_channels=1, hidden_channels=1, out_channels=1),
        sustain=TCN(in_channels=1, hidden_channels=1, out_channels=1),
        fusion=TCN(in_channels=2, hidden_channels=1, out_channels=1),
        transient_length=16,
        use_deterministic_noise=True,
    )
    assert net.use_deterministic_noise
    assert net.noise.shape == (1, 16)
