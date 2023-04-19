import torch
from einops import rearrange

from drumblender.tasks import DrumBlender


def test_drumblender_can_be_instantiated(mocker):
    modal_synth = mocker.stub("modal_synth")
    loss_fn = mocker.stub("loss_fn")

    model = DrumBlender(modal_synth=modal_synth, loss_fn=loss_fn)
    assert model is not None
    assert model.modal_synth == modal_synth
    assert model.loss_fn == loss_fn


def test_drumblender_can_forward_modal(mocker):
    class FakeSynth(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def forward(self, p, length=None):
            return self.output

    loss_fn = mocker.stub("loss_fn")
    expected_output = torch.rand(1, 1)
    modal_synth = FakeSynth(expected_output)
    modal_spy = mocker.spy(modal_synth, "forward")

    batch_size = 7
    num_params = 3
    num_modes = 45
    num_steps = 400
    x = torch.rand(batch_size, 1, 1)
    p = torch.rand(batch_size, num_params, num_modes, num_steps)

    model = DrumBlender(modal_synth=modal_synth, loss_fn=loss_fn)
    y = model(x, p)

    assert y == expected_output
    modal_spy.assert_called_once_with(p, x.size(-1))


def test_drumblender_forwards_all(mocker):
    class FakeModule(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def forward(self, *args):
            return self.output

    batch_size = 7
    num_samples = 1024
    num_params = 3
    num_modes = 45
    num_steps = 400
    embedding_size = 12

    loss_fn = mocker.stub("loss_fn")

    # Fake Encoders
    expected_encoder_output = torch.rand(batch_size, embedding_size)
    encoder = FakeModule(expected_encoder_output)
    encoder_spy = mocker.spy(encoder, "forward")

    expected_modal_encoder_output = torch.rand(batch_size, embedding_size)
    modal_encoder = FakeModule(expected_modal_encoder_output)
    modal_encoder_spy = mocker.spy(modal_encoder, "forward")

    expected_noise_encoder_output = torch.rand(batch_size, embedding_size)
    noise_encoder = FakeModule(expected_noise_encoder_output)
    noise_encoder_spy = mocker.spy(noise_encoder, "forward")

    expected_transient_encoder_output = torch.rand(batch_size, embedding_size)
    transient_encoder = FakeModule(expected_transient_encoder_output)
    transient_encoder_spy = mocker.spy(transient_encoder, "forward")

    # Fake Synths
    expected_modal_output = torch.rand(batch_size, 1, num_samples)
    modal_synth = FakeModule(expected_modal_output)
    modal_spy = mocker.spy(modal_synth, "forward")

    expected_noise_output = torch.rand(batch_size, num_samples)
    noise_synth = FakeModule(expected_noise_output)
    noise_spy = mocker.spy(noise_synth, "forward")

    expected_transient_output = torch.rand(batch_size, 1, num_samples)
    transient_synth = FakeModule(expected_transient_output)
    transient_spy = mocker.spy(transient_synth, "forward")

    x = torch.rand(batch_size, 1, num_samples)
    p = torch.rand(batch_size, num_params, num_modes, num_steps)

    model = DrumBlender(
        loss_fn=loss_fn,
        encoder=encoder,
        modal_autoencoder=modal_encoder,
        noise_autoencoder=noise_encoder,
        transient_autoencoder=transient_encoder,
        modal_synth=modal_synth,
        noise_synth=noise_synth,
        transient_synth=transient_synth,
    )

    y = model(x, p)

    encoder_spy.assert_called_once_with(x)

    modal_encoder_spy.assert_called_once_with(p, expected_encoder_output)
    noise_encoder_spy.assert_called_once_with(expected_encoder_output)
    transient_encoder_spy.assert_called_once_with(expected_encoder_output)

    modal_spy.assert_called_once_with(expected_modal_encoder_output, x.size(-1))
    noise_spy.assert_called_once_with(expected_noise_encoder_output, x.size(-1))

    transient_input = expected_modal_output + rearrange(
        expected_noise_output, "b t -> b () t"
    )
    torch.testing.assert_close(transient_spy.call_args_list[0][0][0], transient_input)
    torch.testing.assert_close(
        transient_spy.call_args_list[0][0][1], expected_transient_encoder_output
    )

    assert torch.all(y == expected_transient_output)
