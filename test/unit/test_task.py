from typing import Optional

import torch
from einops import rearrange

from drumblender.tasks import DrumBlender
from drumblender.tasks import KickSynth


class FakeSynthModel(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.tensor, embedding: Optional[torch.tensor] = None):
        return self.linear(x)


def test_kicksynth_task_uses_modal_conditioning_when_no_conditioning_given(mocker):
    fake_model = FakeSynthModel(1, 1)
    spy = mocker.spy(fake_model, "forward")
    loss_fn = torch.nn.MSELoss()
    task = KickSynth(fake_model, loss_fn)

    # create a fake batch
    audio = torch.rand(1, 1)
    modal = torch.rand(1, 1)

    # run the task
    task.training_step((audio, modal), 0)

    # check that the model was called with the modal conditioning
    spy.assert_called_once_with(modal, None)


def test_kicksynth_task_uses_conditioning_model_when_conditioning_given(mocker):
    fake_model = FakeSynthModel(1, 1)
    fake_conditioning_model = torch.nn.Linear(1, 1)
    spy = mocker.spy(fake_conditioning_model, "forward")
    loss_fn = torch.nn.MSELoss()
    task = KickSynth(fake_model, loss_fn, conditioning_model=fake_conditioning_model)

    # create a fake batch
    audio = torch.rand(1, 1)
    modal = torch.rand(1, 1)

    # run the task
    task.training_step((audio, modal), 0)

    # check that the model was called with the modal conditioning
    spy.assert_called_once_with(audio)


def test_kicksynth_task_uses_embedding_when_embedding_given(mocker):
    fake_model = FakeSynthModel(1, 1)
    fake_embedding_model = torch.nn.Linear(1, 1)
    spy_model = mocker.spy(fake_model, "forward")
    spy_embedding = mocker.spy(fake_embedding_model, "forward")
    loss_fn = torch.nn.MSELoss()
    task = KickSynth(fake_model, loss_fn, embedding_model=fake_embedding_model)

    # create a fake batch
    audio = torch.rand(1, 1)
    modal = torch.rand(1, 1)

    # run the task
    task.training_step((audio, modal), 0)

    # check that the embedding model was called with the original audio and that
    # the synthesis model was called with the modal audio and the embedding
    spy_embedding.assert_called_once_with(audio)
    spy_model.assert_called_once_with(modal, fake_embedding_model(audio))


def test_kicksynth_task_passes_none_when_no_embedding_model_given(mocker):
    fake_model = FakeSynthModel(1, 1)
    spy = mocker.spy(fake_model, "forward")
    loss_fn = torch.nn.MSELoss()
    task = KickSynth(fake_model, loss_fn)

    # create a fake batch
    audio = torch.rand(1, 1)
    modal = torch.rand(1, 1)

    # run the task
    task.training_step((audio, modal), 0)

    # check that the model was called with the modal conditioning
    spy.assert_called_once_with(modal, None)


def test_kick_synth_task_uses_both_conditioning_and_embedding(mocker):
    fake_model = FakeSynthModel(1, 1)
    fake_conditioning_model = torch.nn.Linear(1, 1)
    fake_embedding_model = torch.nn.Linear(1, 1)

    spy_model = mocker.spy(fake_model, "forward")
    spy_conditioning = mocker.spy(fake_conditioning_model, "forward")
    spy_embedding = mocker.spy(fake_embedding_model, "forward")

    loss_fn = torch.nn.MSELoss()
    task = KickSynth(
        fake_model,
        loss_fn,
        conditioning_model=fake_conditioning_model,
        embedding_model=fake_embedding_model,
    )

    # create a fake batch
    audio = torch.rand(1, 1)
    modal = torch.rand(1, 1)

    # run the task
    task.training_step((audio, modal), 0)

    # check that the embedding model was called with the original audio and that
    # the synthesis model was called with the conditioning audio and the embedding
    spy_conditioning.assert_called_once_with(audio)
    spy_embedding.assert_called_once_with(audio)
    spy_model.assert_called_once_with(
        fake_conditioning_model(audio), fake_embedding_model(audio)
    )


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
