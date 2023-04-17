from typing import Optional

import torch

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


def test_drumblender_can_forward(mocker):
    class FakeSynth(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def forward(self, p):
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
    modal_spy.assert_called_once_with(p)
