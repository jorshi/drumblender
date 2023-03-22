import subprocess
from typing import Any

import pytorch_lightning as pl
import torch

from percussionsynth.callbacks import CleanWandbCacheCallback
from percussionsynth.callbacks import LogAudioCallback
from percussionsynth.callbacks import SaveConfigCallbackWanb


class FakeLogger:
    def __init__(self, stub):
        self.stub = stub

    def __getattr__(self, name: str) -> Any:
        if name == "experiment":
            return self
        else:
            return super().__getattr__(name)

    def log(self, *args, **kwargs):
        self.stub(*args, **kwargs)


class FakeModule(pl.LightningModule):
    def __init__(self, fake_logger):
        super().__init__()

        self.fake_logger = fake_logger

    def __getattribute__(self, __name: str) -> Any:
        if __name == "logger":
            return self.fake_logger
        else:
            return super().__getattribute__(__name)

    def forward(self, x, *args, **kwargs):
        return x


def test_callback_correctly_interleaves_audio(monkeypatch, mocker):
    sample_rate = 48000
    callback = LogAudioCallback(
        on_train=True,
        on_val=True,
        on_test=True,
        log_on_epoch_end=True,
        save_audio_sr=sample_rate,
    )

    log_stub = mocker.stub("logger")
    logger = FakeLogger(log_stub)
    model = FakeModule(logger)

    trainer = None
    callback.setup(trainer, model, "fit")

    # patch wandb.Audio inside the percussionsynth.callbacks module
    FAKE_RETURN = "fake return"
    audio_patch = mocker.patch("percussionsynth.callbacks.Audio")
    audio_patch.return_value = FAKE_RETURN

    # If inputs are correctly interleaved, this is the sequence we should receive.
    expected_output = torch.tensor(
        [1, -1, 11, -11, 2, -2, 12, -12, 3, -3, 13, -13]
    ).numpy()

    for i in range(1, 4):
        fake_conditioning = torch.tensor([[[-i]], [[-10 - i]]])
        fake_targets = torch.tensor([[[i]], [[10 + i]]])
        batch = (fake_targets, fake_conditioning)

        callback.on_train_batch_start(trainer, model, batch, 0)
        model(fake_conditioning)
        callback.on_train_batch_end(trainer, model, 0.0, batch, 0)

    callback.on_train_epoch_end(trainer, model)

    assert audio_patch.call_count == 1

    (actual_output,) = audio_patch.call_args.args
    caption = audio_patch.call_args.kwargs["caption"]
    actual_sample_rate = audio_patch.call_args.kwargs["sample_rate"]

    # can't just use assert_called_once_with because we need to use numpy.all
    assert (actual_output == expected_output).all()
    assert caption == "train/audio"
    assert actual_sample_rate == sample_rate

    log_stub.assert_called_once_with({"train/audio": FAKE_RETURN})


def test_clean_wandb_cache_callback_cleans_wandb_cache(monkeypatch, mocker):
    callback = CleanWandbCacheCallback(every_n_epochs=2, max_size_in_gb=1)

    class FakeTrainer:
        current_epoch: int = 0

    trainer = FakeTrainer()
    model = None

    expected_args = ["wandb", "artifact", "cache", "cleanup", "1GB"]
    fake_Popen = mocker.stub("subprocess.Popen")

    monkeypatch.setattr(subprocess, "Popen", fake_Popen)

    for _ in range(4):
        callback.on_train_epoch_end(trainer, model)
        trainer.current_epoch += 1

    fake_Popen.assert_has_calls([mocker.call(expected_args)] * 2)
    assert fake_Popen.call_count == 2


def test_save_config_callback_renames_correctly(mocker, fs):
    # Parent class SaveConfigCallback setup creates a config file
    # in the log_dir, not the experiment dir
    def create_config_file(*args, **kwargs):
        fs.create_file("not_experiment_dir/config.yaml")

    # WandbLogger creates a new experiment dir and returns an experiment object
    # with a dir attribute when the experiment property is accessed
    class FakeExperiment:
        dir = "experiment_dir"

    class FakeLogger(pl.loggers.WandbLogger):
        def __init__(self, *args, **kwargs):
            pass

        @property
        def experiment(self):
            return FakeExperiment()

    # Patch the parent class -- we just want to test that the setup method is
    # called which expected to create a config file in trainer.log_dir
    mock_init = mocker.patch(
        "percussionsynth.callbacks.SaveConfigCallback.__init__", return_value=None
    )
    mock_setup = mocker.patch(
        "percussionsynth.callbacks.SaveConfigCallback.setup",
        side_effect=create_config_file,
    )

    class FakeTrainer:
        logger = FakeLogger()
        log_dir = "not_experiment_dir"

    trainer = FakeTrainer()
    model = None

    callback = SaveConfigCallbackWanb()
    callback.setup(trainer, model, "fit")

    # Make sure the config file was moved and renamed correctly
    assert fs.exists("experiment_dir/model-config.yaml")
    mock_init.assert_called_once()
    mock_setup.assert_called_once()


def test_save_config_callback_just_calls_setup_for_non_wandb_logger(mocker, fs):
    # Parent class SaveConfigCallback setup creates a config file
    # in the log_dir, not the experiment dir
    def create_config_file(*args, **kwargs):
        fs.create_file("not_experiment_dir/config.yaml")

    class FakeNonWandbLogger:
        pass

    # Patch the parent class -- we just want to test that the setup method is
    # called which expected to create a config file in trainer.log_dir
    mock_init = mocker.patch(
        "percussionsynth.callbacks.SaveConfigCallback.__init__", return_value=None
    )
    mock_setup = mocker.patch(
        "percussionsynth.callbacks.SaveConfigCallback.setup",
        side_effect=create_config_file,
    )

    class FakeTrainer:
        logger = FakeNonWandbLogger()
        log_dir = "not_experiment_dir"

    trainer = FakeTrainer()
    model = None

    callback = SaveConfigCallbackWanb()
    callback.setup(trainer, model, "fit")

    # Make sure the config file was not moved
    assert fs.exists("not_experiment_dir/config.yaml")
    mock_init.assert_called_once()
    mock_setup.assert_called_once()
