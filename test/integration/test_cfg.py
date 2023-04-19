import glob
import os
from typing import Callable
from typing import Optional
from typing import Union

import pytest
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.cli import LightningCLI


def import_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


TEST_TYPES = dict(
    loss_cfg=dict(
        pathname="cfg/loss/**/*.yaml",
        recursive=True,
    ),
    data_cfg=dict(
        pathname="cfg/data/**/*.yaml",
        recursive=True,
    ),
    model_cfg=dict(
        pathname="cfg/models/**/*.yaml",
        recursive=True,
    ),
    optimizer_cfg=dict(
        pathname="cfg/optimizer/**/*.yaml",
        recursive=True,
    ),
    synth_cfg=dict(
        pathname="cfg/synths/**/*.yaml",
        recursive=True,
    ),
    experiment_cfg=dict(
        pathname="cfg/*.yaml",
        recursive=False,
    ),
)


def pytest_generate_tests(metafunc):
    for test_type, glob_params in TEST_TYPES.items():
        if test_type in metafunc.fixturenames:
            files = glob.glob(**glob_params)
            metafunc.parametrize(test_type, files)


@pytest.fixture
def parser():
    parser = LightningArgumentParser()
    return parser


def read_cfg(cfg: os.PathLike, wrap: Optional[str] = "cfg"):
    with open(cfg, "r") as f:
        cfg_string = f.read()
    if wrap is not None:
        cfg_string = f"{wrap}:\n{cfg_string}"
        cfg_string = cfg_string.replace("\n", "\n  ")
    return cfg_string


def test_can_instantiate_from_data_config(data_cfg, parser):
    cfg_string = read_cfg(data_cfg)
    parser.add_lightning_class_args(
        LightningDataModule, "cfg", subclass_mode=True, required=True
    )
    args = parser.parse_string(cfg_string)
    assert "class_path" in args.cfg, "No class_path key in config root level"

    class_path = args.cfg["class_path"]
    objs = parser.instantiate_classes(args)

    assert isinstance(objs.cfg, import_class(class_path))


def test_can_instantiate_from_loss_config(loss_cfg, parser):
    cfg_string = read_cfg(loss_cfg)

    parser.add_argument("cfg", type=Union[Callable, torch.nn.Module])
    args = parser.parse_string(cfg_string)
    assert "class_path" in args.cfg, "No class_path key in config root level"

    class_path = args.cfg["class_path"]
    objs = parser.instantiate_classes(args)

    # Check that the instantiated object is of the correct type
    assert isinstance(objs.cfg, import_class(class_path))

    if isinstance(objs.cfg, torch.nn.Module):
        assert hasattr(objs.cfg, "forward"), "Loss function must have a forward method."
    else:
        assert isinstance(objs.cfg, Callable), "Loss function must be callable."


def test_can_instantiate_from_model_config(model_cfg, parser):
    cfg_string = read_cfg(model_cfg)

    parser.add_argument("cfg", type=torch.nn.Module)
    args = parser.parse_string(cfg_string)
    assert "class_path" in args.cfg, "No class_path key in config root level"

    class_path = args.cfg["class_path"]
    objs = parser.instantiate_classes(args)

    # Check that the instantiated object is of the correct type
    assert isinstance(objs.cfg, import_class(class_path))


def test_can_instantiate_from_optimizer_config(optimizer_cfg, parser, monkeypatch):
    cfg_string = read_cfg(optimizer_cfg, wrap="optimizer")
    # set up LightningCLI config
    cfg_string = (
        f"model: pytorch_lightning.demos.boring_classes.BoringModel\n{cfg_string}"
    )

    parser.add_optimizer_args((torch.optim.Optimizer,), "optimizer")
    parser.add_lightning_class_args(
        LightningModule, "model", subclass_mode=True, required=True
    )
    args = parser.parse_string(cfg_string)
    assert "class_path" in args.optimizer, "No class_path key in config root level"
    class_path = args.optimizer["class_path"]

    with monkeypatch.context() as m:
        # stop LightningCLI from accessing pytest argv
        import sys

        m.setattr(sys, "argv", ["fake_file.py"])

        cli = LightningCLI(args=args, run=False)
        optimizer = cli.model.configure_optimizers()

    # Check that the instantiated object is of the correct type
    assert isinstance(optimizer, import_class(class_path))


def test_can_instantiate_from_experiment_config(experiment_cfg, monkeypatch):
    with monkeypatch.context() as m:
        # stop LightningCLI from accessing pytest argv
        import sys

        m.setattr(
            sys,
            "argv",
            [
                "fake_file.py",
                "-c",
                str(experiment_cfg),
                "--trainer.accelerator",
                "cpu",
                "--trainer.devices",
                "1",
            ],
        )

        cli = LightningCLI(run=False)

    assert isinstance(cli.model, LightningModule)
