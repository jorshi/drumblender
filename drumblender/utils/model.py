"""
Helpful utils for handling pre-trained models
"""
import yaml
from jsonargparse import ArgumentParser

from drumblender.data import AudioDataModule
from drumblender.tasks import DrumBlender


def load_model(config: str, ckpt: str, include_data: bool = False):
    """
    Load model from checkpoint
    """
    # Load the config file and instantiate the model
    config_parser = ArgumentParser()
    config_parser.add_subclass_arguments(DrumBlender, "model", fail_untyped=False)
    config_parser.add_argument("--trainer", type=dict, default={})
    config_parser.add_argument("--seed_everything", type=int)
    config_parser.add_argument("--ckpt_path", type=str)
    config_parser.add_argument("--optimizer", type=dict)
    config_parser.add_argument("--lr_scheduler", type=dict)

    if include_data:
        config_parser.add_subclass_arguments(AudioDataModule, "data")
    else:
        config_parser.add_argument("--data", type=dict, default={})

    config = config_parser.parse_path(config)
    init = config_parser.instantiate_classes(config)

    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt}...")
    model = init.model.load_from_checkpoint(
        ckpt,
        modal_synth=init.model.modal_synth,
        loss_fn=init.model.loss_fn,
        noise_synth=init.model.noise_synth,
        transient_synth=init.model.transient_synth,
        modal_autoencoder=init.model.modal_autoencoder,
        noise_autoencoder=init.model.noise_autoencoder,
        transient_autoencoder=init.model.transient_autoencoder,
        encoder=init.model.encoder,
    )

    # Instantiate the datamodule if required
    if include_data:
        datamodule = init.data
        return model, datamodule

    return model, None


def load_datamodule(config: str):
    """
    Load a datamodule from a config file
    """
    datamodule_parser = ArgumentParser()
    datamodule_parser.add_subclass_arguments(AudioDataModule, "datamodule")
    if config is not None:
        with open(config, "r") as f:
            config = yaml.safe_load(f)
            config = {"datamodule": config}
            datamodule_args = datamodule_parser.parse_object(config)
            datamodule = datamodule_parser.instantiate_classes(
                datamodule_args
            ).datamodule

    return datamodule
