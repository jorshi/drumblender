"""
drumblender command line interface entrypoint.
"""
import argparse
import sys

import yaml
from dotenv import load_dotenv
from jsonargparse import ArgumentParser
from pytorch_lightning import LightningDataModule
from pytorch_lightning.cli import LightningCLI
from tqdm import tqdm

import drumblender.utils.data as data_utils
from drumblender.callbacks import SaveConfigCallbackWanb
from drumblender.data import AudioDataModule
from drumblender.tasks import DrumBlender


def run_cli():
    """ """
    _ = LightningCLI(save_config_callback=SaveConfigCallbackWanb)


def main():
    """ """
    load_dotenv()
    run_cli()


def dataset():
    """CLI entrypoint for the dataset preparation script."""
    load_dotenv()
    parser = ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to a config file with arguments.",
    ),
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Whether to download and preprocess the raw dataset.",
    )
    parser.add_argument(
        "--preprocess_features",
        action="store_true",
        help="Preprocess features on an already downloaded dataset.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=None,
        help="If set, will archve the preprocessed dataset into the given path.",
    )
    parser.add_argument(
        "--upload",
        type=str,
        default=None,
        help="If set, will upload an archived dataset to the bucket KickDataset bucket",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="If set, will check that all dataset files are present.",
    )

    args = parser.parse_args(sys.argv[1:])

    # Manually parse the datamodule arguments from the config file and
    # instantiate the datamodule specified in the config file.
    # We're Doing this manually so we can use the same config file that
    # LightningCLI uses. ArgumentParser requires us to add a root node
    # to the config file -- so we do that here.
    datamodule_parser = ArgumentParser()
    datamodule_parser.add_subclass_arguments(AudioDataModule, "datamodule")
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {"datamodule": config}
            datamodule_args = datamodule_parser.parse_object(config)
            datamodule = datamodule_parser.instantiate_classes(
                datamodule_args
            ).datamodule

    if args.archive is not None:
        datamodule.archive_dataset(args.archive)
        return

    if args.upload is not None:
        data_utils.upload_file_r2(
            args.upload,
            datamodule.url,
            datamodule.bucket,
        )
        return

    # Preprocess features for a downloaded dataset
    # This is useful if you want to use a different feature extractor
    # but don't want to re-download the dataset.
    if args.preprocess_features:
        datamodule.preprocess_features(overwrite=True)
        return

    if args.verify:
        verify_dataset(datamodule)
        return

    # Prepare the dataset
    datamodule.prepare_data(use_preprocessed=not args.preprocess)


def verify_dataset(datamodule: LightningDataModule):
    """
    Verify that all files in the dataset are present.
    """
    for split in ["fit", "validate", "test"]:
        datamodule.setup(split)
        if split == "fit":
            dataset = datamodule.train_dataloader().dataset
        elif split == "validate":
            dataset = datamodule.val_dataloader().dataset
        else:
            dataset = datamodule.test_dataloader().dataset

        for i in tqdm(range(len(dataset))):
            _ = dataset[i]


def inference():
    """
    Given an input audio, compute reconstruction.

    Optionally, can pass in different audio files for sinusoidal, noise, and transient
    embeddings.
    """
    load_dotenv()
    parser = ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to a config file with arguments.",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to a checkpoint file.",
    )

    args = parser.parse_args(sys.argv[1:])

    # Load trained model
    config_parser = ArgumentParser()
    config_parser.add_subclass_arguments(DrumBlender, "model", fail_untyped=False)

    config = config_parser.parse_path(args.config)
    print(config)
