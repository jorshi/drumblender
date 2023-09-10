"""
drumblender command line interface entrypoint.
"""
import argparse
import inspect
import sys

import torch
import torchaudio
import yaml
from dotenv import load_dotenv
from einops import rearrange
from jsonargparse import ArgumentParser
from pytorch_lightning import LightningDataModule
from pytorch_lightning.cli import LightningCLI
from tqdm import tqdm

import drumblender.utils.data as data_utils
import drumblender.utils.model as model_utils
from drumblender.callbacks import SaveConfigCallbackWanb
from drumblender.data import AudioDataModule
from drumblender.utils.modal_analysis import CQTModalAnalysis


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
    parser.add_argument("input", type=str, help="Path to input audio file")
    parser.add_argument("output", type=str, help="Path to save audio to")

    args = parser.parse_args(sys.argv[1:])

    # Load the model from checkpoint
    model, _ = model_utils.load_model(args.config, args.checkpoint)

    # Load audio and convert to mono (just selecting the first channel)
    audio, input_sr = torchaudio.load(args.input)
    audio = audio[:1]

    # Resample to the sample rate used in training
    data_config = model_utils.load_config_yaml(args.config)["data"]["init_args"]
    sample_rate = data_config["sample_rate"]
    if input_sr != sample_rate:
        audio = torchaudio.transforms.Resample(
            orig_freq=input_sr, new_freq=sample_rate
        )(audio)

    # Pad (CQT has a minimum length)
    if audio.shape[1] < data_config["num_samples"]:
        num_pad = data_config["num_samples"] - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, num_pad))

    # Perform modal analysis on input file
    cqt_args = inspect.getfullargspec(CQTModalAnalysis.__init__).args
    cqt_kwargs = {key: data_config[key] for key in cqt_args if key in data_config}

    modal = CQTModalAnalysis(**cqt_kwargs)
    modal_freqs, modal_amps, modal_phases = modal(audio)

    # Frequencies are returned in Hz, convert to angular
    modal_freqs = 2 * torch.pi * modal_freqs / sample_rate

    modal_tensor = torch.stack([modal_freqs, modal_amps, modal_phases])
    modal_tensor = rearrange(modal_tensor, "s 1 m f -> 1 s m f")

    # Run the model!
    y_hat = model(audio.unsqueeze(0), modal_tensor)

    torchaudio.save(args.output, y_hat.squeeze(0), sample_rate)
