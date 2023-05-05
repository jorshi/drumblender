"""
drumblender command line interface entrypoint.
"""
import argparse
import sys

import torch
import yaml
from dotenv import load_dotenv
from jsonargparse import ArgumentParser
from pytorch_lightning import LightningDataModule
from pytorch_lightning.cli import LightningCLI
from tqdm import tqdm

import drumblender.utils.data as data_utils
import drumblender.utils.model as model_utils
from drumblender.callbacks import SaveConfigCallbackWanb
from drumblender.data import AudioDataModule
from drumblender.models.soundstream import SoundStreamAttentionEncoder


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


def get_dataset_for_split(datamodule: AudioDataModule, split: str):
    if split == "train":
        datamodule.setup("fit")
        dataset = datamodule.train_dataloader().dataset
    elif split == "val":
        datamodule.setup("validate")
        dataset = datamodule.val_dataloader().dataset
    elif split == "test":
        datamodule.setup("test")
        dataset = datamodule.test_dataloader().dataset
    else:
        raise ValueError(f"Invalid split: {split}")

    return dataset


def export_film_embeddings():
    """
    CLI Entrypint for loading a pretrained model and exporting the film embeddings
    """

    load_dotenv()
    parser = ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to a model config file.",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to a checkpoint file. Use [random] for random weights.",
    )
    parser.add_argument(
        "outdir",
        type=str,
        help="Path to a directory to save the generated FiLM embeddings",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to use for generating embeddings. [train, val, test]]",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override the dataset to use",
    )

    args = parser.parse_args(sys.argv[1:])

    # Load the model and datamodule
    include_data = args.data is None

    if args.checkpoint == "random":
        print("[INFO] Loading random weights.")
        model = SoundStreamAttentionEncoder(
            input_channels=1,
            hidden_channels=16,
            output_channels=128,
            strides=[2, 2, 4, 8],
            causal=False,
            transpose_output=False,
        )
    else:
        model, datamodule = model_utils.load_model(
            args.config, args.checkpoint, include_data=include_data
        )

    # If a different dataset is specified, load it here.
    if args.data is not None:
        datamodule = model_utils.load_datamodule(args.data)

    # Move model to device and set to eval mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
    model.eval()

    # Get dataset for split
    dataset = get_dataset_for_split(datamodule, args.split)
    # Iterate through dataset
    param_list = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            # Get dataset item and extract FiLM embedding, save to outdir.
            example = dataset[i]
            original = example[0].to(device).unsqueeze(0)

            if args.checkpoint == "random":
                transient_params = model(original)
            else:
                transient_params = model.transient_autoencoder(original)

            param_list.append(transient_params.to("cpu").clone())

        params = torch.cat(param_list, dim=0)
    print(f"Collected tensor {params.shape}")
    torch.save(params, args.outdir)
