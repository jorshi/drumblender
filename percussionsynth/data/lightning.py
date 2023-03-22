"""Provide LightningDataModule wrappers for datasets.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type

import pytorch_lightning as pl
import torch
import torchaudio
from einops import rearrange
from torch.utils.data import DataLoader
from torchaudio.datasets.utils import extract_archive
from tqdm import tqdm

import kick2kick.utils.audio as audio_utils
import kick2kick.utils.data as data_utils
from kick2kick.data.audio import AudioDataset
from kick2kick.data.audio import AudioPairDataset
from kick2kick.data.audio import AudioPairWithFeatureDataset
from kick2kick.data.synthetic import MultiplicationDataset
from kick2kick.utils.modal_analysis import CQTModalAnalysis


# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MultiplicationDataModule(pl.LightningDataModule):
    """LightningDataModule for MultiplicationDataset.

    Args:
      train_size(int): Training set size, defaults to 5000
      val_size(int): Validation set size, defaults to 750
      test_size(int): Test set size, defaults to 750
      batch_size(int): Batch size, defaults to 32
      num_workers(int): Number of workers, defaults to 4
      train_seed(int): Training set PRNG seed, defaults to 0
      val_seed(int): Validation set PRNG seed, defaults to 1
      test_seed(int): Test set PRNG seed, defaults to 2

    Returns:

    """

    def __init__(
        self,
        train_size: int = 5000,
        val_size: int = 750,
        test_size: int = 750,
        batch_size: int = 32,
        num_workers: int = 4,
        train_seed: int = 0,
        val_seed: int = 1,
        test_seed: int = 2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """

        Args:
          stage: (Default value = None)

        Returns:

        """
        self.train_dataset = MultiplicationDataset(1000)
        self.val_dataset = MultiplicationDataset(100)
        self.test_dataset = MultiplicationDataset(100)

    def train_dataloader(self):
        """ """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """ """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        """ """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class KickDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the Kick dataset. This class is responsible for downloading
    and extracting a preprocessed dataset, or downloading and preprocessing the
    raw audio files if the preprocessed dataset is not available.

    Args:
        batch_size: Batch size, defaults to 32
        num_workers: Number of workers, defaults to 0
        dataset_class: Dataset class, defaults to AudioDataset
        dataset_kwargs: Additional keyword arguments to pass to the dataset class
            constructor, defaults to None
        url: URL to download the dataset from
        bucket: R2 bucket to download the dataset from
        archive: Gzip archive containing the dataset
        meta_file: JSON file containing metadata about the dataset
        data_dir: Directory to extract the dataset to
        data_dir_unprocessed: Directory to extract the unprocessed dataset to
        sample_rate: Sample rate of the audio files, defaults to 48000
        num_samples: Number of samples to load from each audio file, defaults to
            48000 * 2
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        dataset_class: Type[AudioDataset] = AudioDataset,
        dataset_kwargs: Optional[Dict] = None,
        url="https://d5d740b2d880827ae0c8f465bf180715.r2.cloudflarestorage.com",
        bucket="drum-dataset",
        archive="k2k-dataset-audio-only-v0.tar.gz",
        meta_file="kick-drums.json",
        data_dir="dataset/audio-only",
        data_dir_unprocessed="dataset-unprocessed/audio-only",
        sample_rate=48000,
        num_samples=48000 * 2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_cls = dataset_class
        self.dataset_kwargs = dataset_kwargs or {}
        self.url = url
        self.bucket = bucket
        self.archive = archive
        self.meta_file = meta_file
        self.data_dir = Path(data_dir)
        self.data_dir_unprocessed = Path(data_dir_unprocessed)
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def prepare_data(self, use_preprocessed: bool = True) -> None:
        """
        Download and extract the dataset.

        Args:
            use_preprocessed: Whether to use preprocessed data, defaults to True. If
                False, the raw data will be downloaded and processed. This will only
                need to be set to False if the preprocessed data archive isn't available
        """
        if use_preprocessed:
            # Download and extract the archived dataset
            if not Path(self.data_dir).exists():
                # Download the archive if it doesn't exist
                if not Path(self.archive).exists():
                    log.info("Downloading processed dataset...")
                    data_utils.download_file_r2(self.archive, self.url, self.bucket)
                log.info("Extracting processed dataset...")
                extract_archive(self.archive, self.data_dir)
            else:
                log.info("Dataset already exists.")
        else:
            # Raise an error if the preprocessed dataset exists
            if Path(self.data_dir).exists():
                raise RuntimeError(
                    "Preprocessed dataset already exists. Remove it to reprocess."
                )

            # Download and process the unprocessed dataset
            if not Path(self.data_dir_unprocessed).exists():
                log.info("Downloading unprocessed dataset...")
                data_utils.download_full_dataset(
                    self.url, self.bucket, self.meta_file, self.data_dir_unprocessed
                )

            log.info("Processing unprocessed dataset...")
            self.preprocess_dataset()

    def setup(self, stage: str):
        """
        Assign train/val/test datasets for use in dataloaders.

        Args:
            stage: Current stage (fit, validate, test)
        """
        args = [self.data_dir, self.meta_file, self.sample_rate, self.num_samples]
        if stage == "fit":
            self.train_dataset = self.dataset_cls(
                *args,
                split="train",
                **self.dataset_kwargs,
            )
            self.val_dataset = self.dataset_cls(
                *args,
                split="val",
                **self.dataset_kwargs,
            )
        elif stage == "validate":
            self.val_dataset = self.dataset_cls(
                *args,
                split="val",
                **self.dataset_kwargs,
            )
        elif stage == "test":
            self.test_dataset = self.dataset_cls(
                *args,
                split="test",
                **self.dataset_kwargs,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def preprocess_dataset(self) -> None:
        """
        Preprocess the dataset.
        """
        log.info("Preprocessing dataset...")
        audio_dir = Path(self.data_dir).joinpath("audio")
        audio_dir.mkdir(parents=True, exist_ok=False)

        # Load the metadata
        meta_file = Path(self.data_dir_unprocessed).joinpath(self.meta_file)
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        new_metadata = {}
        for key, item in tqdm(metadata.items()):
            sample_type = item["type"]
            folders = item["folders"]
            files = data_utils.get_files_from_folders(
                self.data_dir_unprocessed, folders, "*.wav"
            )
            assert len(files) > 0, f"No files founds in folders: {folders}"

            for file in files:
                # Create a hashed name for the file based on its path
                # minus the root directory
                output_hash = data_utils.str2int(str(Path(*file.parts[1:])))
                output_file = Path(audio_dir).joinpath(f"{output_hash}.wav")

                # Preprocess the audio file
                try:
                    audio_utils.preprocess_audio_file(
                        file,
                        output_file,
                        self.sample_rate,
                        self.num_samples,
                    )
                except ValueError as e:
                    log.warning(f"Error processing file {file}: {e}. Skipping...")
                    continue

                # Add the new metadata
                new_metadata[output_hash] = {
                    "filename": str(output_file.relative_to(self.data_dir)),
                    "type": sample_type,
                    "sample_pack_key": key,
                }

        # Save the new metadata
        with open(Path(self.data_dir).joinpath(self.meta_file), "w") as f:
            json.dump(new_metadata, f)

    def archive_dataset(self, archive_name: str) -> None:
        """
        Archive the dataset.

        Args:
            archive_name: Name of the archive.
        """
        log.info("Creating a tarfile of the dataset")
        data_utils.create_tarfile(archive_name, self.data_dir)


class KickModalDataModule(KickDataModule):
    """
    DataModule for the modal kick dataset. In addition to the origin kick waveform,
    this also contains a synthesized waveform containing only the modal components,
    extracted from the original waveform using sinusoidal modeling.

    Dataset items are returned as pairs of (original, modal) waveforms.

    Args:
        batch_size: Batch size, defaults to 32
        num_workers: Number of workers, defaults to 0
        dataset_class: Dataset class, defaults to AudioPairDataset
        dataset_kwargs: Additional keyword arguments to pass to the dataset class
            constructor, defaults to None
        url: URL to download the dataset from
        bucket: R2 bucket to download the dataset from
        archive: Gzip archive containing the dataset
        meta_file: JSON file containing metadata about the dataset
        data_dir: Directory to extract the dataset to
        data_dir_unprocessed: Directory to extract the unprocessed dataset to
        sample_rate: Sample rate of the audio files, defaults to 48000
        num_samples: Number of samples to load from each audio file, defaults to
            48000 * 2
        num_modes: Number of modes to extract from the original waveform, modes are
            extracted using sinusoidal modeling and are sorted by their amplitude
            in descending order, defaults to 1
        min_length: Minimum length of the extracted modes in frames, defaults to 10
        threshold: Threshold for the amplitude (in dB) of the extracted sinusoids
            to be considered, defaults to -80.0
        hop_length: Hop length for the CQT used in sinusoidal modelling, defaults to 64
        fmin: Minimum frequency for the CQT used in sinusoidal modelling, defaults to 20
        n_bins: Number of bins for the CQT used in sinusoidal modelling, defaults to 96
        bins_per_octave: Number of bins per octave for the CQT used in sinusoidal
            modelling, defaults to 12
        save_modal_audio: Whether to save the modal audio files, defaults to True
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        dataset_class: Type[AudioPairDataset] = AudioPairDataset,
        dataset_kwargs: Optional[Dict] = None,
        url="https://d5d740b2d880827ae0c8f465bf180715.r2.cloudflarestorage.com",
        bucket="drum-dataset",
        archive="k2k-dataset-audio-only-v0.tar.gz",
        meta_file="kick-drums.json",
        data_dir="dataset/modal",
        data_dir_unprocessed="dataset-unprocessed/modal",
        sample_rate=48000,
        num_samples=48000 * 2,
        num_modes=1,
        min_length=10,
        threshold=-80.0,
        hop_length=64,
        fmin=20,
        n_bins=96,
        bins_per_octave=12,
        save_modal_audio=True,
    ):
        # Set default values for the file_keys in the Dataset -- these are the filename
        # keys in the metadata file that will be used to load the pairs of audio files
        dataset_kwargs = dataset_kwargs or {}
        if dataset_kwargs.get("file_key_a") is None:
            dataset_kwargs["file_key_a"] = "filename"

        if dataset_kwargs.get("file_key_b") is None:
            dataset_kwargs["file_key_b"] = "filename_modal"

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_class=dataset_class,
            dataset_kwargs=dataset_kwargs,
            url=url,
            bucket=bucket,
            archive=archive,
            meta_file=meta_file,
            data_dir=data_dir,
            data_dir_unprocessed=data_dir_unprocessed,
            sample_rate=sample_rate,
            num_samples=num_samples,
        )
        self.num_modes = num_modes
        self.min_length = min_length
        self.threshold = threshold
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.save_modal_audio = save_modal_audio

    def preprocess_dataset(self) -> None:
        """
        Overwrite the parent method to add modal features to the dataset.
        First preprocess the dataset as normal, then add then extracts the
        modal features and saves them alongside the audio files.
        """
        super().preprocess_dataset()

        log.info("Extracting modal features...")
        with open(Path(self.data_dir).joinpath(self.meta_file), "r") as f:
            metadata = json.load(f)

        # Create a feature directory
        feature_dir = Path(self.data_dir).joinpath("features")
        feature_dir.mkdir(parents=True, exist_ok=False)

        modal = CQTModalAnalysis(
            self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            min_length=self.min_length,
            num_modes=self.num_modes,
            threshold=self.threshold,
        )

        for key, item in tqdm(metadata.items()):
            audio_file = Path(self.data_dir).joinpath(item["filename"])
            waveform, _ = torchaudio.load(audio_file)
            modal_freqs, modal_amps, modal_phases = modal(waveform)

            # Stack the freqs, amps, and phases into a single tensor and remove
            # the single batch dimension, since there is only one sample
            # Output shape is (3, num_modes, num_frames)
            modal_tensor = torch.stack([modal_freqs, modal_amps, modal_phases])
            modal_tensor = rearrange(modal_tensor, "s 1 m f -> s m f")

            # Save the modal features
            modal_file = feature_dir.joinpath(audio_file.name.replace(".wav", ".pt"))
            torch.save(modal_tensor, modal_file)

            # Save the modal audio
            if self.save_modal_audio:
                modal_audio = audio_utils.modal_synth(
                    modal_freqs,
                    modal_amps,
                    self.sample_rate,
                    self.num_samples,
                )
                modal_audio_file = audio_file.parent.joinpath(
                    audio_file.name.replace(".wav", "_modal.wav")
                )
                torchaudio.save(modal_audio_file, modal_audio, self.sample_rate)

                # Update the metadata
                metadata[key]["filename_modal"] = str(
                    modal_audio_file.relative_to(self.data_dir)
                )

            # Update the metadata
            metadata[key]["feature_file"] = str(modal_file.relative_to(self.data_dir))

        # Save the new metadata
        with open(Path(self.data_dir).joinpath(self.meta_file), "w") as f:
            json.dump(metadata, f)


class KickModalEmbeddingDataModule(KickModalDataModule):
    """
    DataModule for training a model using audio pairs of an original kick drum
    and a modal synthesis of the original kick drum, along with a pre-computed
    feature embedding of the original kick drum.

    #TODO: Maybe there is a way to generalize this along with the modal feature
    extraction as opposed to this child class.
    For example, a DataModule that takes a list of feature extractors
    as a list of callables.

    Args:
        embedding_model: A callable that takes a waveform and returns an embedding
        feature_prefix: The prefix of the feature file to use for the embedding,
            this is appended to the filename of the audio of the original kick drum
        flatten: Whether to flatten the embedding feature before saving it
        dataset_class: The dataset class to use for the dataloader. Defaults to
            `AudioPairWithFeatureDataset`, which returns pairs of audio plus a feature
            tensor
        dataset_kwargs: Additional keyword arguments to pass to the dataset class
        **kwargs: Additional keyword arguments to pass to KickModalDataModule
    """

    def __init__(
        self,
        embedding_model: Callable,
        feature_prefix: str,
        flatten: bool = False,
        dataset_class: Type[AudioPairWithFeatureDataset] = AudioPairWithFeatureDataset,
        dataset_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        # Create the feature key that will be saved in the metadata and will
        # tell the dataset which feature file to load
        self.feature_prefix = feature_prefix
        self.feature_key = f"{feature_prefix}_feature_file"

        dataset_kwargs = dataset_kwargs or {}
        dataset_kwargs["feature_key"] = self.feature_key
        super().__init__(
            dataset_class=dataset_class, dataset_kwargs=dataset_kwargs, **kwargs
        )

        self.feature_embedding = embedding_model
        self.flatten = flatten

    def preprocess_dataset(self) -> None:
        super().preprocess_dataset()
        self.preprocess_features()

    def preprocess_features(self, overwrite: bool = False) -> None:
        log.info(f"Extracting embedding features using {repr(self.feature_embedding)}")
        with open(Path(self.data_dir).joinpath(self.meta_file), "r") as f:
            metadata = json.load(f)

        # Create a feature directory
        feature_dir = Path(self.data_dir).joinpath("features")
        feature_dir.mkdir(parents=True, exist_ok=True)

        for key, item in tqdm(metadata.items()):
            audio_file = Path(self.data_dir).joinpath(item["filename"])
            waveform, sr = torchaudio.load(audio_file)
            assert (
                sr == self.sample_rate
            ), f"Sample rate mismatch: {sr} != {self.sample_rate}"

            # Extract the features
            embedding = self.feature_embedding(waveform)
            if self.flatten:
                embedding = torch.flatten(embedding)

            # Save the modal features
            feature_file = f"{audio_file.stem}_{self.feature_prefix}.pt"
            feature_file = feature_dir.joinpath(feature_file)

            # Avoid overwriting existing features
            if feature_file.exists() and not overwrite:
                raise FileExistsError("Feature file already exists.")

            torch.save(embedding, feature_file)

            # Update the metadata
            metadata[key][self.feature_key] = str(
                feature_file.relative_to(self.data_dir)
            )

        # Save the updated metadata
        with open(Path(self.data_dir).joinpath(self.meta_file), "w") as f:
            json.dump(metadata, f)
