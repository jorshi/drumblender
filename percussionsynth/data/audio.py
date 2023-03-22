"""
Audio datasets
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional
from typing import Union

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import random_split


# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class AudioDataset(Dataset):
    """
    Dataset of audio files.

    Args:
        data_dir: Path to the directory containing the dataset.
        meta_file: Name of the json metadata file.
        sample_rate: Expected sample rate of the audio files.
        num_samples: Expected number of samples in the audio files.
        split (optional): Split to return. Must be one of 'train', 'val', or 'test'.
            If None, the entire dataset is returned.
        seed: Seed for random number generator used to split the dataset.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        meta_file: str,
        sample_rate: int,
        num_samples: int,
        split: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.meta_file = meta_file
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.seed = seed

        # Confirm that preprocessed dataset exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed dataset not found. Expected: {self.data_dir}"
            )

        # Load the metadata
        with open(self.data_dir.joinpath(self.meta_file), "r") as f:
            self.metadata = json.load(f)

        self.file_list = list(self.metadata.keys())

        # Split the dataset
        if split is not None:
            self._split(split)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_filename = self.metadata[self.file_list[idx]]["filename"]
        waveform, sample_rate = torchaudio.load(self.data_dir.joinpath(audio_filename))

        # Confirm sample rate and shape
        assert sample_rate == self.sample_rate, "Sample rate mismatch."
        assert waveform.shape == (1, self.num_samples), "Incorrect input audio shape."

        return waveform

    def _split(self, split: str):
        """
        Split the dataset into train, validation, and test sets.

        TODO: Do we need something more than a random split here?
        i.e. balancing between acoustic vs. electronic kicks or
        ensuring that samples from the same sample pack in the same split

        Args:
            split: Split to return. Must be one of 'train', 'val', or 'test'.
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Invalid split. Must be one of 'train', 'val', or 'test'.")

        splits = random_split(
            self.file_list,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Set the file list based on the split
        if split == "train":
            self.file_list = splits[0]
        elif split == "val":
            self.file_list = splits[1]
        elif split == "test":
            self.file_list = splits[2]


class AudioPairDataset(AudioDataset):
    """
    Dataset of audio pairs.

    Args:
        data_dir: Path to the directory containing the dataset.
        meta_file: Name of the json metadata file.
        sample_rate: Expected sample rate of the audio files.
        num_samples: Expected number of samples in the audio files.
        file_key_a: Key in the metadata file for the first audio file in a pair.
        file_key_b: Key in the metadata file for the second audio file in a pair.
        split (optional): Split to return. Must be one of 'train', 'val', or 'test'.
            If None, the entire dataset is returned.
        seed: Seed for random number generator used to split the dataset.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        meta_file: str,
        sample_rate: int,
        num_samples: int,
        file_key_a: str = "filename_a",
        file_key_b: str = "filename_b",
        split: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            meta_file=meta_file,
            sample_rate=sample_rate,
            num_samples=num_samples,
            split=split,
            seed=seed,
        )
        self.file_key_a = file_key_a
        self.file_key_b = file_key_b

    def __getitem__(self, idx):
        """
        Returns a tuple of audio samples.
        """
        audio_filename = self.metadata[self.file_list[idx]][self.file_key_a]
        waveform_a, sr_a = torchaudio.load(self.data_dir.joinpath(audio_filename))

        audio_filename = self.metadata[self.file_list[idx]][self.file_key_b]
        waveform_b, sr_b = torchaudio.load(self.data_dir.joinpath(audio_filename))

        # Confirm sample rate and shape
        assert sr_a == self.sample_rate, "Sample rate mismatch."
        assert waveform_a.shape == (1, self.num_samples), "Incorrect input audio shape."
        assert sr_b == self.sample_rate, "Sample rate mismatch."
        assert waveform_b.shape == (1, self.num_samples), "Incorrect input audio shape."

        return waveform_a, waveform_b


class AudioPairKroneckerDeltaDataset(AudioPairDataset):
    """
    Dataset of audio pairs with an impulse as the second audio file in the pair.
    Mainly used for testing a model's ability to reconstruct an impulse response.
    """

    def __getitem__(self, idx):
        waveform_a, waveform_b = super().__getitem__(idx)
        dirac = torch.zeros_like(waveform_a)
        dirac[0, 0] = 1.0
        return waveform_a, dirac


class AudioPairWithFeatureDataset(AudioPairDataset):
    """
    Dataset of audio pairs with an additional feature tensor.

    Args:
        data_dir: Path to the directory containing the dataset.
        meta_file: Name of the json metadata file.
        sample_rate: Expected sample rate of the audio files.
        num_samples: Expected number of samples in the audio files.
        feature_key: Key in the metadata file for the feature file.
        **kwargs: Additional arguments to pass to AudioPairDataset.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        meta_file: str,
        sample_rate: int,
        num_samples: int,
        feature_key: str,
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            meta_file=meta_file,
            sample_rate=sample_rate,
            num_samples=num_samples,
            **kwargs,
        )
        self.feature_key = feature_key

    def __getitem__(self, idx):
        waveform_a, waveform_b = super().__getitem__(idx)
        feature_file = self.metadata[self.file_list[idx]][self.feature_key]
        feature = torch.load(self.data_dir.joinpath(feature_file))
        return waveform_a, waveform_b, feature
