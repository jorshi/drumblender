"""
Tests for the `kick2kick.data.audio` module.
"""
from pathlib import Path

import pytest
import torch

from percussionsynth.data import AudioDataset
from percussionsynth.data import AudioPairWithFeatureDataset

TESTED_MODULE = "kick2kick.data.audio"
TEST_DATA_DIR = "test_data"
TEST_META_FILE = "metadata.json"
TEST_SAMPLE_RATE = 44100
TEST_NUM_SAMPLES = 44100 * 2


def test_kick_dataset_init_no_data(fs):
    with pytest.raises(FileNotFoundError):
        AudioDataset(
            "nonexistent_dir",
            "nonexistent_file.json",
            TEST_SAMPLE_RATE,
            TEST_NUM_SAMPLES,
        )


def processed_metadata(filename: str):
    # Test metadata to mock a processed dataset
    expected_filename = Path(TEST_DATA_DIR).joinpath(TEST_META_FILE)
    if filename.name != expected_filename:
        raise FileNotFoundError

    metadata = {}
    for i in range(100):
        metadata[i] = {
            "filename": f"kick_{i}.wav",
            "sample_pack_key": "pack_a",
            "type": "electro",
        }

    return metadata


def audio_dataset(fs, mocker, **kwargs):
    # Create a KickDataset with a mocked metadata file
    fs.create_dir(TEST_DATA_DIR)
    fs.create_file(Path(TEST_DATA_DIR).joinpath(TEST_META_FILE))
    mocker.patch("json.load", side_effect=processed_metadata)
    return AudioDataset(
        TEST_DATA_DIR, TEST_META_FILE, TEST_SAMPLE_RATE, TEST_NUM_SAMPLES, **kwargs
    )


def test_audio_dataset_init_no_split(fs, mocker):
    dataset = audio_dataset(fs, mocker)
    assert len(dataset.file_list) == 100


def test_audio_dataset_init_train(fs, mocker):
    dataset = audio_dataset(fs, mocker, split="train")
    assert len(dataset.file_list) == 80


def test_audio_dataset_init_test(fs, mocker):
    dataset = audio_dataset(fs, mocker, split="test")
    assert len(dataset.file_list) == 10


def test_audio_dataset_init_val(fs, mocker):
    dataset = audio_dataset(fs, mocker, split="val")
    assert len(dataset.file_list) == 10


def test_audio_dataset_init_invalid_split(fs, mocker):
    with pytest.raises(ValueError):
        audio_dataset(fs, mocker, split="invalid")


def test_audio_dataset_init_reproducible(fs, mocker):
    dataset_a = audio_dataset(fs, mocker)
    dataset_b = AudioDataset(
        TEST_DATA_DIR, TEST_META_FILE, TEST_SAMPLE_RATE, TEST_NUM_SAMPLES
    )
    assert dataset_a.file_list == dataset_b.file_list


def test_audio_dataset_len(fs, mocker):
    dataset = audio_dataset(fs, mocker)
    len(dataset) == 100


def test_audio_dataset_getitem(fs, mocker):
    dataset = audio_dataset(fs, mocker)
    test_audio = torch.rand(1, TEST_NUM_SAMPLES)
    mocker = mocker.patch(
        f"{TESTED_MODULE}.torchaudio.load",
        return_value=(test_audio, TEST_SAMPLE_RATE),
    )
    audio = dataset[0]
    assert audio.shape == (1, TEST_NUM_SAMPLES)
    assert torch.all(audio == test_audio)


def test_audio_pair_with_feature_dataset(fs, monkeypatch):
    # Create a mock dataset
    fs.create_dir(TEST_DATA_DIR)
    fs.create_file(Path(TEST_DATA_DIR).joinpath(TEST_META_FILE))

    # Fake loading metadata
    def fake_load_json(filename):
        metadata = {}
        for i in range(100):
            metadata[i] = {
                "filename_a": f"kick_a{i}.wav",
                "filename_b": f"kick_b{i}.wav",
                "feature_file": f"features/feature_{i}.pt",
            }
        return metadata

    monkeypatch.setattr(f"{TESTED_MODULE}.json.load", fake_load_json)

    # Fake loading of audio files
    def fake_load_audio(filename):
        return torch.ones(1, TEST_NUM_SAMPLES), TEST_SAMPLE_RATE

    monkeypatch.setattr(f"{TESTED_MODULE}.torchaudio.load", fake_load_audio)

    # Fake loading of feature files
    def fake_load_feature(filename):
        return torch.ones(128)

    monkeypatch.setattr(f"{TESTED_MODULE}.torch.load", fake_load_feature)

    dataset = AudioPairWithFeatureDataset(
        TEST_DATA_DIR,
        TEST_META_FILE,
        TEST_SAMPLE_RATE,
        TEST_NUM_SAMPLES,
        feature_key="feature_file",
    )

    assert len(dataset) == 100
    audio_a, audio_b, feature = dataset[0]
    assert audio_a.shape == (1, TEST_NUM_SAMPLES)
    assert audio_b.shape == (1, TEST_NUM_SAMPLES)
    assert feature.shape == (128,)
