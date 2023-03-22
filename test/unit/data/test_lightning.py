"""
Tests for the data.lightning module.
"""
import unittest.mock as mock
from functools import partial
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

import kick2kick.utils.data as data_utils
from kick2kick.data import KickDataModule
from kick2kick.data import KickModalDataModule
from kick2kick.data import KickModalEmbeddingDataModule
from kick2kick.utils.modal_analysis import CQTModalAnalysis

# from kick2kick.data import KickDataset

TESTED_MODULE = "kick2kick.data.lightning"


@pytest.fixture
def fakefs(fs, mocker):
    """
    Fake FS for testing with a mocked tqdm, which behaves
    poorly with fakefs
    """
    mocker.patch(f"{TESTED_MODULE}.tqdm", side_effect=lambda x: x)
    return fs


def test_kick_data_module_init():
    KickDataModule()


def test_kick_datamodule_prepare_download_archive(fs, mocker):
    # Archive and dataset do not exist, so download and extract
    mocked_download = mocker.patch(f"{TESTED_MODULE}.data_utils.download_file_r2")
    mocked_extract = mocker.patch(f"{TESTED_MODULE}.extract_archive")

    data = KickDataModule()
    data.prepare_data()

    assert mocked_download.call_args_list == [
        mock.call(data.archive, data.url, data.bucket)
    ]
    assert mocked_extract.call_args_list == [mock.call(data.archive, data.data_dir)]


def test_kick_datamodule_prepare_datadir_exists(fs, mocker):
    # Dataset exists, so do not download or extract
    mocked_download = mocker.patch(f"{TESTED_MODULE}.data_utils.download_file_r2")
    mocked_extract = mocker.patch(f"{TESTED_MODULE}.extract_archive")

    data = KickDataModule()
    fs.create_dir(data.data_dir)
    data.prepare_data()

    assert mocked_download.call_args_list == []
    assert mocked_extract.call_args_list == []


def test_kick_datamodule_prepare_archive_exists(fs, mocker):
    # Archive exists, so do not download but extract
    mocked_download = mocker.patch(f"{TESTED_MODULE}.data_utils.download_file_r2")
    mocked_extract = mocker.patch(f"{TESTED_MODULE}.extract_archive")

    data = KickDataModule()
    fs.create_file(data.archive)
    data.prepare_data()

    assert mocked_download.call_args_list == []
    assert mocked_extract.call_args_list == [mock.call(data.archive, data.data_dir)]


def test_kick_datamodule_prepare_unprocessed_raise(fs, mocker):
    # The dataset dir already exists, we expect a RuntimeError is prepare_data is
    # called with use_preprocessed=False
    data = KickDataModule()
    fs.create_dir(data.data_dir)
    with pytest.raises(RuntimeError):
        data.prepare_data(use_preprocessed=False)


def test_kick_datamodule_prepare_unprocessed_downloaded(fs, mocker):
    # Preprocessing the dataset from an existing download of the raw data
    mocked_download = mocker.patch(f"{TESTED_MODULE}.data_utils.download_full_dataset")

    # Mock the preprocessing method of the KickdataModule
    mocked_preprocess = mocker.patch(
        f"{TESTED_MODULE}.KickDataModule.preprocess_dataset"
    )

    data = KickDataModule()
    fs.create_dir(data.data_dir_unprocessed)
    data.prepare_data(use_preprocessed=False)

    assert mocked_download.call_args_list == []
    mocked_preprocess.assert_called_once()


def test_kick_datamodule_prepare_unprocessed_with_downloaded(fs, mocker):
    # Preprocessing the dataset from an existing download of the raw data
    mocked_download = mocker.patch(f"{TESTED_MODULE}.data_utils.download_full_dataset")

    # Mock the preprocessing method of the KickdataModule
    mocked_preprocess = mocker.patch(
        f"{TESTED_MODULE}.KickDataModule.preprocess_dataset"
    )

    data = KickDataModule()
    data.prepare_data(use_preprocessed=False)

    assert mocked_download.call_args_list == [
        mock.call(data.url, data.bucket, data.meta_file, data.data_dir_unprocessed)
    ]
    mocked_preprocess.assert_called_once()


def unprocessed_metadata(filename: str):
    # Mock call to return unprocessed metadata
    data = KickDataModule()
    expected_filename = Path(data.data_dir_unprocessed).joinpath(data.meta_file)
    if filename.name != expected_filename:
        raise FileNotFoundError

    metadata = {
        "sample_group_1": {"type": "cool-sounds", "folders": ["folder1", "folder2"]},
        "sample_group_2": {"type": "even-cooler-sounds", "folders": ["folder3"]},
    }

    return metadata


def create_fake_dataset(metadata: dict, num_files: int, fakefs):
    # Create some fake unprocessed files given and metadata
    data = KickDataModule()
    for group in metadata.values():
        for folder in group["folders"]:
            for i in range(num_files):
                fakefs.create_file(
                    Path(data.data_dir_unprocessed)
                    .joinpath(folder)
                    .joinpath(f"file_{i}.wav")
                )


def expected_hashed_ouput(filename: str, audio_dir: str):
    # This mimics the hashing function used in the dataset
    file = Path(filename)
    output_hash = data_utils.str2int(str(Path(*file.parts[1:])))
    output_file = Path(audio_dir).joinpath(f"{output_hash}.wav")
    return output_file


def test_kick_dataset_preprocess(fakefs, mocker):
    """
    A bit of a complex test to make sure that all functions and files are
    called as expected from the preprocess_dataset class method.
    """
    # Fake the metadata file
    data = KickDataModule()
    meta_file = Path(data.data_dir_unprocessed).joinpath(data.meta_file)
    fakefs.create_file(meta_file)
    mocker.patch("json.load", side_effect=unprocessed_metadata)

    # Create a bunch of fake files
    with open(meta_file, "r") as f:
        metadata = unprocessed_metadata(f)

    num_files = 10
    create_fake_dataset(metadata, num_files, fakefs)

    mocked_preprocess = mocker.patch(
        f"{TESTED_MODULE}.audio_utils.preprocess_audio_file"
    )

    # Mock json dump
    mocked_jsondump = mocker.patch("json.dump")

    # Run the function
    data.preprocess_dataset()

    # Expected input and output filenames
    expected_metadata = {}
    expected_input = {}
    expected_outfile = {}
    for key, group in metadata.items():
        group_type = group["type"]
        for folder in group["folders"]:
            for i in range(num_files):
                in_file = (
                    Path(data.data_dir_unprocessed)
                    .joinpath(folder)
                    .joinpath(f"file_{i}.wav")
                )
                out_file = expected_hashed_ouput(
                    in_file, Path(data.data_dir).joinpath("audio")
                )
                hash_key = int(out_file.stem)
                expected_metadata[hash_key] = {
                    "filename": str(out_file.relative_to(data.data_dir)),
                    "type": group_type,
                    "sample_pack_key": key,
                }
                expected_input[hash_key] = in_file
                expected_outfile[hash_key] = out_file

    # Check that expected calls to preprocess were made
    expected_calls = []
    for key in expected_input.keys():
        expected_calls.append(
            mock.call(
                expected_input[key],
                expected_outfile[key],
                data.sample_rate,
                data.num_samples,
            )
        )

    assert mocked_preprocess.call_args_list == expected_calls

    # Expected calls to json.dump
    json_call = mocked_jsondump.call_args_list[0]
    output_metadata = json_call[0][0]
    for key, value in output_metadata.items():
        assert value == expected_metadata[key]

    # Check that the metadata file was written to the correct location
    assert json_call[0][1].name == Path(data.data_dir).joinpath(data.meta_file)


def test_kick_dataset_archive(mocker):
    data = KickDataModule()
    mocked_archive = mocker.patch(f"{TESTED_MODULE}.data_utils.create_tarfile")
    data.archive_dataset("test.tar.gz")
    mocked_archive.assert_called_once_with("test.tar.gz", data.data_dir)


def processed_metadata(filename: str):
    # Test metadata to mock a processed dataset
    data = KickDataModule()
    expected_filename = Path(data.data_dir).joinpath(data.meta_file)
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


@pytest.fixture
def kick_datamodule(fs, mocker):
    # Mock the dataset directory and metadata file
    data = KickDataModule()
    fs.create_dir(data.data_dir)
    fs.create_file(Path(data.data_dir).joinpath(data.meta_file))
    mocker.patch("kick2kick.data.audio.json.load", side_effect=processed_metadata)
    return KickDataModule()


def test_kick_data_module_setup_train(kick_datamodule):
    kick_datamodule.setup("fit")
    assert len(kick_datamodule.train_dataset) == 80
    assert len(kick_datamodule.val_dataset) == 10
    with pytest.raises(AttributeError):
        kick_datamodule.test_dataset


def test_kick_data_module_setup_val(kick_datamodule):
    kick_datamodule.setup("validate")
    assert len(kick_datamodule.val_dataset) == 10
    with pytest.raises(AttributeError):
        kick_datamodule.train_dataset
    with pytest.raises(AttributeError):
        kick_datamodule.test_dataset


def test_kick_data_module_setup_test(kick_datamodule):
    kick_datamodule.setup("test")
    assert len(kick_datamodule.test_dataset) == 10
    with pytest.raises(AttributeError):
        kick_datamodule.train_dataset
    with pytest.raises(AttributeError):
        kick_datamodule.val_dataset


def test_kick_data_module_train_data(kick_datamodule, mocker):
    # Test that the train data loader works and returns the correct shape
    kick_datamodule.setup("fit")
    train_loader = kick_datamodule.train_dataloader()
    assert isinstance(train_loader, DataLoader)

    mocker = mocker.patch(
        f"{TESTED_MODULE}.torchaudio.load",
        return_value=(
            torch.rand(1, kick_datamodule.num_samples),
            kick_datamodule.sample_rate,
        ),
    )

    batch = next(iter(train_loader))
    assert batch.shape == (kick_datamodule.batch_size, 1, kick_datamodule.num_samples)


def test_kick_modal_data_module_init():
    data = KickModalDataModule()
    assert isinstance(data, KickModalDataModule)
    assert isinstance(data, KickDataModule)


def test_kick_modal_datamodule_prepare_download_archive(fs, mocker):
    # Archive and dataset do not exist, so download and extract
    # Should behave the same as KickDataModule
    mocked_download = mocker.patch(f"{TESTED_MODULE}.data_utils.download_file_r2")
    mocked_extract = mocker.patch(f"{TESTED_MODULE}.extract_archive")

    data = KickModalDataModule()
    data.prepare_data()

    assert mocked_download.call_args_list == [
        mock.call(data.archive, data.url, data.bucket)
    ]
    assert mocked_extract.call_args_list == [mock.call(data.archive, data.data_dir)]


def test_kick_modal_datamodule_prepare_datadir_exists(fs, mocker):
    # Dataset exists, so do not download or extract
    # Should behave the same as KickDataModule
    mocked_download = mocker.patch(f"{TESTED_MODULE}.data_utils.download_file_r2")
    mocked_extract = mocker.patch(f"{TESTED_MODULE}.extract_archive")

    data = KickModalDataModule()
    fs.create_dir(data.data_dir)
    data.prepare_data()

    assert mocked_download.call_args_list == []
    assert mocked_extract.call_args_list == []


def test_kick_modal_datamodule_prepare_archive_exists(fs, mocker):
    # Archive exists, so do not download but extract
    # Should behave the same as KickDataModule
    mocked_download = mocker.patch(f"{TESTED_MODULE}.data_utils.download_file_r2")
    mocked_extract = mocker.patch(f"{TESTED_MODULE}.extract_archive")

    data = KickModalDataModule()
    fs.create_file(data.archive)
    data.prepare_data()

    assert mocked_download.call_args_list == []
    assert mocked_extract.call_args_list == [mock.call(data.archive, data.data_dir)]


def test_kick_modal_datamodule_prepare_unprocessed_raise(fs, mocker):
    # The dataset dir already exists, we expect a RuntimeError is prepare_data is
    # called with use_preprocessed=False
    # Should behave the same as KickDataModule
    data = KickModalDataModule()
    fs.create_dir(data.data_dir)
    with pytest.raises(RuntimeError):
        data.prepare_data(use_preprocessed=False)


def test_kick_modal_datamodule_prepare_unprocessed_downloaded(fs, mocker):
    # Preprocessing the dataset from an existing download of the raw data
    mocked_download = mocker.patch(f"{TESTED_MODULE}.data_utils.download_full_dataset")

    # Mock the preprocessing method of KickModalDataModule
    mocked_preprocess = mocker.patch(
        f"{TESTED_MODULE}.KickModalDataModule.preprocess_dataset"
    )

    data = KickModalDataModule()
    fs.create_dir(data.data_dir_unprocessed)
    data.prepare_data(use_preprocessed=False)

    assert mocked_download.call_args_list == []
    mocked_preprocess.assert_called_once()


def mock_modal_audio_load(filename, sample_rate, num_samples):
    # Mock torchaudio.load
    # We'll assume that the audio files are already in the correct format
    # and just return a tensor of ones
    filename_parts = Path(filename).parts
    assert filename_parts[0] == "dataset"
    assert filename_parts[-1].endswith(".wav")
    return torch.rand(1, num_samples), sample_rate


def mock_cqt_call(x, num_samples, num_frames, num_bins):
    assert x.shape == (1, num_samples)
    freqs = torch.rand(1, num_frames, num_bins)
    amps = torch.rand(1, num_frames, num_bins)
    phases = torch.rand(1, num_frames, num_bins)
    return freqs, amps, phases


def processed_modal_metadata(filename: str):
    # Test metadata to mock a processed dataset
    data = KickModalDataModule()
    expected_filename = Path(data.data_dir).joinpath(data.meta_file)
    if filename.name != expected_filename:
        raise FileNotFoundError

    metadata = {}
    for i in range(100):
        metadata[i] = {
            "filename": f"kick_{i}.wav",
            "filename_modal": f"kick_{i}_modal.wav",
            "features": f"kick_{i}.pt",
            "sample_pack_key": "pack_a",
            "type": "electro",
        }

    return metadata


def mock_json_dump_update(metadata, outfile, expected_outfile):
    # Mock json.dump
    # We'll assume that the metadata file is already in the correct format
    # and just return the metadata
    assert outfile.name == expected_outfile


def run_preprocess_test(data, fakefs, mocker):
    """
    Make sure that the modal preprocessing is calling all the right
    methods with the expected inputs and ouputs. This involves mocking
    several methods.
    """
    # Mock the metadata file
    fakefs.create_dir(data.data_dir)
    fakefs.create_file(Path(data.data_dir).joinpath(data.meta_file))
    mocker.patch("json.load", side_effect=processed_modal_metadata)

    # Mock the base class preprocess method -- we'll assume all the audio files
    # have been converted to the correct format and are now in data_dir
    mocked_preprocess = mocker.patch(
        f"{TESTED_MODULE}.KickDataModule.preprocess_dataset"
    )

    # Mock torchaudio.load
    mocked_load = mocker.patch(
        f"{TESTED_MODULE}.torchaudio.load",
        side_effect=partial(
            mock_modal_audio_load,
            sample_rate=data.sample_rate,
            num_samples=data.num_samples,
        ),
    )

    # Mock cqt analysis
    num_hops = data.num_samples // data.hop_length + 1
    mocker.patch.object(CQTModalAnalysis, "__init__", return_value=None)
    mocker.patch.object(
        CQTModalAnalysis,
        "__call__",
        side_effect=partial(
            mock_cqt_call,
            num_samples=data.num_samples,
            num_frames=num_hops,
            num_bins=data.n_bins,
        ),
    )

    # Mock saving the feature file
    mocked_save = mocker.patch(f"{TESTED_MODULE}.torch.save")

    # Mock savaing metadata file
    mocked_jsondump = mocker.patch(
        f"{TESTED_MODULE}.json.dump",
        side_effect=partial(
            mock_json_dump_update,
            expected_outfile=Path(data.data_dir).joinpath(data.meta_file),
        ),
    )

    data.preprocess_dataset()

    # Check that the base class preprocess method was called
    mocked_preprocess.assert_called_once()

    # Check that torchaudio.load was called for each audio file
    filenames = []
    load_calls = []
    with open(Path(data.data_dir).joinpath(data.meta_file), "r") as f:
        metadata = processed_modal_metadata(f)

    for idx in metadata:
        filename = Path(data.data_dir).joinpath(metadata[idx]["filename"])
        load_calls.append(mocker.call(filename))

    mocked_load.assert_has_calls(load_calls)

    # Check that the feature file was saved to the correct location
    feature_dir = Path(data.data_dir).joinpath("features")
    mocked_save.assert_has_calls(
        [
            mocker.call(mocker.ANY, feature_dir.joinpath(Path(f).with_suffix(".pt")))
            for f in filenames
        ]
    )

    mocked_jsondump.assert_called_once()


def test_kick_modal_dataset_preprocess_no_save_audio(fakefs, mocker):
    """
    Make sure that the modal preprocessing is calling all the right
    methods with the expected inputs and ouputs. This involves mocking
    several methods.
    """

    data = KickModalDataModule(
        sample_rate=16000,
        num_samples=16000,
        n_bins=64,
        hop_length=256,
        save_modal_audio=False,
    )
    run_preprocess_test(data, fakefs, mocker)


def test_kick_modal_dataset_preprocess_save_audio(fakefs, mocker):
    """
    Make sure that the modal preprocessing is calling all the right
    methods with the expected inputs and ouputs. This involves mocking
    several methods.
    """

    data = KickModalDataModule(
        sample_rate=16000,
        num_samples=16000,
        n_bins=64,
        hop_length=256,
        save_modal_audio=True,
    )
    mocked_synth = mocker.patch(
        f"{TESTED_MODULE}.audio_utils.modal_synth",
        return_value=torch.rand(1, data.num_samples),
    )
    mocked_save = mocker.patch(f"{TESTED_MODULE}.torchaudio.save")
    run_preprocess_test(data, fakefs, mocker)

    # Check that mocked synth and mocked save were called 100 times
    assert mocked_synth.call_count == 100
    assert mocked_save.call_count == 100


@pytest.fixture
def kick_modal_datamodule(fs, mocker):
    # Mock the dataset directory and metadata file
    data = KickModalDataModule()
    fs.create_dir(data.data_dir)
    fs.create_file(Path(data.data_dir).joinpath(data.meta_file))
    mocker.patch("kick2kick.data.audio.json.load", side_effect=processed_modal_metadata)
    return KickModalDataModule()


def test_kick_modal_data_module_setup_train(kick_modal_datamodule):
    dm = kick_modal_datamodule
    dm.setup("fit")
    assert len(dm.train_dataset) == 80
    assert len(dm.val_dataset) == 10
    with pytest.raises(AttributeError):
        dm.test_dataset


def test_kick_modal_data_module_setup_val(kick_modal_datamodule):
    dm = kick_modal_datamodule
    dm.setup("validate")
    assert len(dm.val_dataset) == 10
    with pytest.raises(AttributeError):
        dm.test_dataset
    with pytest.raises(AttributeError):
        dm.train_dataset


def test_kick_modal_data_module_setup_test(kick_modal_datamodule):
    dm = kick_modal_datamodule
    dm.setup("test")
    assert len(dm.test_dataset) == 10
    with pytest.raises(AttributeError):
        dm.val_dataset
    with pytest.raises(AttributeError):
        dm.train_dataset


def test_kick_modal_data_module_train_data(kick_modal_datamodule, mocker):
    # Test that the train data loader works and returns the correct shape
    dm = kick_modal_datamodule
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    assert isinstance(train_loader, DataLoader)

    mocker = mocker.patch(
        f"{TESTED_MODULE}.torchaudio.load",
        return_value=(
            torch.rand(1, dm.num_samples),
            dm.sample_rate,
        ),
    )

    audio_batch_a, audio_batch_b = next(iter(train_loader))
    assert audio_batch_a.shape == (dm.batch_size, 1, dm.num_samples)
    assert audio_batch_b.shape == (dm.batch_size, 1, dm.num_samples)


def test_kick_modal_embedding_data_module_preprocess(mocker):
    mock_parent_preprocess = mocker.patch(
        "kick2kick.data.KickModalDataModule.preprocess_dataset"
    )
    mock_preprocess_features = mocker.patch(
        "kick2kick.data.KickModalEmbeddingDataModule.preprocess_features"
    )

    def fake_embedding_model(x):
        return x

    dm = KickModalEmbeddingDataModule(
        embedding_model=fake_embedding_model, feature_prefix="feat"
    )

    dm.preprocess_dataset()
    mock_parent_preprocess.assert_called_once()
    mock_preprocess_features.assert_called_once()


def fake_feature_preprocess(mocker, monkeypatch, fakefs, metadata_file, metadata):
    # Patch loading of metadata
    def mock_load_metadata(path):
        assert path.name == metadata_file
        return metadata

    monkeypatch.setattr(f"{TESTED_MODULE}.json.load", mock_load_metadata)

    # Patch loading of audio
    def mock_load_audio(path):
        assert path.name == "0.wav"
        return torch.rand(1, 16000), 16000

    monkeypatch.setattr(f"{TESTED_MODULE}.torchaudio.load", mock_load_audio)

    # Patch saving of features
    def mock_save_feature(feature, path):
        assert path.name == "0_feat.pt"
        assert feature.shape == (128,)
        torch.testing.assert_close(feature, torch.ones(128))

    monkeypatch.setattr(f"{TESTED_MODULE}.torch.save", mock_save_feature)

    def embedding_model(feature):
        assert feature.shape == (1, 16000)
        return torch.ones(128)

    dm = KickModalEmbeddingDataModule(
        embedding_model=embedding_model,
        feature_prefix="feat",
        data_dir="data",
        meta_file="meta.json",
        sample_rate=16000,
    )
    dm.preprocess_features()


def test_kick_modal_embedding_data_module_preprocess_feature_correctly(
    mocker, monkeypatch, fakefs
):
    metadata_file = Path("data").joinpath("meta.json")
    fakefs.create_file(metadata_file)
    metadata = {
        "0": {
            "filename": "0.wav",
        }
    }

    # Mock the json dump
    mock_json = mocker.patch(f"{TESTED_MODULE}.json.dump")

    fake_feature_preprocess(mocker, monkeypatch, fakefs, metadata_file, metadata)

    # Check that the json dump was called with the correct metadata
    metadata["0"]["feat_feature_file"] = Path("features").joinpath("0_feat.pt")
    assert mock_json.call_args[0][0] == metadata
    assert mock_json.call_args[0][1].name == metadata_file


def test_kick_modal_embedding_data_module_preprocess_feature_file_exists(
    mocker, monkeypatch, fakefs
):
    metadata_file = Path("data").joinpath("meta.json")
    fakefs.create_file(metadata_file)
    metadata = {
        "0": {
            "filename": "0.wav",
        }
    }

    fakefs.create_file(Path("data").joinpath("features", "0_feat.pt"))
    with pytest.raises(FileExistsError):
        fake_feature_preprocess(mocker, monkeypatch, fakefs, metadata_file, metadata)
