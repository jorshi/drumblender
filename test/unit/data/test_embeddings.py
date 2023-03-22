import numpy as np
import pytest
import torch

from kick2kick.data.embeddings import OpenL3


def test_openl3_is_none_raises_import_error_on_call(mocker):
    mocker.patch("kick2kick.data.embeddings.openl3", None)
    emb = OpenL3(sample_rate=44100)
    with pytest.raises(ImportError):
        emb(torch.rand(1, 100))


def test_openl3_asserts_on_incorrect_input_shape(mocker):
    mocker.patch("kick2kick.data.embeddings.openl3")
    emb = OpenL3(sample_rate=44100)
    with pytest.raises(AssertionError):
        emb(torch.rand(2, 100))


def test_openl3_is_called_with_correct_arguments(mocker):
    mock = mocker.patch("kick2kick.data.embeddings.openl3")
    mock.get_audio_embedding.return_value = np.zeros((1, 6144)), None
    emb = OpenL3(
        sample_rate=48000,
        embedding_size=512,
        content_type="env",
        input_repr="linear",
        hop_size=0.5,
    )

    x = torch.testing.make_tensor((1, 100), device="cpu", dtype=torch.float32)
    emb(x)

    mock.get_audio_embedding.assert_called_once_with(
        mocker.ANY,
        48000,
        embedding_size=512,
        content_type="env",
        input_repr="linear",
        hop_size=0.5,
        center=True,
        verbose=0,
    )
    assert (
        mock.get_audio_embedding.call_args[0][0] == x.detach().cpu().numpy().T
    ).all()


def test_openl3_summarize_with_mean(mocker):
    mock = mocker.patch("kick2kick.data.embeddings.openl3")
    mock.get_audio_embedding.return_value = np.ones((10, 6144)), None

    emb = OpenL3(sample_rate=44100, summarize="mean")
    x = torch.testing.make_tensor((1, 100), device="cpu", dtype=torch.float32)
    y = emb(x)

    assert y.shape == (6144,)
    assert (y == 1).all()


def test_openl3_summarize_with_flatten(mocker):
    mock = mocker.patch("kick2kick.data.embeddings.openl3")
    mock.get_audio_embedding.return_value = np.ones((10, 512)), None

    emb = OpenL3(sample_rate=44100, summarize="flatten")
    x = torch.testing.make_tensor((1, 100), device="cpu", dtype=torch.float32)
    y = emb(x)

    assert y.shape == (512 * 10,)
    assert (y == 1).all()


def test_openl3_summarize_with_callable(mocker):
    mock = mocker.patch("kick2kick.data.embeddings.openl3")
    mock.get_audio_embedding.return_value = np.ones((3, 512)), None

    def summarize(x):
        return x.sum(dim=0)

    emb = OpenL3(sample_rate=44100, summarize=summarize)
    x = torch.testing.make_tensor((1, 100), device="cpu", dtype=torch.float32)
    y = emb(x)

    assert y.shape == (512,)
    assert (y == 3).all()
