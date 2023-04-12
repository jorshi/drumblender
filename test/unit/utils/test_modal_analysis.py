import numpy as np
import pytest
import torch

import drumblender.utils.audio as audio_utils
import drumblender.utils.modal_analysis as modal_analysis

# These tests will only run if nnAudio is installed
nnAudio = pytest.importorskip("nnAudio")


def test_modal_analysis_init():
    sample_rate = 48000
    x = modal_analysis.CQTModalAnalysis(sample_rate)
    assert x.sample_rate == sample_rate


def test_modal_analysis_spectrogram():
    sample_rate = 48000
    hop_length = 256
    num_bins = 64
    x = modal_analysis.CQTModalAnalysis(
        sample_rate, hop_length=hop_length, n_bins=num_bins
    )
    waveform = torch.randn(1, 48000)

    # Test shape
    spec = x.spectrogram(waveform)
    num_frames = waveform.shape[1] // hop_length + 1
    assert spec.shape == (1, num_bins, num_frames, 2)

    # Test with magnitude spectrogram
    spec = x.spectrogram(waveform, complex=False)
    assert spec.shape == (1, num_bins, num_frames)


def test_modal_analysis_modal_tracking():
    sample_rate = 48000
    hop_length = 256
    num_bins = 64
    x = modal_analysis.CQTModalAnalysis(
        sample_rate, hop_length=hop_length, n_bins=num_bins
    )

    freq_bin = 36
    waveform = audio_utils.generate_sine_wave(
        x.frequencies()[freq_bin], 48000, sample_rate
    )
    spec = x.spectrogram(waveform, complex=True)
    spec = spec[0].numpy()

    freqs, amps, phases = x.modal_tracking(spec)

    # Test that the frequency is correct -- there is quite a bit of spectral
    # leakage, especially at the waveform edges, as a result there are a number
    # of extra sinusodal tracks.
    # Pick the one with the highest amplitude and make sure it's close to the
    # target frequency
    max_track = 0
    max_i = 0
    for i, track in enumerate(amps):
        if sum(track) > max_track:
            max_track = sum(track)
            max_i = i

    assert np.isclose(np.mean(freqs[max_i]), 36, atol=0.025)


def test_modal_analysis_create_modal_tensors():
    sample_rate = 48000
    hop_length = 256
    num_bins = 64
    x = modal_analysis.CQTModalAnalysis(
        sample_rate, hop_length=hop_length, n_bins=num_bins
    )
    assert x.sample_rate == sample_rate


def test_modal_analysis_call():
    sample_rate = 16000
    waveform = audio_utils.generate_sine_wave(
        440, num_samples=sample_rate, sample_rate=sample_rate
    )

    x = modal_analysis.CQTModalAnalysis(
        sample_rate,
        hop_length=256,
        n_bins=60,
        min_length=10,
        num_modes=1,
        threshold=-80.0,
    )
    freqs, amps, phases = x(waveform)

    # Make sure the output shapes are correct
    expected_hops = waveform.shape[1] // 256 + 1
    assert freqs.shape == (1, 1, expected_hops)
    assert amps.shape == (1, 1, expected_hops)
    assert phases.shape == (1, 1, expected_hops)
