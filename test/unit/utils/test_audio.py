import numpy as np
import pytest
import torch
import torchaudio

import kick2kick.utils.audio as audio_utils


def preprocess_audio_file(
    path_factory,
    in_sr=16000,
    out_sr=16000,
    in_dur=1.0,
    out_dur=1.0,
    in_stereo=False,
    amp=1.0,
):
    # Generate a test audio file and save it
    n = int(in_dur * in_sr)
    audio = audio_utils.generate_sine_wave(440, n, in_sr, in_stereo) * amp
    input_file = path_factory.mktemp("data") / "test_input.wav"
    torchaudio.save(input_file, audio, in_sr)

    assert audio.shape[0] == 2 if in_stereo else 1

    # Preprocess the audio file
    output_file = path_factory.mktemp("data") / "test_preprocessed.wav"
    audio_utils.preprocess_audio_file(
        input_file=input_file,
        output_file=output_file,
        sample_rate=out_sr,
        num_samples=int(out_dur * out_sr),
    )
    assert output_file.exists()
    return output_file


def test_preprocess_audio_file_noresample(tmp_path_factory):
    input_sample_rate = 16000
    target_sample_rate = 16000
    output_file = preprocess_audio_file(
        tmp_path_factory, in_sr=input_sample_rate, out_sr=target_sample_rate
    )

    # Load the preprocessed audio file
    waveform, sample_rate = torchaudio.load(output_file)
    assert sample_rate == target_sample_rate


def test_preprocess_audio_file_resample(tmp_path_factory):
    input_sample_rate = 16000
    target_sample_rate = 48000
    output_file = preprocess_audio_file(
        tmp_path_factory, in_sr=input_sample_rate, out_sr=target_sample_rate
    )

    # Load the preprocessed audio file
    waveform, sample_rate = torchaudio.load(output_file)
    assert sample_rate == target_sample_rate


def test_preprocess_audio_file_resample_stereo(tmp_path_factory):
    input_sample_rate = 16000
    target_sample_rate = 48000
    output_file = preprocess_audio_file(
        tmp_path_factory,
        in_sr=input_sample_rate,
        out_sr=target_sample_rate,
        in_stereo=True,
    )

    # Load the preprocessed audio file
    waveform, sample_rate = torchaudio.load(output_file)
    assert sample_rate == target_sample_rate
    assert waveform.shape[0] == 1


def test_preprocess_audio_file_resample_pad(tmp_path_factory):
    input_sample_rate = 16000
    target_sample_rate = 48000
    input_duration = 1.0
    target_duration = 2.0
    output_file = preprocess_audio_file(
        tmp_path_factory,
        in_sr=input_sample_rate,
        out_sr=target_sample_rate,
        in_dur=input_duration,
        out_dur=target_duration,
    )

    # Load the preprocessed audio file
    waveform, sample_rate = torchaudio.load(output_file)
    assert sample_rate == target_sample_rate
    assert waveform.shape[1] == int(target_duration * target_sample_rate)


def test_preprocess_audio_file_resample_trim(tmp_path_factory):
    input_sample_rate = 16000
    target_sample_rate = 48000
    input_duration = 1.0
    target_duration = 0.5
    output_file = preprocess_audio_file(
        tmp_path_factory,
        in_sr=input_sample_rate,
        out_sr=target_sample_rate,
        in_dur=input_duration,
        out_dur=target_duration,
    )

    # Load the preprocessed audio file
    waveform, sample_rate = torchaudio.load(output_file)
    assert sample_rate == target_sample_rate
    assert waveform.shape[1] == int(target_duration * target_sample_rate)


def test_preprocess_audio_file_raises_warning_on_quiet_sound(tmp_path_factory):
    with pytest.raises(ValueError, match="Entire wavfile below threshold level"):
        preprocess_audio_file(
            tmp_path_factory,
            amp=1e-6,
        )


def test_modal_synth(tmp_path):
    num_modes = 3
    num_frames = 24
    sample_rate = 16000
    num_samples = 16000

    # Create modal frequencies, amplitudes, and phases
    modal_freqs = torch.ones(1, num_modes, num_frames)
    modal_freqs[:, 0, :] = 440
    modal_freqs[:, 1, :] = 880
    modal_freqs[:, 2, :] = 1320

    modal_amps = torch.ones(1, num_modes, num_frames)
    modal_amps[:, 0, :] = 0.5 * torch.linspace(1.0, 0.0, num_frames)
    modal_amps[:, 1, :] = 0.3 * torch.linspace(0.0, 1.0, num_frames)
    modal_amps[:, 2, :] = 0.2

    audio = audio_utils.modal_synth(modal_freqs, modal_amps, sample_rate, num_samples)

    assert audio.shape == (1, num_samples)


def test_first_non_silent_sample_returns_correct_sample():
    # Create a fake signal with a silent start and energy in the second half
    waveform = torch.zeros(1000)
    waveform[500:] = 1.0

    # Find the first non-silent sample
    first_non_silent_sample = audio_utils.first_non_silent_sample(
        waveform, frame_size=100, hop_size=100
    )

    assert first_non_silent_sample == 500


def test_first_non_silent_sample_thresholding_works_correctly():
    # Create a fake signal with a start below the threshold
    # and second half above the threshold
    threshold_db = -20.0
    waveform = torch.zeros(1000)
    waveform[:500] = np.power(10.0, threshold_db / 20.0) * 0.99
    waveform[500:] = np.power(10.0, threshold_db / 20.0) * 1.01

    # Find the first non-silent sample
    first_non_silent_sample = audio_utils.first_non_silent_sample(
        waveform, frame_size=100, hop_size=100, threshold_db=threshold_db
    )

    assert first_non_silent_sample == 500


def test_first_non_silent_sample_below_threshold_returns_none():
    # Create a fake signal entirely below the threshold
    threshold_db = -20.0
    waveform = torch.ones(1000) * np.power(10.0, threshold_db / 20.0) * 0.99

    # Find the first non-silent sample
    first_non_silent_sample = audio_utils.first_non_silent_sample(
        waveform, frame_size=100, hop_size=100, threshold_db=threshold_db
    )

    assert first_non_silent_sample is None


def test_cut_start_silence_raises_error_for_incorrect_tensor_shape():
    waveform = torch.zeros(1000)

    with pytest.raises(AssertionError):
        audio_utils.cut_start_silence(waveform)


def test_cut_start_silence_clips_tensor(mocker):
    mock = mocker.patch("kick2kick.utils.audio.first_non_silent_sample")
    mock.side_effect = [200, 100]

    waveform = torch.zeros(2, 1000)
    waveform = audio_utils.cut_start_silence(waveform)

    assert waveform.shape == (2, 900)


def test_cut_start_silence_raises_error_for_silent_input(mocker):
    _ = mocker.patch("kick2kick.utils.audio.first_non_silent_sample", return_value=None)

    waveform = torch.zeros(1, 1000)
    with pytest.raises(ValueError):
        audio_utils.cut_start_silence(waveform)
