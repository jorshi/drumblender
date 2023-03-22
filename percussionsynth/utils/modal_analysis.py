"""
Methods for performing modal analysis on a CQT spectrogram
Based on Sinusoidal Modeling Synthesis (SMS) Toolbox by Xavier Serra and Julius Smith
Uses the nnAudio CQT implementation for better resolution with low frequncies
"""
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch

# nnAudio may not be installed -- some of the methods will not work here,
# but they are only required for pre-processing the audio so we won't
# raise an error, unless the user tries to use a method that requires nnAudio
try:
    import nnAudio.features as features
except ImportError:
    features = None


class CQTModalAnalysis:
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 64,
        fmin: float = 24.0,
        n_bins: int = 96,
        bins_per_octave: int = 12,
        min_length: int = 4,
        num_modes: Optional[int] = None,
        threshold: float = -80.0,
        **kwargs,
    ) -> None:
        """
        Class for performing sinusoidal modelling using a CQT spectrogram.

        Args:
            sample_rate: Sample rate of the incoming audio
            hop_length: Hop length of the CQT spectrogram
            fmin: Minimum frequency of the CQT spectrogram
            n_bins: Number of bins in the CQT spectrogram
            bins_per_octave: Number of bins per octave in the CQT spectrogram
            min_length: Minimum length of a track in frames
            num_modes: Number of modes to return. If None, will return all
            threshold: Threshold for amplitude in dB for a mode to be considered
            **kwargs: Additional keyword arguments to pass to the nnAudio CQT
        """

        if features is None:
            raise ImportError(
                "nnAudio is not installed. Please install it to use this class."
                "You can install it with `pip install kick2kick[modal]`"
            )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.min_length = min_length
        self.num_modes = num_modes
        self.threshold = threshold

        self.cqt = features.CQT(
            sr=sample_rate,
            fmin=fmin,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            n_bins=n_bins,
            output_format="Complex",
            **kwargs,
        )

    def __call__(
        self,
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs modal analysis on a batch of audio waveforms

        Args:
            audio: Audio waveform of shape (batch, samples)

        Returns:
            Tuple of (frequencies, amplitudes, phases) of shape (batch, modes, frames)
        """

        assert audio.ndim == 2, "Audio must be a batch of waveforms"
        x = self.spectrogram(audio, complex=True)
        (_, _, num_hops, _) = x.shape

        # After nnAudio spectrogram, everything is on the CPU with numpy
        x = x.cpu().numpy()

        # For each item in the batch perform modal tracking
        batch_freqs = []
        batch_amps = []
        batch_phases = []
        for i in range(x.shape[0]):
            freqs, amps, phases = self.modal_tracking(x[i], threshold=self.threshold)
            freqs, amps, phases = self.create_modal_tensors(
                freqs, amps, phases, num_hops=num_hops, min_length=self.min_length
            )

            # Convert to Hz
            freqs = torch.pow(2.0, freqs / self.bins_per_octave) * self.fmin

            # Filter out modes
            if self.num_modes is not None:
                modal_energy = amps.sum(dim=1)
                modal_energy, idx = torch.sort(modal_energy, descending=True)
                freqs = freqs[idx[: self.num_modes]]
                amps = amps[idx[: self.num_modes]]
                phases = phases[idx[: self.num_modes]]

            batch_freqs.append(freqs)
            batch_amps.append(amps)
            batch_phases.append(phases)

        batch_freqs = torch.stack(batch_freqs, dim=0)
        batch_amps = torch.stack(batch_amps, dim=0)
        batch_phases = torch.stack(batch_phases, dim=0)
        return batch_freqs, batch_amps, batch_phases

    def spectrogram(self, audio: torch.Tensor, complex: bool = True) -> torch.Tensor:
        """
        CQT spectrogram of the audio

        Args:
            audio: Audio waveform of shape (batch, samples)
            complex: Whether to return a complex spectrogram, or a magnitude spectrogram

        Returns:
            A complex CQT spectrogram of shape (batch, bins, frames, 2) or a magnitude
            spectrogram of shape (batch, bins, frames) if complex is false
        """
        x = self.cqt(audio, normalization_type="wrap")
        if not complex:
            x = torch.sqrt(torch.sum(torch.square(x), dim=-1))

        return x

    def modal_tracking(
        self, spec: np.ndarray, threshold: float = -80.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs modal tracking on a CQT spectrogram -- i.e. finds sinusoidal tracks
        and attempts to continue them across frames.

        Args:
            spec: A complex CQT spectrogram of shape (bins, frames, 2)
            threshold: Threshold for peak detection in dB

        Returns:
            Tuple of (frequencies, amplitudes) of shape (modes, frames)
        """

        assert spec.ndim == 3, "Spectrogram must a complex spectrogram"
        assert spec.shape[-1] == 2, "Spectrogram must be complex"

        # Reverse the spectrogram along the temporal axis
        # Assuming percussive audio here and that the most
        # important modes will have the longest decay
        spec = np.flip(spec, axis=1)

        # Initialize lists to store each mode's frequency, amplitude, and phase
        freqs = []
        amps = []
        phases = []

        # Select modes from the spectrogram
        for i in range(spec.shape[1]):
            # Get the magnitude, dB, and phase of the current frame
            frame = spec[:, i]
            assert frame.shape == (self.n_bins, 2)

            X_mag = np.sqrt(np.sum(np.square(frame), axis=-1))
            X_db = 20.0 * np.log10(X_mag + 1e-8)
            X_phase = np.arctan2(frame[:, 1], frame[:, 0])

            # Find peaks in the current frame
            peaks = peak_detection(X_db, threshold)
            if len(peaks) == 0:
                continue

            peaks_loc, peaks_mag, peaks_phase = peak_interpolation(
                X_mag, X_phase, peaks
            )

            # Initialize modes
            if len(freqs) == 0:
                freqs.append(list(peaks_loc))
                amps.append(list(peaks_mag))
                phases.append(list(peaks_phase))
                continue

            # Try to continue the mode from the previous frame
            for j in range(len(freqs)):
                if len(peaks_loc) > 0:
                    # Find difference between previous peak and current peaks
                    prev_freq = freqs[j][-1]
                    peak_diff = np.abs(peaks_loc - prev_freq)

                    # If the difference is less than 2.5% of the previous frequency,
                    # then we assume that the peak is the same mode
                    if np.min(peak_diff) < prev_freq * 0.025:
                        closest_peak = np.argmin(peak_diff)
                        freqs[j].append(peaks_loc[closest_peak])
                        amps[j].append(peaks_mag[closest_peak])
                        phases[j].append(peaks_phase[closest_peak])

                        # Remove the peak from the list
                        peaks_loc = np.delete(peaks_loc, closest_peak)
                        peaks_mag = np.delete(peaks_mag, closest_peak)
                    else:
                        # No good matching peaks, just copy the last peak, but
                        # with an amplitude of 0.
                        freqs[j].append(freqs[j][-1])
                        amps[j].append(0.0)
                        phases[j].append(phases[j][-1])
                else:
                    # If there are no more peaks, just copy the last peak
                    freqs[j].append(freqs[j][-1])
                    amps[j].append(0.0)
                    phases[j].append(phases[j][-1])

            # Add any remaining peaks as new modes
            for peak in peaks_loc:
                freqs.append([peak])

            for peak in peaks_mag:
                amps.append([peak])

            for peak in peaks_phase:
                phases.append([peak])

        return freqs, amps, phases

    def create_modal_tensors(
        self,
        freqs: List[List[float]],
        amps: List[List[float]],
        phases: List[List[float]],
        num_hops: int,
        min_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts lists of modal freqs, amps, and phases into
        tensors of shape (modes, num_hops). Input lists will be of
        different lengths so they will be padded with zeros to be of length num_hops.
        Modes are also filtered out if they are shorter than min_length.

        Args:
            freqs: List of list of frequencies of shape (modes, frames)
            amps: List of list of amplitudes of shape (modes, frames)
            phases: List of list of phases of shape (modes, frames)
            num_hops: Number of frames in the spectrogram
            min_length: Minimum length of a mode to be included

        Returns:
            Tuple of (frequencies, amplitudes, phases) of shape (modes, num_hops)
        """

        num_modes = len(freqs)
        freq_env = []
        amp_env = []
        phase_env = []

        for i in range(num_modes):
            # Check if the mode is long enough
            if len(freqs[i]) < min_length:
                continue

            freq_env.append(torch.zeros(num_hops))
            amp_env.append(torch.zeros(num_hops))
            phase_env.append(torch.zeros(num_hops))

            freqs[i].reverse()
            amps[i].reverse()
            phases[i].reverse()

            for h in range(num_hops):
                if h < len(freqs[i]):
                    freq_env[-1][h] = freqs[i][h]
                    amp_env[-1][h] = amps[i][h]
                    phase_env[-1][h] = phases[i][h]
                else:
                    break

        freq_env = torch.stack(freq_env)
        amp_env = torch.stack(amp_env)
        phase_env = torch.stack(phase_env)

        return freq_env, amp_env, phase_env

    def frequencies(self) -> np.ndarray:
        """
        Returns the frequencies of the CQT bins
        """
        return self.cqt.frequencies


# TODO: Maybe switch this out for scipy.signal.find_peaks?
def peak_detection(
    x: np.ndarray,  # magnitude spectrum
    threshold: float,  # threshold
) -> np.ndarray:
    """
    Detect spectral peak locations
    From: https://github.com/MTG/sms-tools

    Args:
        x: magnitude spectrum
        threshold: threshold for peak picking

    Returns:
        Peak locations
    """
    thresh = np.where(
        np.greater(x[1:-1], threshold), x[1:-1], 0
    )  # locations above threshold
    next_minor = np.where(
        x[1:-1] > x[2:], x[1:-1], 0
    )  # locations higher than the next one
    prev_minor = np.where(
        x[1:-1] > x[:-2], x[1:-1], 0
    )  # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor  # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1  # add 1 to compensate for previous steps
    return ploc


def peak_interpolation(
    magnitude: np.ndarray,
    phase: np.ndarray,
    ploc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate peak values using parabolic interpolation
    From: https://github.com/MTG/sms-tools

    Args:
        magnitude: magnitude spectrum
        phase: phase spectrum
        ploc: peak locations

    Returns:
        Interpolated peak locations, magnitudes, and phases
    """
    # Magnitude of the peak bin and its neighbours
    val = magnitude[ploc]
    lval = magnitude[ploc - 1]
    rval = magnitude[ploc + 1]

    # Parabolic interpolation
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)
    ipphase = np.interp(iploc, np.arange(0, phase.size), phase)

    return iploc, ipmag, ipphase
