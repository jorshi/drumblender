from drumblender.synths.modal import modal_synth
from drumblender.synths.modal import ModalSynthFreqs
from drumblender.synths.noise import NoiseGenerator
from drumblender.synths.transient import idwt_functional
from drumblender.synths.transient import WaveletConvOLA
from drumblender.synths.transient import WaveletTransform

__all__ = [
    "idwt_functional",
    "NoiseGenerator",
    "ModalSynthFreqs",
    "modal_synth",
    "WaveletConvOLA",
    "WaveletTransform",
]
