import torch

from drumblender.models.decoders import SoundStreamDecoder


def test_soundstreamdecoder_forwards():
    # Test that the SoundStreamDecoder forwards correctly with a random input.
    # This test checks that a latent embedding can be decoded into parameters
    # for noise: (batch_size, latent_size) -> (batch_size, time, 16 filters)
    x = torch.rand(2, 128)
    model = SoundStreamDecoder(128, 16, [9, 5, 2], transpose_output=True)
    y = model(x[..., None])
    assert y.shape == (2, 192, 16)
