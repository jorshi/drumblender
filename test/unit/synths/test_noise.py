import pytest
import torch

from drumblender.synths import NoiseGenerator


@pytest.fixture
def noise_gen():
    return NoiseGenerator(
        window_size=512,
    )


def test_noise_generator_produces_correct_output_size(noise_gen):
    hop_size = 256
    batch_size = 16
    frame_length = 512
    num_filters = 120

    x = torch.rand(batch_size, frame_length, num_filters)

    y = noise_gen(x)
    assert y.shape == (batch_size, hop_size * (frame_length - 1))
