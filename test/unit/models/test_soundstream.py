import torch

from drumblender.models.soundstream import SoundStreamAttentionEncoder


def test_soundstream_attention_encoder_forwards(mocker):
    batch_size = 3
    input_channels = 1
    hidden_channels = 2
    output_channels = 12

    x = torch.rand(batch_size, 1, 512)
    encoder = SoundStreamAttentionEncoder(
        input_channels, hidden_channels, output_channels
    )

    result = encoder(x)

    assert result.shape == (batch_size, output_channels)
