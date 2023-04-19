import torch

from drumblender.models.encoders import AutoEncoder
from drumblender.models.encoders import DummyParameterEncoder
from drumblender.models.encoders import ModalAmpParameters


def test_dummy_parameter_encoder_can_be_instantiated():
    model = DummyParameterEncoder((1, 1))
    assert model is not None


def test_dummy_parameter_encoder_can_forward():
    model = DummyParameterEncoder((1, 1))
    output = model(torch.rand(1, 1))
    assert output.shape == (1, 1)
    assert output.requires_grad


def test_modal_amp_parameters_can_forward():
    batch_size = 7
    num_params = 3
    num_modes = 45
    num_steps = 400
    fake_modal_params = torch.rand(batch_size, num_params, num_modes, num_steps)

    # May receive a batch of modal parameters with a different number of modes
    model = ModalAmpParameters(num_modes + 10)

    output = model(None, fake_modal_params)
    assert output.shape == (batch_size, num_params, num_modes, num_steps)


def test_autoencoder_can_init(mocker):
    encoder = mocker.stub("encoder")
    decoder = mocker.stub("decoder")

    model = AutoEncoder(
        encoder=encoder,
        decoder=decoder,
        latent_size=10,
    )
    assert model is not None


def test_autoencoder_can_forward(mocker):
    fake_encoder = torch.nn.Linear(10, 3)
    encoder_spy = mocker.spy(fake_encoder, "forward")

    fake_decoder = torch.nn.Linear(3, 10)
    mocked_decoder = mocker.patch.object(fake_decoder, "forward")
    mocked_decoder.return_value = torch.ones(10, 10)

    model = AutoEncoder(fake_encoder, fake_decoder, latent_size=3)

    x = torch.zeros(10, 10)
    output, _ = model(x)

    torch.testing.assert_close(output, torch.ones(10, 10))
    encoder_spy.assert_called_once_with(x)


def test_autoencoder_can_forward_with_latent(mocker):
    fake_encoder = torch.nn.Linear(10, 3)
    encoder_spy = mocker.spy(fake_encoder, "forward")

    fake_decoder = torch.nn.Linear(3, 10)
    decoder_spy = mocker.spy(fake_decoder, "forward")

    model = AutoEncoder(fake_encoder, fake_decoder, latent_size=3)

    x = torch.zeros(10, 10)
    output, latent = model(x)

    assert output.shape == (10, 10)
    assert latent.shape == (10, 3)

    encoder_spy.assert_called_once_with(x)
    decoder_spy.assert_called_once_with(latent)
