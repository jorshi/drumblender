import torch
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
    model = ModalAmpParameters(num_modes+10)
    
    output = model(fake_modal_params)
    assert output.shape == (batch_size, num_params, num_modes, num_steps)
