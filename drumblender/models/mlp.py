"""Multi-layer perceptron models."""
from torch import nn


def _get_activation(activation: str):
    return getattr(nn, activation)()


class MLP(nn.Module):
    """A simple multi-layer perceptron model.

    Args:
      input_size(int): The dimension of the input vector
      hidden_size(int): The dimension of the hidden layers
      output_size(int): The dimension of the output vector
      hidden_layers(int): The number of hidden layers

    Returns:

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        hidden_layers: int,
        activation: str,
        output_activation: str = None,
    ):
        super().__init__()

        layers = [nn.Linear(input_size, hidden_size), _get_activation(activation)]
        for _ in range(hidden_layers):
            layers.extend(
                [nn.Linear(hidden_size, hidden_size), _get_activation(activation)]
            )
        layers.append(nn.Linear(hidden_size, output_size))
        if output_activation:
            layers.append(_get_activation(output_activation))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
