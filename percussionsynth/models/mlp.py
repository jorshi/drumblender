"""Multi-layer perceptron models."""
from torch import nn


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
    ):
        super().__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """

        Args:
          x:

        Returns:

        """
        return self.model(x)
