"""Provide synthetic data for development and testing purposes.
"""
import numpy as np
from torch.utils.data import Dataset


class MultiplicationDataset(Dataset):
    """Dataset of pairs of floating point numbers in [0, 1) and their products as
        targets.

    Args:
      length(int): Dataset length
      seed(int): PRNG seed, defaults to 0

    Returns:

    """

    def __init__(self, length: int, seed: int = 0):
        self.length = length
        rng = np.random.default_rng(seed)
        self.data = rng.random((length, 2), dtype=np.float32)
        self.targets = self.data.prod(axis=-1, keepdims=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
