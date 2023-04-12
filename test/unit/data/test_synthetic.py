"""
Unit tests for the synthetic data module.
"""
import pytest

from drumblender.data import MultiplicationDataset


@pytest.fixture
def multiplication_dataset():
    return MultiplicationDataset(1000)


def test_multiplication_dataset_has_correct_length(multiplication_dataset):
    multiplication_dataset = MultiplicationDataset(1000)
    assert len(multiplication_dataset) == 1000


def test_multiplication_dataset_has_correct_shape(multiplication_dataset):
    assert multiplication_dataset[0][0].shape == (2,)
    assert multiplication_dataset[0][1].shape == (1,)


def test_multiplication_dataset_has_correct_product(multiplication_dataset):
    assert multiplication_dataset[0][0].prod() == multiplication_dataset[0][1]
