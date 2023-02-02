"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from random import randint
from typing import Iterable

import pytest

from providence.datasets.utils import train_test_split_sizes, train_val_test_split_sizes


@pytest.fixture
def random_ints(count=1000):
    return [randint(2, 1_000_000_000) for _ in range(count)]


@pytest.mark.parametrize("test_percentage", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test__train_test__standard_params(random_ints: Iterable[int], test_percentage: float):
    failures = []
    for test_length in random_ints:
        sizes = train_test_split_sizes(test_length, test_percentage)
        if (delta := test_length - sum(sizes)) != 0:
            failures.append((test_length, delta))

    message = "Encountered {} invalid lengths for split percentage = {}".format(len(failures), test_percentage)
    deltas = list(map(lambda t: t[1], failures))
    if deltas:
        max_d, min_d = max(deltas), min(deltas)
        message += f"{max_d=}, {min_d=}"
    assert len(failures) == 0, message


@pytest.mark.parametrize("test_percentage", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test__train_val_test__standard_params(random_ints: Iterable[int], test_percentage: float):
    failures = []
    for test_length in random_ints:
        sizes = train_val_test_split_sizes(test_length, test_percentage)
        if (delta := test_length - sum(sizes)) != 0:
            failures.append((test_length, delta))

    message = "Encountered {} invalid lengths for split percentage = {}".format(len(failures), test_percentage)
    deltas = list(map(lambda t: t[1], failures))
    if deltas:
        max_d, min_d = max(deltas), min(deltas)
        message += f"{max_d=}, {min_d=}"
    assert len(failures) == 0, message


