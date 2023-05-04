# -*- coding: utf-8 -*-
"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import List

import numpy as np
import pytest
import torch as pt

from providence.dataloaders import providence_collate_fn
from providence.dataloaders import providence_pad_sequence
from providence.utils import multilevel_sort


@pytest.fixture
def list_of_tensors(n_devices: int = 5, n_features: int = 4, sequence_bounds=(10, 20)) -> List[pt.Tensor]:
    "Create five (devices|entities) (of n_features) with sequence lengths between sequence_bounds"
    tens = [
        pt.rand(pt.randint(sequence_bounds[0], sequence_bounds[1], (1,)).item(), n_features) for _ in range(n_devices)
    ]
    return tens


class TestPadding:
    "Test methods for the padding sequence. These are done whitebox style"

    def test_padding_values(self, list_of_dataframes):
        "Test that padded tensors have the same values as the DF list"
        padded, lenghts = providence_pad_sequence(list_of_dataframes)

        for i in range(padded.size()[1]):
            got = np.array(padded[: lenghts[i], i, :])
            want = list_of_dataframes[i]
            # need to use almost equal here since the Tensor <-> np.array can lead to floats
            # being slightly off
            np.testing.assert_array_almost_equal(got, want)

    def test_padding_dims(self, list_of_dataframes):
        "Test that our padding is producing the correct dimensions"
        padded, lenghts = providence_pad_sequence(list_of_dataframes)
        got = np.array(padded.size())
        print(got)

        max_size = max([x.shape[0] for x in list_of_dataframes])

        want = np.array([max_size, 5, 4])  # NOTE: coupled to the dataframe definitions
        np.testing.assert_array_equal(got, want)

    def test_padding_lengths(self, list_of_dataframes):
        "Test that our lengths are correct"
        _, lenghts = providence_pad_sequence(list_of_dataframes)

        got = lenghts

        want = [x.shape[0] for x in list_of_dataframes]
        np.testing.assert_array_equal(got, want)

    def test_padding_values__with_tensor(self, list_of_tensors):
        "Test that padded tensors have the same values as the DF list"
        padded, lenghts = providence_pad_sequence(list_of_tensors)

        for i in range(padded.size()[1]):
            got = np.array(padded[: lenghts[i], i, :])
            want = list_of_tensors[i]
            # need to use almost equal here since the Tensor <-> np.array can lead to floats
            # being slightly off
            np.testing.assert_array_almost_equal(got, want)

    def test_padding_dims__with_tensor(self, list_of_tensors):
        "Test that our padding is producing the correct dimensions"
        padded, lenghts = providence_pad_sequence(list_of_tensors)
        got = np.array(padded.size())
        print(got)

        max_size = max([x.shape[0] for x in list_of_tensors])

        want = np.array([max_size, 5, 4])  # NOTE: coupled to the dataframe definitions
        np.testing.assert_array_equal(got, want)

    def test_padding_lengths__with_tensor(self, list_of_tensors):
        "Test that our lengths are correct"
        _, lenghts = providence_pad_sequence(list_of_tensors)

        got = lenghts

        want = [x.shape[0] for x in list_of_tensors]
        np.testing.assert_array_equal(got, want)


class TestCollateFn:
    def test_sequence_lengths(self, list_of_dataframes, list_of_targets):
        "Test that sequence lengths are correct"
        batch = list(zip(list_of_dataframes, list_of_targets))

        _, got, _ = providence_collate_fn(batch)

        want = [df.shape[0] for df in list_of_dataframes]
        want = sorted(want, reverse=True)

        np.testing.assert_array_equal(got, want)

    def test_target_padding(self, list_of_dataframes, list_of_targets):
        batch = list(zip(list_of_dataframes, list_of_targets))
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        inputs, _, targets = providence_collate_fn(batch)

        got = targets.shape[0]
        want = inputs.shape[0]

        assert got == want

    def test_batch_size(self, list_of_dataframes, list_of_targets):
        batch = list(zip(list_of_dataframes, list_of_targets))
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        inputs, _, _ = providence_collate_fn(batch)

        got = inputs.shape[1]
        want = len(list_of_dataframes)

        assert got == want


class TestMultilevelCompare:
    @classmethod
    @pytest.fixture
    def example1(cls):
        return dict(a=12, b=7)

    @classmethod
    @pytest.fixture
    def example2(cls):
        return dict(a=10, b=8)

    @classmethod
    @pytest.fixture
    def example3(cls):
        return dict(a=7, b=6)

    def test__standin_for_direct_comparison__key_a(self, example1: dict, example2: dict):
        result = multilevel_sort([example1, example2], keys=["a"])
        assert [
            example2,
            example1,
        ] == result, "Multi-level compare is not a valid, drop-in replacement for direct comparison"

    def test__standin_for_direct_comparison__key_b(self, example1: dict, example2: dict):
        result = multilevel_sort([example1, example2], keys=["b"])
        assert [
            example1,
            example2,
        ] == result, "Multi-level compare is not a valid, drop-in replacement for direct comparison"

    def test__standin_for_direct_sort__key_a(self, example1: dict, example2: dict, example3: dict):
        result = multilevel_sort([example1, example2, example3], keys=["a"])
        assert [
            example3,
            example2,
            example1,
        ] == result, "Multi-level sort is not a valid, drop-in replacement for sort(ed)"

    def test__standin_for_direct_sort__keys_a_and_b(self, example1: dict, example2: dict, example3: dict):
        result = multilevel_sort([example1, example2, example3], keys="b a".split(" "))
        assert [
            example3,
            example1,
            example2,
        ] == result, "Multi-level sort is not a valid, drop-in replacement for sort(ed)"
