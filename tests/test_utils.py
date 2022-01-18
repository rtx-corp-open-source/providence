# -*- coding: utf-8 -*-
import os
from typing import List
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from providence.utils import utils


class TestPadding:
    "Test methods for the padding sequence"

    def test_padding_values(self, list_of_dataframes):
        "Test that padded tensors have the same values as the DF list"
        padded, lenghts = utils.pad(list_of_dataframes)

        for i in range(padded.size()[1]):
            got = np.array(padded[: lenghts[i], i, :])
            want = list_of_dataframes[i]
            # need to use almost equal here since the Tensor <-> np.array can lead to floats
            # being slightly off
            np.testing.assert_array_almost_equal(got, want)

    def test_padding_dims(self, list_of_dataframes):
        "Test that our padding is producing the correct dimensions"
        padded, lenghts = utils.pad(list_of_dataframes)
        got = np.array(padded.size())
        print(got)

        max_size = max([x.shape[0] for x in list_of_dataframes])

        want = np.array([max_size, 5, 4])
        np.testing.assert_array_equal(got, want)

    def test_padding_lengths(self, list_of_dataframes):
        "Test that our lengths are correct"
        _, lenghts = utils.pad(list_of_dataframes)

        got = lenghts

        want = [x.shape[0] for x in list_of_dataframes]
        np.testing.assert_array_equal(got, want)


class TestCollateFn:
    def test_sequence_lengths(self, list_of_dataframes, list_of_targets):
        "Test that sequence lengths are correct"
        batch = list(zip(list_of_dataframes, list_of_targets))

        _, got, _ = utils.collate_fn(batch)

        want = [df.shape[0] for df in list_of_dataframes]
        want = sorted(want, reverse=True)

        np.testing.assert_array_equal(got, want)

    def test_target_padding(self, list_of_dataframes, list_of_targets):
        batch = list(zip(list_of_dataframes, list_of_targets))
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        inputs, _, targets = utils.collate_fn(batch)

        got = targets.shape[0]
        want = inputs.shape[0]

        assert got == want

    def test_batch_size(self, list_of_dataframes, list_of_targets):
        batch = list(zip(list_of_dataframes, list_of_targets))
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        inputs, _, _ = utils.collate_fn(batch)

        got = inputs.shape[1]
        want = len(list_of_dataframes)

        assert got == want
