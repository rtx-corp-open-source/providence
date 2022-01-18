# -*- coding: utf-8 -*-
import os
from pathlib import Path
from providence.utils.datasets.backblaze import make_serial_subsets
from typing import List

import numpy as np
import pytest
import torch
from providence.utils import datasets


class TestNasaDataSet:
    def test___init__(self):
        data = datasets.NasaDataSet(train=True)

        assert len(data.stddev) == 24
        assert len(data.mean) == 24

    def test_train(self, nasa_train):
        got_rul = nasa_train.validation_rul
        want_rul = None

        got_len = len(nasa_train)
        want_len = 709

        assert got_len == want_len
        assert got_rul is None

    def test_test(self, nasa_test):
        got_rul = len(nasa_test.validation_rul)
        want_rul = 707

        got_len = len(nasa_test)
        want_len = 707

        assert got_len == want_len
        assert got_rul == want_rul

    def test__read_data(self, nasa_train):
        got_train = nasa_train._read_data("train", 1).shape
        got_test = nasa_train._read_data("test", 1).shape
        got_rul = nasa_train._read_data("RUL", 1).shape

        assert got_train == (20631, 26)
        assert got_test == (13096, 26)
        assert got_rul == (100,)

    def test__format_data(self, nasa_train):
        raw_data = nasa_train._read_data("train", 1)

        # Only test the first row in array to keep the size of the test small
        got = nasa_train._format_feature_data(raw_data).iloc[0, :].values

        want_array = np.array(
            [
                1.00100e03,
                1.00000e00,
                -7.00000e-04,
                -4.00000e-04,
                1.00000e02,
                5.18670e02,
                6.41820e02,
                1.58970e03,
                1.40060e03,
                1.46200e01,
                2.16100e01,
                5.54360e02,
                2.38806e03,
                9.04619e03,
                1.30000e00,
                4.74700e01,
                5.21660e02,
                2.38802e03,
                8.13862e03,
                8.41950e00,
                3.00000e-02,
                3.92000e02,
                2.38800e03,
                1.00000e02,
                3.90600e01,
                2.34190e01,
            ]
        )

        np.testing.assert_allclose(got, want_array)

    def test__getitem__(self, nasa_test):
        features, target = nasa_test[0]

        got_target_sum = torch.sum(target, dim=0)
        want_target_sum = torch.Tensor([3968.0, 31.0])

        got_feature_sum = torch.sum(features, dim=0)
        want_feature_sum = torch.Tensor(
            [
                -32.3037,
                -34.5655,
                10.7111,
                33.4719,
                32.8841,
                31.3850,
                32.3561,
                34.3599,
                34.5897,
                34.6415,
                24.8771,
                30.9847,
                31.9797,
                28.4807,
                34.6187,
                10.7037,
                16.9500,
                -26.2570,
                30.2205,
                31.2972,
                24.8641,
                10.7111,
                34.5449,
                34.5090,
            ]
        )
        # rather than explicitly testing for tensor equality we're going
        # to reduce them into a 1D tensor via sum
        assert torch.allclose(got_target_sum, want_target_sum)
        assert torch.allclose(got_feature_sum, want_feature_sum)

    def test__prep_features(self, nasa_train):
        _, engine_data = nasa_train.data[0]
        got = nasa_train._prep_features(engine_data.values)

        assert got.shape[1] == 24


@pytest.mark.download_test
class TestBackblazeDataset:
    @pytest.fixture
    def backblaze_train(self):
        return datasets.BackblazeDataset(train=True, download=True, use_feather=True)

    def test___init___no_download(self):
        try:
            ds = datasets.BackblazeDataset(train=True, download=False, prepare=False)
            assert ds is not None, "Instantiating successful"
        except FileNotFoundError:
            assert True, "Test failed as expected: this test is leaky to the file system"
            # more: designing a work around would demand working the implementation to not only fail when a file is not found,
            # but also with empty temp files in place
            # In reality, if you instantiate without downloading that should fail.
            # It would be additional engineering effort to refactor the internals to be configurable such that this test
            # passes, while not making the code opaque. It is more useful to articulate the opinions of the codebase
            # internally, rather that be so generic that useful functionality and expectations aren't discernable.

    def test___init___with_download(self, backblaze_train: datasets.BackblazeDataset):
        ds = backblaze_train
        assert len(np.shape(ds.data)) == 2, "Should have dataset of serial to two aggregate stats"

        np.shape(ds.min) == (len(ds.numerical_features))
        np.shape(ds.max) == (len(ds.numerical_features))

    def test_make_serial_subsets__default_case(self):
        numbers = list(range(5))
        l = make_serial_subsets(numbers, [5])
        assert len(l[0]) == len(numbers), "Should have the same length as the source list"
        assert set(l[0]) == set(numbers), "Should have withdrawn everything that there was to choose"

    def test_make_serial_subsets__default_case_handling_strings(self):
        numbers = [f"frogs times {i}" for i in range(5)]
        l = make_serial_subsets(numbers, [5])
        assert len(l[0]) == len(numbers), "Should have the same length as the source list"
        assert set(l[0]) == set(numbers), "Should have withdrawn everything that there was to choose"

    def test__get_item__(self, backblaze_train: datasets.BackblazeDataset):
        features, targets = backblaze_train[0]
        features.size()[0] == len(datasets.BackblazeDataset.numerical_features)

        features_2, targets_2 = backblaze_train[0]
        assert torch.allclose(features, features_2), "__get_item__ should be idempotent"
        assert torch.allclose(targets, targets_2), "__get_item__ should be idempotent"

    def test_batchability(self, backblaze_train: datasets.BackblazeDataset) -> None:
        from torch.utils.data import DataLoader

        test_bs = 1
        features_1, targets_1 = next(iter(DataLoader(backblaze_train, batch_size=test_bs)))
        assert features_1.shape[0] == test_bs, "Should be able to construct a batch of one"

        max_time_steps = 92  # the maximum number of days in a three month period. This is a quarterly dataset.
        test_bs = 32
        features, sequence_lengths, targets = next(iter(datasets.ProvidenceDataLoader(backblaze_train, batch_size=test_bs)))
        assert features.size() == torch.Size(
            [max_time_steps, test_bs, len(datasets.BackblazeDataset.numerical_features)]
        ), "Should be able to construct a trainable, time-ordered batch"
