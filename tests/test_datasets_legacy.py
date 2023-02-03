# -*- coding: utf-8 -*-
"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.dataloaders import ProvidenceDataLoader
from providence.datasets.adapters import BACKBLAZE_FEATURE_NAMES, NASA_METADATA_COLUMNS, NasaTurbofanTest, load_nasa_dataframe
from providence.datasets.backblaze import BackblazeDataset

import numpy as np
import pytest
import torch as pt
from providence import datasets
from providence.datasets.core import DataSubsetId, ProvidenceDataset
from providence.datasets.nasa import NasaDataset



@pytest.fixture
def nasa_train():
    return NasaDataset(DataSubsetId.Train)


@pytest.fixture
def nasa_test():
    return NasaDataset(DataSubsetId.Test)


class TestNasaDataset:
    # TODO: this whole test needs reworking or deletion. It's good documentation.
    def test___init__(self):
        data = datasets.NasaDataset(DataSubsetId.Train)

        assert len(data.feature_columns) == 24

    def test_train(self, nasa_train):
        got_len = len(nasa_train)
        want_len = 709

        assert got_len == want_len

    def test_test(self):
        nasa_test = datasets.NasaDataset(DataSubsetId.Test)

        got_len = len(nasa_test)
        want_len = 707

        assert got_len == want_len

    def test__read_data(self):
        train_df = load_nasa_dataframe(NasaTurbofanTest.FD001, split_name="train", data_root='./.data')
        test_df = load_nasa_dataframe(NasaTurbofanTest.FD001, split_name="test", data_root='./.data')

        got_train = train_df.drop(columns=["RUL"]).shape
        got_test = test_df.drop(columns=["RUL"]).shape

        n_unique_devices = test_df["unit number"].unique().shape

        assert got_train == (20631, 26)
        assert got_test == (13096, 26)
        assert n_unique_devices == (100,)

    def test__load_from_raw(self):
        """Test that the loaded, near-raw Dataframe is exactly what it should be."""

        got = load_nasa_dataframe(NasaTurbofanTest.FD001, split_name="train", data_root='./.data').values[0, :-1]
    

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

    def test__getitem__(self, nasa_test: ProvidenceDataset):
        features, target = nasa_test[0]

        got_target_sum = pt.sum(target, dim=0)
        print(f"{got_target_sum = }")
        want_target_sum = pt.Tensor([3968.0, 31.0])

        got_feature_sum = pt.sum(features, dim=0)
        want_feature_sum = pt.Tensor(
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
        assert pt.allclose(got_target_sum.to(pt.double), want_target_sum.to(pt.double))

        # this is neglible difference at the raw level, as it gets washed out post-normalization
        assert pt.allclose(got_feature_sum.to(pt.double), want_feature_sum.to(pt.double), rtol=1e-4)


@pytest.mark.download_test
class TestBackblazeDataset:
    @pytest.fixture
    def backblaze_train(self):
        return BackblazeDataset(DataSubsetId.Train)

    def test___init___with_download(self, backblaze_train: ProvidenceDataset):
        ds = backblaze_train
        assert len(np.shape(ds.data)) == 2, "Should have dataset of serial to two aggregate stats"
        backblaze_train.grouped.size().size == len(ds)

    def test__get_item__(self, backblaze_train: ProvidenceDataset):
        features, targets = backblaze_train[0]
        max_time_steps = 92  # the maximum number of days in a three month period. This is a quarterly dataset.

        assert features.size()[0] == max_time_steps

        features_2, targets_2 = backblaze_train[0]
        assert pt.allclose(features, features_2), "__get_item__ should be idempotent"
        assert pt.allclose(targets, targets_2), "__get_item__ should be idempotent"

    def test_batchability(self, backblaze_train: ProvidenceDataset) -> None:
        from torch.utils.data import DataLoader

        test_bs = 1
        features_1, targets_1 = next(iter(DataLoader(backblaze_train, batch_size=test_bs)))
        assert features_1.shape[0] == test_bs, "Should be able to construct a batch of one"

        max_time_steps = 92  # the maximum number of days in a three month period. This is a quarterly dataset.
        test_bs = 32
        features, sequence_lengths, targets = next(iter(ProvidenceDataLoader(backblaze_train, batch_size=test_bs)))
        assert features.size() == pt.Size(
            [max_time_steps, test_bs, len(BACKBLAZE_FEATURE_NAMES)]
        ), "Should be able to construct a trainable, time-ordered batch"
