"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.datasets import BackblazeDataset
from providence.datasets.backblaze import BackblazeDatasets, BackblazeQuarter
from torch.utils.data import ConcatDataset

import pytest

from providence.datasets.core import DataSubsetId


@pytest.fixture(scope="module")
def ds1() -> BackblazeDataset:
    """
    This dataset requires that the source download `data_Q1_2020.zip` has been unzipped to
    `.data/backblaze-download/data_Q1_2020`
    """
    return BackblazeDataset(DataSubsetId.Train, quarter=BackblazeQuarter._2020_Q1, train_percentage=1)


@pytest.fixture(scope="module")
def ds2() -> BackblazeDataset:
    """
    This dataset requires that the source download `data_Q2_2020.zip` has been unzipped to
    `.data/backblaze-download/data_Q2_2020`
    """
    return BackblazeDataset(DataSubsetId.Train, quarter=BackblazeQuarter._2020_Q2, train_percentage=1)

class Test_DatasetConstruction:
    """Tests that the BackblazeDataset objects are correctly loading the data for their quarter"""

    @classmethod
    def assert_length(cls, ds: BackblazeDataset, length: int, msg: str = None):
        if msg is None:
            msg = f"Initialization should have {length} records. Found {len(ds)}"
        assert len(ds) == length, msg

    def test_ds1__has_correct_default_record_count(self, ds1):
        self.assert_length(ds1, 223 * 2) # n-failures * 2

    def test_ds2__has_correct_default_record_count(self, ds2):
        self.assert_length(ds2, 159 * 2) # n-failures * 2

    def test_concat_dataset__wraps_correctl(self, ds1, ds2):
        catds = ConcatDataset([ds1, ds2])
        self.assert_length(catds, len(ds1) + len(ds2), "concatenated dataset should be correctly sized.")

    def test_train_test__init(self):
        # torch will throw internally if the lengths don't line up, so if this doesn't throw the test passes
        dses = BackblazeDatasets(quarter=BackblazeQuarter._2019_Q4, include_validation=False, random_seed=1234)
        assert len(dses) == 2
        print(f"{[len(ds) for ds in dses] = }")

    def test_train_val_test__init(self):
        # torch will throw internally if the lengths don't line up, so if this doesn't throw the test passes
        dses = BackblazeDatasets(quarter=BackblazeQuarter._2019_Q4, include_validation=True)
        assert len(dses) == 3
        print(f"{[len(ds) for ds in dses] = }")
