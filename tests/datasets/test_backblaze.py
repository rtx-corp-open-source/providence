# # Purpose
#
# A first pass on using multiple quarters of Backblaze data at the same time.

from providence.utils.datasets import BackblazeDataset
from providence.utils.datasets.backblaze import BackblazeQuarterName
from torch.utils.data import ConcatDataset

import pytest

@pytest.fixture
def ds1() -> BackblazeDataset:
    """
    This dataset requires that the source download `data_Q1_2020.zip` has been unzipped to
    `.data/backblaze-download/data_Q1_2020`
    """
    return BackblazeDataset(quarter_for_download=BackblazeQuarterName._2020_Q1, use_feather=True, download=False)


@pytest.fixture
def ds2() -> BackblazeDataset:
    """
    This dataset requires that the source download `data_Q2_2020.zip` has been unzipped to
    `.data/backblaze-download/data_Q2_2020`.
    See `providence/utils/datasets/scripts/download_backblaze_zip.sh` to pull those manually.
    Alternatively, change these fixtures to `download=True`
    """
    return BackblazeDataset(quarter_for_download=BackblazeQuarterName._2020_Q2, use_feather=True, download=False)


class Test_DatasetConstruction:
    """Tests that the BackblazeDataset objects are correctly loading the data for their quarter"""
    @classmethod
    def assert_length(cls, ds: BackblazeDataset, length: int):
        assert len(ds) == length, f"Default initialization should have {length} records. Found {len(ds)}"

    def test_ds1__has_correct_default_record_count(self, ds1):
        self.assert_length(ds1, 237)

    def test_ds2__has_correct_default_record_count(self, ds2):
        self.assert_length(ds2, 176)

    def test_concat_dataset__wraps_correctl(self, ds1, ds2):
        catds = ConcatDataset([ds1, ds2])
        assert len(catds) == (len(ds1) + len(ds2)), "concatenated dataset is correct"
