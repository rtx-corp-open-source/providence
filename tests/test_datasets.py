"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from pandas import isna
import pytest

from providence.datasets import DataFrameSplit
from providence.datasets import NasaPreprocessing
from providence.datasets.adapters import load_nasa_dataframe

@pytest.mark.requires_data
def test_nasa__preprocessing():
    splits = DataFrameSplit(
        load_nasa_dataframe(1, split_name="train", data_root=".data"),
        load_nasa_dataframe(1, split_name="test", data_root=".data"),
    )
    results = NasaPreprocessing(splits, nasa_subset=1)
    assert (isna(results.train) == False).all().all(), "NASA data should make it through transforms without NaN"
    assert (isna(results.test) == False).all().all(), "NASA data should make it through transforms without NaN"
    assert results.validation is None, "No validation set is used"
