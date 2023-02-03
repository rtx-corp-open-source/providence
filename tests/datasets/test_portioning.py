"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import pytest

from providence.datasets.adapters import NasaTurbofanTest
from providence.datasets import downsample_to_event_portion, BackblazeDataset, DataSubsetId, NasaFD00XDatasets


@pytest.mark.parametrize("nasa_subset_num", NasaTurbofanTest.all(), ids=lambda nasa_subset: nasa_subset.name)
def test_nasa__train__subset_should_be_strictly_substractive(nasa_subset_num: NasaTurbofanTest):
    """
    Removing a portion of the failures from the NASA dataset should effect the total volume of data,
    but there aren't any censored events in the training sets
    """
    ds_train, ds_test = NasaFD00XDatasets(nasa_subset_num)
    test_portion = 0.8

    original_length = len(ds_train)
    downsample_to_event_portion(ds_train, portion=test_portion)

    expected_length = int(original_length * test_portion)
    print(f"{original_length = } {expected_length = }")
    assert expected_length == len(ds_train), "Subsetting dataset should've removed the appropriate portion of devices"

    original_length_test = len(ds_test)

    expected_length_test = int(original_length_test - original_length_test * round(1 - test_portion, 2))
    print(f"{original_length_test = } {expected_length_test = }")

    downsample_to_event_portion(ds_test, portion=test_portion)

    assert len(ds_test) == expected_length_test, "Subsetting the test set should only remove the failures"


def test__subset_zero_should_leave_unchanged():
    ds = BackblazeDataset(DataSubsetId.Train)
    original_length = len(ds)
    downsample_to_event_portion(ds, portion=1) # keep all the data

    assert len(ds) == original_length, "Mutation should not occur if subsetting is given 1 i.e. 100%"
    print("Train length:", len(ds))

    ds = BackblazeDataset(DataSubsetId.Test)
    original_length = len(ds)
    downsample_to_event_portion(ds, portion=1) # keep all the data

    assert len(ds) == original_length, "Mutation should not occur if subsetting is given 1 i.e. 100%"
    print("Test length:", len(ds))


def test__subset_ten_percent_should_leave_unchanged():
    ds = BackblazeDataset(DataSubsetId.Train)

    n_uncensored = sum((df[ds.event_indicator_column].sum() > 0) for _, df in ds.data)
    n_censored = len(ds) - n_uncensored
    test_portion = 0.1
    downsample_to_event_portion(ds, portion=test_portion) # keep all the data

    expected_length = n_censored + (test_portion * n_uncensored)

    # FIXME: we should know exactly how many devices are in a given dataset. Compute this number exactly and put in the test
    # this accounts for rounding as int(x) ≈ floor and int(x) + 1 ≈ ceil
    assert len(ds) in {int(expected_length), int(expected_length) + 1}, "Mutation subsetting with 0.1 i.e. 10%"