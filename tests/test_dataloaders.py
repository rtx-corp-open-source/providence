"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Tuple
import pytest
from torch import device
from providence.datasets.adapters import NasaTurbofanTest

from providence.datasets import ProvidenceDataset, NasaFD00XDatasets
from providence.dataloaders import DataLoaders, ProvidenceDataLoader


def NasaFD001Datasets() -> Tuple[ProvidenceDataset, ProvidenceDataset]:
    return NasaFD00XDatasets(NasaTurbofanTest.FD001)


class TestDataLoadersWrapper:
    @pytest.fixture
    def nasa_example(self) -> DataLoaders:
        train_ds, test_ds = NasaFD001Datasets()
        train_dl, test_dl = ProvidenceDataLoader(train_ds), ProvidenceDataLoader(test_ds)

        wrapper = DataLoaders(train_dl, test_dl)
        return wrapper

    def test_init(self):
        train_ds, test_ds = NasaFD001Datasets()
        train_dl, test_dl = ProvidenceDataLoader(train_ds), ProvidenceDataLoader(test_ds)

        wrapper = DataLoaders(train_dl, test_dl)
        assert wrapper.train is not None, wrapper.validation is test_dl

    # def test_to_device(self, nasa_example: DataLoaders):
    #     # test preconditions
    #     assert isinstance(nasa_example.train.dataset, ProvidenceDataset)
    #     assert isinstance(nasa_example.validation.dataset, ProvidenceDataset)

    #     # actual test preamble
    #     def assert_ds_is_on_device(ds: ProvidenceDataset, destination_device: device) -> bool:
    #         assert isinstance(ds.device, (device, str))
    #         assert ds.device == destination_device

    #     initial_device = device('cpu')
    #     assert_ds_is_on_device(nasa_example.train.dataset, initial_device)
    #     assert_ds_is_on_device(nasa_example.validation.dataset, initial_device)

    #     # actual test action that changes the state of the system
    #     test_device = 'cpu'
    #     nasa_example.to_device(test_device)

    #     # assert that the change propogated through the system
    #     assert_ds_is_on_device(nasa_example.train.dataset, test_device)
    #     assert_ds_is_on_device(nasa_example.validation.dataset, test_device)


def test_num_features():
    train_ds, test_ds = NasaFD001Datasets()
    train_dl, test_dl = ProvidenceDataLoader(train_ds), ProvidenceDataLoader(test_ds)

    from providence.datasets.adapters import NASA_FEATURE_NAMES
    assert train_dl.num_features == test_dl.num_features == len(NASA_FEATURE_NAMES), 'Number of features should be ' \
                                                                                     'identical throughout '
