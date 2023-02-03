"""
A "bottom" of the import chain, this is the home for types that are used through Providence,
such that keeping them in any other place was a slight to one class or another.
Notice that this module doesn't import anything from anyone except PyTorch. This is intentional

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from torch import device
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset

from typing import NamedTuple, Optional


class DataLoaders(NamedTuple):
    train: TorchDataLoader
    validation: TorchDataLoader
    test: Optional[TorchDataLoader] = None

    @property
    def train_ds(self) -> TorchDataset:
        return self.train.dataset

    @property
    def validation_ds(self) -> TorchDataset:
        return self.validation.dataset

    @property
    def test_ds(self) -> TorchDataset:
        return self.test.dataset if self.test else None

    def to_device(self, device: device) -> None:
        for dl in self:
            if dl is not None:
                dl.dataset.device = device
