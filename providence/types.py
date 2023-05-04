"""
A "bottom" of the import chain, this is the home for types that are used through Providence,
such that keeping them in any other place was a slight to one class or another.
Notice that this module doesn't import anything from anyone except PyTorch. This is intentional

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import NamedTuple
from typing import Optional

import sys

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


from jaxtyping import Float
from jaxtyping import Int
import torch as pt
from torch import device
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset


class DataLoaders(NamedTuple):
    """Multiple DataLoader instances that are meant to be used together.

    ``test`` is optional because
    1. "validation" and "test" are used interchangeably depending on context
    2. You may have designs to constuct a hold-out set by other means, and this type shouldn't require you store
        something. (Propogating a ``None`` would be troublesome)

    Works well with ProvidenceDataloaders as well.
    """

    train: TorchDataLoader
    validation: TorchDataLoader
    test: Optional[TorchDataLoader] = None

    @property
    def train_ds(self) -> TorchDataset:
        """Training dataset for this DataLoaders.

        Returns:
            TorchDataset: dataset from the dataloader under ``train``
        """
        return self.train.dataset

    @property
    def validation_ds(self) -> TorchDataset:
        """Validation dataset for this DataLoaders.

        Returns:
            TorchDataset: dataset from the dataloader under ``validation``
        """
        return self.validation.dataset

    @property
    def test_ds(self) -> TorchDataset:
        """Test dataset for this DataLoaders, if it is set.

        Returns:
            TorchDataset: dataset from the dataloader under ``test`` or None if so.
        """
        return self.test.dataset if self.test else None

    def to_device(self, device: device) -> None:
        for dl in self:
            if dl is not None:
                dl.dataset.device = device


LengthsTensor = Int[pt.Tensor, "entity+0"]
TimeTensor = Int[pt.Tensor, "time+0"]
ProvidenceTensor: TypeAlias = Float[pt.Tensor, "*batch time entity feature"]
