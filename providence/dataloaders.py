"""
Dataloaders and supporting utility functions for loading data in a Providence-compliant format.
See `.providence_pad_sequence` for an explanation

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import List, Sequence, Tuple, Type, TypeVar, Union
from torch import device, Tensor, cuda
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence

from providence.datasets import BackblazeDatasets, NasaDatasets, NasaFD00XDatasets, ProvidenceDataset
from providence.datasets.adapters import BackblazeQuarter, NasaTurbofanTest
from providence.datasets.backblaze import BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER, BackblazeExtendedDatasets
from providence.types import DataLoaders

################################################################################
#
# Dataloader Support
#
################################################################################


def is_list_of(maybe_list, type_: Type) -> bool:
    return isinstance(maybe_list, (Sequence, list)) and all(map(lambda x: isinstance(x, type_), maybe_list))


# type of a dataset element: features and targets
# TODO: centralize some of these types. This one is much more general than the dataloaders...
ProvidenceItem = Tuple[Tensor, Tensor]
T = TypeVar('T')


def providence_pad_sequence(
    data: Union[Tensor, Sequence[Tensor], List[T]], target_device: device = device('cpu')
) -> Tuple[Tensor, Tensor]:
    """
    Padding function for variable length sequences
    This function concatenates a list of panels. The result
    will resemble something akin to the following:
    .. code-block::

            /     FEATURE2   /     FEATURE2   /     FEATURE3    /|
           /_______________ /________________/________________ / |
          /     FEATURE1   /     FEATURE1   /     FEATURE1    /| |
         /_______________ / _______________/_______________  / |/|
    T1   |   Subject1    |   Subject2    |   Subject3       | /| |
         |_______________|_______________|__________________|/ |/|
    T2   |   Subject1    |   Subject2    |   Subject3       | /| |
         |_______________|_______________|__________________|/ | |
         |               |               |                  |  | |
         ...
    :param data: List of NxM matricies
    :return: Tuple[Tensor, Tensor[int]]
    """
    lengths = Tensor([len(x) for x in data]).long()

    if is_list_of(data, Tensor):
        padded = pad_sequence(data, batch_first=False)
    else:
        padded = pad_sequence([Tensor(sequence, device=target_device) for sequence in data])

    return padded, lengths


def providence_collate_fn(batch: List[ProvidenceItem]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Collate a batch around the temporal / sequence dimension, rather than batch-first.
    We pad to the longest sequence. See `providence_pad_sequence()` for more intuition there.

    :params batch: Dataset batch
    :returns: Padded inputs, the real sequence lengths, and padded targets

    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # we want to sort the largest array first
    inputs, targets = zip(*batch)
    inputs, inputs_lengths = providence_pad_sequence(inputs)

    # target sequence length is already captured by inputs_length
    targets, _ = providence_pad_sequence(targets)
    return inputs, inputs_lengths, targets


################################################################################
#
# Actual Dataloaders
#
################################################################################


class ProvidenceDataLoader(TorchDataLoader):
    def __init__(self, *args, collate_fn=providence_collate_fn, **kwargs) -> None:
        """
        Standard PyTorch dataloader, with a change of defaulting the collate function to use
        the jagged sequence-support collate function we have internal to the library
        """
        kwargs['collate_fn'] = collate_fn
        super().__init__(*args, **kwargs)

    @property
    def num_features(self) -> int:
        if isinstance(self.dataset, ProvidenceDataset):
            return len(self.dataset.feature_columns)
        # assume we can count the features, as the last dimension of the data
        return count_providence_ds_features(self.dataset)

    # copy the parent documentation
    __doc__ = (__doc__ or "") + "\nParent documentation:\n" + TorchDataLoader.__doc__


def count_providence_ds_features(ds: TorchDataset) -> int:
    """Takes a `ds` and returns the number of feature columns a given "row" has"""
    return ds[0][0].size(-1)


def BasicDataloaders(train_ds: ProvidenceDataset, val_ds: ProvidenceDataset, batch_size: int) -> DataLoaders:
    "Provides ProvidenceDataloaders that will be used in the training and validation passes, respectively"
    return DataLoaders(
        train=ProvidenceDataLoader(train_ds, shuffle=True, batch_size=batch_size),
        validation=ProvidenceDataLoader(val_ds, batch_size=batch_size),
    )


def CustomProvidenceDataloaders(
    train_ds: ProvidenceDataset, val_ds: ProvidenceDataset, *, batch_size: int, **kwargs
) -> DataLoaders:
    "Provides ProvidenceDataloaders that will be used in the training and validation passes, respectively"
    return DataLoaders(
        train=ProvidenceDataLoader(train_ds, shuffle=True, batch_size=batch_size, **kwargs),
        validation=ProvidenceDataLoader(val_ds, batch_size=batch_size, **kwargs)
    )


def BackblazeDataLoaders(
    *,
    quarter: BackblazeQuarter,
    include_validation: bool = False,
    split_percentage: float = 0.8,
    batch_size: int,
    censoring_proportion: int = 1,
    data_root: str = "./.data",
    random_seed: int = 1234
) -> DataLoaders:
    """
    Constructs a DataLoaders of `batch_size` for the `ProvidenceDataset`s built on Backblaze data in the given `quarter`
    NOTE: A version of this function may move to paper_reproductions.py
    """
    datasets = BackblazeDatasets(
        quarter=quarter,
        include_validation=include_validation,
        split_percentage=split_percentage,
        censoring_proportion=censoring_proportion,
        data_root=data_root,
        random_seed=random_seed
    )

    return DataLoaders(*[ProvidenceDataLoader(ds, batch_size=batch_size) for ds in datasets])


def BackblazeExtendedDataLoaders(
    *,
    quarters: BackblazeQuarter = BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER,
    include_validation: bool = False,
    split_percentage: float = 0.8,
    batch_size: int,
    censoring_proportion: int = 1,
    data_root: str = "./.data",
    random_seed: int = 1234
) -> DataLoaders:
    """
    Constructs a DataLoaders of `batch_size` for the `ProvidenceDataset`s built on BackblazeExtended data in the given `quarters`h
    NOTE: A version of this function may move to paper_reproductions.py
    """
    datasets = BackblazeExtendedDatasets(
        *quarters,
        include_validation=include_validation,
        split_percentage=split_percentage,
        censoring_proportion=censoring_proportion,
        data_root=data_root,
        random_seed=random_seed
    )

    return DataLoaders(*[ProvidenceDataLoader(ds, batch_size=batch_size) for ds in datasets])


def NasaFD00XDataLoaders(turbofan_test: NasaTurbofanTest, *, batch_size: int, data_root: str = './.data'):
    train_ds, test_ds = NasaFD00XDatasets(turbofan_test, data_root=data_root)
    train_dl = ProvidenceDataLoader(train_ds, batch_size=batch_size)
    test_dl = ProvidenceDataLoader(test_ds, batch_size=batch_size)
    return DataLoaders(train_dl, test_dl)


def NasaDataLoaders(*, batch_size: int, data_root: str = './.data'):
    train_ds, test_ds = NasaDatasets(data_root=data_root)
    train_dl = ProvidenceDataLoader(train_ds, batch_size=batch_size)
    test_dl = ProvidenceDataLoader(test_ds, batch_size=batch_size)
    return DataLoaders(train_dl, test_dl)
