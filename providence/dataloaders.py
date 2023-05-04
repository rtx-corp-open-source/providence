"""
Dataloaders and supporting utility functions for loading data in a Providence-compliant format.
See ``.providence_pad_sequence`` for an explanation

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Callable
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from torch import device
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from providence.datasets import BackblazeDatasets
from providence.datasets import NasaDatasets
from providence.datasets import NasaFD00XDatasets
from providence.datasets import ProvidenceDataset
from providence.datasets.adapters import BackblazeQuarter
from providence.datasets.adapters import NasaTurbofanTest
from providence.datasets.backblaze import BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER
from providence.datasets.backblaze import BackblazeExtendedDatasets
from providence.types import DataLoaders

################################################################################
#
# Dataloader Support
#
################################################################################


def is_list_of(maybe_list, type_: Type) -> bool:
    """Check that ``maybe_list`` is a list of homogenous ``type_``.

    Args:
        maybe_list (Any): any python object
        type_ (Type): the type of element desired to be found in the list

    Returns:
        bool: True if ``maybe_list`` is a Sequence or list, and that all elements are of Type ``type_``.
            False otherwise.
    """
    return isinstance(maybe_list, (Sequence, list)) and all(map(lambda x: isinstance(x, type_), maybe_list))


# type of a dataset element: features and targets
# TODO: centralize some of these types. This one is much more general than the dataloaders...
ProvidenceItem = Tuple[Tensor, Tensor]
T = TypeVar("T")


def providence_pad_sequence(
    data: Union[Tensor, Sequence[Sequence[T]], Sequence],
    target_device: device = device("cpu"),
) -> Tuple[Tensor, Tensor]:
    """Apply padding for variable length sequences.

    This function concatenates a list of matrices, representing the multivariate time series of a given entity.
    The result will resemble something akin to the following, where time is the outer-most dimension, subject / entity
    is the second dimension, and the last dimension is the feature dimension:


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

    Args:
        data: fixed-length, index-able of NxM matricies

    Returns:
        Tuple[Tensor, LongTensor]: the padded sequences and the lengths of each entity in the dataset
    """
    lengths = Tensor([len(x) for x in data]).long()

    if is_list_of(data, Tensor):
        padded = pad_sequence(data, batch_first=False)
    else:
        padded = pad_sequence([Tensor(sequence, device=target_device) for sequence in data])

    return padded, lengths


def providence_collate_fn(batch: List[ProvidenceItem]) -> Tuple[Tensor, Tensor, Tensor]:
    """Collate a batch around the temporal / sequence dimension, rather than batch-first.

    We pad to the longest sequence. See ``providence_pad_sequence()`` for more intuition there.

    Args:
        batch (List[ProvidenceItem]): a list of tuples of features and targets

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Padded inputs, the real sequence lengths, and padded targets
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
    """Load data as a standard Pytorch dataloader, defaulting to use ``providence_collate_fn`` support jagged sequences.

    Args:
        collate_fn (Callable[[List[ProvidenceItem]], Tuple[Tensor, ...]], optional): A valid collate function. Defaults to providence_collate_fn.
    """

    def __init__(
        self, *args, collate_fn: Callable[[List[ProvidenceItem]], Tuple[Tensor, ...]] = providence_collate_fn, **kwargs
    ) -> None:
        kwargs["collate_fn"] = collate_fn
        super().__init__(*args, **kwargs)

    @property
    def num_features(self) -> int:
        """The number of features returned by Tensors of this DataLoader.

        Returns:
            int: number of features
        """
        if isinstance(self.dataset, ProvidenceDataset):
            return len(self.dataset.feature_columns)
        # assume we can count the features, as the last dimension of the data
        return count_providence_ds_features(self.dataset)

    # copy the parent documentation
    __doc__ = (__doc__ or "") + "\nParent documentation:\n" + (TorchDataLoader.__doc__ or "")


def count_providence_ds_features(ds: TorchDataset) -> int:
    """Take a ``ds`` and return the number of feature columns a given "row" has.

    Returns:
        int: number of features on a ProvidenceDataset
    """
    return ds[0][0].size(-1)


def BasicDataloaders(train_ds: ProvidenceDataset, val_ds: ProvidenceDataset, batch_size: int) -> DataLoaders:
    """Construct ProvidenceDataloaders that will be used in the training and validation passes, respectively.

    Args:
        train_ds (ProvidenceDataset): training dataset
        val_ds (ProvidenceDataset): validation or testing dataset
        batch_size (int): batch size desired in the contained DataLoader instances

    Returns:
        DataLoaders: containing an initialized ``train`` and ``validation`` field
    """
    return DataLoaders(
        train=ProvidenceDataLoader(train_ds, shuffle=True, batch_size=batch_size),
        validation=ProvidenceDataLoader(val_ds, batch_size=batch_size),
    )


def CustomProvidenceDataloaders(
    train_ds: ProvidenceDataset, val_ds: ProvidenceDataset, *, batch_size: int, **kwargs
) -> DataLoaders:
    """Construct ProvidenceDataloaders that will be used in the training and validation passes, respectively.

    Args:
        train_ds (ProvidenceDataset): training dataset
        val_ds (ProvidenceDataset): validation or testing dataset
        batch_size (int): batch size desired in the contained DataLoader instances

    Returns:
        DataLoaders: containing an initialized ``train`` and ``validation`` field
    """
    return DataLoaders(
        train=ProvidenceDataLoader(train_ds, shuffle=True, batch_size=batch_size, **kwargs),
        validation=ProvidenceDataLoader(val_ds, batch_size=batch_size, **kwargs),
    )


def BackblazeDataLoaders(
    *,
    quarter: BackblazeQuarter,
    include_validation: bool = False,
    split_percentage: float = 0.8,
    batch_size: int,
    censoring_proportion: int = 1,
    data_root: str = "./.data",
    random_seed: int = 1234,
) -> DataLoaders:
    """Construct a DataLoaders of ``batch_size`` for the ``ProvidenceDataset``s built on Backblaze data in the given ``quarter``.

    Args:
        quarter (BackblazeQuarter): quarter of the calendar year from which you want to extract the data
        include_validation (bool, optional): ``True`` if you want the validation and test in the returned DataLoaders.
            ``False`` returns in the test set will be in the ``validation`` field of the result.
            Defaults to False.
        split_percentage (float): real number between 0 and 1, propogated to ``BackblazeDatasets``. Defaults to 0.8.
        batch_size (int): batch size of returned Tensors
        censoring_proportion (float): the number of non-event devices per eventful device.
            This is for a downsampling procedure. See ``censored_subset`` for more.
        data_root (str): the parent directory used for downloading, caching, and retrieving this dataset.
            Multiple child directories will be created, so ensure adequate permissions when you supply a directory.
            Defaults to "./.data".
        random_seed (int, optional): for deterministic splits. Defaults to 1234.

    Returns:
        DataLoaders: containing ``train`` and ``validation`` (based on supplied arguments). ``test`` is only set if
            ``include_validation=True``
    """
    datasets = BackblazeDatasets(
        quarter=quarter,
        include_validation=include_validation,
        split_percentage=split_percentage,
        censoring_proportion=censoring_proportion,
        data_root=data_root,
        random_seed=random_seed,
    )

    return DataLoaders(*[ProvidenceDataLoader(ds, batch_size=batch_size) for ds in datasets])


def BackblazeExtendedDataLoaders(
    *,
    quarters: list = BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER,
    include_validation: bool = False,
    split_percentage: float = 0.8,
    batch_size: int,
    censoring_proportion: int = 1,
    data_root: str = "./.data",
    random_seed: int = 1234,
) -> DataLoaders:
    """Construct a DataLoaders of ``batch_size`` for the ``ProvidenceDataset``s built on BackblazeExtended data.

    NOTE: A version of this function may move to paper_reproductions.py

    Args:
        quarters (BackblazeQuarter, optional): quarter of the calendar year from which you want to extract the data.
            Quarters are sorted before processing, to each cross-quarter timeseries concatenation.
            Defaults to ``BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER``.
        include_validation (bool, optional): ``True`` if you want the validation and test in the returned DataLoaders.
            ``False`` returns in the test set will be in the ``validation`` field of the result.
            Defaults to False.
        split_percentage (float): real number between 0 and 1, propogated to ``BackblazeDatasets``. Defaults to 0.8.
        batch_size (int): batch size of returned Tensors
        censoring_proportion (float): the number of non-event devices per eventful device.
            This is for a downsampling procedure. See ``censored_subset`` for more.
        data_root (str): the parent directory used for downloading, caching, and retrieving this dataset.
            Multiple child directories will be created, so ensure adequate permissions when you supply a directory.
            Defaults to "./.data".
        random_seed (int, optional): for deterministic splits. Defaults to 1234.

    Returns:
        DataLoaders: containing ``train`` and ``validation`` (based on supplied arguments). ``test`` is only set if
            ``include_validation=True`` and is set with the test dataset."""
    datasets = BackblazeExtendedDatasets(
        *quarters,
        include_validation=include_validation,
        split_percentage=split_percentage,
        censoring_proportion=censoring_proportion,
        data_root=data_root,
        random_seed=random_seed,
    )

    return DataLoaders(*[ProvidenceDataLoader(ds, batch_size=batch_size) for ds in datasets])


def NasaFD00XDataLoaders(turbofan_test: NasaTurbofanTest, *, batch_size: int, data_root: str = "./.data"):
    """Construct a ``DataLoaders`` of ``batch_size`` on the FD00X datasets for ``turbofan_test``.

    Args:
        turbofan_test_num (NasaTurbofanTest): the NASA turbofan test runs to be concatenated together to make a dataset
        batch_size (int): batch size of the returned batches.
        data_root (str, optional): the parent directory used for downloading, caching, and retrieving this dataset.
            Multiple child directories will be created, so ensure adequate permissions when you supply a directory.
            Defaults to "./.data".

    Returns:
        DataLoaders: ``train`` set to the training dataloader, ``validation`` set to the test dataloader
    """
    train_ds, test_ds = NasaFD00XDatasets(turbofan_test, data_root=data_root)
    train_dl = ProvidenceDataLoader(train_ds, batch_size=batch_size)
    test_dl = ProvidenceDataLoader(test_ds, batch_size=batch_size)
    return DataLoaders(train_dl, test_dl)


def NasaDataLoaders(*, batch_size: int, data_root: str = "./.data"):
    """Construct a ``DataLoaders`` of ``batch_size`` for the full, aggregate NASA dataset.

    Args:
        batch_size (int): batch size of the returned batches.
        data_root (str, optional): the parent directory used for downloading, caching, and retrieving this dataset.
            Multiple child directories will be created, so ensure adequate permissions when you supply a directory.
            Defaults to "./.data".

    Returns:
        DataLoaders: ``train`` set to the training dataloader, ``validation`` set to the test dataloader
    """
    train_ds, test_ds = NasaDatasets(data_root=data_root)
    train_dl = ProvidenceDataLoader(train_ds, batch_size=batch_size)
    test_dl = ProvidenceDataLoader(test_ds, batch_size=batch_size)
    return DataLoaders(train_dl, test_dl)
