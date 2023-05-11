"""
Utility functions that support ProvidenceDataset construction and everything leading up to it.
Normalization, (opinionated) train-val-test splitting, and more wil be available to you

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Any  # noqa: F401
from typing import Callable  # noqa: F401
from typing import List
from typing import Tuple
from typing import Union

import torch as pt
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.utils.data import random_split

from providence.utils import validate

from .core import DataFrameSplit
from .core import ProvidenceDataset


################################################################################
#
# Dataset subsetting / splitting
#
################################################################################


# don't bother generalizing. There are two cases and the code would become unnecessarily opaque
def train_test_split_sizes(length: int, split_percentage: float) -> Tuple[int, int]:
    """Split ``ds`` into train-, val- and test portion by its own item semantics.

    If you wanted to do a train-test split at 80% of the data, you might code
    >>> train_portion = 0.8
    >>> test_portion = 0.2  # 1 - 0.8

    You can instead call this function
    >>> train_portion, test_portion = train_test_split_sizes(ds_length, 0.8)

    NOTE: this doesn't work well for 1e-1 * [1, 3, 5, 7, 9]. This has to do with float-point representation of these
    values.

    Args:
        length (int): non-negative. ``length < 0`` is undefined
        split_percentage (float): value in range [0.0, 1.0]. ``split_percentage > 1`` produces an effective no-op
            (train_size >= length, test_size == 0), while negative numbers will beget a negative train_size and
            test_size great than 1.

    Returns:
        Tuple[int, int]: train_size, test_size
    """
    train_portion = split_percentage
    test_portion = 1 - split_percentage
    train_size, test_size = round(length * train_portion), round(length * test_portion)
    if (delta := length - (train_size + test_size)) != 0:
        train_size += delta
    return train_size, test_size


def train_test_split(ds: Dataset, split_percentage: float = 0.8, *, seed=42):
    """Split ``ds`` into train-, val- and test portion by its own item semantics.

    Produces a train-val-test split, where a train-test split is first taken based on ``split_percentage`` and a second
    train-test (i.e. val-test) split is taken on the previous "test" dataset.
    See ``train_val_test_split_sizes(...)`` for additoinal details.

    Args:
        ds (Dataset): dataset to split into the portions based on ``split_percentage``.
        split_percentage (float): value in range [0.0, 1.0], producing a training portion of the data is
            ``split_percentage`` and the testing portion is ``1.00 - split_percentage`` percentage of the devices.
            Defaults to ``0.``.
        seed (int, optional): Set the random seed for a reproducible result. Defaults to ``42``.

    Returns:
        List[pt.utils.data.Subset]: where the elements are the train and test subset of ``ds``
    """
    sizes = train_test_split_sizes(len(ds), split_percentage)
    return random_split(ds, sizes, generator=pt.Generator().manual_seed(seed))


def train_val_test_split_sizes(length: int, split_percentage: float) -> Tuple[int, int, int]:
    """Calculate the sizes for a train-val-test split.

    If you wanted e.g. 80% splits, you could compute the following
    >>> train_portion = 0.64  # 0.8 ^ 2
    >>> val_portion = 0.16  # 0.8 * (1 - 0.8)
    >>> test_portion = 0.2  # 1 - 0.8

    With this function, this is reduced to one call
    >>> train_portion, val_portion, test_portion = train_val_test_split_sizes(ds_length, 0.8)

    We realize this is not the only logic by which one may want to do a train-val-test split. It is merely preference.

    Args:
        length (int): non-negative. ``length < 0`` is undefined
        split_percentage (float): value in range [0.0, 1.0]. ``split_percentage > 1`` produces an effective no-op
            (train_size == length, val_size == test_size == 0), while negative numbers will beget overlapping sizes.

    Returns:
        Tuple[int, int, int]: sizes of the (train, validation, test) sets, based on ``length``
    """
    test_portion = 1 - split_percentage
    train_portion = split_percentage**2
    val_portion = split_percentage - train_portion
    train_size, val_size, test_size = (
        round(length * train_portion),
        round(length * val_portion),
        round(length * test_portion),
    )

    if (delta := length - (train_size + val_size + test_size)) != 0:
        # if we over-rounded, give the extra to the validation set (which is the smallest of the 3 datasets)
        # if we've undercut, take the extra from the validation set because adding a negative is subtraction.
        val_size += delta
    return train_size, val_size, test_size


def train_val_test_split(ds: Dataset, split_percentage=0.8, *, seed=42):
    """Split ``ds`` into train-, val- and test portion by its own item semantics.

    Produces a train-val-test split, where a train-test split is first taken based on ``split_percentage`` and a second
    train-test (i.e. val-test) split is taken on the previous "test" dataset.

    Example:
    >>> train_, val_, test_ = train_val_test_split(ds, 0.5)
    >>> train, pseudo_test = train_test_split(ds, 0.5)
    >>> val, test = train_test_split(pseudo_test, 0.5)
    >>> assert all([
            train_ == train,
            val_ == val,
            test_ = test
        ])

    Args:
        ds (Dataset): dataset to split into the portions based on ``split_percentage``.
        split_percentage (float): value in range [0.0, 1.0], producing a training portion of the data is
            ``split_percentage`` and the testing portion is ``1.00 - split_percentage`` percentage of the entities.
            The testing portion is then similarly split into a validation and testing portion.
            See ``train_val_test_split_sizes(...)`` for additoinal details. Defaults to ``0.8``.
        seed (int, optional): Set the random seed for a reproducible result. Defaults to ``42``.

    Returns:
        List[pt.utils.data.Subset]: where the elements are the train, val, and test subset of ``ds``.
    """
    return random_split(
        ds,
        train_val_test_split_sizes(len(ds), split_percentage),
        generator=pt.Generator().manual_seed(seed),
    )


def downsample_to_event_portion(ds: ProvidenceDataset, portion: float):
    """Downsample, in-place ``ds`, reducing the number of event entities present.

    We want to reduce the number of eventful entities in the ``ds``, and do so by deletion for simplicity.
    As this mutates in place, for e.g. comparative analysis, we recommend using this in conjunction with the cached
    loading methods in this module.

    Args:
        ds (ProvidenceDataset): an instance to remove entities from.
        portion (float): the portion of events to keep. ``portion < 0`` functions as though ``portion == 0``.
            ``portion > 1`` results in a no-op
    """

    # count the number of events in the dataset
    n_events = sum((df[ds.event_indicator_column].sum() > 0) for _, df in ds.data)

    # n_to_remove := "number of events" times "1 - the portion"
    portion_to_remove = round(1 - portion, 2)
    n_to_remove = n_to_remove_initial = round(n_events * portion_to_remove)

    # run through the `data` field and remove the last `n_to_remove` eventful devices
    cursor = len(ds) - 1
    backing_data = ds.data
    while cursor > -1 and n_to_remove > 0:
        if is_event_df := ((item_df := backing_data[cursor][1])[ds.event_indicator_column].sum() > 0):
            del backing_data[cursor]
            n_to_remove -= 1
        cursor -= 1
    # Because we're mutating data in-place, no further assignment is needed
    if n_to_remove:
        print(f"Had more devices to remove than present in the dataset {n_to_remove_initial > len(ds) = }")


################################################################################
#
# DataFrame manipulations
#
################################################################################


def censored_subset(
    df: DataFrame,
    censoring_proportion: Union[int, float],
    *,
    entity_id="serial_number",
    event_indicator_column="failure",
) -> DataFrame:
    """Downsample ``df`` such that eventful entities are proportional to ``censoring_proportion``.

    Take a subset of ``df`` including all uncensored entities + ``censoring_proportion`` of censored entities.
    i.e. beginning from eventful entities, we take ``censoring_proportion * n_eventful``-many entities (rounded) from
    the censored entities to produce the aggregate of censored and uncensored entities.
    This function was motivated by a censoring experiment we were conducting with the Backblaze dataset and the default
    match correspondingly.

    Args:
        df (DataFrame): the dataset to be portioned. Must include both ``entity_id`` and ``event_indicator_column``
            in ``df.columns``
        censoring_proportion (Union[int, float]): Can be greater than 1.0.
        entity_id (str, optional): column name identifying the entity, e.g. "device_id", "eng_num", "serial_number"
            Defaults to "serial_number".
        event_indicator_column (str, optional): column of ``df`` representing event, interpreted as either
            1. an event column, where there is a ``1`` at the time step when the event occurs in said column, or
            2. a "censor indicator column", where there is a ``1`` for the entire column.
            Other formats produce undefined behavior.
            Defaults to "failure".

    Returns:
        DataFrame:
    """
    # count the number of uncensored entities in train
    print(f"df.columns are {df.columns.tolist()}")
    failure_serials = df[df[event_indicator_column] == 1][entity_id].unique()
    # we don't use "df['failure'] == 0" because that could catch entities that have a failure in one part of the df
    is_censored_mask = ~df[entity_id].isin(failure_serials)
    censored_serials = df[is_censored_mask][entity_id].unique()  # this is a deterministic unique per the documentation
    # multiply that by censoring proportion
    n_censored_ids = round(len(failure_serials) * censoring_proportion)
    # select that many censored entities
    censored_serials = censored_serials[:n_censored_ids]
    # *THAT* is the training set
    subset_serials = set(failure_serials.tolist() + censored_serials.tolist())
    print("number of serials in joint censored-uncensored dataset =", len(subset_serials))
    return df[df[entity_id].isin(subset_serials)]


def df_train_test_split(
    df: DataFrame,
    entity_identifier: str,
    *,
    split_percentage: float = 0.8,
    seed: int = 42,
) -> DataFrameSplit:
    """Split ``df`` into train- and test portion by ``entity_indentifier``.

    This step in the preprocessing pipeline might concern itself with swapping devices around to achieve
    a more representation of censored or uncensored devices in the test set.

    Args:
        df (DataFrame): the dataset to be split
        entity_identifier (str): column name identifying the entity, e.g. "device_id", "serial_number"
        split_percentage (float): value in range [0.0, 1.0], producing a training portion of the data is
            ``split_percentage`` and the testing portion is ``1.00 - split_percentage`` percentage of the devices.
            Defaults to ``0.8``.
        seed (int): random seed to be set for the train-test split.

    Returns:
        DataFrameSplit(train=train_portion(df), test=test_portion(df), validation=None)
    """
    validate(0 <= split_percentage <= 1, f"Invalid input: {split_percentage=} is not a percentage")
    validate(entity_identifier in df, "Entity identifier is not a listed column in the supplied DataFrame")

    # pandas.Series.unique() returns a numpy array, so it's np.ndarray -> Python List -> Python Set
    ids = df[entity_identifier].unique()

    train_id_ds, test_id_ds = train_test_split(ids, split_percentage, seed=seed)

    train_ids = set(train_id_ds)  # iterate the returned torch...Subset with the iterator protocol
    test_ids = set(test_id_ds)
    test_ids_ = set(ids) - train_ids
    assert test_ids_ == test_ids, "Leveraging PyTorch for reproducible train-test split failed"

    is_in_train_ids = df[entity_identifier].isin(train_ids)
    is_in_test_ids = df[entity_identifier].isin(test_ids)
    return DataFrameSplit(train=df[is_in_train_ids], test=df[is_in_test_ids])


def df_train_val_test_split(
    df: DataFrame,
    entity_identifier: str,
    *,
    split_percentage: float = 0.8,
    seed: int = 42,
) -> DataFrameSplit:
    """Split ``df`` into train-, val- and test portion by ``entity_indentifier``.

    Args:
        df (DataFrame): the dataset to be split
        entity_identifier (str): column name identifying the entity, e.g. "device_id", "serial_number"
        split_percentage (float): value in range [0.0, 1.0], producing a training portion of the data is
            ``split_percentage`` and the testing portion is ``1.00 - split_percentage`` percentage of the devices.
            Defaults to ``0.8``.
        seed (int): random seed to be set for the train-test split.

    Returns:
        DataFrameSplit(train=train_portion(df), test=test_portion(df), validation=validation_portion)
    """
    validate(0 <= split_percentage <= 1, f"Invalid input: {split_percentage=} is not a percentage")
    validate(entity_identifier in df, "Entity identifier is not a listed column in the supplied DataFrame")

    ids = df[entity_identifier].unique()

    ids_as_subsets = train_val_test_split(ids, split_percentage, seed=seed)
    train_ids, val_ids, test_ids = [set(id_subset) for id_subset in ids_as_subsets]

    val_ids_ = set(ids) - train_ids - test_ids
    assert val_ids_ == val_ids, "Leveraging PyTorch for reproducible train-test split failed"
    test_ids_ = set(ids) - train_ids - val_ids
    assert test_ids_ == test_ids, "Leveraging PyTorch for reproducible train-test split failed"

    is_in_train_ids = df[entity_identifier].isin(train_ids)
    is_in_val_ids = df[entity_identifier].isin(val_ids)
    is_in_test_ids = df[entity_identifier].isin(test_ids)
    return DataFrameSplit(train=df[is_in_train_ids], test=df[is_in_test_ids], validation=df[is_in_val_ids])


def extract_normalization_stats(df: DataFrame, *, method: str = "min_max", feature_names: List[str]) -> DataFrame:
    """Extract the necessary (summary) statistics from ``df`` - based on ``method`` - for a down-stream normalization.

    Args:
        df (DataFrame): data to normalize
        method (str): 'min_max' or 'standardize', the method for normalization
        feature_names (List[str]): feature column names in the DataFrame ``df``

    Returns:
        DataFrame columns ['min', 'max'] or ['mean', 'std'] and rows for each feature
        i.e. shape == [len(feature_names), 2]

    Raises:
        ValueError: If the supplied ``method`` of normalization is not 'standardize' or 'min_max'.
    """
    if method == "min_max":
        aggregations = ["min", "max"]  # type: List[Union[Callable[..., Any], str]]
    elif method == "standardize":
        aggregations = ["mean", "std"]
    else:
        raise ValueError(f"{method = } is invalid for per-device normalization")

    return df[feature_names].agg(aggregations).T  # so callers can use result['min'] in the calculations


def normalize_with_stats(df: DataFrame, stats: DataFrame, *, method="min_max", feature_names: List[str]) -> DataFrame:
    """Normalize `df` with given `stats`, expected stats to correspond to `feature_names` 1:1 in order.

    Args:
        df (DataFrame): data to normalize
        method (str, optional): 'min_max' or 'standardize', the method for normalization. Defauts to "min_max".
        feature_names (List[str]): feature column names in the DataFrame `df`

    Returns:
        A new DataFrame containing the normalized data in `df`.

    Raises:
        ValueError: If the supplied `method` of normalization is not 'standardize' or 'min_max'.
    """
    if method == "min_max":
        # 0-1 min-max norm
        denominator = (stats["max"].to_numpy() - stats["min"]).to_numpy() + 1e-7
        normed = (df[feature_names] - stats["min"].to_numpy()) / denominator
    elif method == "standardize":
        denominator = stats["std"].to_numpy() + 1e-7
        normed = (df[feature_names] - stats["mean"].to_numpy()) / denominator
    else:
        raise ValueError(f"{method = } is invalid for prescriptive normalization")

    metaddata_dropped = list(set(df.columns.tolist()) - set(normed.columns.tolist()))
    normed[metaddata_dropped] = df[metaddata_dropped]

    return normed


def normalize_by_device(
    df: DataFrame, *, method="min_max", entity_id: str = "serial_number", feature_names: List[str]
) -> DataFrame:
    """Normalize the `df` once grouped by `entity_id`, by `method` using features indicated by `feature_names`.

    Normalizing within the statistics of a given entity, rather than a providing statistics to normalize the
    population in `df`.

    Args:
        df (DataFrame): data to normalize
        method (str): 'min_max' or 'standardize', the method for normalization
        entity_id (str): the column identifying the entity, which will be normalized by its own statistics
        feature_names (List[bool]): feature column names in the DataFrame `df`

    Returns:
        A new DataFrame containing the normalized data in `df`.

    Raises:
        ValueError: If the supplied `method` of normalization is not 'standardize' or 'min_max'.
    """
    if method == "min_max":
        do_norm = lambda sub: (sub - sub.min()) / (sub.max() - sub.min() + 1e-6)
    elif method == "standardize":
        # everything should have variance, but just incase vvvvvv add epsilon
        do_norm = lambda sub: (sub - sub.mean()) / (sub.std() + 1e-6)
    else:
        raise ValueError(f"{method = } is invalid for per-device normalization")
    normed = df.groupby(entity_id)[feature_names].transform(do_norm)

    metaddata_dropped = list(set(df.columns.tolist()) - set(normed.columns.tolist()))
    normed[metaddata_dropped] = df[metaddata_dropped]

    return normed
