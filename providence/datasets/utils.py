"""
Utility functions that support ProvidenceDataset construction and everything leading up to it.
Normalization, (opinionated) train-val-test splitting, and more wil be available to you

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import List, Tuple, Union

import torch as pt
from pandas import DataFrame
from torch.utils.data import Dataset, random_split

from .core import DataFrameSplit, ProvidenceDataset




################################################################################
#
# Dataset subsetting / splitting
#
################################################################################


# don't bother generalizing. There are two cases and the code would become unnecessarily opaque
def train_test_split_sizes(length: int, split_percentage: float) -> Tuple[int, int]:
    """
    If you wanted to do a train-test, then a train-val-test on 80% split percentage a la
    >>> train_portion = 0.8
    >>> test_portion = 0.2  # 1 - 0.8

    Instead you'd call
    >>> train_portion, test_portion = train_test_split_sizes(ds_length, 0.8)

    NOTE: this doesn't work well for 1e-1 * [1, 3, 5, 7, 9]. Working on it.
    """
    train_portion = split_percentage
    test_portion = 1 - split_percentage
    train_size, test_size = round(length * train_portion), round(length * test_portion)
    if (delta := length - (train_size + test_size)) != 0:
        train_size += delta
    return train_size, test_size


def train_test_split(ds: Dataset, split_percentage: float = 0.8, *, seed=42):
    "Perform a train-test split with a fixed seed"
    sizes = train_test_split_sizes(len(ds), split_percentage)
    return random_split(ds, sizes, generator=pt.Generator().manual_seed(seed))


def train_val_test_split_sizes(length: int, split_percentage: float) -> Tuple[int, int, int]:
    """
    If you wanted to do a train-test, then a train-val-test on 80% split percentage a la
    >>> train_portion = 0.64  # 0.8 ^ 2
    >>> val_portion = 0.16  # 0.8 * (1 - 0.8)
    >>> test_portion = 0.2  # 1 - 0.8

    Instead you'd call
    >>> train_portion, val_portion, test_portion = train_val_test_split_sizes(ds_length, 0.8)
    """
    test_portion = 1 - split_percentage
    train_portion = split_percentage**2
    val_portion = split_percentage - train_portion
    train_size, val_size, test_size = round(length * train_portion), round(length * val_portion), round(length * test_portion)

    if (delta := length - (train_size + val_size + test_size)) != 0:
        # if we over-rounded, give the extra to the validation set (which is the smallest of the 3 datasets)
        # if we've undercut, take the extra from the validation set because adding a negative is subtraction.
        val_size += delta
    return train_size, val_size, test_size


def train_val_test_split(ds: Dataset, split_percentage=0.8, *, seed=42):
    """
    Produce a train-val-test split, where a train-test split is taken wrt `split_percentage` and
    train-test (i.e. val-test) is taken on the output "test" dataset. Effectively performing the following:
    >>> train, pseudo_test = train_test_split(ds, 0.5)
    >>> val, test = train_test_split(pseudo_test, 0.5)

    See `train_val_test_split_sizes(...)` for details. Set the random seed for a reproducible result.
    """
    return random_split(
        ds, train_val_test_split_sizes(len(ds), split_percentage), generator=pt.Generator().manual_seed(seed)
    )


def downsample_to_event_portion(ds: ProvidenceDataset, portion: float):
    "In-place mutation of the dataset, reducing the number of event devices present"
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
    event_indicator_column="failure"
) -> DataFrame:
    """
    Take a subset of `df` including all uncensored devices + `censoring_portion` of censored devices.
    i.e. we upsample the censored devices relative to the number of uncensored devices.

    This function was motivated by a censoring experiment we were conducting with the Backblaze dataset and the default
    match correspondingly.
    """
    # count the number of uncensored devices in train
    print(f"df.columns are {df.columns.tolist()}", )
    failure_serials = df[df[event_indicator_column] == 1][entity_id].unique()
    # we don't use "df['failure'] == 0" because that could catch devices that have a failure in one quarter and not another
    is_censored_mask = ~df[entity_id].isin(failure_serials)
    censored_serials = df[is_censored_mask][entity_id].unique()  # this is a deterministic unique per the documentation
    # multiply that by censoring proportion
    n_censored_ids = round(len(failure_serials) * censoring_proportion)
    # select that many censored devices
    censored_serials = censored_serials[:n_censored_ids]
    # *THAT* is the training set
    subset_serials = set(failure_serials.tolist() + censored_serials.tolist())
    print("number of serials in joint censored-uncensored dataset =", len(subset_serials))
    return df[df[entity_id].isin(subset_serials)]


def df_train_test_split(
    df: DataFrame, entity_identifier: str, *, split_percentage: float = 0.8, seed: int = 42
) -> DataFrameSplit:
    """
    Performs an id-based (using `entity_identifier`) train-test split,
    where train is `split_percentage` ∈ [0.0, 1.0] of the ids and test is `1 - split_percentage` of the ids.
    
    Another step in the preprocessing pipeline might concern itself with swapping devices around to achieve (say) more representation of
    censored or uncensored devices in the test set.

    Arguments:
        df: dataframe of the dataset to be split
        entity_identifier: specifying column, like 'device_id' or 'personelle_number' or 'serial_number' or 'wallet_address'
        split_percentage: see above. Defaults to `0.8`
    
    Returns: DataFrameSplit as described above.
    """
    assert 0 <= split_percentage <= 1, "Split percentage is not a percentage"
    assert entity_identifier in df, "Entity identifier is not a listed column in the supplied DataFrame"

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
    df: DataFrame, entity_identifier: str, *, split_percentage: float = 0.8, seed: int = 42
) -> DataFrameSplit:
    """
    Performs an id-based (using `entity_identifier`) train-val-test split.
    See `train_val_test()` for semantics and implications of `split_percentage`
    
    Arguments:
        df: dataframe of the dataset to be split
        entity_identifier: specifying column, like 'device_id' or 'personelle_number' or 'serial_number' or 'wallet_address'
        split_percentage: see above. Defaults to `0.8`. 
    
    Returns: DataFrameSplit as described above.
    """
    assert 0 <= split_percentage <= 1, "Split percentage is not a percentage"
    assert entity_identifier in df, "Entity identifier is not a listed column in the supplied DataFrame"

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


def extract_normalization_stats(df: DataFrame, *, method: str = 'min_max', feature_names: List[str]) -> DataFrame:
    """
    Extract the necessary (summary) statistics from `df` for performing some normalization down-stream, based on `method`
    """
    if method == 'min_max':
        aggregations = ['min', 'max']
    elif method == 'standardize':
        aggregations = ['mean', 'std']
    else:
        raise ValueError(f"{method = } is invalid for per-device normalization")

    return df[feature_names].agg(aggregations).T  # so callers can use result['min'] in the calculations


def normalize_with_stats(df: DataFrame, stats: DataFrame, *, method='min_max', feature_names: List[str]) -> DataFrame:
    "Out-of-place normalization of `df`, using the `stats` supplied. It is assumed that len(stats) == len(feature_names)"
    if method == 'min_max':
        # 0-1 min-max norm
        denominator = (stats['max'].to_numpy() - stats['min']).to_numpy() + 1e-7
        normed = (df[feature_names] - stats['min'].to_numpy()) / denominator
    elif method == 'standardize':
        denominator = stats['std'].to_numpy() + 1e-7
        normed = (df[feature_names] - stats['mean'].to_numpy()) / denominator
    else:
        raise ValueError(f"{method = } is invalid for prescriptive normalization")

    metaddata_dropped = list(set(df.columns.tolist()) - set(normed.columns.tolist()))
    normed[metaddata_dropped] = df[metaddata_dropped]

    return normed


def normalize_by_device(df: DataFrame, *, method='min_max', entity_id: str = 'serial_number', feature_names: List[str]):
    "Normalized the `df` once grouped by `entity_id`, by `method` using features indicated by `feature_names`."
    if method == 'min_max':
        do_norm = lambda sub: (sub - sub.min()) / (sub.max() - sub.min() + 1e-6)
    elif method == 'standardize':
        # everything should have variance, but just incase vvvvvv add epsilon
        do_norm = lambda sub: (sub - sub.mean()) / (sub.std() + 1e-6)
    else:
        raise ValueError(f"{method = } is invalid for per-device normalization")
    normed = df.groupby(entity_id)[feature_names].transform(do_norm)

    metaddata_dropped = list(set(df.columns.tolist()) - set(normed.columns.tolist()))
    normed[metaddata_dropped] = df[metaddata_dropped]

    return normed
