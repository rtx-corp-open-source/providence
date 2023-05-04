"""
Functionality specific to the BackblazeDataset. A key aspect of the redesign was to get away from unnecessary classes and to generalize
the things that were truly general. With an equivalent amount of code, we have more configurability, more functionality, and more clarity
as to what makes the Backblaze dataset (object).
Function-level documentation should clarify any ambiguities

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from operator import attrgetter
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

import numpy as np
from pandas import concat as concat_dataframes
from pandas import DataFrame
from pandas import read_csv
from pandas import Series
from typing_extensions import Literal

from providence.datasets.adapters import assemble_trainable_by_entity_id
from providence.datasets.adapters import BACKBLAZE_FEATURE_NAMES
from providence.datasets.adapters import BackblazeQuarter
from providence.datasets.adapters import load_backblaze_csv
from providence.datasets.core import DataFrameSplit
from providence.datasets.core import DataSubsetId
from providence.datasets.core import ProvidenceDataset
from providence.datasets.utils import censored_subset
from providence.datasets.utils import df_train_test_split
from providence.datasets.utils import df_train_val_test_split
from providence.datasets.utils import extract_normalization_stats
from providence.datasets.utils import normalize_by_device
from providence.datasets.utils import normalize_with_stats
from providence.utils import cached_dataframe
from providence.utils import validate

################################################################################
#
# Backblaze Dataset goodies
#
################################################################################
_NormalizationMode = Literal["device", "fleet"]


def BackblazeDataset(
    subset_choice: DataSubsetId,
    *,
    quarter: BackblazeQuarter = BackblazeQuarter._2019_Q4,
    censoring_proportion: int = 1,
    normalization_by: _NormalizationMode = "device",
    train_percentage: float = 0.8,
    consider_validation: bool = False,
    data_dir: str = "./.data",
    random_seed: int = 42,
):
    """Construct the ``subset`` of Backblaze dataset for the given ``quarter``

    API was designed with this usage in mind
    >>> bb_paper_train_ds = BackblazeDataset(subset_choice=DataSubsetId.Train)
    >>> bb_paper_test_ds = BackblazeDataset(subset_choice=DataSubsetId.Test)
    >>> bb_paper_val_ds = BackblazeDataset(subset_choice=DataSubsetId.Validation)
    >>> bb_val_ds_2019_Q1 = BackblazeDataset(subset_choice=DataSubsetId.Validation, quarter=BackblazeQuarter._2019_Q1)

    So the user of the API can plug-and-play with the complexity that they want to deal with.
    It is slower to use this methods rather one which instantiates two datasets at once.

    Args:
        subset_choice (DataSubsetId): whether, you want the train, validation or test portion of the dataset
        quarter (BackblazeQuarter)   : quarter of the calendar year from which you want to extract the data
        censoring_proportion (float): the number of non-event devices per eventful device.
            This is for a downsampling procedure
        train_percentage (float)     : real number between 0 and 1, passed to ``df_train_test_split`` or
            ``df_train_val_test_split`` depending on `consider_validation`
        consideration_validation (bool): whether to consider the validation set when taking the train or test seth
            (implied True when subset_choice=Validation)
        data_dir (str): the parent directory used for downloading, caching, and retrieving this dataset. Multiple child
            directories will be created, so ensure adequate permissions when you supply a directory.
        random_seed: for deterministic splits

    Returns:
        A Providence Dataset filled with the Backblaze data
    """

    def create_trainable() -> DataFrame:
        # load a subset of the censored cases based on censoring proportion
        bb_df_full: DataFrame = load_cached_backblaze_quarter(quarter, data_root=data_dir)
        use_validation = consider_validation or subset_choice == DataSubsetId.Validation

        split_func = df_train_val_test_split if use_validation else df_train_test_split
        bb_df_split = split_func(
            bb_df_full,
            "serial_number",
            split_percentage=train_percentage,
            seed=random_seed,
        )

        desired_sub_df = {
            DataSubsetId.Train: bb_df_split.train,
            DataSubsetId.Test: bb_df_split.test,
            DataSubsetId.Validation: bb_df_split.validation,
        }[subset_choice]

        assert desired_sub_df is not None
        desired_sub_df = censored_subset(desired_sub_df, censoring_proportion)

        # only normalize the raw_features
        raw_features = [n for n in BACKBLAZE_FEATURE_NAMES if n.endswith("_raw")]
        if normalization_by == "fleet":
            # here, we're always normalizing by the training set.
            norm_stats = extract_normalization_stats(bb_df_split.train, method="min_max", feature_names=raw_features)

            normalized_sub_df = normalize_with_stats(
                desired_sub_df, norm_stats, method="min_max", feature_names=raw_features
            )

        else:  # normalize by device, for a dataset you'd treat independently
            # NOTE: normalizing by device can resolve some nan-filling issues. If this is a problem with your preprocessing pipeline
            # please address your NaNs before passing to this function
            normalized_sub_df = normalize_by_device(desired_sub_df, feature_names=raw_features)

        # apply the tte column
        return assemble_trainable_by_entity_id(
            normalized_sub_df,
            entity_id="serial_number",
            temporality_indicator="date",
            event_occurence_column="failure",
        )

    cache_name = f"backblaze-{subset_choice.name}-{quarter.name}-censored:{censoring_proportion}.csv"
    trainable_sub_df = cached_dataframe(create_trainable, Path(data_dir, "backblaze", cache_name))

    as_dataset = _BackblazeProvidenceDataset(trainable_sub_df)
    return as_dataset


def BackblazePreprocessing(df_splits: DataFrameSplit):
    """Apply the Backblaze-specific preprocessing to the DataFrames contained in `df_splits`

    Args:
        df_splits (DataFrameSplit): a grouping of train, validation (optionally None), and test dataframes
            with the Backblaze dataset schema

    Returns:
        DataFrameSplit that is (in pseudo-code) df_splits.map(backblaze_preprocessing),
            where train, vaidation (if not None), and test are processed for Backblaze training.

    Raises:
        AnyError: If anything bad happens
    """
    # normalize and extract stats on the training set
    norm_stats = extract_normalization_stats(df_splits.train, method="min_max", feature_names=BACKBLAZE_FEATURE_NAMES)
    df_splits = DataFrameSplit(
        # normalize per device in the training set,
        train=normalize_by_device(
            assemble_trainable_by_entity_id(
                df_splits.train,
                entity_id="serial_number",
                temporality_indicator="date",
                event_occurence_column="failure",
            ),
            method="min_max",
            feature_names=BACKBLAZE_FEATURE_NAMES,
        ),
        validation=(
            None
            if df_splits.validation is None
            else normalize_by_device(
                assemble_trainable_by_entity_id(
                    df_splits.validation,
                    entity_id="serial_number",
                    temporality_indicator="date",
                    event_occurence_column="failure",
                ),
                method="min_max",
                feature_names=BACKBLAZE_FEATURE_NAMES,
            )
        ),
        # use aggregate population statistics to normalize the test set, which we would get at production time
        # normalize by applying stats on the test set
        test=normalize_with_stats(
            assemble_trainable_by_entity_id(
                df_splits.test,
                entity_id="serial_number",
                temporality_indicator="date",
                event_occurence_column="failure",
            ),
            norm_stats,
            method="min_max",
            feature_names=BACKBLAZE_FEATURE_NAMES,
        ),
    )

    return df_splits


def BackblazeDatasets(
    *,
    quarter: BackblazeQuarter,
    include_validation: bool = False,
    split_percentage: float = 0.8,
    censoring_proportion: int = 1,
    data_root: str = "./.data",
    random_seed: int = 1234,
) -> Union[
    Tuple[ProvidenceDataset, ProvidenceDataset, ProvidenceDataset],
    Tuple[ProvidenceDataset, ProvidenceDataset],
    Tuple[Any, ...],  # supplied for mypy type check
]:
    """Construct the Backblaze Train, (optional Validation) and Test datasets.

    Because there is more to configure with the Backblaze dataset, we require the user specify these parameters for
    1. clear documentation at the site of experimentation
    2. not prescribing inappropriate defaults, and thus facilitating undesirable behavior for the user.

    Args:
        quarter (BackblazeQuarter): quarter of the calendar year from which you want to extract the data
        censoring_proportion (float): the number of non-event devices per eventful device.
            This is for a downsampling procedure
        split_percentage (float): real number between 0 and 1, passed to ``df_train_test_split`` or
            ``df_train_val_test_split`` depending on `consider_validation`
        consideration_validation (bool): whether to consider the validation set when taking the train or test seth
            (implied True when subset_choice=Validation)
        data_root (str): the parent directory used for downloading, caching, and retrieving this dataset. Multiple child
            directories will be created, so ensure adequate permissions when you supply a directory.
        random_seed: for deterministic splits

    Returns:
    - 'Train, Validation, Test' if ``include_validation`` such as
        >>> train_ds, val_ds, test_ds = BackblazeDatasets(quarter=BackblazeQuarter._2019_Q4, include_validation=True, split_percentage=0.8)
    - 'Train, Test' if otherwise i.e.
        >>> train_ds, test_ds = BackblazeDatasets(quarter=BackblazeQuarter._2019_Q4, include_validation=False, split_percentage=0.8)
    """
    df_split = load_backblaze_splits(
        quarter=quarter,
        include_validation=include_validation,
        split_percentage=split_percentage,
        censoring_proportion=censoring_proportion,
        data_root=data_root,
        random_seed=random_seed,
    )

    print(
        f"Backblaze df_split lengths: train={len(df_split.train)}, test={len(df_split.test)},"
        f" validation={None if df_split.validation is None else len(df_split.validation)}"
    )

    df_split = BackblazePreprocessing(df_split)
    dss = BackblazeDatasets_from_split(df_split)
    return dss


def load_backblaze_splits(
    *,
    quarter: BackblazeQuarter,
    include_validation: bool,
    split_percentage: float,
    censoring_proportion: int = 1,
    data_root: str = "./.data",
    random_seed: int = 1234,
) -> DataFrameSplit:
    """Load the backblaze CSV for the ``quarter``, split based on ``split_percentage`` and ``include_validation``

    Additionally, censor with ``censoring_proportion``.

    Args:
        quarter (BackblazeQuarter): quarter of the calendar year from which you want to extract the data
        censoring_proportion (float): the number of non-event devices per eventful device.
            This is for a downsampling procedure
        split_percentage (float): real number between 0 and 1, passed to ``df_train_test_split`` or
            ``df_train_val_test_split`` depending on `consider_validation`
        include_validation (bool): whether to include the validation set, following the split procedure
        data_root (str): the parent directory used for downloading, caching, and retrieving this dataset. Multiple child
            directories will be created, so ensure adequate permissions when you supply a directory.
        random_seed: for deterministic splits

    Returns:
        DataFrameSplit that is (in pseudo-code) df_splits.map(backblaze_preprocessing),
            where train, vaidation (if not None), and test are processed for Backblaze training.
    """
    validate(0 < split_percentage < 1, f"Invalid {split_percentage=} provided. Must be within the open interval (0, 1)")
    df_full: DataFrame = load_cached_backblaze_quarter(quarter, data_root)

    split_func = df_train_val_test_split if include_validation else df_train_test_split
    df_split = split_func(
        df_full,
        entity_identifier="serial_number",
        split_percentage=split_percentage,
        seed=random_seed,
    )
    df_split = censor_backblaze_splits(df_split, censoring_proportion)
    return df_split


def censor_backblaze_splits(df_split: DataFrameSplit, censoring_proportion: Union[float, int]) -> DataFrameSplit:
    """Downsample ``df_split``, according to the ``censoring_proportion``, creating a 1-to-``censoring_proportion`` DataFrame

    Args:
        censoring_proportion (float): the right side of the proportion `uncensored:censored`, such that there is a
            correspondence between uncensored/eventful entities and the censored

    Returns:
        DataFrameSplit, where each DataFrame is downsampled if it exists.
    """
    return DataFrameSplit(
        train=censored_subset(df_split.train, censoring_proportion),
        test=censored_subset(df_split.test, censoring_proportion),
        validation=(
            None if df_split.validation is None else censored_subset(df_split.validation, censoring_proportion)
        ),
    )


def BackblazeDatasets_from_split(
    df_split: DataFrameSplit,
) -> Union[
    Tuple[ProvidenceDataset, ProvidenceDataset, ProvidenceDataset],
    Tuple[ProvidenceDataset, ProvidenceDataset],
    Tuple[Any, ...],  # supplied for mypy type check
]:
    """Construct a ProvidenceDataset for each dataframe in `df_split`, expecting Backblaze data set properties.

    Args:
        df_splits (DataFrameSplit): a grouping of train, validation (optionally None), and test dataframes
            with the Backblaze dataset schema

    Returns:
        Tuple[ProvidenceDataset...] of length 2 or 3, corresponding to (train, test) or (train, val, test), conditional
        of `df_split.validation` being `None`
    """
    return tuple(
        _BackblazeProvidenceDataset(df) for df in (df_split.train, df_split.validation, df_split.test) if df is not None
    )


def _BackblazeProvidenceDataset(df: DataFrame) -> ProvidenceDataset:
    """Convenience wrapper for the defaults that apply to the Backblaze-sourced DataFrames"""
    return ProvidenceDataset(
        df,
        grouping_field="serial_number",
        feature_columns=BACKBLAZE_FEATURE_NAMES,
        tte_column="tte",
        event_indicator_column="failure",
    )


################################################################################
#
# Backblaze Extended Dataset
#
################################################################################
BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER = [
    BackblazeQuarter._2019_Q1,
    BackblazeQuarter._2019_Q2,
    BackblazeQuarter._2019_Q3,
    BackblazeQuarter._2019_Q4,
]


def BackblazeExtendedDataset(
    *quarters: BackblazeQuarter,
    connsider_validation: bool = False,
    subset_choice: DataSubsetId = DataSubsetId.Train,
    split_percentage: float = 0.8,
    censoring_proportion: int = 1,
    data_root: str = "./.data",
    random_seed: int = 1234,
) -> ProvidenceDataset:
    """Parallel to BackblazeDataset(), but with the stipulation that you must provide plural quarters

    Args:
        quarters (BackblazeQuarter): quarter of the calendar year from which you want to extract the data.
            Quarters are sorted before processing, to each cross-quarter timeseries concatenation.
        consideration_validation (bool): whether to consider the validation set when taking the train or test seth
            (implied ``True`` when `subset_choice=Validation`)
        subset_choice (DataSubsetId): whether, you want the train, validation or test portion of the dataset
        censoring_proportion (float): the number of non-event devices per eventful device.
            This is for a downsampling procedure
        split_percentage (float): real number between 0 and 1, passed to ``df_train_test_split`` or
            ``df_train_val_test_split`` depending on `consider_validation`
        data_root (str): the parent directory used for downloading, caching, and retrieving this dataset. Multiple child
            directories will be created, so ensure adequate permissions when you supply a directory.
        random_seed: for deterministic splits

    Returns:
        A Providence Dataset filled with the Backblaze data corresponding to the ``quarters`` given.
    """
    validate(
        len(quarters) > 1,
        "Please supply multiple quarters. If you want to use a single quarter, use BackblazeDataset",
    )

    df = DataFrame()

    for quarter in sorted(quarters, key=lambda q: q.value):
        # load the quarter csv
        quarter_nan_filtered = load_cached_backblaze_quarter(quarter, data_root)
        # do the proportion
        sub_df = censored_subset(quarter_nan_filtered, censoring_proportion)
        # concat to DataFrame
        df = concat_dataframes([df, sub_df])

    # once all are assembled...
    # split and find the desired subset
    consider_validation = connsider_validation or subset_choice == DataSubsetId.Validation

    split_func = df_train_val_test_split if consider_validation else df_train_test_split
    bb_df_split = split_func(df, "serial_number", split_percentage=split_percentage, seed=random_seed)

    desired_sub_df = {
        DataSubsetId.Train: bb_df_split.train,
        DataSubsetId.Test: bb_df_split.test,
        DataSubsetId.Validation: bb_df_split.validation,
    }[subset_choice]

    assert desired_sub_df is not None
    desired_sub_df = censored_subset(desired_sub_df, censoring_proportion)

    # normalize
    raw_features = [n for n in BACKBLAZE_FEATURE_NAMES if n.endswith("_raw")]
    normalized_sub_df = normalize_by_device(desired_sub_df, feature_names=raw_features)
    # do the assemblage to traintable
    trainable_df = assemble_trainable_by_entity_id(normalized_sub_df, entity_id="serial_number")
    return _BackblazeProvidenceDataset(trainable_df)


def BackblazeExtendedDatasets(
    *quarters: BackblazeQuarter,
    include_validation: bool,
    split_percentage: float = 0.8,
    censoring_proportion: int = 1,
    data_root: str = "./.data",
    random_seed: int = 1234,
) -> Union[
    Tuple[ProvidenceDataset, ProvidenceDataset, ProvidenceDataset],
    Tuple[ProvidenceDataset, ProvidenceDataset],
    Tuple[Any, ...],  # supplied for mypy type check
]:
    """Produce the BackblazeExtended Train, (optional Validation) and Test datasets.

    BackblazeExtended differs from the Backblaze dataset in that the data is drawn from multiple quarters at once,
    concatenating common devices across quarters whenever possible before downsampling according to the
    `censoring_proportion`.

    Args:
        quarters (BackblazeQuarter): quarter of the calendar year from which you want to extract the data.
            Quarters are sorted before processing, to each cross-quarter timeseries concatenation.
        include_validation (bool): whether to include the validation set, following the split procedure
        split_percentage (float, optional): real number between 0 and 1, passed to ``df_train_test_split`` or
            ``df_train_val_test_split`` depending on ``consider_validation``. Defaults to 0.8.
        censoring_proportion (float, optional): the number of non-event devices per eventful device.
            This is for a downsampling procedure. See ``censored_subset`` for more.
        data_root (str, optional): the parent directory used for downloading, caching, and retrieving this dataset. Multiple child
            directories will be created, so ensure adequate permissions when you supply a directory.
        random_seed: for deterministic splits

    Returns:
        DataFrameSplit that is (in pseudo-code) df_splits.map(backblaze_preprocessing),
            where train, vaidation (if not None), and test are processed for Backblaze training.
    """
    validate(
        len(quarters) > 1,
        "Please supply multiple quarters. If you want to use a single quarter, use BackblazeDataset",
    )
    splits = [
        load_backblaze_splits(
            quarter=quarter,
            include_validation=include_validation,
            split_percentage=split_percentage,
            censoring_proportion=censoring_proportion,
            data_root=data_root,
            random_seed=random_seed,
        )
        for quarter in sorted(quarters, key=lambda q: q.value)
    ]
    df_split = DataFrameSplit(
        train=concat_dataframes([quarter_split.train for quarter_split in splits]),
        test=concat_dataframes([quarter_split.test for quarter_split in splits]),
        validation=concat_dataframes([quarter_split.validation for quarter_split in splits])  # type: ignore[misc]
        if include_validation
        else None,
    )
    df_split = BackblazePreprocessing(df_split)
    dss = BackblazeDatasets_from_split(df_split)
    return dss


def load_cached_backblaze_quarter(quarter: BackblazeQuarter, data_root: str = "./.data") -> DataFrame:
    """Load a `quarter`'s data, which is stored under ``data_root`` and cached adjacent to the ``quarter`` subdirectory.

    Args:
        quarter (BackblazeQuarter): quarter of the calendar year from which you want to extract the data
        data_root (str, optional): the parent directory used for downloading and caching this dataset.
            Multiple child directories will be created, so ensure adequate permissions when you supply a directory.

    Returns:
        DataFrame: the ``quarter`` stripped of NaN-containing devices.

    Raises:
        PermissionError: see ``cached_dataframe`` for more.
    """
    return cached_dataframe(
        lambda: (
            load_backblaze_csv(quarter, data_root=data_root)
            .groupby("serial_number")
            .filter(lambda df: not df.isnull().any().any())  # drop devices with NaN-ful sequences, shouldn't be many
        ),
        Path(data_root, "backblaze-download", quarter.value, "filtered.csv"),
    )


def BackblazeExtended_Legacy(
    *, data_root: str = "./.data", verbose: bool = False
) -> Tuple[ProvidenceDataset, ProvidenceDataset]:
    """Loads the datasets which were used in the paper, per the appendix of the research publication.

    Prefer the newer implementation above, or the DataLoader wrapper in providence.dataloaders, which are more inline
    with the narrative of Providence: Neural Time-to-Event Framework (https://doi.org/10.1109/AERO53065.2022.9843469)

    Args:
        data_root (str): the parent directory used for downloading, caching, and retrieving this dataset. Multiple child
            directories will be created, so ensure adequate permissions when you supply a directory.

    Returns:
        Tuple[ProvidenceDataset, ProvidenceDataset] which are two BackblazeExtended datasets
    """
    train_ids = read_csv(f"{data_root}/backblaze/bbe_legacy_ids_train.csv")["ids"]
    test_ids = read_csv(f"{data_root}/backblaze/bbe_legacy_ids_test.csv")["ids"]
    return _BackblazeExtended_from_legacy_ids(train_ids, test_ids, verbose=verbose, data_root=data_root)


def _BackblazeExtended_from_legacy_ids(
    train_ids: Union[List[str], Series],
    test_ids: Union[List[str], Series],
    *,
    split_percentage: float = 0.8,
    data_root: str = "./.data",
    verbose: bool = False,
) -> Tuple[ProvidenceDataset, ProvidenceDataset]:
    if verbose:
        print("In the function")
        print(f"{len(train_ids) = } {len(test_ids) = }")
    all_ids = set(train_ids) | set(test_ids)

    if verbose:
        print(f"{len(all_ids) = }")

    def calc_split_index(length: int, percentage: float) -> int:
        return int(length * percentage)

    class CompactQuarter(NamedTuple):
        quarter: BackblazeQuarter
        data: DataFrame
        failure_serials: List[str]  # set, list, doesn't really matter
        non_failure_serials: List[str]

    quarter_to_ids: Dict[BackblazeQuarter, CompactQuarter] = dict()
    for quarter in BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER:
        if verbose:
            print("Loading quarter", quarter)  # yapf: disable
        df_full = load_cached_backblaze_quarter(quarter, data_root)

        if verbose:
            print("DF loaded:")
            print("current n_devices:", len(df_full["serial_number"].unique()))
            print("current data shape:", df_full.shape)

        failure_mask = df_full["failure"] == 1
        failure_ids: np.ndarray = df_full[failure_mask]["serial_number"].unique()
        if verbose:
            print(f"{len(failure_ids) = }")  # yapf: disable

        assert all(map(lambda id_: id_ in all_ids, failure_ids)), "Got a failure id that's not in the legacy dataset"

        censored_mask = ~failure_mask
        censored_ids = df_full[censored_mask]["serial_number"].unique()
        if verbose:
            print(f"{len(censored_ids) = }")  # yapf: disable

        censored_ids_selected_subset = []
        for c_id in censored_ids:
            if c_id in all_ids:
                censored_ids_selected_subset.append(c_id)
            if len(censored_ids_selected_subset) == len(failure_ids):
                break
        else:
            if verbose:
                print(
                    f"Only found {len(censored_ids_selected_subset)} ids. What's up? You're about to get an exception"
                )

        if verbose:
            print(f"{len(censored_ids_selected_subset) = }")  # yapf: disable

        assert all(
            map(lambda id_: id_ in all_ids, censored_ids_selected_subset)
        ), "Got a censored id that's not in the legacy dataset"

        subset_ids = set(failure_ids) | set(censored_ids_selected_subset)
        data_for_quarter = df_full[df_full["serial_number"].isin(subset_ids)]
        quarter_to_ids[quarter] = CompactQuarter(
            quarter,
            data_for_quarter,
            failure_ids.tolist(),
            censored_ids_selected_subset,
        )

    if verbose:
        print("Processing DFs")  # yapf: disable
    quarter_to_processed_df: Dict[BackblazeQuarter, DataFrameSplit] = dict()

    for q, compaction in quarter_to_ids.items():
        if verbose:
            print(
                f"Processing {q}: {len(compaction.failure_serials) = } {len(compaction.non_failure_serials) = }"
            )  # yapf: disable
        split_ind = calc_split_index(len(compaction.failure_serials), split_percentage)
        train_serials = compaction.failure_serials[:split_ind] + compaction.non_failure_serials[:split_ind]
        test_serials = compaction.failure_serials[split_ind:] + compaction.non_failure_serials[split_ind:]
        df = compaction.data
        train_df_quarter = df[df["serial_number"].isin(train_serials)]
        test_df_quarter = df[df["serial_number"].isin(test_serials)]

        # this is how it was done in the original implementation. Yes, this is premature
        processed_dfs = BackblazePreprocessing(DataFrameSplit(train_df_quarter, test_df_quarter))
        if verbose:
            print("Completed processing of quarter", q)  # yapf: disable
        quarter_to_processed_df[q] = processed_dfs
        # feel free to put it in the paper_reproductions code...

    del quarter_to_ids

    # NOTE: each tuple should just be (train, test) for each of the quarters. Using the zip trick, you get two long tuples: each containing 4 quarters
    train_dses, test_dses = zip(
        *[BackblazeDatasets_from_split(df_split) for df_split in quarter_to_processed_df.values()]
    )  # type: Tuple[list, list]

    if verbose:
        for ds in train_dses:
            print(f"{len(ds) = }")

        for ds in test_dses:
            print(f"{len(ds) = }")

    def wrapping_ProvidenceDataset_from_many_bb_ds(
        backing_dses: List[ProvidenceDataset],
    ) -> ProvidenceDataset:
        """A simple wiring together of multiple BackblazeDatasets into one Providence dataset that is effectively the legacy
        BackblazeExtended-compatible.
        This is _much_ easier than hacking around with ConcatDataset, or carrying around all references in memory.
        Here we just pull apart the Providence Datasets, do the semantic concatenation, and return
        """
        from copy import deepcopy

        assert len(backing_dses), "Should have at least one backing dataframe"
        ds_instance = ProvidenceDataset.__new__(ProvidenceDataset)
        ds_instance.feature_columns = deepcopy(backing_dses[0].feature_columns)
        ds_instance.tte_column = deepcopy(backing_dses[0].tte_column)
        ds_instance.grouping_field = deepcopy(backing_dses[0].grouping_field)
        ds_instance.event_indicator_column = deepcopy(backing_dses[0].event_indicator_column)
        ds_instance.device = deepcopy(backing_dses[0].device)
        all_data = [
            (group_name, df) for groupby in map(attrgetter("grouped"), backing_dses) for group_name, df in groupby
        ]

        del backing_dses

        ds_instance.grouped = concat_dataframes([df for _, df in all_data]).groupby(ds_instance.grouping_field)
        ds_instance.data = all_data
        return ds_instance

    if verbose:
        print("Wrapping many Providence Datasets: train")  # yapf: disable
    train_ds = wrapping_ProvidenceDataset_from_many_bb_ds(train_dses)
    if verbose:
        print("Wrapping many Providence Datasets: test")  # yapf: disable
    test_ds = wrapping_ProvidenceDataset_from_many_bb_ds(test_dses)
    return train_ds, test_ds
