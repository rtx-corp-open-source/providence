"""
Providence-friedly implementation of a Dataset object for the NASA Turbofan Engine Degredation Simulation dataset. What follows is a README from that dataset.

The dataset can be found here: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
- New URL: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

Data Set: FD001
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

Data Set: FD002
Train trjectories: 260
Test trajectories: 259
Conditions: SIX
Fault Modes: ONE (HPC Degradation)

Data Set: FD003
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004
Train trjectories: 248
Test trajectories: 249
Conditions: SIX
Fault Modes: TWO (HPC Degradation, Fan Degradation)


Experimental Scenario

Data sets consists of multiple multivariate time series.
Each data set is further divided into training and test subsets.
Each time series is from a different engine i.e., the data can be considered to be from a fleet of engines of the same type.
Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user.

This wear and variation is considered normal, i.e., it is not considered a fault condition.
There are three operational settings that have a substantial effect on engine performance.
These settings are also included in the data.
The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
1)	unit number
2)	time, in cycles
3)	operational setting 1
4)	operational setting 2
5)	operational setting 3
6)	sensor measurement  1
7)	sensor measurement  2
...
26)	sensor measurement  23

Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, ìDamage Propagation Modeling for Aircraft Engine Run-to-Failure Simulationî, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

Implementation is
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Literal  # noqa: F401
from typing import Tuple

from pandas import concat as concat_dataframes
from pandas import DataFrame

from providence.datasets.adapters import get_nasa_subset_normalization_stats
from providence.datasets.adapters import load_nasa_dataframe
from providence.datasets.adapters import NASA_FEATURE_NAMES
from providence.datasets.adapters import NasaTurbofanTest
from providence.datasets.adapters import T_NASA_SUBSET_ID
from providence.datasets.core import DataFrameSplit
from providence.datasets.core import DataSubsetId
from providence.datasets.core import ProvidenceDataset
from providence.datasets.utils import normalize_by_device
from providence.datasets.utils import normalize_with_stats
from providence.utils import validate

################################################################################
#
# NASA Turbofan Dataset goodies
#
################################################################################


def NasaDataset(subset_choice: DataSubsetId, *, data_dir: str = "./.data") -> ProvidenceDataset:
    """Prepare the NASA Aggregate Dataset, either 'train' or 'test' based on `subset_choice`.

    The functionality herein is akin to effectively:
    >>> pipeline(load_dataframes, normalization, lambda df: df.assign(event=1), _NasaProvidenceDataset)

    Args:
        subset_choice (DataSubsetId): returns the training set if `subset_choice=Train`, otherwise the test set
        data_root (str): the parent directory used for downloading, caching, and retrieving this dataset. Multiple child
            directories will be created, so ensure adequate permissions when you supply a directory.

    Returns:
        A ProvidenceDataset for the NASA Data
    """
    train_or_test = "train" if subset_choice == DataSubsetId.Train else "test"  # type: Literal["train", "test"]
    sub_dfs = [
        load_nasa_dataframe(nasa_subset, split_name=train_or_test, data_root=data_dir)
        for nasa_subset in NasaTurbofanTest.all()
    ]
    all_df = concat_dataframes(sub_dfs)
    norm_stats = get_nasa_subset_normalization_stats("all")
    normed = normalize_with_stats(all_df, norm_stats, method="standardize", feature_names=NASA_FEATURE_NAMES)
    # NOTE: event = 1 because we _know_ the engine will fail at the end of its run. We've just censored that.
    ds = _NasaProvidenceDataset(normed.assign(event=1))

    return ds


def NasaDatasets(*, data_root: str = "./.data") -> Tuple[ProvidenceDataset, ProvidenceDataset]:
    """Constructs the NASA Aggregate train-test datasets

    Args:
        data_root (str): the parent directory used for downloading, caching, and retrieving this dataset. Multiple child
            directories will be created, so ensure adequate permissions when you supply a directory.

    Returns:
        Tuple[ProvidenceDataset, ProvidenceDataset] of the NASA train and test datasets.
    """
    splits_at_tests = [
        load_nasa_train_test_split(test_number, data_root=data_root) for test_number in NasaTurbofanTest.all()
    ]
    df_split = DataFrameSplit(
        train=concat_dataframes([test_split.train for test_split in splits_at_tests]),
        test=concat_dataframes([test_split.test for test_split in splits_at_tests]),
    )
    df_split = NasaPreprocessing(df_split, nasa_subset="all")
    df_split = DataFrameSplit(train=df_split.train.assign(event=1), test=df_split.test.assign(event=1))
    train_ds, test_ds = NasaDatasets_from_split(df_split)
    return train_ds, test_ds


def NasaPreprocessing(df_splits: DataFrameSplit, *, nasa_subset: T_NASA_SUBSET_ID):
    """Normalize and shape DataFrames based on statistics of the given ``nasa_subset``

    # TODO(stephen): expand to support optional validation set


    Args:
        df_splits (DataFrameSplit): train and test DataFrames, intended to be for the given ``nasa_subset``
        nasa_subset (T_NASA_SUBSET_ID): see type definition

    Returns:
        DataFrameSplit: train, test datasets normalized per the procedure used in the paper
    """
    norm_stats = get_nasa_subset_normalization_stats(nasa_subset)
    df_splits = DataFrameSplit(
        # normalize per device in the training set,
        train=normalize_by_device(
            df_splits.train,
            entity_id="unit number",
            method="standardize",
            feature_names=NASA_FEATURE_NAMES,
        ),
        test=normalize_with_stats(
            df_splits.test,
            norm_stats,
            method="standardize",
            feature_names=NASA_FEATURE_NAMES,
        ),
    )

    return df_splits


def NasaFD00XDataset(
    *turbofan_test_num: NasaTurbofanTest,
    subset_choice: DataSubsetId,
    data_dir: str = "./.data",
) -> ProvidenceDataset:
    """Construct the concatenation of n-many ``NasaTurbfanTest`` data sets.

    Use this function if you want just the 'train' or 'test' portion of this CMAPPS data.
    The name ``FD00X`` brings to mind that the runs were named fd00{integer}.

    Args:
        turbofan_test_num (NasaTurbofanTest): the NASA turbofan test runs to be concatenated together to make a dataset
        data_root (str, optional): the parent directory used for downloading, caching, and retrieving this dataset.
            Multiple child directories will be created, so ensure adequate permissions when you supply a directory.
            Defaults to "./.data".

    Returns:
        A ``ProvidenceDataset`` of the given ``subset_choice`` of tests ``turbofan_test_num``. By example:
        1. Initialization of a single subset of the PHM08 CMAPPS data, fully normalized and ready for training
            >>> nasa_trainFD001 = NasaFD00XDataset(NasaTurbofanTest.FD001, subset_choice=DataSubsetId.Train)
            >>> nasa_trainFD004 = NasaFD00XDataset(NasaTurbofanTest.FD004, subset_choice=DataSubsetId.Train)

        2. Initialization of any choice of subset of the same CMAPPS data, as Martinsson does in section 4.3 of his thesis
            >>> martinsson_ds = NasaFD00XDataset(NasaTurbofanTest.FD002, NasaTurbofanTest.FD004, subset_choice=DataSubsetId.Train)
    """
    validate(subset_choice != DataSubsetId.Validation, "Only train and test subset choices are valid.")
    train_or_test = "train" if subset_choice == DataSubsetId.Train else "test"  # type: Literal["train", "test"]
    # I need this function to return the normalization of n-many turbofan_test-s
    sub_dfs = [
        normalize_with_stats(
            load_nasa_dataframe(nasa_subset, split_name=train_or_test, data_root=data_dir),
            stats=get_nasa_subset_normalization_stats(nasa_subset),
            method="standardize",
            feature_names=NASA_FEATURE_NAMES,
        )
        for nasa_subset in turbofan_test_num
    ]

    normed = concat_dataframes(sub_dfs)

    normed = normed.assign(event=1)  # event indicator column. Everything fails eventually.

    return _NasaProvidenceDataset(normed)


def NasaFD00XDatasets(
    nasa_subset_num: NasaTurbofanTest, *, data_root: str = "./.data"
) -> Tuple[ProvidenceDataset, ProvidenceDataset]:
    """Construct the train and test set of the given `nasa_subset_num`, with data downloaded to `data_root`

    Args:
        nasa_subset_num (NasaTurbfanTest): The run of the NASA simulator that you want train-test from.
        data_root (str, optional): the parent directory used for downloading, caching, and retrieving this dataset.
        Multiple child directories will be created, so ensure adequate permissions when you supply a directory.
        Defaults to "./.data"

    Returns:
        Tuple[ProvidenceDataset, ProvidenceDataset]: train and test datasets for the given ``NasaTurbofanTest``
    """
    df_split = load_nasa_train_test_split(nasa_subset_num, data_root=data_root)
    df_split = NasaPreprocessing(df_split, nasa_subset=nasa_subset_num)
    df_split = DataFrameSplit(train=df_split.train.assign(event=1), test=df_split.test.assign(event=1))
    train_ds, test_ds = NasaDatasets_from_split(df_split)
    return train_ds, test_ds


def _NasaProvidenceDataset(df: DataFrame) -> ProvidenceDataset:
    return ProvidenceDataset(
        df,
        grouping_field="unit number",
        feature_columns=NASA_FEATURE_NAMES,
        tte_column="RUL",
        event_indicator_column="event",
    )


def load_nasa_train_test_split(
    nasa_subset: NasaTurbofanTest = NasaTurbofanTest.FD001,
    *,
    data_root: str = "./.data",
) -> DataFrameSplit:
    """Load the nasa dataset

    Args:
        nasa_subset (NasaTurbofanTest): The turbofan test you want loaded
        data_root (str): Root directory under which a subdirectory for download and caching will be constructed
            for storage and retrieval of this dataset

    Returns:
        DataFrameSplit containing the train and test set, and `validation=None`.
    """
    return DataFrameSplit(
        train=load_nasa_dataframe(nasa_subset, split_name="train", data_root=data_root),
        test=load_nasa_dataframe(nasa_subset, split_name="test", data_root=data_root),
    )


def NasaDatasets_from_split(df_split: DataFrameSplit):
    """Prepare NASA train and test datasets from ``df_split``

    Args:
        df_split (DataFrameSplit): train and test set for some NASA-compatible DataFrames

    Returns:
        Tuple[ProvidenceDataset, ProvidenceDataset]: instantiated on the train, test field of ``df_split``
    """
    train_ds = _NasaProvidenceDataset(df_split.train)
    test_ds = _NasaProvidenceDataset(df_split.test)

    return train_ds, test_ds
