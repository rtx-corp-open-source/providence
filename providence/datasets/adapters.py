"""
Purpose:
This file represents the bridge between the old system and the new without a ton of code duplication.
In this way, we'll
1. generate new functionality alongside existing functionality,
2. show weaknesses of the old implementation,
3. grow or augment the good atoms of the old system into something better

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import io
from enum import Enum
import json
from logging import getLogger
from multiprocessing.pool import ThreadPool
from pathlib import Path
from shutil import move
from shutil import unpack_archive
from typing import Final, Optional
from typing import List
from typing import Literal
from typing import Union

import numpy as np
import progressbar
import requests  # type: ignore[import]
from pandas import concat as concat_dataframes
from pandas import DataFrame
from pandas import read_csv
from pandas import Series

from providence.utils import cached_dataframe, validate

logger = getLogger(__name__)


################################################################################
#
# Backblaze Specifc
#
################################################################################
class BackblazeQuarter(Enum):
    _2019_Q1 = "data_Q1_2019"
    _2019_Q2 = "data_Q2_2019"
    _2019_Q3 = "data_Q3_2019"
    _2019_Q4 = "data_Q4_2019"
    _2020_Q1 = "data_Q1_2020"
    _2020_Q2 = "data_Q2_2020"
    _2020_Q3 = "data_Q3_2020"
    _2020_Q4 = "data_Q4_2020"

    @property
    def drive_stats_name(self) -> str:
        _, quarter, year = self.value.split("_")
        # e.g. drive_stats_2019_Q1
        return "_".join(["drive_stats", year, quarter])


def _file_download(source_url: str, dest: Path) -> None:
    """Download the file from ``source_url`` to ``dest`` on the local system.

    Args:
        source_url (str): URL to download content from
        dest (Path): Desination to download the file contents into.
            Functions as a temporary directory that will be deleted if errors occur.

    Raises:
        AnyError: If anything bad happens
        requests.exceptions.Timeout:
            Typically occurs if your network proxy is in the way, which is resolvable outside of this code.

        requests.exceptions.TooManyRedirects:
            This can theoretically happen, but hasn't happened in any usage of the library

        requests.exceptions.RequestException:
            This might happen if your internet access is cut somewhat out of the blue.
    """
    try:
        with open(dest.as_posix(), mode="wb") as write_out:
            big_response = requests.get(source_url, {"stream": True})
            for chunk in progressbar.progressbar(big_response.iter_content(1024 * 1024)):  # iterate megabytes
                write_out.write(chunk)
    except Exception as e:
        logger.info("Cleaning up empty zip")
        import os

        os.remove(dest.as_posix())
        raise


def _extract_zip(zip_file_path: Path, extraction_directory: Path):
    """Extract the contents of the zip file at ``zip_file_path`` to `extraction_directory`

    Args:
        zip_file_path (Path): path to the zip file to extract. Not necessarily absolute, but YMMV

    Raises:
        FileNotFoundError: if ``zip_file_path`` is not found on your machine.
    """
    if not extraction_directory:
        extraction_directory = zip_file_path.parent
    logger.info(f"Extract zip: {zip_file_path = }, {extraction_directory = }")
    unpack_archive(zip_file_path.as_posix(), extraction_directory.as_posix(), format="zip")


def _download_zip(
    url: str,
    zip_download_dest: Path,
    make_parent_dirs: bool = True,
):
    """Download the zip file to `dest`, creating the storage directory `zip_download_dest`

    Args:
        url (str): valid url to download a zip file from.
        zip_download_dest (Path): the destination file path and name i.e. the path to a zip file to be created.
        make_parent_dirs (bool):  `make_parent_dirs=False` will provoke an exception if
            the parent directory of the ``zip_download_dest`` doesn't exist.

    Raises:
        AssertionError: If the supplied destination
    """
    assert zip_download_dest.suffix == ".zip"
    zip_download_dest.parent.mkdir(parents=make_parent_dirs, exist_ok=True)

    if zip_download_dest.exists():
        logger.info("Zip already downloaded. Skipping")
    else:
        _file_download(url, dest=zip_download_dest)


def _verify_download(directory: Path, quarter_downloaded: BackblazeQuarter):
    """Inspect ``directory`` for the expected contents of the ``quarter_downloaded`` from Backblaze
    unraveling their naming/packaging heuristics to get the 90-ish CSVs within accessible at the root of the folder.

    Args:
        directory (Path): path to directory (i.e. not an individual file)
        quarter_downloaded (BackblazeQuarter): the quarter of Backblaze's dataset to validate.
            Necessary because quarters aren't zipped uniformly, and the BackblazeQuarter type provides meaningful
            information to assist in checking the full quarter is accessible at the top of the ``directory`` given.

    Raises:
        AssertionError: if the ``directory`` and ``quarter_downloaded`` do not correspond
    """
    assert (
        directory.name == quarter_downloaded.value
    ), "Expected downloaded directory to have the same name as the quarter source file"
    directory_contents = [p.name for p in directory.iterdir()]
    # this magic 5 represents a significant number of files that would exceed the expected OS-related metadata
    # that may be shipped in the zip that has been extracted.
    # NOTE(stephen): if we ever store the data RTX-side, this pre-post-processing can be done elsewhere
    if len(directory_contents) < 5:
        # it is clear that this CSVs aren't at *this* level, but it's possible that the extraction directory is nested
        # so we check for the name matches that we have seen thus far
        potential_extraction_names = [
            directory.name,
            quarter_downloaded.value,
            quarter_downloaded.drive_stats_name,
        ]
        any_directory_matches = any(map(lambda name: name in directory_contents, potential_extraction_names))
        logger.info(f"{potential_extraction_names = }")
        logger.info(f"{any_directory_matches = }")
        if any_directory_matches:
            # move all of the files from the nested directory up one level, into the owning directory
            # being careful to avoid the __MACOSX/._<filename>.csv's
            for p in directory.glob("**/20*.csv"):
                move(p.as_posix(), (directory / p.name).as_posix())


def download_backblaze_dataset(quarter_for_download: BackblazeQuarter, *, data_root: str):
    """Download the `quarter_for_download`, extracting its content to a named child of `'data_root'/backblaze-download/`.

    Args:
        quarter_for_download (BackblazeQuarter): the BackblazeQuarter desired to be downloaded.
        data_root (str): directory under which `backblaze-download` directory will be generated, and all
            downloaded files and temporary files will be stored. Intended to be kept in case of corruption of the
            compact representation.
    """
    url = f"https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/{quarter_for_download.value}.zip"
    csv_dest_directory = Path(data_root, "backblaze-download", quarter_for_download.value)
    logger.info("Downloading...")

    zip_download_dest = Path(data_root, quarter_for_download.value).with_suffix(".zip")
    _download_zip(url, zip_download_dest)
    _extract_zip(zip_download_dest, extraction_directory=csv_dest_directory)

    logger.info("Verifying...")
    _verify_download(csv_dest_directory, quarter_for_download)


def load_quarterly_data_parallel(quarter_dir: Path, feature_set: List[str]) -> DataFrame:
    """Load a directory of CSVs across 4 threads, taking only ``feature_set`` from a given CSV (for speed).

    Originally intended to be used for reading the CSVs from the Backblaze dataset download in parallel, but
    can easily be used in other contexts.

    Args:
        quarter_dir (Path): Path to a directory of CSVs
        feature_set (List[str]): Features to take from a given CSV in the directory

    Returns:
        DataFrame: concatenation of the all CSVs found in ``quarter_dir``

    Raises:
        FileNotFoundError: If the directory supplied does not exist.
    """

    def read_csv_with_dates(p):
        return read_csv(p, usecols=feature_set, parse_dates=["date"])

    logger.info("Starting parallel download")
    with ThreadPool(4) as p:
        logger.info("Initialized thread pool")
        dataframe = concat_dataframes(  # type: ignore[call-overload]
            progressbar.progressbar(p.map(read_csv_with_dates, quarter_dir.glob("*.csv")), max_value=93),
            axis="rows",
        )
    logger.info("Reseting dataframe index")
    return dataframe.reset_index(drop=True)


def load_backblaze_csv(quarter: BackblazeQuarter, *, data_root: str) -> DataFrame:
    """Download (or load cached) the ``quarter`` from the Backblaze remote, storing under ``data_root/'backblaze'/``.

    Args:
        quarter (BackblazeQuarter): the BackblazeQuarter desired to be loaded from the file tree under `data_root`
            If this quarter has not been downloaded, it will be.
        data_root (str): directory under which ``"backblaze-download"`` directory will be generated, and all
            downloaded files and temporary files will be stored. Intended to be kept in case of corruption of the
            compact representation.

    Raises:
        PermissionError: If you are trying to access a ``data_root`` that your user doesn't have permission for.
        Error: If anything else bad happens. See the other methods in this file for specifics
    """
    csv_directory = Path(data_root, "backblaze-download", quarter.value)
    if not csv_directory.is_dir():
        download_backblaze_dataset(quarter, data_root=data_root)
    columns_to_use = BACKBLAZE_METADATA_COLUMNS + BACKBLAZE_FEATURE_NAMES
    cached_path = Path(data_root, "backblaze", quarter.value) / "cached.csv"
    data = cached_dataframe(lambda: load_quarterly_data_parallel(csv_directory, columns_to_use), cached_path)
    logger.info("Loading done")
    return data


################################################################################
#
# NASA dataset specifics
#
################################################################################

_NASA_REMOTE_URL_STEM = "https://raw.githubusercontent.com/AlephNotation/nasa_turbofan/main/"
_Train_or_test = Literal["train", "test"]


class NasaTurbofanTest(int, Enum):
    FD001 = 1
    FD002 = 2
    FD003 = 3
    FD004 = 4

    @classmethod
    def all(cls) -> List["NasaTurbofanTest"]:
        """Produce all of the turbofan tests as a list

        Returns:
            List[NasaTurbofanTest]: containing each of the turbofan tests above
        """
        return [cls.FD001, cls.FD002, cls.FD003, cls.FD004]


def load_nasa_dataframe(nasa_subset: NasaTurbofanTest, *, split_name: _Train_or_test, data_root: str) -> DataFrame:
    """Download (or load from cache) the raw dataframe corresponding to supplied NASA Turbofan test `nasa_subset`.

    Args:
        nasa_subset (NasaTurbofanTest): The given NASA dataset you want to load
        split_name (str): the literal 'train' or 'test' to determine which portion of the ``nasa_subset`` loaded by
            this method.
            Necessary because they are stored and loaded separately.
        data_root (str): directory under which `nasas-download` directory will be generated, and all
            downloaded files and temporary files will be stored. Intended to be kept in case of corruption of the
            compact representation.

    Returns:
        DataFrame: all timeseries from every device in the given ``nasa_subset``, with features as explained in nasa.py

    Raises:
        AssertionError: when `not split_name in {"train", "test"}`
    """
    validate(split_name in {"train", "test"}, f"{split_name=} is invalid.")
    download_name = lambda prefix: f"{prefix}_FD{nasa_subset:03d}"

    cache_path = Path(data_root, "nasa-download", download_name(split_name)).with_suffix(".csv")

    def load_data_from_remote():
        url = _NASA_REMOTE_URL_STEM + download_name(split_name) + ".txt"
        logger.info(f"Fetching NASA raw data from {url = }")

        http_response = requests.get(url)
        np_data = np.loadtxt(io.BytesIO(http_response.content))

        data = DataFrame(np_data, columns=NASA_METADATA_COLUMNS + NASA_FEATURE_NAMES)

        if split_name == "test":
            url = _NASA_REMOTE_URL_STEM + download_name("RUL") + ".txt"
            logger.info(f"Fetching RUL for {nasa_subset}, from {url = }")
            http_response = requests.get(url)
            rul_data = np.loadtxt(io.BytesIO(http_response.content))

            # rul_data is the starting RULs for all these devices and needs to be mapped to the individual devices
            # grab the rul_data[device_id] which is the final tte. Then count backwards.

        def calculate_first_and_final_rul(cycles: Series, device_id: str):
            # this abuses Pythons late binding to reduce the amount of code here.
            last_rul_for_device = rul_data[int(device_id) - 1] if split_name == "test" else 0
            first_rul = cycles.iloc[-1] + last_rul_for_device
            final_rul = first_rul - len(cycles)
            return first_rul, final_rul

        _nasa_device_id_col = NASA_METADATA_COLUMNS[0]

        disaggregated = [
            # tte_targets = arange(tte := (max tte + offset for "validation)", tte - sequence_length, -1)
            df.assign(
                RUL=lambda df: np.arange(*calculate_first_and_final_rul(df["cycle"], device_id), -1)  # noqa: B023
            )
            for device_id, df in data.groupby(_nasa_device_id_col)
        ]
        # reassign df. stops wasting memory, among other things
        data = concat_dataframes(disaggregated)

        data[_nasa_device_id_col] += nasa_subset * 1000  # prefix ids of 10XX for FD001, 20XX for FD002, etc.

        return data

    return cached_dataframe(load_data_from_remote, cache_path)


T_NASA_SUBSET_ID = Union[Literal["all"], None, NasaTurbofanTest]


def get_nasa_subset_normalization_stats(subset_id: T_NASA_SUBSET_ID) -> DataFrame:
    """Produce the precomputed, feature-wise mean and standard deviation of the given subset of NASA data.

    Args:
        subset_id (T_NASA_SUBSET_ID): The given NASA dataset you want to load statistics for.
            `subset_id="all"` or `subset_id=None` produce the aggregated statistics over the entire dataset.
        - It should be noted that training on this version of the data is different that taking each subset in turn,
            normalizing, then concatenating.

    Returns:
        A DataFrame: of shape [len(NASA_FEATURE_NAMES), 2], containing the normalization statistics,
            where columns are the statistic facilitating `normalization_stats['std']`.
            with the convention of 'standard deviation' -> 'std' over (say) 'stddev' or 'sd'.

    Raises:
        ValueError: If you use an invalid `subset_id`. See ``T_NASA_SUBSET_ID`` for valid values.
    """
    if subset_id is not None and subset_id not in (valid_identifiers := {"all", *NasaTurbofanTest.all()}):
        raise ValueError(f"{subset_id = } is an invalid identifier for a NASA subset. Try one of {valid_identifiers}")

    stats = {
        "mean": _NasaNormalizationSummary.get_feature_means(subset_id),
        "std": _NasaNormalizationSummary.get_feature_stddevs(subset_id),
    }

    df = DataFrame(stats)
    return df


class _NasaNormalizationSummary:
    """
    Holds precomputed mean and standard deviation values for each NASA Turbofan dataset.
    If you prefer an OOP-usage, you can instantiate per subset. Typically, this is used as a static namespace.

    Standard deviations close to zero (< 1e-15) are set equal to one for numerical stability
    """

    statistics = json.load(open((Path(__file__).parent / "nasa_summary_stats.json"))) # type: dict

    def __init__(self, dataset_number: int = None):
        self.dataset_number = dataset_number

    # class methods

    @classmethod
    def get_feature_means(cls, dataset_number: Optional[int]) -> List[float]:
        key = "" if dataset_number is None else str(dataset_number)
        return cls.statistics[key]["mean"]

    @classmethod
    def get_feature_stddevs(cls, dataset_number: Optional[int]) -> List[float]:
        key = "" if dataset_number is None else str(dataset_number)
        return cls.statistics[key]["stddev"]

    # instance methods.
    # NOTE: the following are unused and likely deprecated.

    def get_means(self) -> List[float]:
        return self.statistics[self.dataset_number]["mean"]

    def get_stddevs(self) -> List[float]:
        return self.statistics[self.dataset_number]["stddev"]


################################################################################
#
# Aliases
#
################################################################################
# SMART stats. See "Dataset details"
_smart_features = [
    1,
    5,
    12,
    187,
    188,
    192,
    193,
    195,
    197,
    198,
]
_raw_features = [f"smart_{d1}_raw" for d1 in _smart_features]
_normalized_features = [f"smart_{d2}_normalized" for d2 in _smart_features]

# NOTE: I prefer lowercase constant names, but this is Python's PEP-8 convention
BACKBLAZE_METADATA_COLUMNS: Final = [
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
]
BACKBLAZE_FEATURE_NAMES: Final = _raw_features + _normalized_features

NASA_METADATA_COLUMNS: Final = ["unit number", "cycle"]
NASA_FEATURE_NAMES: Final = [
    *[f"operational_setting_{i}" for i in range(1, 4)],  # 1 - 3
    *[f"sensor_measurement_{i}" for i in range(21)],  # 0 - 20
]

DS_NAME_TO_FEATURE_COUNT = {
    "backblaze": len(BACKBLAZE_FEATURE_NAMES),
    "bbe": len(BACKBLAZE_FEATURE_NAMES),
    "nasa": len(NASA_FEATURE_NAMES),
}


################################################################################
#
# TTE-utilities
#
################################################################################


def compute_tte(s: Series, event_name: str = "failure", tte_offset: int = 1, tte_name: str = "tte") -> Series:
    """Compute a TTE series from ``s`` and ``event_name``.

    Can be used independently, but use in conjunction with ``assemble_trainable_by_entity_id(...)``, for best results.
    See ``assemble_trainable_by_entity_id(...)`` for more.

    Compute TTE on Series indexed by datetime (pd.Timestamp), given there's a failure in the series.
    Otherwise, returns an empty series.

    Args:
        s (Series):
        tte_offset (int): How far from the event should we consider the time to event?
            I.e. is the final time to event 0, 1 or n when we have a "true" after a long string of "not occured"?

    Returns:
        Series with a descending time-to-event

    Example:

        >>> compute_tte(grab_time_sorted_df('ZHZ3RZW4').set_index('date')['failure'])
    """
    validate(s.name == event_name, f"Expecting to be working off event '{event_name}'. Got series named: '{s.name}'")

    # asserting, as this is both internal and external state, but key to the coherence of the library
    assert all(s.value_counts().keys().isin([0, 1])), "Expected binary labels of specific event occurence"  # type: ignore[attr-defined]

    # if we didn't get datetime-sorted going in, make sure it happens now.
    if "datetime" in s.index.dtype.name:
        s = s.sort_index()  # just in case

    last_day = s.index[-1]

    def days_from_end(x):
        return last_day - x  # .days

    # not using the vectorized subtraction because that doesn't work (for some reason). Will investigate later
    ret = (s.index.map(days_from_end) + tte_offset).to_series().astype(int)

    return ret.rename(tte_name).reset_index(drop=True)


def assemble_trainable_by_entity_id(
    data: DataFrame,
    *,
    entity_id: str,
    temporality_indicator: str = "date",
    event_occurence_column: str = "failure",
) -> DataFrame:
    """Convert a timeseries dataset into one with a time-to-event ``tte`` that counts down to 1.

    We assume that the last recording came before the event, which if a failure can't be recorded.

    Args:
        data (DataFrame): the DataFrame that you would like converted into being `ProvidenceDataset`-friendly
        entity_id (str): the entity identifier field in the `data`
                It will be a marginal performance hit and make this function much more generic
        temporality_indicator (str): the notion of time you would like to sort each entity by. This is typically a date or timestamp
        event_occurence_column (str): the column that indicates where the entity experienced an event within the dataset

    Returns:
        DataFrame: shaped and organizing for preprocessing, which can be done before training.
    """

    def grab_time_sorted_df(serial) -> DataFrame:
        return data[data[entity_id] == serial].sort_values(by=[temporality_indicator]).reset_index(drop=True)

    logger.info("Assembling trainable, by entity id")
    all_dfs = [
        grab_time_sorted_df(serial_number).assign(tte=lambda df: compute_tte(df[event_occurence_column]))
        # TODO(stephen): update this function to actually use the entity_id field to generate unique entries
        for serial_number in progressbar.progressbar(data.serial_number.unique())
    ]

    return concat_dataframes(all_dfs, axis="rows", ignore_index=True)  # type: ignore[call-overload]
