from enum import Enum
import logging
from pathlib import Path
from shutil import move, unpack_archive
from typing import List, Set, Tuple, Union

import numpy as np
import pandas as pd
import progressbar
import torch
from providence.utils import logging_utils

_Random = np.random.RandomState(452)

#### Logging ####
logging.basicConfig(
    level=logging.NOTSET,
    format="[%(asctime)s] PROVIDENCE:%(levelname)s - %(name)s - %(message)s",
    handlers=[logging_utils.ch, logging_utils.fh],
)

logging.captureWarnings(True)
logger = logging.getLogger(__name__)
#################

logger.debug("Set debug ON")


class BackblazeQuarterName(Enum):
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
        data, quarter, year = self.value.split("_")
        # drive_stats_2019_Q1
        return '_'.join(["drive_stats", year, quarter])


def calc_split_index(l: List[int], percentage: float) -> int:
    return int(len(l) * percentage)


def load_quarterly_data(quarter_dir: Path, feature_set: List[str]) -> pd.DataFrame:
    """Loads the quarterly data as extracted from Backblaze."""
    logger.info("Loading Backblaze data for the quarter")
    dataframe = pd.concat(
        [
            pd.read_csv(p, usecols=feature_set, parse_dates=["date"])
            for p in progressbar.progressbar(quarter_dir.glob(f"*.csv"), max_value=93)  # maximum days in a quarter
        ],
        axis="rows",
    )
    return dataframe.reset_index(drop=True)


def create_serial_id_mapping(ids: Union[pd.Series, np.ndarray], exclusions: pd.Series):
    """
    Coupled to the DataFrame giving me a series back. Don't want to throw around a ton of constructors.
    """
    ids = pd.Series(ids)
    subset = ids[~ids.isin(exclusions)]
    starting_id = len(exclusions)
    logger.info(f"Starting_id = {starting_id}")
    return pd.DataFrame(zip(subset, range(starting_id, starting_id + len(subset))), columns=["serial_number", "id"])


def raw_features(feature_ids: List[int]) -> List[str]:
    return [(f"smart_{d}_raw") for d in feature_ids]


def normalized_features(feature_ids: List[int]) -> List[str]:
    return [(f"smart_{d}_normalized") for d in feature_ids]


def mixed_features(feature_ids: List[int]) -> List[str]:
    """Return the mixed list of feature names from the numerical identifiers of the features"""
    return raw_features(feature_ids) + normalized_features(feature_ids)


def get_candidate_serials(base_serials, previously_chosen=set(), sample_size=5) -> Tuple[List[str], Set[str]]:
    indices = set(base_serials)
    chosen = _Random.choice(list(indices - previously_chosen), replace=False, size=sample_size)

    ret = list(chosen), previously_chosen | set(chosen)
    return ret


def make_serial_subsets(base_serials: List[str], sample_sizes: List[int]) -> List[List[str]]:
    return [get_candidate_serials(base_serials, sample_size=sample_size)[0] for sample_size in sample_sizes]


def compute_tte(s: pd.Series, event_name: str = "failure", tte_offset: int = 1) -> pd.Series:
    """
    Compute TTE on Series indexed by datetime (pd.Timestamp), given there's a failure in the series.
    Otherwise, returns an empty series
    Our standard is that TTE is a countdown to **1**. So,
    1. The first plot should count from 93 to 1
    2. the second should count 33 to 1 (31 days of october + one day in november)
    :params tte_offset: How far from the event should we consider the time to event?
                        I.e. is the time to event 0, 1 or n when we have a "true" after a long string of "false"?

    # usage:
    >>> compute_tte(grab_time_sorted_df('ZHZ3RZW4').set_index('date')['failure']) 
    """
    assert s.name == event_name, f"Expecting to be working off event '{event_name}'. Got series named: '{s.name}'"
    assert all(s.value_counts().keys().isin([0, 1])), "Expected binary labels of specific event occurence"

    # if we didn't get datetime-sorted going in, make sure it happens now.
    if "datetime" in s.index.dtype.name:
        s = s.sort_index()  # just in case

    last_day = s.index[-1]

    def days_from_end(x):
        return last_day - x  # .days

    # not using the vectorized subtraction because that doesn't work (for some reason). Will investigate later
    ret = (s.index.map(days_from_end) + tte_offset).to_series().astype(np.int32)

    return ret.rename("tte").reset_index(drop=True)


def assemble_persistable_trainable(data: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the assemblage of all the days of the quarterly dataset into a drive-by-drive TTE dataset
    """

    def grab_time_sorted_df(serial) -> pd.DataFrame:
        return data[data["serial_number"] == serial].sort_values(by=["date"]).reset_index(drop=True)

    all_dfs = [
        grab_time_sorted_df(serial_number).assign(tte=lambda df: compute_tte(df["failure"]))
        for serial_number in progressbar.progressbar(data.serial_number.unique())
    ]

    return pd.concat(all_dfs, axis="rows", ignore_index=True)


def now_dt_string() -> str:
    from datetime import datetime

    # training isn't fast enough to record the microsecond
    return datetime.now().replace(microsecond=0).isoformat()


class BackblazeDataset(torch.utils.data.Dataset):
    """
    Dataset object for the Backblaze SMART Stats Failure dataset.

    The dataset can be found here: https://www.backblaze.com/b2/hard-drive-test-data.html
    - A example quarter can be found at URL: https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q4_2019.zip

    Dataset details:
    - The "failure" event is actually a marker of an intervention on a hard drive, either a failed drive that needed
      to be removed or a just that Backblaze predicted to fail and remove preemptively.
    - The feature included are a subset, out of both bandwidth and utility requirements. The source dataset has dozens more
      features, most of which aren't profitable but are neveretheless reported for historics' sake. These features are
      SMART stats 1, 5, 12, 187, 188, 197, and 198. Each is related to some harddrive auto reported statistic from an
      industry-standard suite for industrial harddrive usage.
      - Read more at the Backblaze blog here: https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/
    """

    metadata_columns = ["date", "serial_number", "model", "capacity_bytes", "failure"]
    _smart_features = [1, 5, 12, 187, 188, 197, 198]  # SMART stats. See "Dataset details"
    _raw_features = raw_features(_smart_features)
    numerical_features = mixed_features(_smart_features)

    @staticmethod
    def _file_download(source_url: str, dest: Path) -> None:
        """Download the file from `source_url` to `dest` on the local system"""
        import subprocess

        subprocess.check_call([(Path(__file__).parent / "scripts/download_backblaze_zip.sh").as_posix(), source_url, dest.as_posix()])

    @staticmethod
    def _extract_zip(zip_file_path: Path, extraction_directory: str = None):
        """Extract the contents of the zip file at `zip_file_path` to `extraction_directory` or adjacent to the file if None provided"""
        if not extraction_directory:
            extraction_directory = (zip_file_path.parent).as_posix()
        logger.info(f"Extract zip: {zip_file_path = }, {extraction_directory = }")
        unpack_archive(zip_file_path.as_posix(), extraction_directory, format="zip")

    def _download_and_extract_zip(self, url: str, download_root: Path, extraction_dir_name: str, make_parent_dirs: bool = True):
        """Download and extract the contents of the file downloaded to `dest`"""
        download_root.mkdir(parents=make_parent_dirs, exist_ok=True)

        zip_download_dest = download_root / f"data-{now_dt_string()}.zip"
        self._file_download(url, dest=zip_download_dest)
        self._extract_zip(zip_download_dest, extraction_directory=(download_root / extraction_dir_name).as_posix())

    @staticmethod
    def _verify_download(directory: Path, quarter_downloaded: BackblazeQuarterName):
        """Inspect `directory` for the expected contents of the `quarter_downloaded`"""
        assert directory.name == quarter_downloaded.value, "Expected downloaded directory to have the same name as the quarter source file"
        directory_contents = [p.name for p in directory.iterdir()]
        # this magic 5 represents a significant number of files that would exceed the expected OS-related metadata
        # that may be shipped in the zip that has been extracted.
        # NOTE(stephen): if we ever store the data RTX-side, this can almost certainly change
        if len(directory_contents) < 5:
            # it is clear that this CSVs aren't at *this* level, but it's possible that the extraction directory is nested
            # so we check for the name matches that we have seen thus far
            potential_extraction_names = [directory.name, quarter_downloaded.value, quarter_downloaded.drive_stats_name]
            any_directory_matches = any(map(lambda name: name in directory_contents, potential_extraction_names))
            logger.info(f"{potential_extraction_names = }")
            logger.info(f"{any_directory_matches = }")
            if any_directory_matches:
                # move all of the files from the nested directory up one level, into the owning directory
                # being careful to avoid the __MACOSX/._<filename>.csv's
                for p in directory.glob("**/20*.csv"):
                    move(p, directory / p.name)

    @staticmethod
    def _to_trainable_dataset(
        source_directory: Path, output_directory: Path, *, create_if_missing: bool = True, use_feather: bool = False
    ) -> None:
        """
        Convert the extracted zip files (dozens of CSVs with a few million data points in total) to something trainable.

        :param source_directory: directory within the zip archive was extracted
        :param output_director: directory within the trainable artifacts will be written
        """

        if create_if_missing:
            (output_directory).mkdir(exist_ok=True)

        df_all = load_quarterly_data(source_directory, feature_set=BackblazeDataset.metadata_columns + BackblazeDataset.numerical_features,)

        logger.info(f"Filtering dataframe by nans rather than imputing over the missing values. Will remove approx ~25% of the dataset")
        # remove all nans from the dataset, because it's already low-signal. Don't need to make our lives any harder.
        df_all = df_all.groupby("serial_number").filter(lambda df: df.isna().sum().sum() == 0)

        def in_output_dir(filename: str) -> Path:
            return output_directory / filename

        def write_metadata_serials(serials, name: str):
            pd.Series(serials, name="serial_number").to_csv(in_output_dir(f"metadata.{name}.csv"), index=False)

        failure_serials = df_all[df_all["failure"] == 1]["serial_number"].unique()

        write_metadata_serials(failure_serials, "failures")

        subset_sizes = [len(failure_serials) * multiplier for multiplier in [1, 2, 5, 10]]

        # NOTE: uniqueness from create_serial_id_mapping seems incompatible with the obvious route of set(A) - set(B)
        # Investigate when you have the capacity to do so.
        logger.info("Creating serial->id mappings")
        censored_serial_id_mapping = create_serial_id_mapping(df_all["serial_number"].unique(), exclusions=failure_serials)
        indices_for_subsets = make_serial_subsets(censored_serial_id_mapping["serial_number"], sample_sizes=subset_sizes)

        # persist to disk in a way that's usable for training, regardless of how the "tte", "failure" or other columns need to be adjusted
        selected = df_all[df_all["serial_number"].isin(failure_serials)]

        assemble_persistable_trainable(selected).pipe(
            lambda df: (
                df.to_feather(in_output_dir("data.failures.feather"))
                if use_feather
                else df.to_csv(in_output_dir("data.failures.csv"), index=False)
            )
        )

        if create_if_missing:
            in_output_dir("censored").mkdir(parents=True, exist_ok=True)

        for subset_serial_names in indices_for_subsets:
            selected = df_all[df_all["serial_number"].isin(subset_serial_names)]

            size = len(subset_serial_names)

            write_metadata_serials(selected["serial_number"].unique(), f"censored-{size}")

            assemble_persistable_trainable(selected).pipe(
                lambda df: (
                    df.to_feather(in_output_dir(f"censored/data-{size}.feather"))
                    if use_feather
                    else df.to_csv(in_output_dir(f"censored/data-{size}.csv"), index=False),
                )
            )

    def __init__(
        self,
        dest: str = "./.data/",
        *,
        download: bool = True,
        prepare: bool = True,
        train: bool = True,
        use_feather: bool = False,
        quarter_for_download: BackblazeQuarterName = BackblazeQuarterName._2019_Q4,
    ) -> None:
        """
        :param dest: destination directory for the downloaded, extracted, and constructed files
        :param download: whether to download the dataset. If you ignore this step, the dataset won't be prepared for usage and you will get an exception
        :param use_feather: whether to use feather to load the dataframes with feather or not.
                            Feather files are expected to have extension ".feather", and CSVs are expected to have extension ".csv"
        """
        super().__init__()
        self.train = train
        self.use_feather = use_feather
        self.data_source_quarter = quarter_for_download

        data_root = Path(dest)
        extraction_dir = "backblaze-download"
        trainables_dir = "backblaze"
        logger.debug(
            f"Initial params: \n" f"{data_root = }, {extraction_dir = }, {trainables_dir = }, {download = }, {train = }, {use_feather = }"
        )

        quarter_download_directory = Path(data_root, extraction_dir, quarter_for_download.value)
        # if the download hasn't been previously completed.
        # need a directory to look for stuff. Maybe https://github.com/explosion/catalogue but just for strings?
        # "magic numbers" explaination: the quarter should have strictly more than 89 files (think about it), but less than 100
        downloaded_and_extracted = quarter_download_directory.exists() and (89 < len(list(quarter_download_directory.glob("*.csv"))))
        logger.debug(f"{downloaded_and_extracted = }")
        extension = "feather" if use_feather else "csv"

        # only download if we have nothing. If someone deletes the download artifacts directory, that's okay
        # if we're not prepared for training or not downloaded
        will_download = download and not downloaded_and_extracted
        logger.debug(f"{will_download = }")
        if will_download:
            logger.info("Starting download")
            logger.info(f"{quarter_download_directory = }")
            logger.info(f"{extraction_dir = }")
            self._download_and_extract_zip(
                url=f"https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/{quarter_for_download.value}.zip",
                download_root=data_root,
                extraction_dir_name=Path(extraction_dir, quarter_for_download.value),  # relative to support the generality of that method
            )
            logger.info("Extraction complete")
        self._verify_download(quarter_download_directory, quarter_downloaded=quarter_for_download)


        prepared_data_dir = Path(data_root, trainables_dir, quarter_for_download.value)
        prepared_for_training = prepared_data_dir.exists() and len(list(prepared_data_dir.glob(f"**/*.{extension}"))) > 4
        logger.debug(f"{prepared_for_training = }")
        if prepare and not prepared_for_training:
            # if not prepared_for_training:
            logger.info("Starting construction of usable dataset")
            self._to_trainable_dataset(
                source_directory=quarter_download_directory,
                output_directory=prepared_data_dir,
                use_feather=use_feather,
            )
            logger.info("Completed construction of usable dataset")

        # named as such because the working Dataset isn't going to be concerned with the download artifacts, just the usable dataset
        self.data_root = prepared_data_dir
        self.data = self._load_data(train)

    def _split_dataset(self, indexable, percentage: float, is_train: bool):
        """Split the `indexable` at the index corresponding to `percentage`, giving the left-hand side if `is_train=True`"""
        split_at = calc_split_index(indexable, percentage)
        return indexable[:split_at] if is_train else indexable[split_at:]

    def _read_failure_data(self, file_name_no_ext: str, **pandas_kwargs) -> pd.DataFrame:
        """
        Thin wrapper over pandas.read_X() functions
        Read in data from a same-name-different-extension to support feathering without abstraction leaking all over the place
        """
        # branch for the two different functions and the distinct concatentation
        if self.use_feather:
            return pd.read_feather(self.data_root / f"{file_name_no_ext}.feather", **pandas_kwargs)
        else:
            return pd.read_csv(self.data_root / f"{file_name_no_ext}.csv", **pandas_kwargs)

    def _read_censored_data(self, file_name_no_ext: str, **pandas_kwargs) -> pd.DataFrame:
        """
        Thin wrapper over pandas.read_X() functions
        Read in data from a same-name-different-extension to support feathering without abstraction leaking all over the place
        """
        # branch for the two different functions and the distinct concatentation
        if self.use_feather:
            return pd.read_feather(self.data_root / "censored" / f"{file_name_no_ext}.feather", **pandas_kwargs)
        else:
            return pd.read_csv(self.data_root / "censored" / f"{file_name_no_ext}.csv", **pandas_kwargs)

    def _read_metadata_file(self, file_name_no_ext: str) -> pd.DataFrame:
        return pd.read_csv(self.data_root / f"metadata.{file_name_no_ext}.csv")

    def _load_data(self, train: bool) -> List[pd.DataFrame]:
        failure_serials = self._read_metadata_file("failures")
        # splitting the failures 80:20 train:test
        base_size = len(failure_serials)
        split_percentage = 0.8
        failure_serials = self._split_dataset(failure_serials, split_percentage, train)

        # assume the above does the right thing (it does, but it's going to have to be repeated.)
        failure_data = self._read_failure_data("data.failures")
        failure_data = failure_data[failure_data["serial_number"].isin(failure_serials)]

        # load and filter down to the data set: train vs test
        censored_data = self._read_censored_data(f"data-{base_size}")
        censored_serials = self._split_dataset(censored_data["serial_number"].unique(), split_percentage, train)

        censored_data = censored_data[censored_data["serial_number"].isin(censored_serials)]

        # want this fast, don't need a new dataframe anyways
        all_data = pd.concat((failure_data, censored_data), axis=0, copy=False)
        raw_numerical_features = self._raw_features
        statistics = all_data[raw_numerical_features].agg([np.nanmin, np.nanmax]).T
        assert len(statistics) == len(
            raw_numerical_features
        ), f"Computed summ stats along the wrong axis. Expected {len(raw_numerical_features)} got {len(statistics)}"
        self.min = statistics["nanmin"]
        self.max = statistics["nanmax"]

        return list(all_data.groupby("serial_number"))

    def _prep_features(self, df: pd.DataFrame) -> Tuple[torch.Tensor, bool]:
        """
        Convert the device-specific timeseries to a Torch tensor that matches specifications.
        Normalizes currently non-normalized fields by min-max norm, so the dataset is exposed to both
        standard (z-score) normalization and min-max norm (from -1 to 1).

        :return: Tuple of formatted tensor, and whether or not this device had a failure/intervention
        """
        df = df.copy()
        features = df.drop(columns=self.metadata_columns + ["tte"])
        is_failure = any(df["failure"])  # this is 7x faster than pandas/ on my machine.

        # min-max algorithm for a [-1, 1] spread
        features[self._raw_features] = (features[self._raw_features] - self.min) * 2 / (self.max - self.min + 1e-6) - 1

        return torch.tensor(features.values), is_failure

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, data = self.data[index]

        features, is_failure = self._prep_features(data)
        targets = torch.tensor(data["tte"].values)  # this should be correct, because TTE is computed per device

        # target tensor is meant to be a 2-D Tensor of [tte, censor]
        binary_column = torch.ones(targets.size()) if is_failure else torch.zeros(targets.size())
        targets = torch.stack((targets, binary_column), dim=1)

        return features, targets

    def __len__(self) -> int:
        return len(self.data)
