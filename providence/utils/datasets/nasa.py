# -*- coding: utf-8 -*-
import io
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import torch

from providence.utils import logging_utils

#### Logging ####
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] PROVIDENCE:%(levelname)s - %(name)s - %(message)s",
    handlers=[logging_utils.ch, logging_utils.fh],
)

logging.captureWarnings(True)
logger = logging.getLogger(__name__)
#################


class NasaDataSet(torch.utils.data.Dataset):
    """
    Dataset object for the NASA Turbofan Engine Degredation Simulation dataset.

    The dataset can be found here: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

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

    Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine ñ i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

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
    26)	sensor measurement  26

    Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, ìDamage Propagation Modeling for Aircraft Engine Run-to-Failure Simulationî, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.



    :Example:
    >>> from providence.utils.datasets import NasaDataset
    >>> train_data = NasaDataset(train=True)
    >>> engine0_feature, engine0 target = train_data[0]
    """

    def __init__(self, train: bool = True, mean: List[float] = None, stddev: List[float] = None, data_path: str = None):
        """
        :param train: Boolean flag for training and test datasets.
        :param mean: (Optional) Mean values of feature columns
        :param stddev: (Optional) Std. deviation values of feature columns
        :param data_path: Path or URL to directory containing the turbofan dataset
        """
        self.train = train
        self.validation_rul = None
        self.data_path = data_path

        # If no datapath is provided we will download it from github
        if data_path is None:
            self.data_path = "https://raw.githubusercontent.com/AlephNotation/nasa_turbofan/main/"

        # These are pre-computed mean and std dev for the complete training dataset.
        if mean is None:
            self.mean = [
                1.72245631e01,
                4.10352811e-01,
                9.57339325e01,
                4.85821285e02,
                5.97278898e02,
                1.46612718e03,
                1.25948690e03,
                9.89215430e00,
                1.44201815e01,
                3.59590653e02,
                2.27384447e03,
                8.67514621e03,
                1.15343657e00,
                4.41669423e01,
                3.38650301e02,
                2.34971089e03,
                8.08726502e03,
                9.05152236e00,
                2.51273110e-02,
                3.60457294e02,
                2.27378874e03,
                9.83927591e01,
                2.59455399e01,
                1.55675691e01,
            ]

        if stddev is None:
            self.stddev = [
                1.65287845e01,
                3.67992288e-01,
                1.23467959e01,
                3.04228203e01,
                4.24595469e01,
                1.18054339e02,
                1.36086086e02,
                4.26553251e00,
                6.44366063e00,
                1.74102867e02,
                1.42310999e02,
                3.74093117e02,
                1.42073445e-01,
                3.41967503e00,
                1.64160057e02,
                1.11057510e02,
                7.99949844e01,
                7.50328573e-01,
                4.99837893e-03,
                3.09876001e01,
                1.42395993e02,
                4.65165534e00,
                1.16951710e01,
                7.01722544e00,
            ]

        self.data = self._make_data(train)

    def _prep_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Format features and apply standardization

        :param data: Tensor of feature data
        :return: Formatted tensor data
        """

        # randomly truncate between 1 and 5 observations from the data
        # TODO: is truncation a good idea?
        if self.train:
            truncation = np.random.randint(data.shape[0] - 5, data.shape[0] - 1)
        else:
            truncation = data.shape[0]

        # First two columsn are `unit_number` and time so we drop them
        features = data[:truncation, 2:]

        # Standardize Features
        features = (features - self.mean) / self.stddev

        return torch.FloatTensor(features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, data = self.data[index]  # convert to numpy array
        data = data.values

        features = self._prep_features(data)
        # Get sequence length
        seq_length = len(features)

        # Get TTE
        max_time = data[-1, 1]
        # Test data ends after feature observations
        tte = int(max_time + (0 if self.train else +self.validation_rul[index]))

        # Construct target matrix
        targets = torch.arange(tte, tte - seq_length, -1)  # construct the target as a countdown

        # the target var is a 2d vector of [tte, censor]
        # since all events are observed in this specific dataset, our censor == 1

        targets = torch.stack((targets, torch.ones(targets.size())), 1)
        return features, targets

    def __len__(self):
        return len(self.data)

    def _read_data(self, file: str, index: int):
        """
        Read NASA data from url // path

        :param: File name prefix
        :param: File index
        :return: NASA engine DataFrame
        """
        file_url = f"{self.data_path}{file}_FD{index:03d}.txt"
        logger.info(f"Reading file from {file_url}")

        response = requests.get(file_url)
        data = np.loadtxt(io.BytesIO(response.content))

        if file in ["test", "train"]:
            data[:, 0] = data[:, 0] + 1000 * index

        return data

    def _format_feature_data(self, data: np.ndarray) -> pd.DataFrame:
        """
        Convert numpy array to Pandas dataframe

        :param data: Data array
        :return: Formatted dataframe
        """
        # construct column names
        id_col = "unit_number"
        time_col = "time"
        feature_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + ["sensor_measurement_{}".format(x) for x in range(1, 22)]
        column_names = [id_col, time_col] + feature_cols

        # set types so our indicies are not floats
        df = pd.DataFrame(data, columns=column_names).astype({"unit_number": int, "time": int})

        return df

    def _make_data(self, train: bool) -> List[pd.DataFrame]:
        """
        Load and preprocess the training or test dataset and return a list of
        dataframes partitioned by engine id.

        :param train: Flag to download train or test data
        :return: List of dataframes
        """

        # we're gonna flex a bit with some functional programming
        # this lambda will read raw data, format it, then concatenate it into a single dataframe
        getdata = lambda x: pd.concat(map(self._format_feature_data, [self._read_data(x, i) for i in range(1, 5)]))

        # get training data
        if train:
            data = getdata("train")

        # get test data
        else:
            data = getdata("test")
            # we will need to reference the test RUL in the __getitem__ method
            # but given that we dont need to do this with the training data we
            # will bind it to an instance variable
            self.validation_rul = np.concatenate([self._read_data("RUL", i) for i in range(1, 5)])

        return list(data.groupby(["unit_number"]))
