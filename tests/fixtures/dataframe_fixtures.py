# -*- coding: utf-8 -*-
import pytest
from providence.utils import utils
import pandas as pd
import numpy as np
from typing import List


@pytest.fixture(scope="module")
def list_of_dataframes() -> List[pd.DataFrame]:
    "Create five dataframes with lengths between 10 and 20"
    dfs = [pd.DataFrame(np.random.random(size=(np.random.randint(10, 20), 4)), columns=list("ABCD")) for _ in range(5)]
    return [x.values for x in dfs]


@pytest.fixture(scope="module")
def list_of_targets(list_of_dataframes) -> List[pd.DataFrame]:
    target_list = []
    for df in list_of_dataframes:
        target_list.append(pd.DataFrame(np.random.random((df.shape[0], 1)), columns=["E"]))
    return [x.values for x in target_list]
