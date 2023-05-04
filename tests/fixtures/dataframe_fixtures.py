# -*- coding: utf-8 -*-
"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import List

import numpy as np
import pandas as pd
import pytest


# TODO: rename, this should be list of ndarrays
@pytest.fixture(scope="module")
def list_of_dataframes() -> List[np.ndarray]:
    "Create five dataframes with lengths between 10 and 20"
    dfs = [pd.DataFrame(np.random.random(size=(np.random.randint(10, 20), 4)), columns=list("ABCD")) for _ in range(5)]
    # return dfs
    return [x.values for x in dfs]


@pytest.fixture(scope="module")
def list_of_targets(list_of_dataframes) -> List[np.ndarray]:
    target_list = []
    for df in list_of_dataframes:
        target_list.append(np.random.random((df.shape[0], 1)))
    # return [x.values for x in target_list]
    return target_list
