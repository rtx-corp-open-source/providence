"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Tuple

from providence.datasets import ProvidenceDataset
from providence.nn.module import ProvidenceModule

from pandas import DataFrame, concat
import pytest
import torch as pt

@pytest.fixture(scope="module")
def simple_providence_ds():
    from random import normalvariate
    df_length = 5
    e1 = DataFrame(
        {
            'id': [1] * df_length,
            'a': [normalvariate(0, 1) for _ in range(df_length)],
            'b': [normalvariate(1, 2) for _ in range(df_length)]
        }
    ).assign(countdown=range(df_length, 0, -1))
    e2 = e1 * 2
    e3 = (e1**2).assign(id=3) # don't want squared ids
    df = concat([e1, e2, e3]).assign(failure=1) # everything is a failure column.
    features = 'a b'.split()
    return ProvidenceDataset(df, grouping_field='id', feature_columns=features, tte_column='countdown')

@pytest.fixture(scope="module")
def simple_weibull_model() -> ProvidenceModule:
    from providence.nn.weibull import WeibullHead
    class SimplestLinear(pt.nn.Module):
        def __init__(self):
            super().__init__()
            self.out = WeibullHead()
            self.device = pt.device('cpu')

        def forward(self, input, input_lengths) -> Tuple[pt.Tensor, pt.Tensor]:
            return self.out(input)

    return SimplestLinear() #.to(pt.float64)
