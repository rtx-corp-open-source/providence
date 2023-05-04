"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.datasets import BackblazeDataset
from providence.datasets import DataSubsetId
from providence.datasets.adapters import BackblazeQuarter
from providence.paper_reproductions import BackblazeTransformer
from providence.paper_reproductions import GeneralMetrics
from providence.training import EpochLosses
from providence.training import LossAggregates

import pytest


@pytest.mark.requires_data
def test_working():
    ds = BackblazeDataset(DataSubsetId.Test, quarter=BackblazeQuarter._2019_Q4)
    model = BackblazeTransformer()  # .to(dtype=torch.float64) # because default floats in losses default to float64
    lossaggs = LossAggregates([], [])
    for n in range(10, 1, -1):
        lossaggs.append_losses(EpochLosses(n**2, 2 * n))

    metrics = GeneralMetrics(model, ds, lossaggs)
    assert True
