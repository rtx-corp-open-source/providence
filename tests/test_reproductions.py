"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.datasets.adapters import BackblazeQuarter
from providence.datasets import BackblazeDataset, DataSubsetId
from providence.training import EpochLosses, LossAggregates
from providence.paper_reproductions import GeneralMetrics, BackblazeTransformer

def test_working():
    ds = BackblazeDataset(DataSubsetId.Test, quarter=BackblazeQuarter._2019_Q4)
    model = BackblazeTransformer() #.to(dtype=torch.float64) # because default floats in losses default to float64
    lossaggs = LossAggregates([], [])
    for n in range(10, 1, -1):
        lossaggs.append_losses(EpochLosses(n ** 2, 2 * n))
    
    metrics = GeneralMetrics(model, ds, lossaggs)
    assert True
