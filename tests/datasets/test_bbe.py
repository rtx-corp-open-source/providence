"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import pytest

from providence.datasets import BackblazeExtendedDataset
from providence.datasets.adapters import BackblazeQuarter

@pytest.mark.requires_data
class TestBackblazeExtended:
    def test_init__works(self):
        ret = BackblazeExtendedDataset(
            BackblazeQuarter._2019_Q3,
            BackblazeQuarter._2019_Q4,
            connsider_validation=True,
        )
        assert ret is not None, "Valid construction would result in a valid return"

    def test_init__fails(self):
        with pytest.raises(ValueError) as e:
            BackblazeExtendedDataset(BackblazeQuarter._2019_Q3, connsider_validation=True)

        assert e.match("single quarter")
