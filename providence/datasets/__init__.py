"""
This module is everything related to datasets that we could care to use, that isn't already accounted for by (say)
a DataLoader. Herein you will find
1. Constructors for the NASA and Backblaze datasets
2. Many different utilities for working with DataFrames.
3. Unifying encapsulation of a ProvidenceDataset, which should allow you to easily support of new datasets
  - All you really have to do is load a dataframe (say) with `assemble_persistable_trainable(...)`

Throughout working with this code, I found a great distaste for predicating weth `pd.` only to use two functions from
the library. Zen of Python encourages otherwise i.e. explict statement of what's being used, and we do that here.

The code generally follows an idiom of 
1. XYZDataset - the constructor for a particular split of XYZ i.e. NASA or Backblaze
2. XYZDatasets - the contructor that will return the train-test split for NASA; train-test or train-val-test for Backblave
3. _XYZDataset - a thin constructor of ProvidenceDataset, with prefilled arguments for the TTE, event indicitor, and such for the given dataset.

There are smaller helper functions that you are free to use. As will be seen, the pattern is only:
1. Load the data
2. Split into either train-test or train-validation-test
3. Preprocess appropriately
4. Wrap in a ProvidenceDataset

Anyone working with just the helper functions could follow a similar pattern to easily integrate a new dataset.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from . import *

from .adapters import *
from .backblaze import *
from .core import *
from .nasa import *
from .utils import *