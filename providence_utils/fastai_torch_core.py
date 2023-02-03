"""
Pulled from
https://github.com/fastai/fastai/blob/c5b9aa050e1ed382b40a7f772a07d74453fdcacb/fastai/torch_core.py#L619

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from providence_utils.fastcore_meta import PrePostInitMeta
from torch import nn

class Module(nn.Module, metaclass=PrePostInitMeta):
    "Same as `nn.Module`, but no need for subclasses to call `super().__init__`"
    def __pre_init__(self, *args, **kwargs): super().__init__()
    def __init__(self): pass
