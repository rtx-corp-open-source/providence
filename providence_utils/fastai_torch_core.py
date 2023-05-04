"""
PrePostInitMeta pulled from
https://github.com/fastai/fastcore/blob/90efea31c9b24cde9b9d9800b65dc4e74616bf13/fastcore/meta.py

Module pulled from
https://github.com/fastai/fastai/blob/c5b9aa050e1ed382b40a7f772a07d74453fdcacb/fastai/torch_core.py#L619

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from torch import nn

import inspect


# %% ../nbs/07_meta.ipynb 7
def _rm_self(sig):
    sigd = dict(sig.parameters)
    sigd.pop("self")
    return sig.replace(parameters=sigd.values())


# %% ../nbs/07_meta.ipynb 8
class FixSigMeta(type):
    """A metaclass that fixes the signature on classes that override ``__new__``."""

    def __new__(cls, name, bases, dict):
        res = super().__new__(cls, name, bases, dict)
        if res.__init__ is not object.__init__:
            res.__signature__ = _rm_self(inspect.signature(res.__init__))
        return res


# %% ../nbs/07_meta.ipynb 23
class PrePostInitMeta(FixSigMeta):
    """A metaclass that calls optional ``__pre_init__`` and ``__post_init__`` methods."""

    def __call__(cls, *args, **kwargs):
        res = cls.__new__(cls)
        if type(res) == cls:
            if hasattr(res, "__pre_init__"):
                res.__pre_init__(*args, **kwargs)
            res.__init__(*args, **kwargs)
            if hasattr(res, "__post_init__"):
                res.__post_init__(*args, **kwargs)
        return res


class Module(nn.Module, metaclass=PrePostInitMeta):
    """Same as ``nn.Module``, but no need for subclasses to call ``super().__init__``."""

    def __pre_init__(self, *args, **kwargs):
        super().__init__()

    def __init__(self):
        pass
