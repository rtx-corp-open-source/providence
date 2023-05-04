"""
Utilities for the FastAI-dependent code

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import inspect
import re
from typing import Any

from torch import nn


def basic_repr(flds=None):
    """Minimal ``__repr__``."""
    if isinstance(flds, str):
        flds = re.split(", *", flds)
    flds = list(flds or [])

    def _f(self):
        res = f"{type(self).__module__}.{type(self).__name__}"
        if not flds:
            return f"<{res}>"
        sig = ", ".join(f"{o}={getattr(self,o)!r}" for o in flds)
        return f"{res}({sig})"

    return _f


def delegates(to=None, keep=False):
    """Decorator: replace `**kwargs` in signature with params from ``to``.

    This version is pulled from https://www.fast.ai/2019/08/06/delegation/, not the new fastcore (or fastai) code base
    """

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop("kwargs")
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        }
        sigd.update(s2)
        if keep:
            sigd["kwargs"] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def ifnone(a: Any, b: Any) -> Any:
    """``a`` if ``a`` is not None, otherwise ``b``."""
    return b if a is None else a


def is_listy(x):
    """`isinstance(x, (tuple,list,slice))`."""
    return isinstance(x, (tuple, list, slice))


def merge(*ds):
    """Merge all dictionaries in ``ds``."""
    return {k: v for d in ds if d is not None for k, v in d.items()}


def module(*flds, **defaults):
    """Decorator to create an ``nn.Module`` using ``f`` as ``forward`` method."""
    pa = [inspect.Parameter(o, inspect.Parameter.POSITIONAL_OR_KEYWORD) for o in flds]
    pb = [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=v) for k, v in defaults.items()]
    params = pa + pb
    all_flds = [*flds, *defaults.keys()]

    def _f(f):
        class c(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                for i, o in enumerate(args):
                    kwargs[all_flds[i]] = o
                kwargs = merge(defaults, kwargs)
                for k, v in kwargs.items():
                    setattr(self, k, v)

            __repr__ = basic_repr(all_flds)
            forward = f

        c.__signature__ = inspect.Signature(params)
        c.__name__ = c.__qualname__ = f.__name__
        c.__doc__ = f.__doc__
        return c

    return _f


def pv(message, verbose: bool = False):
    if verbose:
        print(message)
