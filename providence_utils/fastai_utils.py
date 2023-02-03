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
    "Minimal `__repr__`"
    if isinstance(flds, str):
        flds = re.split(', *', flds)
    flds = list(flds or [])

    def _f(self):
        res = f'{type(self).__module__}.{type(self).__name__}'
        if not flds:
            return f'<{res}>'
        sig = ', '.join(f'{o}={getattr(self,o)!r}' for o in flds)
        return f'{res}({sig})'

    return _f


def custom_dir(c, add):
    return dir(type(c)) + list(c.__dict__.keys()) + add

def delegate_attr(self, k, to):
    "Use in `__getattr__` to delegate to attr `to` without inheriting from `GetAttr`"
    if k.startswith('_') or k==to: raise AttributeError(k)
    try: return getattr(getattr(self,to), k)
    except AttributeError: raise AttributeError(k) from None


class GetAttr:
    "Base class for attr accesses in `self._xtra` passed down to `self.default`"
    "https://github.com/fastai/fastcore/blob/8185a912b2a9ffc7884d0d8ae16c589856bfad7e/fastcore/basics.py#L485"
    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _default='default'
    def _component_attr_filter(self,k):
        if k.startswith('__') or k in ('_xtra',self._default): return False
        xtra = getattr(self,'_xtra',None)
        return xtra is None or k in xtra
    def _dir(self): return [k for k in dir(getattr(self,self._default)) if self._component_attr_filter(k)]
    def __getattr__(self,k):
        if self._component_attr_filter(k):
            attr = getattr(self,self._default,None)
            if attr is not None: return getattr(attr,k)
        raise AttributeError(k)
    def __dir__(self): return custom_dir(self,self._dir())
#     def __getstate__(self): return self.__dict__
    def __setstate__(self,data): self.__dict__.update(data)


def delegates(to=None, keep=False):
    """Decorator: replace `**kwargs` in signature with params from `to`
    
    This version is pulled from https://www.fast.ai/2019/08/06/delegation/, not the new fastcore (or fastai) code base
    """
    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        }
        sigd.update(s2)
        if keep:
            sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def is_listy(x):
    "`isinstance(x, (tuple,list,slice))`"
    return isinstance(x, (tuple, list, slice))


def merge(*ds):
    "Merge all dictionaries in `ds`"
    return {k: v for d in ds if d is not None for k, v in d.items()}


def module(*flds, **defaults):
    "Decorator to create an `nn.Module` using `f` as `forward` method"
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
