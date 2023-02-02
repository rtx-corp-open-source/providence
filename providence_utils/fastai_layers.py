"""
Layers from the FastAI library that were utilitized by the TSAI code that we borrowed.
Kept for educational purposes.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import torch as pt
from torch import Tensor, nn
from torch.nn import functional as F

from providence_utils.fastai_torch_core import Module
from providence_utils.fastai_utils import is_listy, module


def init_lin_zero(m):
    if isinstance(m, (nn.Linear)):
        if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 0)
    for l in m.children(): init_lin_zero(l)

lin_zero_init = init_lin_zero

class MaxPPVPool1d(Module):
    "Drop-in replacement for AdaptiveConcatPool1d - multiplies nf by 2"
    def forward(self, x):
        _max = x.max(dim=-1).values
        _ppv = pt.gt(x, 0).sum(dim=-1).float() / x.shape[-1]
        return pt.cat((_max, _ppv), dim=-1).unsqueeze(2)

class AdaptiveConcatPool1d(Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"
    def __init__(self, size=None):
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)
    def forward(self, x): return pt.cat([self.mp(x), self.ap(x)], 1)


class AdaptiveWeightedAvgPool1d(Module):
    '''Global Pooling layer that performs a weighted average along the temporal axis

    It can be considered as a channel-wise form of local temporal attention. Inspired by the paper:
    Hyun, J., Seong, H., & Kim, E. (2019). Universal Pooling--A New Pooling Method for Convolutional Neural Networks. arXiv preprint arXiv:1907.11440.'''

    def __init__(self, n_in, seq_len, mult=2, n_layers=2, ln=False, dropout=0.5, act=nn.ReLU(), zero_init=True):
        layers = nn.ModuleList()
        for i in range(n_layers):
            inp_mult = mult if i > 0 else 1
            out_mult = mult if i < n_layers -1 else 1
            p = dropout[i] if is_listy(dropout) else dropout
            layers.append(LinLnDrop(seq_len * inp_mult, seq_len * out_mult, ln=False, p=p,
                                    act=act if i < n_layers-1 and n_layers > 1 else None))
        self.layers = layers
        self.softmax = SoftMax(-1)
        if zero_init: init_lin_zero(self)

    def forward(self, x):
        wap = x
        for l in self.layers: wap = l(wap)
        wap = self.softmax(wap)
        return pt.mul(x, wap).sum(-1)

# https://docs.fast.ai/torch_core.html#tensorbase
# class TensorBase(Tensor):
#     "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
#     debug,_opt = False,defaultdict(list)
#     def __new__(cls, x, **kwargs):
#         res = cast(tensor(x), cls)
#         for k,v in kwargs.items(): setattr(res, k, v)
#         return res

#     @classmethod
#     def _before_cast(cls, x): return tensor(x)
#     def __repr__(self): return re.sub('tensor', self.__class__.__name__, super().__repr__())

#     def __reduce_ex__(self,proto):
#         torch.utils.hooks.warn_if_has_hooks(self)
#         args = (self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
#         if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
#         args = args + (self.requires_grad, OrderedDict())
#         f = torch._utils._rebuild_qtensor if self.is_quantized else  torch._utils._rebuild_tensor_v2
#         return (_rebuild_from_type, (f, type(self), args, self.__dict__))

#     @classmethod
#     def register_func(cls, func, *oks): cls._opt[func].append(oks)

#     @classmethod
#     def __torch_function__(cls, func, types, args=(), kwargs=None):
#         if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
#         if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
#         res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
#         dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
#         if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)
#         return res

#     def new_tensor(self, size, dtype=None, device=None, requires_grad=False):
#         cls = type(self)
#         return self.as_subclass(Tensor).new_tensor(size, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

#     def new_ones(self, data, dtype=None, device=None, requires_grad=False):
#         cls = type(self)
#         return self.as_subclass(Tensor).new_ones(data, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

#     def new(self, x=None):
#         cls = type(self)
#         res = self.as_subclass(Tensor).new() if x is None else self.as_subclass(Tensor).new(x)
#         return res.as_subclass(cls)
    
#     def requires_grad_(self, requires_grad=True):
#         # Workaround https://github.com/pytorch/pytorch/issues/50219
#         self.requires_grad = requires_grad
#         return self


@module(full=False)
def Flatten(self, x: Tensor):
    "Flatten `x` to a single dimension, e.g. at end of a model. `full` for rank-1 tensor"
    # NOTE it's a little weird that Tensor{Base} was used, when a view will suffice and *is* correct.
    return (x.view(-1) if self.full else x.view(x.size(0), -1))



class GAP1d(Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Flatten()
    def forward(self, x):
        return self.flatten(self.gap(x))

class GACP1d(Module):
    "Global AdaptiveConcatPool + Flatten"
    def __init__(self, output_size=1):
        self.gacp = AdaptiveConcatPool1d(output_size)
        self.flatten = Flatten()
    def forward(self, x):
        return self.flatten(self.gacp(x))


class GAWP1d(Module):
    "Global AdaptiveWeightedAvgPool1d + Flatten"
    def __init__(self, n_in, seq_len, n_layers=2, ln=False, dropout=0.5, act=nn.ReLU(), zero_init=False):
        self.gacp = AdaptiveWeightedAvgPool1d(n_in, seq_len, n_layers=n_layers, ln=ln, dropout=dropout, act=act, zero_init=zero_init)
        self.flatten = Flatten()
    def forward(self, x):
        return self.flatten(self.gacp(x))


class LinLnDrop(nn.Sequential):
    "Module grouping `LayerNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, ln=True, p=0., act=None, lin_first=False):
        layers = [nn.LayerNorm(n_out if lin_first else n_in)] if ln else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not ln)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)


class Pad1d(nn.ConstantPad1d):
    def __init__(self, padding, value=0.):
        super().__init__(padding, value)

def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return pt.sigmoid(x) * (high - low) + low

@module('low','high')
def SigmoidRange(self, x):
    "Sigmoid module with range `(low, high)`"
    return sigmoid_range(x, self.low, self.high)


class SoftMax(Module):
    "SoftMax layer"
    def __init__(self, dim=-1):
        self.dim = dim
    def forward(self, x):
        return F.softmax(x, dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'


class Transpose(Module):
    def __init__(self, *dims, contiguous=False): self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
    def __repr__(self):
        if self.contiguous: return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])}).contiguous()"
        else: return f"{self.__class__.__name__}({', '.join([str(d) for d in self.dims])})"

