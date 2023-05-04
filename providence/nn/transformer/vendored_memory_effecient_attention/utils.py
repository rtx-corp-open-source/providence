"""
Utilities needed to make pytorch behave like JAX for the sake of the memory-efficient implementation.
These may be obseleted by the `functorch` module whenever it reaches v1.0 (or equivalent)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import numpy as np
import torch as pt


def dynamic_slice(x: pt.Tensor, starts, sizes) -> pt.Tensor:
    """Dynamically access finer grain slices of deeper dimensions of input tensor ``x``

    Built to get around the clunky interfaces to pt.index_select etc. See unit tests for examples

    Args:
        x (pt.Tensor): input Tensor
        starts (Union[List[int], pt.IntTensor]): list or tensor where ``len(starts) <= len(x.shape)`` and
            ``len(starts) == len(sizes)``
        sizes (Union[List[int], pt.IntTensor]): list or tensor where ``len(sizes) <= len(x.shape)`` and
            ``len(sizes) == len(starts)``

    Returns:
        pt.Tensor: view into ``x`` at the specified ranges per dimension
    """
    # start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i] - size_indices[i])
    starts = [np.clip(starts[i], 0, x.shape[i] - sizes[i]) for i in range(len(starts))]
    for i, (start, size) in enumerate(zip(starts, sizes)):
        x = pt.index_select(x, i, pt.tensor(range(start, start + size), device=x.device))
    return x


def map_pt(f, xs):
    """Map ``f`` over ``xs``, supporting tupled / parallel returns

    If ``f`` returns only one value, you will need to unpack the result i.e.
        result, = map_pt(my_func, tensors) # note the comma

    Args:
        f (Callable[[Any], Union[pt.Tensor, Tuple[pt.Tensor, ...]]]): function to map
        xs ([Iterable[pt.Tensor]]): elements for ``f`` to receive

    Returns:
        Tuple[pt.Tensor, ...]: tuple of n-many Tensors, where
            ret[i] is a stack of each result i.e. ``ret[i] = f(xs[j])[i]``, and
            n is the number of return values of ``f`` i.e. ``len(f(x))``
    """
    t = [f(x) for x in xs]
    return tuple(map(pt.stack, zip(*t)))


def scan(f, init, xs, length=None):
    """Iterative call ``f`` on ``xs``, recording intermediate results whose concatenation is final output

    Args:
        f (Union[Callable[[Any, type(xs[0])]], Function]): some function or callable which returns a Tuple[T_carry, pt.Tensor]
        init: the initial value for ``carry``
        xs: an iterable. If ``None``, behaviors as a list of ``None` of length ``length``
        length (int, optional): used if ``xs is None``. Defaults to None.

    Returns:
        Tuple[Any, pt.Tensor]: _description_
    """
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, pt.stack(ys)
