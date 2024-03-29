"""
The following three functions were vendored from fastai/fastcore as of May 13, 2021
https://github.com/fastai/fastcore/blob/d3dfa7f99c872d68836a334f0ada31edd5f2b0cc/fastcore/basics.py#L781

These functions were taken because we needed a strong way to update types at a point in time that Python
doesn't like, but fits nicely with our design. The good folks at FastAI made a way for clean and smooth
dynamic patching of classes to have better functions. It's the kind of documented "extension functions"
that I wish Python made easier to use. Yes, we can monkey-patch willy nilly, setting properties left and
right. But the other niceties of IDE / notebook support, state management, and classes/types vs instances
get goofy to handle. They did the work and we thank them 1000x over for it. But we won't import the library
because it's kind of big :)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import builtins
import functools
from copy import copy
from types import FunctionType
from types import MethodType

# maybe don't need

NoneType = type(None)
################################################################################
#
# FastAI's Fast Core @patch functionality
#
################################################################################
"""
The following three functions were vendored from fastai/fastcore as of May 13, 2021
https://github.com/fastai/fastcore/blob/d3dfa7f99c872d68836a334f0ada31edd5f2b0cc/fastcore/basics.py#L781
"""


def copy_func(f):
    """Copy a non-builtin function ``f``.

    Args:
        f (Union[FunctionType, MethodType]): some function declared as such. i.e. ``def my_func(inst: MyType)``

    Returns:
        The copy of ``f``
    """
    if not isinstance(f, FunctionType):
        return copy(f)
    fn = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    fn.__kwdefaults__ = f.__kwdefaults__
    fn.__dict__.update(f.__dict__)
    return fn


def patch_to(cls, as_prop=False, cls_method=False):
    """Create the decorating function that adds ``f`` to ``cls``.

    The ``f`` is the argument to the returned function.
    This function is best used through ``patch``, rather than invoking correctly. Kept public for additional
    experimentation.

    Args:
        as_prop (bool, optional): whether to make ``f`` a property. Only effective if ``cls_method == False``.
        cls_method (bool, optional): whether to make ``f`` bound to the class, rather than instances of the type.

    Returns:
        Function that will some argument ``f`` on invocation.
    """
    if not isinstance(cls, (tuple, list)):
        cls = (cls,)

    def _inner(f):
        for c_ in cls:
            nf = copy_func(f)
            nm = f.__name__
            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually
            for o in functools.WRAPPER_ASSIGNMENTS:
                setattr(nf, o, getattr(f, o))
            nf.__qualname__ = f"{c_.__name__}.{nm}"
            if cls_method:
                setattr(c_, nm, MethodType(nf, c_))
            else:
                setattr(c_, nm, property(nf) if as_prop else nf)
        # Avoid clobbering existing functions
        return globals().get(nm, builtins.__dict__.get(nm, None))

    return _inner


def patch(f=None, *, as_prop=False, cls_method=False):
    """Decorator: add ``f`` to the first parameter's class (based on f's type annotations).

    Args:
        f (Union[FunctionType, MethodType]): some function declared as such. i.e. ``def my_func(inst: MyType)``
        as_prop (bool, optional): whether to make ``f`` a property. Only effective if ``cls_method == False``.
        cls_method (bool, optional): whether to make ``f`` bound to the class, rather than instances of the type.

    Returns:
        The wrapped function.
    """
    if f is None:
        return functools.partial(patch, as_prop=as_prop, cls_method=cls_method)
    cls = next(iter(f.__annotations__.values()))
    if cls_method:
        cls = f.__annotations__.pop("cls")
    return patch_to(cls, as_prop=as_prop, cls_method=cls_method)(f)


def once(to_decorate):
    """Wrap ``to_decorate``, be it a function or method (class, static, or instance), to only be invoked once.

    After the first invocation, the returned function is an effective no-op.

    Args:
        to_decorate (Union[FunctionType, MethodType]): the function-like callable (but not a general ``Callable``) to
            be invoked only once.

    Returns:
        Function: wrapped ``to_decorate`` that will only perform its operation once
    """
    # valid when there's a type annotation, otherwise it'll fail
    is_method = {
        "static": isinstance(to_decorate, staticmethod),
        "class": isinstance(to_decorate, classmethod),
    }
    if is_cls_method := is_method["static"] or is_method["class"]:
        to_decorate = to_decorate.__func__
        wrapper = staticmethod if is_method["static"] else classmethod

    @functools.wraps(to_decorate)
    def _decorator(*args, **kwargs):
        if getattr(to_decorate, "_used_once", False):
            return
        ret = to_decorate(*args, **kwargs)
        setattr(to_decorate, "_used_once", True)
        return ret

    _decorator._original = to_decorate
    if is_cls_method:
        _decorator = wrapper(_decorator)

    return _decorator


################################################################################
#
# Other functions
#
################################################################################


def type_name(x) -> str:
    """The type name of argument ``x``.

    Args:
        x (Any): some Python object

    Returns:
        str: the name of the type of ``x``
    """
    if isinstance(x, (type, FunctionType, MethodType)):
        return x.__name__
    return type(x).__name__
