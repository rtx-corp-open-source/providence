"""
This module is home to multiple utility functions that we found need for over the course of developing the library
or wrote multiple times during experimentation (`now_dt_string()` especially).

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import inspect
import logging
from crypt import mksalt
from logging import getLogger
from operator import itemgetter
from pathlib import Path
from typing import Any  # noqa: F401
from typing import Callable  # noqa: F401
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch as pt
from pandas import DataFrame
from pandas import read_csv
from pandas import read_feather


logger = getLogger(__file__)


def cached_dataframe(operation: Callable[[], DataFrame], path: Union[str, Path]):
    """Cache a DataFrame on disk at the path.

    If a file is found, we attempt to load that.
    Otherwise we run the operation that would produce a DataFrame, cache it, and return the DataFrame

    Args:
        operation (Callable[[], DataFrame]): an operation that produces a DataFrame
        path (PathLike): Second argument description..

    Returns:
        Return description line.

    Raises:
        AnyError: If anything bad happens
    """
    path = Path(path)
    suffix = path.suffix[1:]  # all suffixes are preceded by '.' i.e. Path('foo.csv').suffix == '.csv'
    if path.is_file():
        read_func = read_csv if suffix == "csv" else read_feather  # type: function  # noqa: F821
        logger.info(f"Found cached dataframe at {path}")
        return read_func(path)  # type: ignore[operator]
    elif suffix in {"csv", "feather"}:
        df = operation()
        path.parent.mkdir(parents=True, exist_ok=True)  # give us a write path if we don't have one
        if suffix == "csv":
            df.to_csv(path, index=False)
        else:
            df.to_feather(path)
        return df
    else:
        raise ValueError(f"Invalid path suffix in {path.as_posix() = }. Only 'csv' and 'feather' are supported")


def clear_memory(*models):
    """Attempt to reclaim the memory from PyTorch and the ``models`` constructed.

    Args:
        models: nn.Modules to delete, but can be any objects that are claiming memory
    """
    import torch
    import gc

    for m in models:
        del m
    torch.cuda.memory.empty_cache()
    gc.collect()


def configure_logger_in_dir(
    directory: Union[str, Path], *, logger_name: Optional[str] = None, capture_warnings: bool = True
) -> logging.Logger:
    """Create and configure a logger to like logs to ``log.log`` under the given ``directory``.

    Args:
        directory (PathLike): Relative or absolute path
        logger_name (Optional[str]): Second argument description..
        capture_warnings (bool): Whether the logger should capture the warnings

    Returns:
        Return the configured Logger

    Raises:
        PermissionError: If you are trying to access a ``directory`` that your user doesn't have permission for.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    log_file = Path(directory, "log.log").absolute().expanduser().as_posix()

    file_handler = logging.FileHandler(log_file)
    logging.captureWarnings(capture_warnings)

    # calling this every time is *not* great, but logging in Python is already a chore. This minimizes the surface area
    # other loggers (previously instantiated) should be fine with whatever they're doing...
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] PROVIDENCE:%(levelname)s - %(name)s - %(message)s",
        handlers=[file_handler],
    )

    _logger = logging.getLogger(logger_name or "generated-logger-" + ("".join([mksalt() for _ in range(5)])))
    _logger.setLevel(logging.DEBUG)
    # _logger.addHandler(file_handler)
    return _logger


def cps(arg, *funcs) -> Any:
    """Treat the block as a Continuation-Passing Style code such that.

    # NOTE(stephen): this function is to be removed from this project. It's not used anywhere but was facilitating experiments
    >>> cps(x, square) == square(x)
    >>> cps(x, square, sqrt) == x
    >>> cps(x, square, (lambda y: logger.debug(f"After square: {y}")) == None # and performs the obvious

    This is reminiscent to the ``pipe`` vs ``compose`` paradigm, and makes for self-starting blocks that are readable.

    Args:
        arg: first argument to the many functions
        funcs: functions to be invoked in the order given

    Returns:
        The final result of calling funcs[i] with the result of the output of funcs[i-1]
    """
    out = arg
    for f in funcs:
        out = f(out)
    return out


T_arg = TypeVar("T_arg")


def also(arg: T_arg, block: Callable[[T_arg], Any]) -> T_arg:
    """Executes ``block`` on ``arg`` and returns ``arg``.

    Example:

        >>> x = also(calculate_x(), print)
        >>> model_outputs = also(model(X), log_inference_metrics)

    Args:
        arg : some (potentially None) argument
        block : a (likely side-effectful) callable

    Returns:
        Return description line.

    Raises:
        Error: Any error that could be emitted by ``block``
    """
    block(arg)  # result dropped
    return arg


T_co = TypeVar("T_co", covariant=True)


def multilevel_sort(objects: List[T_co], *, keys: List[str], sort_descending: Optional[List[bool]] = None):
    """In-place sort ``objects`` by the values return by ``keys``, in the order given.

    Args:
        objects (list): the objects that you want to sort
        keys (List[str]):
        sort_descending: (Optional[List[bool]]): a list of the same length as ``keys``, where ``True`` indicates you
            want the property at the given index sorted from least to greatest and ``False`` is vice versa.

    Returns:
        Returns a sorted list of the objects

    Raises:
        KeyError: If any of the given ``keys`` is not found in a given element of ``objects``.
    """
    if sort_descending is None:
        sort_descending = [False] * len(keys)
    for key, descending in reversed(list(zip(keys, sort_descending))):
        objects.sort(key=itemgetter(key), reverse=descending)  # type: ignore[arg-type]
    return objects


def name_and_args() -> List[Tuple[str, Any]]:
    """Helper function to print args of the function it is called in.

    Returns:
        List[Tuple[str, Any]]: arg names and values
    """
    caller = inspect.stack()[1][0]
    args, _, _, values = inspect.getargvalues(caller)
    return [(i, values[i]) for i in args]


def now_dt_string() -> str:
    """Calculate and return a datetime string for this instant, truncated to the single second in ISO 8601 format."""
    from datetime import datetime

    # training isn't fast enough to record the sub-second microsecond
    return datetime.now().replace(microsecond=0).isoformat()


def remove_keys_from_dict(d: dict, keys: Iterable[str]) -> None:
    """Delete ``keys`` from dictionary ``d``.

    Args:
        d (dict): dictionary which has key-value pairs to delete
        keys (Iterable[str]): the keys to delet from the dictionary
    """
    for k in keys:
        if k in d:
            del d[k]


def save_df(df: DataFrame, path: Union[str, Path], *, root=".") -> None:
    """Persist ``df`` to ``path`` under ``root``, using `DataFrame.to_csv(...)`

    Args:
        df (DataFrame): Some DataFrame you would like to save
        path (Union[str, Path]): a ``str`` or ``Path`` defining the subdirectory under ``root``

    Raises:
        PermissionError: If you are trying to access a ``root`` or ``path`` that your user doesn't have permission for.
    """
    path = Path(root, path)
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)


def set_seed(seed: int) -> None:
    """Set the numerical seed across every possible system we utilize: NumPy, PyTorch.

    TL;DR: this is the most stable version of setting global seeds we have found. If it's not good enough, we tried.

    This is our best bet at reproducibility with PyTorch. It is genuinely frustrating that getting reproducible research
    results requires freezing so many global state variables.
    Even with these changes, there seems to be variance between models runs, even though there aren't *any* random variable
    usages in the Providence code base.
    (Rather, we use a random sampling to initialize variables, which should be deterministic when seeds are set.)
    Some of this problem might be native to Pytorch, and other technologies may be necessary to get the required determinism.
    Some of it may be model-dependent in a way that is undocumented. For more see:
    - https://pytorch.org/docs/stable/notes/randomness.html
    - https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097

    Args:
        seed (int): the numerical seed to use
    """
    import random
    import os
    import numpy.random as np_rnd
    import torch as pt

    random.seed(seed)
    if seed > (2**32) - 1:
        # python's seed must be in [0, 4294967295] = [0, 2**32 - 1], so we'll put them here
        # if the above throws, this would too - but downstream, so we prevent here.
        python_hash_and_numpy_seed = seed % (2**32)
        # numpy's seed must similarly but an unsigned 32-bit integer, and I don't like the idea
        # of clamping (and using the same seed too many times), so we apply the same transform
        np_rnd.seed(python_hash_and_numpy_seed)
        os.environ["PYTHONHASHSEED"] = str(python_hash_and_numpy_seed)
    else:
        # there's another way to do this that always invokes the mod operator... Clarity > one less branch
        np_rnd.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    pt.set_rng_state(pt.manual_seed(seed).get_state())
    pt.cuda.manual_seed(seed)
    pt.backends.cudnn.deterministic = True
    try:
        pt.backends.cudnn.benchmark = False
    except Exception as e:
        # this has failed in the past, but I don't know why it fails sometimes and doesn't others...
        print(
            f"Exception in set_seed({seed}): couldn't set cudnn.benchmark = False. Caught exception:",
            e,
        )


def TODO(message: str = None):
    """Raises an error to display a TODO message that should be resolved before utilizing the thing at the call site.

    Raises:
        NotImplementedError: containing the TODO message
    """
    raise NotImplementedError(message)


def validate(b: bool, message):
    """Assert with an error message, raising a ValueError rather than an AssertionError.

    Args:
        b (bool): the evaluation of some consideration
        message (Union[Callable[[], str], str]): a (potentially lazily evaluated) string to wrap in the error message

    Raises:
        ValueError: if ``not b``
    """
    if not b:
        raise ValueError(message() if callable(message) else message)


def where_are_your_parameters(model: pt.nn.Module) -> pt.device:
    """Find ``model`` parameters' torch device.

    Args:
        model (nn.Module): a Pytorch model

    Returns:
        torch.device: the device whereon the first parameters of the model live.
    """
    return next(model.parameters()).device
