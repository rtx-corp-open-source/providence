"""
The HyperparameterSweeper and supporting types that make it easier to do a controlled grid search in pure Python.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from itertools import product
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Union

from progressbar import progressbar
from providence.utils import validate

from providence_utils.merge_dict import merge_dictionaries


Hyperparameter = Union[str, int, float]
HyperparameterList = Union[List[str], List[int], List[float]]
TrainingParams = Dict[str, Hyperparameter]
Metrics = Dict[str, float]


def nest_values(d: Mapping, *, sep: str = ".", signal: str = "no_iter", verbose: bool = False) -> dict:
    """Unnest the keys of a flatten (depth of 1) dictionary into a nested dictionary.

    Supply ``signal`` (with any value) at the depth that you do not want recursed on

    Args:
        d (Mapping): dictionary-like object with nested keys (see examples)
        sep (str, optional): Value that seperates semantic keys in ``d``. Defaults to ".".
        signal (str, optional): Value to indicate that no further nesting should occur. Defaults to "no_iter".
        verbose (bool, optional): Defalts to False.

    Returns:
        dict: nesting each key found after seperating ``sep`` in ``d``.

    Examples:
    >>> nest_values({"a.b": 1})
    {"a": {"b": 1}}

    >>> {"a.b.c": 1}
    {"a": {"b": {"c": 1}}}

    >>> {"a.b.c": 1, "a.b.d": 2}
    {"a": {"b": {"c": 1, "d": 2}}}
    """
    validate(all(map(lambda x: isinstance(x, str), d.keys())), "Can only unnest str keys")
    # TODO(stephen): deep copy on the values. Keys are creation (as they are strings on the heap)
    if signal in d:
        return dict(d)

    output: Dict[Any, Any] = dict()
    for k, v in d.items():
        if sep in k:
            top_level, rest = k.split(sep, 1)
            nested_result = nest_values({rest: v}, sep=sep, verbose=verbose)
            if verbose:
                print(f"Working top-level key '{top_level}' and nested key '{rest}'")
                print(f"Got {nested_result = }")
            output[top_level] = merge_dictionaries(output.get(top_level, {}), nested_result)
        else:
            output[k] = v
    return output


def nest_keys(d: Mapping, *, sep: str = ".", signal: str = "no_iter") -> dict:
    """Nest the keys of a nested dictionary and flattens it into a depth-of-1 dictionary.

    Args:
        d (Mapping): dict with keys that have Mappings for values, which aught to be nested.
        sep (str, optional): value to seperate keys, post-nesting. Defaults to "."
        signal (str, optional): indicator to not attempt to unnest the given dictionary. For convenience in
            ``HyperparameterSweeper``, we wrap such "stop dicts" in an array. Defaults to "no_iter".

    Returns:
        dict: depth-of-1 dictionary

    Examples:
    >>> nest_keys({"a": {"b": {"c": 1}}})
    {"a.b.c": 1}

    >>> nest_keys({"a": {"b": {"d1": 1, "d2": 2}, "c": 3}})
    {"a.b.d1": 1, "a.b.d2": 2, "a.c": 3}
    """
    out = dict()
    for k, v in d.items():
        if isinstance(v, Mapping):
            if signal in v:
                new_v = {**v}
                new_v.pop(signal)
                out[k] = [new_v]
            else:
                nested = nest_keys(v, sep=sep)
                for nested_k, nested_v in nested.items():
                    new_key = sep.join((k, nested_k))
                    out[new_key] = nested_v
        else:
            out[k] = v
    return out


class HyperparameterSweeper:
    """GridSearch over all combinations of supplied hyperparameters.

    This can also invoke a training loop with a given configuration (i.e. "cell" in the "grid") and produce a report.
    Name is an homage to Weights&Biases hyperparameter tuning toolkit (see: https://docs.wandb.ai/guides/sweeps)

    Args:
        hyperparams: key=value pairs, where
            key (str): name for the value
            value (HyperparameterList): hyperparameter values to try
    """

    @classmethod
    def from_dict(cls, dict_of_hyperparams: Mapping[str, HyperparameterList]) -> "HyperparameterSweeper":
        """Construct an instance from a pre-written ``Mapping``.

        One would use this over the constructor if your keys aren't valid keyword arguments e.g. "run-limit"
        or "search-depth"

        Args:
            dict_of_hyperparams:

        Returns:
            HyperparameterSweeper: instance
        """
        return cls(**dict_of_hyperparams)

    def __init__(self, **hyperparams: HyperparameterList):
        self.hyperparameters = hyperparams

    def poll_sweeps(self):
        """Iterate the product of the sweeps.

        Upon encountering a list, we interpret this as a list of parameters
        Upon encountering a dict, we iterate the keys. This is probably not desired functionality, so we recomend
        flattening dictionaries and unfolding the values on the dictionary into the top-level.
        We facilitate this behavior with the ``nest_keys``
        TODO(stephen): take aadditional arguments, as ``nest_keys_kwargs``

        Generates:
            dict: nested values, fully hierarchical dictionary, mirroring the structure given at sweeper construction,
                only replacing the lists of values with the single values.
        """
        hparams_nested = nest_keys(self.hyperparameters)
        print("Parameters, post-nesting:", hparams_nested)
        config_keys = list(hparams_nested.keys())
        candidate_values = list(hparams_nested.values())
        for configuration in product(*candidate_values):
            yield nest_values(dict(zip(config_keys, configuration)))

    def sweep(
        self,
        training_loop: Callable[[TrainingParams], Metrics],
        with_progress_bar: bool = True,
    ) -> List[Metrics]:
        """Perform a hyperparameter sweep across the configurations produced by the ``poll_sweep``.

        You supply your own callable so you can leverage your own training loop code, rather than us implement one for you.

        Args:
            training_loop (Callable[[TrainingParams], Metrics]): a reference to function that leverages the configurations.
                Returns: the returned metrics are expected to obey a schema, such that they can be tabulated, ranked
                and organized.

        Returns:
            List[Metrics]: metrics for each training loop run, intended to be wrt the supplied configuration
        """

        def prefixed_config(cfg: Dict[str, Hyperparameter]):
            return {f"hparam:{k}": v for k, v in cfg.items()}

        itr = self.poll_sweeps()
        if with_progress_bar:
            itr = progressbar(itr, max_value=self._count_configurations, redirect_stdout=True)

        metrics = [{**training_loop(configuration), **prefixed_config(configuration)} for configuration in itr]
        return metrics

    @property
    def _count_configurations(self) -> int:
        """Compute the size of the sweeps, iteratively. DO NOT USE if your sweeps can't be iterated twice (e.g. list vs iterator)."""
        from functools import reduce

        return reduce(lambda x, y: x + 1, self.poll_sweeps(), 0)
