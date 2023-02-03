"""
The HyperparameterSweeper and supporting types that make it easier to do a controlled grid search in pure Python.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from itertools import product
from providence_utils.merge_dict import merge_dictionaries
from typing import Callable, Dict, List, Mapping, Union

from progressbar import progressbar


Hyperparameter = Union[str, int, float]
HyperparameterList = Union[List[str], List[int], List[float]]
TrainingParams = Dict[str, Hyperparameter]
Metrics = Dict[str, float]



def nest_values(d: dict, *, sep: str = ".", signal: str = "no_iter", verbose: bool = False) -> dict:
    """Unnest the keys of a flatten (depth of 1) dictionary into a nested dictionary
    Supply `signal` (with any value) at the depth that you do not want recursed on

    >>> nest_values({"a.b": 1})
    {"a": {"b": 1}}

    >>> {"a.b.c": 1}
    {"a": {"b": {"c": 1}}}

    >>> {"a.b.c": 1, "a.b.d": 2}
    {"a": {"b": {"c": 1, "d": 2}}}
    """
    assert all(map(lambda x: isinstance(x, str), d.keys())), "Can only unnest str keys"
    # TODO(stephen): deep copy on the values. Keys are creation (as they are strings on the heap)
    if signal in d:
        return d

    output = dict()
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


def nest_keys(d: dict, *, sep: str = ".", signal: str = "no_iter") -> dict:
    """
    Nest the keys of a nested dictionary and flattens it into a depth-of-1 dictionary
    Much more like a depth-first search in implementation

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
    """
    A supportive tool to iterate all combinations of supplied hyperparameters, invoking a training
    loop with a given configuration and (optionally) producing a report.

    Name is an homage to Weights&Biases hyperparameter tuning toolkit (see: https://docs.wandb.ai/guides/sweeps)
    """
    @classmethod
    def from_dict(cls, dict_of_hyperparams: Dict[str, HyperparameterList]) -> 'HyperparameterSweeper':
        return cls(**dict_of_hyperparams)

    def __init__(self, **hyperparams: Dict[str, HyperparameterList]):
        self.hyperparameters = hyperparams

    def poll_sweeps(self):
        """
        Iterates the product of the sweeps
        - Upon encountering a list, we interpret this as a list of parameters
        - Upon encountering a dict, we iterate the keys. This is probably not desired functionality, so we recomend flattening
          dictionaries and unfolding the values on the dictionary into the top-level.
        """
        hparams_nested = nest_keys(self.hyperparameters)
        print("Parameters, post-nesting:", hparams_nested)
        config_keys = list(hparams_nested.keys())
        candidate_values = list(hparams_nested.values())
        for configuration in product(*candidate_values):
            yield nest_values(dict(zip(config_keys, configuration)))

    def sweep(self, training_loop: Callable[[TrainingParams], Metrics], with_progress_bar: bool = True) -> List[Metrics]:
        """
        Perform a hyperparameter sweep across the configurations produced by the hyperparameter (configuration) generator.
        You supply your own callable so you can leverage your own training loop code, rather than us implement one for you.

        Args:
            :param training_loop: a reference to function that leverages the configurations.
                Returns: the returned metrics are expected to obey a schema, such that they can be tabulated, ranked and
                        organized.
        
        Returns: List of metrics, corresponding to each training loop run (hopefully) based on the supplied configuration
        """

        def prefixed_config(cfg: Dict[str, Hyperparameter]):
            return {f"hparam:{k}": v for k, v in cfg.items()}

        itr = self.poll_sweeps()
        if with_progress_bar:
            itr = progressbar(itr, max_value=self._count_configurations, redirect_stdout=True)

        metrics = [{**training_loop(configuration),**prefixed_config(configuration)} for configuration in itr]
        return metrics

    @property
    def _count_configurations(self) -> int:
        """Compute the size of the sweeps, iteratively. DO NOT USE if your sweeps can't be iterated twice (e.g. list vs iterator)"""
        from functools import reduce

        return reduce(lambda x, y: x + 1, self.poll_sweeps(), 0)

