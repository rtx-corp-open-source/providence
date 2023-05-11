"""
Metrics for Providence modules, emphasizing the 'fleet'-level statistics, but functions like ``output_per_device``
make individual investigation easy

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from logging import getLogger
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
from pandas import concat as concat_dataframes
from pandas import DataFrame
from pandas import Series
from torch import Tensor
from torch.nn import Module as TorchModule

from providence.datasets import ProvidenceDataset
from providence.datasets.core import T_GroupId
from providence.distributions import SurvivalAnalysisDistribution
from providence.nn.module import ProvidenceModule

ProvidenceItem = Tuple[Tensor, Tensor]
FleetType = Union[ProvidenceDataset, Iterable[Tuple[T_GroupId, ProvidenceItem]]]

logger = getLogger(__file__)
# "SDist" is really T_SurvivalAnalysisDistribution i.e. a type "under" SurvivalAnalysisDistribution
SDist = TypeVar("SDist", bound=SurvivalAnalysisDistribution)
# SDist = SurvivalAnalysisDistribution


def output_per_device(model: ProvidenceModule, distribution: SDist, fleet_object: FleetType) -> DataFrame:
    """Calculate distribution measures for each device, carrying the unique identifier of each device.

    Args:
        model: Providence-trained model
        distribution: Module of providence.distributions
        fleet_object: ProvidenceDataset or Iterator[Tuple[Any, Tensor]]. When evaluating after training, this should
            be your holdout / test set.

    Returns:
        Dataframe: each device's distribution measures
    """
    device_arr = []
    parameter_names = distribution.parameter_names()

    model.eval()
    # now we have the id accessible in our metrics
    if isinstance(fleet_object, ProvidenceDataset):
        fleet_object = fleet_object.iter_tensors_with_id()

    for device_id, (feature_tens, prov_targets_tens) in fleet_object:
        # we execute the following on whatever device we're on initially, moving to the CPU after the fact
        # because the GPU on those matrix multiplies is just that much faster, that marshalling is worth it.
        # we send them back to the CPU later.
        params = distribution.compute_distribution_parameters(model, feature_tens)

        device_distribution_measures = {
            "mean": distribution.mean(params).cpu().numpy(),
            "median": distribution.median(params).cpu().numpy(),
            "mode": distribution.mode(params).cpu().numpy(),
        }

        # NOTE: params._fields if all params are Named Tuples.
        param_dict = {p_name: p.to("cpu") for p_name, p in zip(parameter_names, params)}

        device_df = (
            DataFrame(
                {
                    "tte": prov_targets_tens.cpu().numpy()[:, 0],
                    "censor": prov_targets_tens.cpu().numpy()[:, 1],
                }
            )
            .assign(**param_dict)
            .assign(**device_distribution_measures)
            .assign(id=device_id)
        )

        device_arr.append(device_df)

    device_output = concat_dataframes(device_arr)

    return device_output


ArrayLike = Union[Tensor, Series, np.ndarray]


def generate_metrics_table(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = True) -> DataFrame:
    """Generate the metrics table.

    Args:
        y_true (ArrayLike): groud truth values
        y_pred (ArrayLike): predictions from e.g. a model
        ignore_nans (bool, optional): whether to ignore nans in calculation of the metrics. Defaults to True.

    Returns:
        the following metrics:
        - RMSE: square-root of mean squared error
        - MSE: mean squared error
        - MFE: mean forecasting error
        - SMAPE: symmetric mean absolute percentage error
        - SMPE: symmetric mean percentage error
        - *_count
            - Prediction: the number of inferences
            - Real: the number of predictions which were non-NaN after inference
            - NaN: the number of predictions which resulted as NaN
    """
    d = {
        "RMSE": [rmse(y_true, y_pred, ignore_nans=ignore_nans)],
        "MSE": [mse(y_true, y_pred, ignore_nans=ignore_nans)],
        "MFE": [mfe(y_true, y_pred, ignore_nans=ignore_nans)],
        "SMAPE": [smape(y_true, y_pred, ignore_nans=ignore_nans)],
        "SMPE": [smpe(y_true, y_pred, ignore_nans=ignore_nans)],
        "Prediction_count": [y_pred.shape[0]],
        "Real_count": [(~np.isnan(y_pred)).sum()],
        "NaN_count": [np.isnan(y_pred).sum()],
    }
    metrics = DataFrame(data=d)

    return metrics


def fleet_metrics(
    model: ProvidenceModule,
    distribution: SDist,
    fleet_object: FleetType,
    *,
    ignore_nans: bool = False,
) -> DataFrame:
    """Calculate performance metrics for the fleet.

    Args:
        model: Providence-trained model
        distribution: Module of providence.distributions
        fleet_object: ProvidenceDataset or Iterator[Tuple[Any, Tensor]] for which to generate metrics.
            When evaluating after training, this should be your holdout / test set.

    Returns:
        Dataframe: performance metrics for the fleet
    """
    fleet = output_per_device(model, distribution, fleet_object)
    logger.info(f"Computed outputs for fleet with {ignore_nans=}")
    y_true, y_pred = fleet["tte"], fleet["mode"]

    fleet_metrics_ = generate_metrics_table(y_true, y_pred, ignore_nans=ignore_nans)

    return fleet_metrics_


def mfe(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> float:
    """Calculate mean forecast error of two vectors.

    Args:
        y_true (ArrayLike): target vector
        y_pred (ArrayLike): prediction vector
        ignore_nans (bool, optional): whether to ignore nans in calculation of the metrics. Defaults to True.

    Returns:
        float: mean forecasting error
    """
    func = np.nanmean if ignore_nans else np.mean
    mfe_ = func(y_true - y_pred, axis=0)
    return mfe_


def mse(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> ArrayLike:
    """Calculate mean square error of two vectors.

    Args:
        y_true (ArrayLike): target vector
        y_pred (ArrayLike): prediction vector
        ignore_nans (bool, optional): whether to ignore nans in calculation of the metrics. Defaults to True.

    Returns:
        float: mean squared error
    """
    func = np.nanmean if ignore_nans else np.mean
    mse_ = func(np.square(y_true - y_pred), axis=0)
    return mse_


def rmse(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> ArrayLike:
    """Calculate the square root of the mean squared error.

    Args:
        y_true (ArrayLike): target vector
        y_pred (ArrayLike): prediction vector
        ignore_nans (bool, optional): whether to ignore nans in calculation of the metrics. Defaults to True.

    Returns:
        float: root mean squared error
    """
    func = np.nanmean if ignore_nans else np.mean
    rmse_ = np.sqrt(func(np.square(y_true - y_pred)))
    return rmse_


def mape(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> ArrayLike:
    """Calculate the mean absolute percentage error of two vectors.

    As this metric is bound to [0, 1] it is an indicator epistimically akin to accuracy, though lower is better.
    Research literature suggests that values lower than ~10% are tolerable / acceptable, though lower is always better.
    Perfectly predicting models would beget a 0.

    Args:
        y_true (ArrayLike): target vector
        y_pred (ArrayLike): prediction vector
        ignore_nans (bool, optional): whether to ignore nans in calculation of the metrics. Defaults to True.

    Returns:
        float: mean aboslute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    func = np.nanmean if ignore_nans else np.mean
    mape_ = func(np.abs((y_true - y_pred) / y_true))  # if there's a division by zero, you can just call the func again
    return mape_


def smape(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> ArrayLike:
    """Calculate symmetric mean absolute percentage error of two vectors.

    As this metric is bound to [0, 1] it is an indicator epistimically akin to accuracy.
    This differs frome MAPE in that it gives a stronger indication of the directionality of the errors:
    - If there is a mixture of undershoot and overshooting, MAPE will wash that out and can produce the same values.
    - SMAPE, however, will be increase with the total error, penalizing overshooting.

    For a code example compare the outputs the two functions on the two pairs below.

    Example:

        >>> y_true = np.array([100, 110, 120, 130])
        >>> y_pred_1 = np.array([90, 105, 110, 140]) # overshooting
        >>> y_pred_2 = np.array([90, 105, 115, 120]) # strictly undershooting
        >>> y_pred_3 = np.array([90, 105, 110, 120]) # strictly undershooting, closely matching y_pred_1
        >>> print(pd.DataFrame({
        >>>     "smape": [smape(y_true, y_pred_1), smape(y_true, y_pred_2), smape(y_true, y_pred_3)],
        >>>     "mape": [mape(y_true, y_pred_1), mape(y_true, y_pred_2), mape(y_true, y_pred_3)],
        >>>     "total_error": [(y_true - y_pred_1).sum(), (y_true-y_pred_2).sum(), (y_true-y_pred_3).sum()],
        >>>     "total_abs_error": [np.abs(y_true - y_pred_1).sum(), np.abs(y_true-y_pred_2).sum(), np.abs(y_true-y_pred_3).sum()]
        >>> }).to_markdown(tablefmt='grid'))
        +----+-----------+-----------+---------------+-------------------+
        |    |     smape |      mape |   total_error |   total_abs_error |
        +====+===========+===========+===============+===================+
        |  0 | 0.0391007 | 0.0764277 |            15 |                35 |
        +----+-----------+-----------+---------------+-------------------+
        |  1 | 0.034291  | 0.0660111 |            30 |                30 |
        +----+-----------+-----------+---------------+-------------------+
        |  2 | 0.0398414 | 0.0764277 |            35 |                35 |
        +----+-----------+-----------+---------------+-------------------+

    Args:
        y_true (ArrayLike): target vector
        y_pred (ArrayLike): prediction vector
        ignore_nans (bool, optional): whether to ignore nans in calculation of the metrics. Defaults to True.

    Returns:
        float: symmetric mean absolute percentage error
    """
    s = {"pred": y_pred, "act": y_true}
    smape_df = DataFrame(data=s)
    smape_ = (smape_df.pred - smape_df.act).abs() / (smape_df.act.abs() + smape_df.pred.abs())
    if ignore_nans:
        smape_ = smape_[np.isfinite(smape_)]
    else:  # penalize making all absurd calculations
        smape_ = np.nan_to_num(smape_, nan=0, posinf=0, neginf=0)
    return np.mean(smape_)


def smpe(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> ArrayLike:
    """Calculate scaled mean percentage error of two vectors.

    Results will be in [-1, 1], giving a directional indicator of accuracy: over- and undershooting, and by how much?

    Args:
        y_true (ArrayLike): target vector
        y_pred (ArrayLike): prediction vector
        ignore_nans (bool, optional): whether to ignore nans in calculation of the metrics. Defaults to True.

    Returns:
        float: symmetric mean percentage error
    """
    s = {"pred": y_pred, "act": y_true}
    smpe_df = DataFrame(data=s)
    smpe_ = (smpe_df.pred - smpe_df.act) / (smpe_df.act.abs() + smpe_df.pred.abs())
    if ignore_nans:
        smpe_ = smpe_[np.isfinite(smpe_)]
    else:  # penalize making all absurd calculations
        smpe_ = np.nan_to_num(smpe_, nan=0, posinf=0, neginf=0)
    return np.mean(smpe_)


def score_phm08(y_true: ArrayLike, y_pred: ArrayLike, alpha_1: float = 13.0, alpha_2: float = 10.0):
    """Calcluates score function commonly used for turbofan evaluation.

    Defaults are set per the competition standard / norm.

    Args:
        y_true (ArrayLike): target vector
        y_pred (ArrayLike): prediction vector
        ignore_nans (bool, optional): whether to ignore nans in calculation of the metrics. Defaults to True.

        alpha_1 (float, optional): exponential parameter for underestimating RUL. Defaults to 13.0
        alpha_2 (float, optional): exponential parameter for overestimating RUL. Defaults to 10.0

    Returns:
        float: score as determined by the PHM '08 challenge
    """
    d = y_pred - y_true
    d_less = d[d < 0]
    d_greater = d[d >= 0]
    # NOTE: type: ignore here because mypy cannot recognize this numpy syntax.
    score_ = np.sum(np.exp(-d_less / alpha_1) - 1) + np.sum(np.exp(d_greater / alpha_2) - 1)  # type: ignore[operator]
    return score_


_ErrorMessage = str  # TODO(stephen): use NewType


class Result(NamedTuple):
    """A DataFrame computation result where the computation may fail or be a no-op.

    Args:
        succeeded (bool): whether the computation was / could be done
        result (Union[DataFrame, _ErrorMessage]): a DataFrame if ``succeeded``
    """

    succeeded: bool
    result: Union[DataFrame, _ErrorMessage]


class MetricsCalculator:
    """The core metrics found in the file, used principally to accelerate metrics computation.

    Metrics computations are accelerated by
    1. Intelligently caching at key points in the data pipeline (without exposing that to the user)
    2. Allowing the override of those cache elements, enabling users to 'compute once' outside of an
        instance of ``MetricsCalculator``, load the cache, and profit

    This class was created out of a need to generate metrics much faster at training and evaluation time.
    The functions which this class wraps all have fairly narrow, highly similar pipelines, which might have been organized
        around different idioms - but to little benefit for the effort.
        Cutting out the redundant recomputations shaves several minutes per model iteration (in a hyperparameter sweep) evalution.
            On some datasets, we're talking ~15 minutes per model, when sweeping several dozen models.
        So we reap the benefits of the simple API and accelerate with simple caching.

    Args:
        model: Providence-trained model
        distribution: Module of providence.distributions
        fleet_object: ProvidenceDataset or Iterator[Tuple[Any, Tensor]]. When evaluating after training, this should
            be your holdout / test set.
    """

    def __init__(
        self,
        model: TorchModule,
        distribution: SDist,
        fleet_object: FleetType,
        *,
        rul_stat: str = "mode",
    ) -> None:
        self.model = model
        self.dist = distribution
        self.fleet = fleet_object
        self.rul_stat = rul_stat

        import inspect

        def get_function_name():
            return inspect.currentframe().f_back.f_code.co_name

        self.get_function_name = get_function_name
        self.results_cache: dict = dict()

    def outputs_per_device(self) -> DataFrame:
        """Load and cache outputs of ``output_per_device`` (global) on this instances fields.

        Returns:
            DataFrame: fleet-level output per device
        """
        # cache the expensive operation that everyone calls.
        # NOTE: this cache is done manually so this object supports overriding by updating the results_cache.
        # Remember the rules of Python: be an adult.
        if (f_name := self.get_function_name()) in self.results_cache:
            return self.results_cache[f_name]

        res = output_per_device(self.model, self.dist, self.fleet)
        self.results_cache[f_name] = res
        return res

    def window_predictions(self, *, min_timestep: int, max_timestep: int) -> DataFrame:
        """Compute predictions for the fleet and take the subset between ``min_timestep`` and ``max_timestep``.

        This result is cached by the pair of timesteps, as this function is used frequently in visualizations
        and other, higher-order metrics.

        Args:
            min_timestep (int): the smallest timestep of interest
            max_timestep (int): the largest timestep of interest

        Returns:
            Union[Series, DataFrame]: only the output rows for devices with data between the given timesteps
        """
        if (key := (self.get_function_name(), min_timestep, max_timestep)) in self.results_cache:
            return self.results_cache[key]

        fleet = self.outputs_per_device()
        mask = fleet["tte"].between(min_timestep, max_timestep)
        result = fleet[mask]

        self.results_cache[key] = result

        return result

    def metrics_by_timestep(self, *, max_timestep: int, min_timestep: int) -> DataFrame:
        """Calculate MSE and MFE for all devices at each timestep.

        Args:
            min_timestep (int): minimum timestep of desired window
            max_timestep (int): maximum timestep of desired window

        Returns:
            DataFrame: the metrics at every timestep for every device with data between the given timesteps
        """
        window = self.window_predictions(min_timestep=min_timestep, max_timestep=max_timestep)

        metrics_windows: List[DataFrame] = []
        for timestep in window.tte.unique():
            df = window[(window["tte"] == timestep)]
            mse_ = mse(df["tte"], df["mode"])
            mfe_ = mfe(df["tte"], df["mode"])
            output = DataFrame({"tte": [timestep], "mse": [mse_], "mfe": [mfe_]})
            metrics_windows.append(output)

        metrics_window = concat_dataframes(metrics_windows)

        return metrics_window

    def error_by_timestep(self, *, max_timestep: int, min_timestep: int) -> DataFrame:
        """Calculate error between actual TTE and mean, median, and mode of output distribution for each timestep.

        Args:
            min_timestep (int): minimum timestep of desired window
            max_timestep (int): maximum timestep of desired window

        Returns:
            DataFrame: errors and ``{mean|median|mode}_overshoot`` as a boolean for whether the prediction overshot.
        """
        df = self.window_predictions(min_timestep=min_timestep, max_timestep=max_timestep)
        df_update = dict()
        df_update["error_mean"] = df["tte"] - df["mean"]
        df_update["error_median"] = df["tte"] - df["median"]
        df_update["error_mode"] = df["tte"] - df["mode"]
        df_update["mean_overshoot"] = df_update["error_mean"] < 0
        df_update["median_overshoot"] = df_update["error_median"] < 0
        df_update["mode_overshoot"] = df_update["error_mode"] < 0
        return df.assign(**df_update)

    def overshoot_summary(self) -> Result:
        """Generate summary statistics for only the predictions which have overshot actual TTE.

        Returns:
            Result: ``(True, DataFrame)``if any overshooting occurred else ``(False, explanation)``
        """
        error_by_ts = self.outputs_per_device()
        error_by_ts["error_mean"] = error_by_ts["tte"] - error_by_ts["mean"]
        error_by_ts["mean_overshoot"] = np.where((error_by_ts["error_mean"] < 0), True, False)

        mask = error_by_ts["mean_overshoot"] == True  # noqa E712 TODO: replace this with pd.bool()
        if mask.any():
            overshot = error_by_ts[mask]

            d = {
                "Smallest Overshoot": format(abs(max(overshot["error_mean"])), ".2f"),
                "Greatest Overshoot": format(abs(min(overshot["error_mean"])), ".2f"),
                "Mean Overshoot": format(abs(np.mean(overshot["error_mean"])), ".2f"),
                "Variance of Overshoot": format(np.var(overshot["error_mean"]), ".2f"),
                "STD of Overshoot": format(np.std(overshot["error_mean"]), ".2f"),
                "Percent of Predictions Overshot": [round(100 * (len(overshot)) / (len(error_by_ts)), 2)],
                "MFE of Overshoot": format(mfe(overshot["tte"], overshot["mean"]), ".2f"),
                "SMPE of Overshoot": [smpe(overshot["tte"], overshot["mean"])],
            }

            summary = DataFrame(data=d)

            return Result(True, summary)
        else:
            return Result(False, "No predictions overshot actual TTE")

    def percent_overshot_by_tte(self, *, max_timestep: int, min_timestep: int) -> DataFrame:
        """Calculate how many predictions have overshot the actual TTE at each timestep for mean, median, and mode.

        Args:
            max_timestep (int): maximum timestep of desired window
            min_timestep (int): minimum timestep of desired window

        Returns:
            DataFrame: percentage overshot at each time step, broken down by prediction statistic
        """
        error_by_ts = self.error_by_timestep(max_timestep=max_timestep, min_timestep=min_timestep)

        percent_overshot = []
        for i in error_by_ts.tte.unique():
            df = error_by_ts[(error_by_ts["tte"] == i)]
            overshot_mean = (df["mean_overshoot"].astype("bool")).sum() / (len(df))
            overshot_median = (df["mean_overshoot"].astype("bool")).sum() / (len(df))
            overshot_mode = (df["mean_overshoot"].astype("bool")).sum() / (len(df))
            overshot_df = DataFrame(
                {
                    "TTE": [i],
                    "%_Overshot_Mean": [overshot_mean],
                    "%_Overshot_Median": [overshot_median],
                    "%_Overshot_Mode": [overshot_mode],
                }
            )
            percent_overshot.append(overshot_df)

        overshot_by_tte = concat_dataframes(percent_overshot)

        return overshot_by_tte

    def metrics_per_device(self, *, tte_cutoff: int = 150, rul_stat: Optional[str] = None) -> DataFrame:
        """Compute our metrics with respect to temporal windowing.

        Example:
            pseudo code, as we do the vectorized computation in full::

                MSE_divisor = 1/(EOL - FPT)
                MSE_dividend = sum(
                    map( lambda i: (preds[i] - truth[i]) ** 2, # squared error
                        range(FPT, EOL + 1) # interval of the summation
                    )
                )
                MSE = MSE_dividend / MSE_divisor

        Args:
            tte_cutoff (int): First predicting time, before which we don't care about the model's prediction metrics.
            rul_stat (Optional[str], optional): average statistic, one of mean, median, or mode.
                Defaults to None and uses if so ``self.rul_stat``.

        Returns:
            DataFrame: metrics per device, within TTE range of [0, tte_cutoff]
        """
        rul_stat = rul_stat or self.rul_stat
        outputs = self.outputs_per_device()

        per_device = []
        for id, df in outputs.groupby("id"):
            # there's a way to do a groupby-map, but I don't feel like spending an hour
            # recalling the difference between map, apply, applymap, transform, agg, ... Pandas is complicated
            mask = df["tte"].between(0, tte_cutoff + 1)
            truth = df[mask]["tte"]
            pred = df[mask][rul_stat]
            device_metrics = generate_metrics_table(truth, pred)
            device_metrics["id"] = id

            per_device.append(device_metrics)

        metrics = concat_dataframes(per_device)

        return metrics

    def fleet_metrics(
        self,
        *,
        tte_cutoff: int = 150,
        rul_stat: Optional[str] = None,
        method: str = "brutal",
    ) -> DataFrame:
        """Comupute fleet metrics leveraging the internal cache.

        Otherwise, this implementation only adds two methods of evaluation, "charitable"/"naive" or "brutal".
        The former treats the tte_cutoff as an absolute reference from TTE=0, so devices may fall out of the dataset
        altogether. The latter ("brutal") takes all entities and make ``tte_cutoff`` relative for each, potentially
        finding more errors.
        # TODO(stephen): rename "absolute" and "relative"

        Args:
            tte_cutoff (int, optional): The oldest timestep to start considering for metrics calculations.
                Defaults to 150.
            rul_stat (Optional[str], optional): average statistic, one of mean, median, or mode.
                Defaults to None and uses if so ``self.rul_stat``.
            method (str, optional): _description_. Defaults to "brutal".

        Returns:
            DataFrame: fleet metrics; see extended summary for details
        """
        outputs = self.outputs_per_device()
        if method == "charitable" or method == "naive":
            # old implementation with naive cut-off time
            outputs = outputs[outputs["tte"].between(0, tte_cutoff + 1)]
        else:
            # new: cutoff is applied per device
            # make sure the tte's are counting down (e.g. descending)
            outputs = outputs.sort_values(["id", "tte"], ascending=[False, False])
            # grab the end of the sequence, by that tte
            outputs = outputs.groupby("id").tail(tte_cutoff)

        rul_stat = rul_stat or self.rul_stat
        fleetwide = generate_metrics_table(outputs["tte"], outputs[rul_stat], ignore_nans=False).assign(
            # TODO(stephen): this should be a direct call to ``score_phm08``
            score=self.score(rul_stat=rul_stat)
        )
        return fleetwide

    def score(self, *, rul_stat: Optional[str] = None) -> float:
        """Score the fleet across the final time step (per device) using ``rul_stat`` for the RUL extraction.

        Args:
            rul_stat (Optional[str]): One of mean, median, and mode.
                Defaults to None and uses if so ``self.rul_stat``.

        Returns:
            float: score per the PHM '08 challenge
        """
        outputs = self.outputs_per_device()

        rul_stat = rul_stat or self.rul_stat

        # just to be sure, sort in ascending order, then take the first row of each device's TTE and RUL stat
        tte_and_rul_pred = outputs.sort_values(["id", "tte"], ascending=True).groupby("id").head(1)[["tte", rul_stat]]
        tte = tte_and_rul_pred["tte"]
        rul_pred = tte_and_rul_pred[rul_stat]
        return score_phm08(tte, rul_pred)
