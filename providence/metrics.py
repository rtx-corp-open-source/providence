"""
Metrics for Providence modules, emphasizing the 'fleet'-level statistics, but functions like `output_per_device`
make individual investigation easy

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from logging import getLogger
from typing import Iterator, NamedTuple, Optional, Tuple, TypeVar, Union

import numpy as np
from pandas import DataFrame, Series
from pandas import concat as concat_dataframes
from providence.datasets import ProvidenceDataset
from providence.distributions import SurvivalAnalysisDistribution
from providence.nn.module import ProvidenceModule
from torch import Tensor
from torch.nn import Module as TorchModule

ProvidenceItem = Tuple[Tensor, Tensor]
FleetType = Union[ProvidenceDataset, Iterator[Tuple[str, ProvidenceItem]]]

logger = getLogger(__file__)
# "SDist" is really T_SurvivalAnalysisDistribution i.e. a type "under" SurvivalAnalysisDistribution
SDist = TypeVar('SDist', bound=SurvivalAnalysisDistribution)
DistParams = TypeVar('DistParams', bound=SurvivalAnalysisDistribution.Params, covariant=True)


def output_per_device(model: ProvidenceModule, distribution: SDist, fleet_object: FleetType) -> DataFrame:
    """
    Function to calculate distribution measures for each device, carrying the unique identifier of each device

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :fleet_object: ProvidenceDataset or Iterator that provides an id with the Providence inference item.
                    This should be your holdout / test set.

    :return: Dataframe of each device's distribution measures
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
        param_dict = {p_name: p.to('cpu') for p_name, p in zip(parameter_names, params)}

        device_df = DataFrame({
            "tte": prov_targets_tens.cpu().numpy()[:, 0],
            "censor": prov_targets_tens.cpu().numpy()[:, 1],
        }).assign(**param_dict).assign(**device_distribution_measures).assign(id=device_id)

        device_arr.append(device_df)

    device_output = concat_dataframes(device_arr)

    return device_output


ArrayLike = Union[Tensor, Series, np.ndarray]


def generate_metrics_table(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = True) -> DataFrame:
    """
    Produces the following metrics:
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
    model: ProvidenceModule, distribution: SDist, fleet_object: FleetType, *, ignore_nans: bool = False
) -> DataFrame:
    """
    Function to calculate performance metrics for the fleet

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset

    :return: Dataframe of performance metrics for the fleet
    """
    fleet = output_per_device(model, distribution, fleet_object)
    logger.info(f"Computed outputs for fleet with {ignore_nans=}")
    y_true, y_pred = fleet["tte"], fleet["mode"]

    fleet_metrics_ = generate_metrics_table(y_true, y_pred, ignore_nans=ignore_nans)

    return fleet_metrics_


def mfe(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> ArrayLike:
    """
    Calculates mean forecast error of two vectors

    :param y_true: target vector
    :param y_pred: prediction vector

    :return MFE:
    """
    func = np.nanmean if ignore_nans else np.mean
    mfe_ = func(y_true - y_pred, axis=0)
    return mfe_


def mse(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> ArrayLike:
    """
    Calculates mean square error of two vectors

    :param y_true: target vector
    :param y_pred: prediction vector

    :return MSE:
    """
    func = np.nanmean if ignore_nans else np.mean
    mse_ = func(np.square(y_true - y_pred), axis=0)
    return mse_


def rmse(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> ArrayLike:
    """
    Calculates the square root of the mean squared error

    :param y_true: target vector
    :param y_pred: prediction vector

    :return RMSE:
    """
    func = np.nanmean if ignore_nans else np.mean
    rmse_ = np.sqrt(func(np.square(y_true - y_pred)))
    return rmse_


def mape(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> ArrayLike:
    """
    Calculates the mean absolute percentage error of two vectors.
    As this metric is bound to [0, 1] it is an indicator epistimically akin to accuracy, though lower is better.
    Research literature suggests that values lower than ~10% are tolerable / acceptable, though lower is always better.
    Perfectly predicting models would beget a 0.
    
    :param y_true: target vector
    :param y_pred: prediction vector
    
    :return MAPE:
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    func = np.nanmean if ignore_nans else np.mean
    mape_ = func(np.abs((y_true - y_pred) / y_true)) # if there's a division by zero, you can just call the func again
    return mape_


def smape(y_true: ArrayLike, y_pred: ArrayLike, *, ignore_nans: bool = False) -> ArrayLike:
    """
    Calculates symmetric mean absolute percentage error of two vectors.
    As this metric is bound to [0, 1] it is an indicator epistimically akin to accuracy.
    This differs frome MAPE in that it gives a stronger indication of the directionality of the errors:
    - If there is a mixture of undershoot and overshooting, MAPE will wash that out and can produce the same values.
    - SMAPE, however, will be increase with the total error, penalizing overshooting.
    
    For a code example compare the outputs the two functions on the two pairs below.
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

    :param y_true: target vector
    :param y_pred: prediction vector

    :return SMAPE:
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
    """
    Calculates scaled mean percentage error of two vectors.
    Results will be in [-1, 1], giving a directional indicator of accuracy: over- and undershooting, and by how much?

    :param y_true: target vector
    :param y_pred: prediction vector

    :return SMPE:
    """
    s = {"pred": y_pred, "act": y_true}
    smpe_df = DataFrame(data=s)
    smpe_ = (smpe_df.pred - smpe_df.act) / (smpe_df.act.abs() + smpe_df.pred.abs())
    if ignore_nans:
        smpe_ = smpe_[np.isfinite(smpe_)]
    else:  # penalize making all absurd calculations
        smpe_ = np.nan_to_num(smpe_, nan=0, posinf=0, neginf=0)
    return np.mean(smpe_)


def score_phm08(y_true: ArrayLike, y_pred: ArrayLike, alpha_1: float = 13., alpha_2: float = 10.):
    """
    Calcluates score function commonly used for turbofan evaluation

    :param y_true: target vector
    :param y_pred: prediction vector
    :param alpha_1: exponential parameter for underestimating RUL
    :param alpha_2: exponential parameter for overestimating RUL

    :return score:
    """
    d = y_pred - y_true
    d_less = d[d < 0]
    d_greater = d[d >= 0]
    score_ = np.sum(np.exp(-d_less / alpha_1) - 1) + np.sum(np.exp(d_greater / alpha_2) - 1)
    return score_


_ErrorMessage = str
class Result(NamedTuple):
    succeeded: bool
    result: Union[DataFrame, _ErrorMessage]

class MetricsCalculator:
    """
    OOP-wrapper around the core metrics found in the file, used principally to accelerate metrics computation by
    1. Intelligently caching at key points in the data pipeline (without exposing that to the user)
    2. Allowing the override of those cache elements, enabling users to 'compute once' outside of an
        instance of `MetricsCalculator`, load the cache, and profit

    This class was created out of a need to generate metrics much faster at training and evaluation time.
    The functions which this class wraps all have fairly narrow, highly similar pipelines, which might have been organized
        around different idioms - but to little benefit for the effort.
        Cutting out the redundant recomputations of shaved several minutes per model iteration (in a hyperparameter sweep) evalution.
            On some datasets, we're talking ~15 minutes per model, when sweeping several dozen models.
        So we reap the benefits of the simple API and accelerate with simple caching.
    """
    def __init__(
        self, model: TorchModule, distribution: SDist, fleet_object: FleetType, *, rul_stat: str = "mode"
    ) -> None:
        self.model = model
        self.dist = distribution
        self.fleet = fleet_object
        self.rul_stat = rul_stat

        import inspect

        def get_function_name():
            return inspect.currentframe().f_back.f_code.co_name

        self.get_function_name = get_function_name
        self.results_cache = {}

    def outputs_per_device(self) -> DataFrame:
        # cache the expensive operation that everyone calls.
        # NOTE: this cache is done manually so this object supports overriding by updating the results_cache.
        # Remember the rules of Python: be an adult.
        if (f_name := self.get_function_name()) in self.results_cache:
            return self.results_cache[f_name]

        res = output_per_device(self.model, self.dist, self.fleet)
        self.results_cache[f_name] = res
        return res

    def window_predictions(self, *, min_timestep: int, max_timestep: int) -> Series:
        if (key := (self.get_function_name(), min_timestep, max_timestep)) in self.results_cache:
            return self.results_cache[key]

        fleet = self.outputs_per_device()
        mask = fleet['tte'].between(min_timestep, max_timestep)
        result = fleet[mask]

        self.results_cache[key] = result

        return result

    def metrics_by_timestep(self, *, max_timestep: float, min_timestep: float) -> DataFrame:
        """
        Function to calculate MSE and MFE for all devices at each timestep

        :param min_timestep: minimum timestep of desired window
        :param max_timestep: maximum timestep of desired window

        return: DataFrame
        """
        window = self.window_predictions(min_timestep=min_timestep, max_timestep=max_timestep)

        metrics_window = []
        for timestep in window.tte.unique():
            df = window[(window['tte'] == timestep)]
            mse_ = mse(df['tte'], df['mode'])
            mfe_ = mfe(df['tte'], df['mode'])
            output = DataFrame({'tte': [timestep], 'mse': [mse_], 'mfe': [mfe_]})
            metrics_window.append(output)

        metrics_window = concat_dataframes(metrics_window)

        return metrics_window

    def error_by_timestep(self, *, max_timestep: float, min_timestep: float) -> DataFrame:
        """
        Function to calculate error between actual TTE and mean, median, and mode
        of output distribution for each timestep.

        :param min_timestep: minimum timestep of desired window
        :param max_timestep: maximum timestep of desired window

        return: DataFrame
        """
        df = self.window_predictions(min_timestep=min_timestep, max_timestep=max_timestep)
        df_update = dict()
        df_update['error_mean'] = df['tte'] - df['mean']
        df_update['error_median'] = df['tte'] - df['median']
        df_update['error_mode'] = df['tte'] - df['mode']
        df_update['mean_overshoot'] = df_update['error_mean'] < 0
        df_update['median_overshoot'] = df_update['error_median'] < 0
        df_update['mode_overshoot'] = df_update['error_mode'] < 0
        return df.assign(**df_update)

    def overshoot_summary(self) -> Result:
        """
        Function to generate summary statistics for only the predictions which have overshot actual TTE.

        return: Result(True, DataFrame) or Result(False, "Why it doesn't work")
        """
        error_by_ts = self.output_per_device()
        error_by_ts['error_mean'] = error_by_ts['tte'] - error_by_ts['mean']
        error_by_ts['mean_overshoot'] = np.where((error_by_ts['error_mean'] < 0), True, False)

        mask = error_by_ts['mean_overshoot'] == True
        if mask.any():

            overshot = error_by_ts[mask]

            d = {
                "Smallest Overshoot": format(abs(max(overshot['error_mean'])), '.2f'),
                "Greatest Overshoot": format(abs(min(overshot['error_mean'])), '.2f'),
                "Mean Overshoot": format(abs(np.mean(overshot['error_mean'])), '.2f'),
                "Variance of Overshoot": format(np.var(overshot['error_mean']), '.2f'),
                "STD of Overshoot": format(np.std(overshot['error_mean']), '.2f'),
                "Percent of Predictions Overshot": [round(100 * (len(overshot)) / (len(error_by_ts)), 2)],
                "MFE of Overshoot": format(mfe(overshot['tte'], overshot['mean']), '.2f'),
                "SMPE of Overshoot": [smpe(overshot['tte'], overshot['mean'])]
            }

            summary = DataFrame(data=d)

            return Result(True, summary)
        else:
            return Result(False, "No predictions overshot actual TTE")

    def percent_overshot_by_tte(self, *, max_timestep: float, min_timestep: float) -> DataFrame:
        """
        Function to calculate how many predictions have overshot the actual TTE at
        each timestep for mean, median, and mode

        :param max_timestep: maximum timestep of desired window
        :param min_timestep: minimum timestep of desired window

        return: dataframe
        """
        error_by_ts = self.error_by_timestep(max_timestep=max_timestep, min_timestep=min_timestep)

        percent_overshot = []
        for i in error_by_ts.tte.unique():
            df = error_by_ts[(error_by_ts['tte'] == i)]
            overshot_mean = (len(df[(df['mean_overshoot'] == True)])) / (len(df))
            overshot_median = (len(df[(df['median_overshoot'] == True)])) / (len(df))
            overshot_mode = (len(df[(df['mode_overshoot'] == True)])) / (len(df))
            overshot_df = DataFrame(
                {
                    "TTE": [i],
                    "%_Overshot_Mean": [overshot_mean],
                    "%_Overshot_Median": [overshot_median],
                    "%_Overshot_Mode": [overshot_mode]
                }
            )
            percent_overshot.append(overshot_df)

        overshot_by_tte = concat_dataframes(percent_overshot)

        return overshot_by_tte

    def metrics_per_device(self, *, tte_cutoff: int = 150, rul_stat: Optional[str] = None) -> DataFrame:
        """Not just a wrapper, but e.g. a method of computing our metrics with respect to temporal windowing, like the following:
        
        MSE_divisor = 1/(EOL - FPT) 
        MSE_dividend = sum(
            map( lambda i: (preds[i] - truth[i]) ** 2, # squared error
                range(FPT, EOL + 1) # interval of the summation
            )
        )
        MSE = MSE_dividend / MSE_divisor

        Arguments:
        - tte_cutoff: First predicting time, before which we don't care about the model's predictions
        - rul_stat: Optional[str], defaulting to self.rul_stat
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
        self, *, tte_cutoff: int = 150, rul_stat: Optional[str] = None, method: str = "brutal"
    ) -> DataFrame:
        outputs = self.outputs_per_device()
        if method == "charitable" or method == "naive":
            # old implementation with naive cut-off time
            outputs = outputs[outputs["tte"].between(0, tte_cutoff + 1)]
        else:
            # new: cutoff is applied per device
            # make sure the tte's are counting down (e.g. descending)
            outputs = outputs.sort_values(["id", "tte"], ascending=[False, False])
            # grab the end of the sequence, by that tte
            outputs = outputs.groupby('id').tail(tte_cutoff)

        rul_stat = rul_stat or self.rul_stat
        fleetwide = generate_metrics_table(outputs["tte"], outputs[rul_stat], ignore_nans=False).assign(score=self.score(rul_stat=rul_stat))
        return fleetwide

    def score(self, *, rul_stat: Optional[str] = None) -> float:
        """Score the fleet across the final time step (per device) using `rul_stat` or self.rul_stat for the RUL extraction"""
        outputs = self.outputs_per_device()

        rul_stat = rul_stat or self.rul_stat

        # just to be sure, sort in ascending order, then take the first row of each device's TTE and RUL stat
        tte_and_rul_pred = (outputs.sort_values(["id", "tte"], ascending=True).groupby("id").head(1)[["tte", rul_stat]])
        tte = tte_and_rul_pred["tte"]
        rul_pred = tte_and_rul_pred[rul_stat]
        return score_phm08(tte, rul_pred)
