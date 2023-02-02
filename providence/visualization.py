# -*- coding: utf-8 -*-
"""
Visualization code for both introspection and interpretation of the model outputs, at the 'fleet' and 'individual'/'entity'-level.
Usage is explained at the function level.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from contextlib import AbstractContextManager
from typing import Callable, Union
import random

import numpy as np
import pandas as pd
import torch as pt
from providence.datasets import ProvidenceDataset

from providence.distributions import Weibull

from .utils import logger

from providence.metrics import FleetType, MetricsCalculator, SDist as Distribution
from providence.metrics import smpe



class _ShimRotorContext(AbstractContextManager):
    def __exit__(self, __exc_type, __exc_value, __traceback) -> bool:
        return False  # Because we don't want suppress exceptions, we return false


class WarnOnce:
    def __init__(self, action: Callable[[], None]):
        self.action = action
        self.invoked = False

    def __call__(self, *args):
        if not self.invoked:
            self.action()
            self.invoked = True


warn_about_mpl_import = WarnOnce(
    lambda: (logger.warning("\nPlease install the dx-rotor-plot package in order to enable plotting functionality."))
)


def check_for_mpl(rotor_config: dict = {}):
    """Makes sure matplotlib and rotor_plot can be imported.
    Parameters
    ----------
    rotor_config : dict
        A dictionary containing kwargs to pass to the rotor context for plot styling (theme, dark, high_contrast, and palette).
    Returns
    -------
    plt : `matplotlib.pyplot`
    ctx : :class:`RotorContext`
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("\nPlease install the matplotlib package in order to enable plotting functionality.") from e

    try:
        from rotor_plot import RotorContext

        ctx = RotorContext(
            rotor_config.get("theme", "core"),
            rotor_config.get("dark", False),
            rotor_config.get("high_contrast", False),
            rotor_config.get("palette", None),
        )
    except ImportError as e:
        # NOTE: very willing to make this a quieter log level.
        warn_about_mpl_import()
        ctx = _ShimRotorContext()

    try:
        import seaborn as sns

    except ImportError as e:
        raise ImportError("\nPlease install the seaborn package in order to enable plotting functionality.") from e

    return plt, ctx, sns


def plot_weibull(
    alpha: pt.Tensor, beta: pt.Tensor, tte: pt.Tensor, max_time: int = 500, title: str = None, palette="RdBu_r",
    *,
    verbose: bool = False,
    draw_guideline: bool = False,
) -> None:
    """
    Plot Weibull distributions over time

    :param alpha: Weibull alpha parameter
    :param beta: Weibull beta parameter
    :param tte: Tensor of time until event values
    :param max_time: Maximim time value to plot

    :return: None
    """
    # Get color palette
    plt, ctx, sns = check_for_mpl()
    palette = sns.color_palette(palette, alpha.shape[0])
    color_dict = dict(enumerate(palette))

    if draw_guideline:
        # weibull modes are probably the best tool to use here
        modes = Weibull.mode(Weibull.Params(alpha, beta))

    t = pt.arange(start=0, end=max_time)
    tte_array = tte.numpy()

    fig, ax = plt.subplots(1, figsize=(20, 20))


    with ctx:
        # Loop through the tensor values and plot the distribution over t, mode, and actual time of failure
        for i in range(0, alpha.shape[0]):
            if beta[i] >= 1:  # Mode is \inf if beta < 1
                color = color_dict[i]
                pdf = Weibull.pdf(Weibull.Params(alpha[i], beta[i]), t)
                dot_location = [int(tte_array[i])]
                if draw_guideline: # draw to the peak of the curve, which is our prediction.
                    ax.vlines(modes[i], ymin=0, ymax=max(pdf), colors=color, linestyles=":")
                ax.plot(pdf, color=color)
                # bold dot on the curve, representing "actual". Ideally, the peak of the curve is under the dot.
                ax.scatter(dot_location, pdf[dot_location], color=color, s=100)
            elif verbose:
                logger.info(f"β =< 1 at position {i}")

        # ax.set_yticklabels([], fontsize=14)
        ax.set_ylabel("P(t)", fontsize=16)
        ax.set_xlim(right=pt.max(t))
        ax.set_xlabel("Time", fontsize=16)
        if title:
            ax.set_title(title)

    return fig, ax


def make_error_plot(predicted_tte: Union[np.ndarray, pd.Series], actual_tte: Union[np.ndarray, pd.Series], kind="reg"):
    plt, rotor_context, sns = check_for_mpl()
    plt.figure()  # need to reset the figure reference in the global scope because Seaborn is LAAAAAZZZZYYYY and reuses them.

    sns.set(rc={"figure.figsize": (16, 16)})

    with rotor_context:
        grid = sns.jointplot(
            x=predicted_tte,
            y=actual_tte,
            kind=kind,
        )
    grid.set_axis_labels(f"Predicted TTE ({predicted_tte.name})", "Actual TTE")
    grid.fig.suptitle("Predicted vs. Actual TTE", y=1)

    return grid


def plot_mse_by_timestep(
    calc: MetricsCalculator,
    max_timestep: float,
    min_timestep: float,
):
    """
    Plot MSE for the fleet at each timestep in a given window

    :param calc: Metrics calculator, which has a caching computation for the error_by_timestep
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window
    """
    plt, rotor_context, sns = check_for_mpl()

    metrics_by_ts = calc.metrics_by_timestep(max_timestep=max_timestep, min_timestep=min_timestep)

    plt.figure()

    with rotor_context:
        ax = sns.scatterplot(x=metrics_by_ts.tte, y=metrics_by_ts.mse)
        ax.invert_xaxis()
        ax.set_xlabel("Actual TTE")
        ax.set_ylabel("MSE")
        ax.axhline(linewidth=2, color="gray")
        ax.set_title("MSE by Timestep", y=1)

    return ax


def plot_mfe_by_timestep(
    calc: MetricsCalculator,
    max_timestep: float,
    min_timestep: float,
):
    """
    Plot MFE for the fleet at each timestep in a given window

    :param calc: Metrics calculator, which has a caching computation for the error_by_timestep
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window
    """
    plt, rotor_context, sns = check_for_mpl()

    metrics_by_ts = calc.metrics_by_timestep(max_timestep=max_timestep, min_timestep=min_timestep)

    plt.figure()

    with rotor_context:
        ax = sns.scatterplot(x=metrics_by_ts.tte, y=metrics_by_ts.mfe)
        ax.invert_xaxis()
        ax.set_xlabel("Actual TTE")
        ax.set_ylabel("MFE")
        ax.axhline(linewidth=2, color="gray")
        ax.set_title("MFE by Timestep", y=1)

    return ax


def scatterplot_overshoot(
    calc: MetricsCalculator,
    max_timestep: float,
    min_timestep: float,
    *,
    stat: str = "mode"
):
    """
    When the mean of the output distribution is used as the predicted TTE, this plot
    shows the percentage of predictions overshooting the actual TTE at each timestep.

    The behavior of this plot extremely informative of the model's utility for forecasting:
    - linear trends in the plot show a high-bias model, that doesn't make any trade-offs.
    - polynomial decays / convergence to zero (a funnel with mouth on the left, and pinch on the right-hand side)
      is a fair trade-off, so long as there are fewer overshoots than undershoots
    - pseudo-exponential decays to the TTE=0 hold more promise

    You will also want to see few extreme overshoots, otherwise you might have a problem with model capacity and/or over-regularization

    :param calc: Metrics calculator, which has a caching computation for the error_by_timestep
    :param max_timestep: maximum timestep of desired window.
    :param min_timestep: minimum timestep of desired window
    
    Keyword arguments:
    :param stat: stat in {mean, median, mode}
    """
    assert stat in {"mean", "median", "mode"}
    
    plt, rotor_context, sns = check_for_mpl()

    error_key = f"error_{stat}"
    overshoot_key = f"{stat}_overshoot"
    error_title = stat.title()

    error_by_ts = calc.error_by_timestep(max_timestep=max_timestep, min_timestep=min_timestep)

    plt.figure()

    with rotor_context:
        ax = sns.scatterplot(x=error_by_ts["tte"], y=error_by_ts[error_key], hue=error_by_ts[overshoot_key])
        ax.legend(loc="upper right", title="Overshoot")
        ax.set_xlabel("Actual TTE")
        ax.set_ylabel("Prediction Error (Mean of Distribution)")
        ax.invert_xaxis()
        ax.set_title(f"Overshoot by Distribution {error_title} TTE", y=1)
    return ax


def plot_percent_overshot_by_tte(
    calc: MetricsCalculator,
    max_timestep: float,
    min_timestep: float,
):
    """
    Plot the percentage of predictions which have overshot actual TTE at each
    timestep for mean, median, and mode.

    :param calc: Metrics calculator, which has a caching computation for the error_by_timestep
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window
    """
    plt, rotor_context, sns = check_for_mpl()

    overshot = calc.percent_overshot_by_tte(max_timestep=max_timestep, min_timestep=min_timestep)

    df = pd.melt(overshot, id_vars=["TTE"], value_vars=["%_Overshot_Mean", "%_Overshot_Median", "%_Overshot_Mode"], var_name="Measures")
    plt.figure()

    measures_palette = ["#6c91ef", "#e4780c", "#8f00a3"]

    with rotor_context:
        ax = sns.scatterplot(x="TTE", y="value", data=df, hue="Measures", palette=measures_palette)
        ax.legend(loc="lower left", title="Distribution Measure")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Actual TTE")
        ax.set_ylabel("% Predictions Overshot (Mean, Median, Mode)")
        ax.invert_xaxis()
    return ax


def plot_weibull_for_fleet(model, dataset, n_devices: int = 50, seed: int = 123, max_time: int = 500, title=None, *, verbose: bool = False):
    def random_indexes(l, n):
        random.seed(seed)

        indicies = []
        for i in range(n):
            indicies += [random.randrange((len(l)))]
        return indicies

    model.eval()

    last_alpha = []
    last_beta = []
    last_tte = []

    for i in range(len(dataset)):
        data_object = dataset[i]

        alpha, beta = model.forward(data_object[0].unsqueeze(1), [data_object[0].shape[0]])
        last_alpha.append(alpha.squeeze(1).detach())
        last_beta.append(beta.squeeze(1).detach())
        last_tte.append(data_object[1][-1, 0])

    indicies = random_indexes(last_alpha, n_devices)
    print(f"{len(indicies) = }")

    alpha = pt.Tensor([x[-1] for idx, x in enumerate(last_alpha) if idx in indicies])
    beta = pt.Tensor([x[-1] for idx, x in enumerate(last_beta) if idx in indicies])
    tte = pt.Tensor([x for idx, x in enumerate(last_tte) if idx in indicies])

    return plot_weibull(alpha=alpha, beta=beta, tte=tte, max_time=max_time, title=title, palette="muted", verbose=verbose)

def incremental_scoring(feats, targets, model):
    # NOTE: this function was used for EDA of the Weibull progression over the course training. Your mileage may vary.

    f_len = feats.shape[0]

    modes, alphas, betas = pt.zeros(f_len, 1), pt.zeros(f_len, 1), pt.zeros(f_len, 1)

    print(modes.shape, alphas.shape)

    for INT in range(4, f_len):
        alpha, beta = model.forward(feats.unsqueeze(1)[:INT], [INT])
        alpha, beta = alpha.squeeze(1).detach(), beta.squeeze(1).detach()

        alphas[INT, 0] = alpha[-1, 0]
        betas[INT, 0] = beta[-1, 0]
        modes[INT, 0] = Weibull.mode(Weibull.Params(alpha, beta))[-1, 0]

    return modes, alphas, betas


def plot_weibull_given_modes(alpha, beta, modes, ttes: pt.Tensor, max_time: int = 500, title: str = None, *,
    show_vertical_pred = False, subplots_kwargs=None) -> None:
    """
    Plot Weibull distributions over time

    :param alpha: Weibull alpha parameter
    :param beta: Weibull beta parameter
    :param tte: Tensor of time until event values
    :param max_time: Maximim time value to plot

    :return: None
    """
    plt, rotor_context, sns = check_for_mpl()

    # Get color palette
    assert ttes.shape[0] == alpha.shape[0]
    palette = sns.color_palette("RdBu_r", alpha.shape[0])
    color_dict = dict(enumerate(palette))
    t = pt.arange(start=0, end=max_time)
    ttes_numpy = ttes.numpy() if isinstance(ttes, pt.Tensor) else ttes.to_numpy()

    fig, ax = plt.subplots(1, **subplots_kwargs) if subplots_kwargs is not None else plt.subplots(1, figsize=(20, 15))

    with rotor_context:
        # Loop through the tensor values and plot the distribution over t, mode, and actual time of failure
        for i in range(0, alpha.shape[0]):
            if beta[i] >= 1:  # Mode is \inf if beta < 1
                color = color_dict[i]
                pdf = Weibull.pdf(Weibull.Params(alpha[i], beta[i]), t).squeeze(0)
                dot_location = [ttes_numpy[i]]

                ax.plot(pdf, color=color)
                if show_vertical_pred and (modes[i] > 0):
                    ax.vlines(modes[i], ymin=0, ymax=max(pdf), colors=color, linestyles=":")
                ax.scatter(dot_location, pdf[dot_location], color=color, s=100)
            else:
                logger.info(f"β =< 1 at position {i}")

    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    ax.set_ylabel("P(t)", fontsize=18)
    ax.set_xlim(right=pt.max(t))
    ax.set_xlabel("Time", fontsize=18)
    if title:
        ax.set_title(title)

    return fig, ax


def smpe_by_device(
    model: pt.nn.Module,
    distribution: Distribution,
    fleet_object: FleetType,
    *,
    prediction_summary_stat = "mode"
) -> None:
    """
    Plot SMPE for each device, color-coding undershoot vs. overshoot

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset

    Keyword Arguments:
    :param prediction_summary_stat: stat in {mean, median, mode}

    :return: None
    """

    smpe_ = []

    if isinstance(fleet_object, ProvidenceDataset):
        fleet_object = fleet_object.iter_tensors_with_id()

    tte_infer = {
        "mean": distribution.mean,
        "median": distribution.median,
        "mode": distribution.mode,
    }[prediction_summary_stat]

    model.eval()
    for device_id, (feature_tens, prov_targets_tens) in fleet_object:

        params = distribution.compute_distribution_parameters(model, feature_tens)

        # just a reference, the following line
        # device_df = generate_distribution_measures(distribution, ab, prov_target_tens)
        # is replaced by the shorter block below
        device_df = pd.DataFrame({
            "tte": prov_targets_tens.numpy()[:, 0],
            prediction_summary_stat: tte_infer(params)
        })

        y_true, y_pred = device_df["tte"], device_df[prediction_summary_stat]
        d = {'smpe': [smpe(y_true, y_pred)], 'seq_len': [len(device_df)]}
        device_smpe = pd.DataFrame(data=d)
        device_smpe["id"] = device_id

        smpe_.append(device_smpe)

    smpe_ = pd.concat(smpe_)

    smpe_["overshoot"] = np.where(smpe_['smpe'] < 0, 'Overshoot', 'Undershoot')

    plt, rotor_context, sns = check_for_mpl()

    plt.figure()

    with rotor_context:
        ax = sns.scatterplot(x=smpe_["seq_len"], y=smpe_["smpe"], hue=smpe_["overshoot"])
        ax.legend(loc="upper right")
        ax.set_ylim(-1,1)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("SMPE")
        ax.set_title("SMPE by Device", y=1)

    return ax

from functools import lru_cache
class Visualizer:
    """
    The core functions of the visualizations module, with intermediate calculations stripped out or cached to save time.
    For one-off usage, the FleetType can be the iterator. If you supply an iterator, do NOT call warm_up_cache.
    For repeated usage, provide a ProvidenceDataset.
    """

    def __init__(self, model: pt.nn.Module, distribution: Distribution, fleet_object: FleetType) -> None:
        self.model = model
        self.dist = distribution
        self.fleet = fleet_object

        self.metrics_calculator = MetricsCalculator(self.model, self.dist, self.fleet)

    def warm_up_cache(self, *, min_timestep: int, max_timestep: int):
        
        funcs_to_cache = [
            self.metrics_calculator.error_by_timestep,
            self.metrics_calculator.metrics_by_timestep,
            self.metrics_calculator.percent_overshot_by_tte,]

        for func_to_cache in funcs_to_cache:
            func_to_cache(min_timestep=min_timestep, max_timestep=max_timestep)

    @lru_cache
    def error_by_timestep(self, *, min_timestep: int, max_timestep: int):
        res = self.metrics_calculator.error_by_timestep(min_timestep=min_timestep, max_timestep=max_timestep)
        return res

    @lru_cache
    def metrics_by_timestep(self, *, min_timestep: int, max_timestep: int):
        res = self.metrics_calculator.metrics_by_timestep(min_timestep=min_timestep, max_timestep=max_timestep)
        return res

    @lru_cache
    def percent_overshot_by_tte(self, *, min_timestep: int, max_timestep: int):
        res = self.metrics_calculator.percent_overshot_by_tte(min_timestep=min_timestep, max_timestep=max_timestep)
        return res

    def plot_mfe_by_timestep(self, *, min_timestep: int, max_timestep: int):
        plt, rotor_context, sns = check_for_mpl()

        metrics_by_ts = self.metrics_by_timestep(max_timestep=max_timestep, min_timestep=min_timestep)

        plt.figure()

        with rotor_context:
            ax = sns.scatterplot(x=metrics_by_ts.tte, y=metrics_by_ts.mfe)
            ax.invert_xaxis()
            ax.set_xlabel("Actual TTE")
            ax.set_ylabel("MFE")
            ax.axhline(linewidth=2, color="gray")
            ax.set_title("MFE by Timestep", y=1)

        return ax

    
    def plot_mse_by_timestep(self, *, min_timestep: int, max_timestep: int):
        plt, rotor_context, sns = check_for_mpl()

        metrics_by_ts = self.metrics_by_timestep(min_timestep=min_timestep, max_timestep=max_timestep)

        plt.figure()

        with rotor_context:
            ax = sns.scatterplot(x=metrics_by_ts.tte, y=metrics_by_ts.mse)
            ax.invert_xaxis()
            ax.set_xlabel("Actual TTE")
            ax.set_ylabel("MSE")
            ax.axhline(linewidth=2, color="gray")
            ax.set_title("MSE by Timestep", y=1)

        return ax

    
    def scatterplot_overshoot_mode(self, *, min_timestep: int, max_timestep: int):
        plt, rotor_context, sns = check_for_mpl()

        error_by_ts = self.error_by_timestep(min_timestep=min_timestep, max_timestep=max_timestep)

        plt.figure()

        with rotor_context:
            ax = sns.scatterplot(x=error_by_ts["tte"], y=error_by_ts["error_mode"], hue=error_by_ts["mode_overshoot"])
            ax.legend(loc="upper right", title="Overshoot")
            ax.set_xlabel("Actual TTE")
            ax.set_ylabel("Prediction Error (Mode of Distribution)")
            ax.invert_xaxis()
            ax.set_title("Overshoot by Distribution Mode TTE", y=1)

        return ax
    
    def plot_percent_overshot_by_tte(self, *, min_timestep: int, max_timestep: int):
        plt, rotor_context, sns = check_for_mpl()

        overshot = self.percent_overshot_by_tte(min_timestep=min_timestep, max_timestep=max_timestep)
        df = pd.melt(overshot, id_vars=["TTE"], value_vars=["%_Overshot_Mean", "%_Overshot_Median", "%_Overshot_Mode"], var_name="Measures")
        plt.figure()

        measures_palette = ["#6c91ef", "#e4780c", "#8f00a3"]

        with rotor_context:
            ax = sns.scatterplot(x="TTE", y="value", data=df, hue="Measures", palette=measures_palette)
            ax.legend(loc="lower left", title="Distribution Measure")
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Actual TTE")
            ax.set_ylabel("% Predictions Overshot (Mean, Median, Mode)")
            ax.invert_xaxis()
            ax.set_title("Percent Overshot over Time")

        return ax
