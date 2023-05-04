# -*- coding: utf-8 -*-
"""
Visualization code for both introspection and interpretation of the model outputs, at the 'fleet' and 'individual'/'entity'-level.
Usage is explained at the function level.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import random
from contextlib import AbstractContextManager
from typing import Any
from typing import Callable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch as pt
from matplotlib import pyplot as plt

from .utils import logger
from providence.datasets import ProvidenceDataset
from providence.distributions import Weibull
from providence.metrics import FleetType
from providence.metrics import MetricsCalculator
from providence.metrics import SDist as Distribution
from providence.metrics import smpe


class _ShimRotorContext(AbstractContextManager):
    def __exit__(self, __exc_type, __exc_value, __traceback) -> Literal[False]:
        # Because we don't want suppress exceptions, we return false
        return False


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


def check_for_mpl(rotor_config: dict = None):
    """Makes sure matplotlib and rotor_plot can be imported.
    Args:
        rotor_config (dict): A dictionary containing kwargs to pass to the rotor context for plot styling
            (theme, dark, high_contrast, and palette).

    Returns:
        Tuple[plt, ctx, sns]: ``plt`` being ``matplotlib.pyplot``, ``ctx`` being ``RotorContext`` or None, and ``sns``
            being the Seaborn package
    """
    if rotor_config is None:
        rotor_config = {}
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
    alpha: pt.Tensor,
    beta: pt.Tensor,
    tte: pt.Tensor,
    max_time: int = 500,
    title: str = None,
    palette="RdBu_r",
    *,
    verbose: bool = False,
    draw_guideline: bool = False,
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes], np.ndarray]]:
    """Plot Weibull distributions over time.

    Args:
        alpha (pt.Tensor): Weibull alpha parameter
        beta (pt.Tensor): Weibull beta parameter
        tte (pt.Tensor): Tensor of time until event values
        max_time (int): Maximim time value to plot
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
                if draw_guideline:  # draw to the peak of the curve, which is our prediction.
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


def make_error_plot(
    predicted_tte: Union[np.ndarray, pd.Series],
    actual_tte: Union[np.ndarray, pd.Series],
    kind="reg",
):
    """Draw an Seaborn jointplot between ``predicted_tte`` and ``actual_tte``.

    Args:
        predicted_tte (Union[np.ndarray, pd.Series]): predictions from a model
        actual_tte (Union[np.ndarray, pd.Series]): ground truth time-to-event
        kind (str, optional): regression ("reg") or kernel density estimate ("kde"). Defaults to "reg".

    Returns:
        Any: the return value of ``sns.jointplot``
    """
    plt, rotor_context, sns = check_for_mpl()
    plt.figure()  # need to reset the figure reference in the global scope because Seaborn is LAAAAAZZZZYYYY and reuses them.

    sns.set(rc={"figure.figsize": (16, 16)})

    with rotor_context:
        grid = sns.jointplot(
            x=predicted_tte,
            y=actual_tte,
            kind=kind,
        )
    if isinstance(predicted_tte, pd.Series):
        grid.set_axis_labels(f"Predicted TTE ({predicted_tte.name})", "Actual TTE")
    grid.fig.suptitle("Predicted vs. Actual TTE", y=1)

    return grid


def plot_mse_by_timestep(
    calc: MetricsCalculator,
    max_timestep: int,
    min_timestep: int,
):
    """Plot MSE for the fleet at each timestep in a given window.

    Args:
        calc (MetricsCalculator): used for caching computation for the ``error_by_timestep``
        max_timestep (int): maximum timestep of desired window
        min_timestep (int): minimum timestep of desired window

    Returns:
        plt.Axes: axis drawn on to visualize the MSE over time
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
    max_timestep: int,
    min_timestep: int,
):
    """Plot MFE for the fleet at each timestep in a given window.

    Args:
        calc (MetricsCalculator): used for caching computation for the ``error_by_timestep``
        max_timestep (int): maximum timestep of desired window
        min_timestep (int): minimum timestep of desired window

    Returns:
        plt.Axes: axis drawn on to visualize the MFE over time
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
    max_timestep: int,
    min_timestep: int,
    *,
    stat: str = "mode",
):
    """Draw the percentage of predictions overshooting the actual TTE at each timestep.

    When the mean of the output distribution is used as the predicted TTE, this plot
    shows the percentage of predictions overshooting the actual TTE at each timestep.

    The behavior of this plot extremely informative of the model's utility for forecasting:
    - linear trends in the plot show a high-bias model, that doesn't make any trade-offs.
    - polynomial decays / convergence to zero (a funnel with mouth on the left, and pinch on the right-hand side)
      is a fair trade-off, so long as there are fewer overshoots than undershoots
    - pseudo-exponential decays to the TTE=0 hold more promise

    You will also want to see few extreme overshoots, otherwise you might have a problem with model capacity and/or over-regularization

    Args:
        calc (MetricsCalculator): used for caching computation for the ``error_by_timestep``
        max_timestep (int): maximum timestep of desired window
        min_timestep (int): minimum timestep of desired window
        stat (str, optional): average statistic to use for the error computation. Defaults to "mode".

    Returns:
        plt.Axes: axis drawn on to visualize the MSE over time
    """
    assert stat in {"mean", "median", "mode"}

    plt, rotor_context, sns = check_for_mpl()

    error_key = f"error_{stat}"
    overshoot_key = f"{stat}_overshoot"
    error_title = stat.title()

    error_by_ts = calc.error_by_timestep(max_timestep=max_timestep, min_timestep=min_timestep)

    plt.figure()

    with rotor_context:
        ax = sns.scatterplot(
            x=error_by_ts["tte"],
            y=error_by_ts[error_key],
            hue=error_by_ts[overshoot_key],
        )
        ax.legend(loc="upper right", title="Overshoot")
        ax.set_xlabel("Actual TTE")
        ax.set_ylabel("Prediction Error (Mean of Distribution)")
        ax.invert_xaxis()
        ax.set_title(f"Overshoot by Distribution {error_title} TTE", y=1)
    return ax


def plot_percent_overshot_by_tte(
    calc: MetricsCalculator,
    max_timestep: int,
    min_timestep: int,
):
    """Plot the percentage of predictions which have overshot actual TTE at each timestep for mean, median, and mode.

    Args:
        calc (MetricsCalculator): used for caching computation for the ``error_by_timestep``
        max_timestep (int): maximum timestep of desired window
        min_timestep (int): minimum timestep of desired window

    Returns:
        plt.Axes: axis on which the visualization was drawn.
    """
    plt, rotor_context, sns = check_for_mpl()

    overshot = calc.percent_overshot_by_tte(max_timestep=max_timestep, min_timestep=min_timestep)

    df = pd.melt(
        overshot,
        id_vars=["TTE"],
        value_vars=["%_Overshot_Mean", "%_Overshot_Median", "%_Overshot_Mode"],
        var_name="Measures",
    )
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


def plot_weibull_for_fleet(
    model,
    dataset: ProvidenceDataset,
    n_devices: int = 50,
    seed: int = 123,
    max_time: int = 500,
    title: str = None,
    *,
    verbose: bool = False,
):
    """Draw a single Weibull curve from (a uniform random sample) ``n_devices``, predicted by ``model`` on ``dataset``.

    Args:
        model (ProvidenceModule): Pytorch nn.Module that will predict on the ``dataset``
        dataset (ProvidenceDataset): the dataset with devices to draw
        n_devices (int, optional): the number of devices to visualize. Defaults to 50.
        seed (int, optional): numerical seed for the random sampling of devices. Defaults to 123.
        max_time (int, optional): time steps from ``[0, max_time]`` over which to draw ``Weibull`` curves.
            Defaults to 500.
        title (str, optional): Title to give the plot. Defaults to None.
        verbose (bool, optional): Whether to log issue with individual values in the beta sequences predicted by
            ``mode``. Defaults to False.
    """

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

    return plot_weibull(
        alpha=alpha,
        beta=beta,
        tte=tte,
        max_time=max_time,
        title=title,
        palette="muted",
        verbose=verbose,
    )


def incremental_scoring(feats, targets, model):
    """EXPERIMENTAL: Analyze sequence of Weibull-per-Weibull predictions made by ``model``.


    NOTE: this function was used for EDA of the Weibull progression over the course training. Your mileage may vary.

    Args:
        feats (pt.Tensor): first element of the tuple given from a ``ProvidenceDataset``
        targtes (pt.Tensor): second element of the tuple given from a ``ProvidenceDataset``
        model (nn.Module): a module that can predict on Providence data.
    """
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


def plot_weibull_given_modes(
    alpha: pt.Tensor,
    beta: pt.Tensor,
    modes: pt.Tensor,
    ttes: pt.Tensor,
    max_time: int = 500,
    title: str = None,
    *,
    show_vertical_pred=False,
    subplots_kwargs=None,
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes], np.ndarray]]:
    """
    Plot Weibull distributions over time

    Args:
        alpha (pt.Tensor): Weibull alpha parameter
        beta (pt.Tensor): Weibull beta parameter
        tte (pt.Tensor): Tensor of time until event values
        max_time (int): Maximim time value to plot
        title (str, optional): title to the plot. Defaults to None.
        show_vertical_pred (bool, optional): Whether to draw a vertical indicator to the peek of the Weibull curve.
            Defaults to False.
        subplots_kwargs (dict, optional): arguments to given to ``plt.subplots``. Defaults to dict.
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
    fleet_object: Union[FleetType, Iterator[Tuple[str, Tuple[Any, ...]]]],
    *,
    prediction_summary_stat="mode",
) -> None:
    """
    Plot SMPE for each device, color-coding undershoot vs. overshoot

    model (nn.Module): Providence-compatible model
    distribution (Distribution): type of a ``SurvivalAnalysisDistribution`` e.g. ``Weibull``
    fleet_object (FleetType): Providence dataset on which to analyze ``model``'s performance.
    prediction_summary_stat (str, optional): stat in {mean, median, mode}. Defaults to "mode".

    Returns:
        plt.Axes: axis drawn on to visualize the SMPE per device plot.
    """

    smpes_: List[pd.DataFrame] = []

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
        device_df = pd.DataFrame(
            {
                "tte": prov_targets_tens.numpy()[:, 0],
                prediction_summary_stat: tte_infer(params),
            }
        )

        y_true, y_pred = device_df["tte"], device_df[prediction_summary_stat]
        d = {"smpe": [smpe(y_true, y_pred)], "seq_len": [len(device_df)]}
        device_smpe = pd.DataFrame(data=d)
        device_smpe["id"] = device_id

        smpes_.append(device_smpe)

    smpe_ = pd.concat(smpes_)
    # NOTE: mypy cannot understand this pandas syntax.
    smpe_["overshoot"] = np.where(smpe_["smpe"] < 0, "Overshoot", "Undershoot")  # type: ignore[call-overload]

    plt, rotor_context, sns = check_for_mpl()

    plt.figure()

    with rotor_context:
        # NOTE: mypy cannot understand this pandas syntax.
        ax = sns.scatterplot(x=smpe_["seq_len"], y=smpe_["smpe"], hue=smpe_["overshoot"])  # type: ignore[call-overload]
        ax.legend(loc="upper right")
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("SMPE")
        ax.set_title("SMPE by Device", y=1)

    return ax


from functools import lru_cache


class Visualizer:
    """Core functions to visualize modules, leveraging pre-computation from ``MetricsCalculator`` to save time.

    For one-off usage, the FleetType can be the iterator. If you supply an iterator, do NOT call warm_up_cache.
    For repeated usage, provide a ProvidenceDataset.
    """

    def __init__(self, model: pt.nn.Module, distribution: Distribution, fleet_object: FleetType) -> None:
        self.model = model
        self.dist = distribution
        self.fleet = fleet_object

        self.metrics_calculator = MetricsCalculator(self.model, self.dist, self.fleet)

    def warm_up_cache(self, *, min_timestep: int, max_timestep: int):
        """Invoke each of the caching computations, so visualizations run more quickly.

        Args:
            min_timestep (int): minimum timestep of desired window
            max_timestep (int): maximum timestep of desired window
        """
        funcs_to_cache = [
            self.metrics_calculator.error_by_timestep,
            self.metrics_calculator.metrics_by_timestep,
            self.metrics_calculator.percent_overshot_by_tte,
        ]

        for func_to_cache in funcs_to_cache:
            func_to_cache(min_timestep=min_timestep, max_timestep=max_timestep)

    @lru_cache  # noqa: B019
    def error_by_timestep(self, *, min_timestep: int, max_timestep: int):
        """Calculate error between actual TTE and mean, median, and mode of output distribution for each timestep.

        This is a direct pass-through to the ``MetricsCalculator``.
        Please see ``MetricsCalculator.error_by_timestep`` for fuller documentation.
        """
        res = self.metrics_calculator.error_by_timestep(min_timestep=min_timestep, max_timestep=max_timestep)
        return res

    @lru_cache  # noqa: B019
    def metrics_by_timestep(self, *, min_timestep: int, max_timestep: int):
        """Calculate MSE and MFE for all devices at each timestep.

        This is a direct pass-through to the ``MetricsCalculator``.
        Please see ``MetricsCalculator.metrics_by_timestep`` for fuller documentation.
        """
        res = self.metrics_calculator.metrics_by_timestep(min_timestep=min_timestep, max_timestep=max_timestep)
        return res

    @lru_cache  # noqa: B019
    def percent_overshot_by_tte(self, *, min_timestep: int, max_timestep: int):
        """Calculate how many predictions have overshot the actual TTE at each timestep for mean, median, and mode.

        This is a direct pass-through to the ``MetricsCalculator``.
        Please see ``MetricsCalculator.percent_overshot_by_tte`` for fuller documentation.
        """
        res = self.metrics_calculator.percent_overshot_by_tte(min_timestep=min_timestep, max_timestep=max_timestep)
        return res

    def plot_mfe_by_timestep(self, *, min_timestep: int, max_timestep: int):
        """Plot MFE by timestep between (inclusive) ``min_timestep`` and ``max_timestep``.

        Args:
            min_timestep (int): minimum timestep of desired window
            max_timestep (int): maximum timestep of desired window

        Returns:
            plt.Axes: axis drawn on to visualize the MFE over time
        """
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
        """Plot MSE by timestep between (inclusive) ``min_timestep`` and ``max_timestep``.

        Args:
            min_timestep (int): minimum timestep of desired window
            max_timestep (int): maximum timestep of desired window

        Returns:
            plt.Axes: axis drawn on to visualize the MSE over time
        """
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
        """Scatterplot of mode predictions by timestep between (inclusive) ``min_timestep`` and ``max_timestep``.

        This plot is particularly useful to show per-device mean of ``"mode"`` predictions, to see if the model is
        getting predictions nearly correct as we approach the termination of sequence. This is even more useful if
        drawn only against uncensored entities - those entities that experience events. When appled to censored entities
        it begs the question of the intent of the analyst, but do so if you find utility.

        Args:
            min_timestep (int): minimum timestep of desired window
            max_timestep (int): maximum timestep of desired window

        Returns:
            plt.Axes: axis drawn on to visualize the scattplot
        """
        plt, rotor_context, sns = check_for_mpl()

        error_by_ts = self.error_by_timestep(min_timestep=min_timestep, max_timestep=max_timestep)

        plt.figure()

        with rotor_context:
            ax = sns.scatterplot(
                x=error_by_ts["tte"],
                y=error_by_ts["error_mode"],
                hue=error_by_ts["mode_overshoot"],
            )
            ax.legend(loc="upper right", title="Overshoot")
            ax.set_xlabel("Actual TTE")
            ax.set_ylabel("Prediction Error (Mode of Distribution)")
            ax.invert_xaxis()
            ax.set_title("Overshoot by Distribution Mode TTE", y=1)

        return ax

    def plot_percent_overshot_by_tte(self, *, min_timestep: int, max_timestep: int):
        """Plot percentage of predictions overshot, by timestep between (inclusive) ``min_timestep`` and ``max_timestep``.

        Args:
            min_timestep (int): minimum timestep of desired window
            max_timestep (int): maximum timestep of desired window

        Returns:
            plt.Axes: axis drawn on to visualize the percentage of overshooting over time
        """
        plt, rotor_context, sns = check_for_mpl()

        overshot = self.percent_overshot_by_tte(min_timestep=min_timestep, max_timestep=max_timestep)
        df = pd.melt(
            overshot,
            id_vars=["TTE"],
            value_vars=["%_Overshot_Mean", "%_Overshot_Median", "%_Overshot_Mode"],
            var_name="Measures",
        )
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
