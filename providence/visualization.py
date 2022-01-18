# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union, Tuple, List, Dict, types

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .distributions import weibull
from .utils.logging_utils import logger

from .metrics import window_predictions, metrics_by_timestep, error_by_timestep, percent_overshot_by_tte

Distribution = types.ModuleType



def plot_weibull(alpha: torch.tensor, beta: torch.tensor, tte: torch.tensor, max_time: int = 500, title: str = None) -> None:
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
    palette = sns.color_palette("RdBu_r", alpha.shape[0])
    color_dict = dict(enumerate(palette))

    modes = weibull.mode(alpha, beta)  # weibull modes are probably the best tool to use here
    t = torch.arange(start=0, end=max_time)

    fig, ax = plt.subplots(1, figsize=(20, 20))

    # Loop through the tensor values and plot the distribution over t, mode, and actual time of failure
    for i in range(0, alpha.shape[0]):
        if beta[i] >= 1:  # Mode is \inf if beta < 1
            color = color_dict[i]
            pdf = weibull.pdf(alpha[i], beta[i], t)
            dot_location = [int(tte.numpy()[i])]
            ax.vlines(modes[i], ymin=0, ymax=max(pdf), colors=color, linestyles=":")
            ax.plot(pdf, color=color)
            ax.scatter(dot_location, pdf[dot_location], color=color, s=100)
        else:
            logger.info(f"β =< 1 at position {i}")

    ax.set_yticklabels([])
    ax.set_xlim(right=torch.max(t))
    ax.set_xlabel("Time")
    if title:
        ax.set_title(title)

    return fig, ax


def make_error_plot(predicted_tte: Union[np.ndarray, pd.Series], actual_tte: Union[np.ndarray, pd.Series], kind="reg"):
    # need to reset the figure reference in the global scope because Seaborn reuses them.
    plt, ctx, sns = check_for_mpl()
    plt.figure() # TODO: clean up figures manually rather than relying 

    sns.set(rc={"figure.figsize": (16, 16)})

    grid = sns.jointplot(
            x=predicted_tte,
            y=actual_tte,
            kind=kind,
        )
    grid.set_axis_labels(f"Predicted TTE ({predicted_tte.name})", "Actual TTE")
    grid.fig.suptitle("Predicted vs. Actual TTE", y=1)

    return grid


def plot_mse_by_timestep(
    model: torch.nn.Module,
    distribution: Distribution,
    fleet_object: Union[Dataset, List[Tuple[torch.Tensor, torch.Tensor]]],
    max_timestep: float,
    min_timestep: float,
):
    """
    Plot MSE for the fleet at each timestep in a given window

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window
    """

    metrics_by_ts = metrics_by_timestep(model, distribution, fleet_object, max_timestep, min_timestep)

    plt.figure()

    ax = sns.scatterplot(x=metrics_by_ts.tte, y=metrics_by_ts.mse)
    ax.invert_xaxis()
    ax.set_xlabel("Actual TTE")
    ax.set_ylabel("MSE")
    ax.axhline(linewidth=2, color="gray")
    ax.set_title("MSE by Timestep", y=1)

    return ax


def plot_mfe_by_timestep(
    model: torch.nn.Module,
    distribution: Distribution,
    fleet_object: Union[Dataset, List[Tuple[torch.Tensor, torch.Tensor]]],
    max_timestep: float,
    min_timestep: float,
):
    """
    Plot MFE for the fleet at each timestep in a given window

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window
    """

    metrics_by_ts = metrics_by_timestep(model, distribution, fleet_object, max_timestep, min_timestep)

    plt.figure()

    ax = sns.scatterplot(x=metrics_by_ts.tte, y=metrics_by_ts.mfe)
    ax.invert_xaxis()
    ax.set_xlabel("Actual TTE")
    ax.set_ylabel("MFE")
    ax.axhline(linewidth=2, color="gray")
    ax.set_title("MFE by Timestep", y=1)

    return ax


def scatterplot_overshoot_mean(
    model: torch.nn.Module,
    distribution: Distribution,
    fleet_object: Union[Dataset, List[Tuple[torch.Tensor, torch.Tensor]]],
    max_timestep: float,
    min_timestep: float,
):
    """
    When the mean of the output distribution is used as the predicted TTE, this plot
    shows the percentage of predictions overshooting the actual TTE at each timestep.

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window
    """

    error_by_ts = error_by_timestep(model, distribution, fleet_object, max_timestep, min_timestep)

    plt.figure()

    ax = sns.scatterplot(x=error_by_ts["tte"], y=error_by_ts["error_mean"], hue=error_by_ts["mean_overshoot"])
    ax.legend(loc="upper right", title="Overshoot")
    ax.set_xlabel("Actual TTE")
    ax.set_ylabel("Prediction Error (Mean of Distribution)")
    ax.invert_xaxis()
    ax.set_title("Overshoot by Distribution Mean TTE", y=1)

    return ax


def scatterplot_overshoot_median(
    model: torch.nn.Module,
    distribution: Distribution,
    fleet_object: Union[Dataset, List[Tuple[torch.Tensor, torch.Tensor]]],
    max_timestep: float,
    min_timestep: float,
):
    """
    When the median of the output distribution is used as the predicted TTE, this plot
    shows the percentage of predictions overshooting the actual TTE at each timestep.

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window
    """

    error_by_ts = error_by_timestep(model, distribution, fleet_object, max_timestep, min_timestep)

    plt.figure()

    ax = sns.scatterplot(x=error_by_ts["tte"], y=error_by_ts["error_median"], hue=error_by_ts["median_overshoot"])
    ax.legend(loc="upper right", title="Overshoot")
    ax.set_xlabel("Actual TTE")
    ax.set_ylabel("Prediction Error (Median of Distribution)")
    ax.invert_xaxis()
    ax.set_title("Overshoot by Distribution Median TTE", y=1)

    return ax


def scatterplot_overshoot_mode(
    model: torch.nn.Module,
    distribution: Distribution,
    fleet_object: Union[Dataset, List[Tuple[torch.Tensor, torch.Tensor]]],
    max_timestep: float,
    min_timestep: float,
):
    """
    When the mode of the output distribution is used as the predicted TTE, this plot
    shows the percentage of predictions overshooting the actual TTE at each timestep.

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window
    """

    error_by_ts = error_by_timestep(model, distribution, fleet_object, max_timestep, min_timestep)

    plt.figure()

    ax = sns.scatterplot(x=error_by_ts["tte"], y=error_by_ts["error_mode"], hue=error_by_ts["mode_overshoot"])
    ax.legend(loc="upper right", title="Overshoot")
    ax.set_xlabel("Actual TTE")
    ax.set_ylabel("Prediction Error (Mode of Distribution)")
    ax.invert_xaxis()
    ax.set_title("Overshoot by Distribution Mode TTE", y=1)

    return ax


def plot_percent_overshot_by_tte(
    model: torch.nn.Module,
    distribution: Distribution,
    fleet_object: Union[Dataset, List[Tuple[torch.Tensor, torch.Tensor]]],
    max_timestep: float,
    min_timestep: float,
):
    """
    Plot the percentage of predictions which have overshot actual TTE at each
    timestep for mean, median, and mode.

    :param model: Providence-trained model
    :param distribution: Module of providence.distributions
    :param fleet_object: Providence test dataset
    :param max_timestep: maximum timestep of desired window
    :param min_timestep: minimum timestep of desired window
    """

    overshot = percent_overshot_by_tte(model, distribution, fleet_object, max_timestep, min_timestep)
    df = pd.melt(overshot, id_vars=["TTE"], value_vars=["%_Overshot_Mean", "%_Overshot_Median", "%_Overshot_Mode"], var_name="Measures")
    plt.figure()

    measures_palette = ["#6c91ef", "#e4780c", "#8f00a3"]

    ax = sns.scatterplot(x="TTE", y="value", data=df, hue="Measures", palette=measures_palette)
    ax.legend(loc="lower left", title="Distribution Measure")
    ax.set_xlabel("Actual TTE")
    ax.set_ylabel("% Predictions Overshot (Mean, Median, Mode)")
    ax.invert_xaxis()

    return ax
