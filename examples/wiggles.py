# -*- coding: utf-8 -*-
"""
Animation of the Weibull distributions predicted by Providence models

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib import animation
from torch import nn

from providence.distributions import Weibull
from providence.nn import ProvidenceGRU as prov_model


def load_model_from_saved_state_dict(path: str) -> nn.Module:
    model = prov_model(24, 120, num_layers=4, dropout=0.2)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    return model


def animate_weibull_curve_over_time(
    model: nn.Module, data: Tuple[torch.Tensor, torch.Tensor]
) -> animation.FuncAnimation:
    """
    Animate the providence predictions
    :param model: should be a providence trained model
    TODO: incorporate into main graphing component

    :return anim:
    """
    model.eval()
    features, target = data  # [2]

    a__, b__ = params = Weibull.compute_distribution_parameters(model, features)

    tte_ = target[:, 0].type(torch.LongTensor)
    palette = sns.color_palette("RdBu_r", a__.shape[0])
    color_dict = dict(enumerate(palette))

    fig, ax = plt.subplots()

    ax.set_xlim((0, 500))
    ax.set_ylim((0, 0.02))

    (points,) = ax.plot([1, 2], [3, 4], marker="o", ls="")  # for points
    (line,) = ax.plot([2, 1], [4, 3])  # for lines
    time = torch.arange(start=0, end=500)

    def _animate(i):
        pdf = Weibull.pdf(Weibull.Params(a__[i], b__[i]), time).numpy()
        line.set_data(time, pdf)
        line.set_color(color_dict[i])
        dot_x = int(tte_[i])
        points.set_data(dot_x, pdf[dot_x])
        points.set_color(color_dict[i])
        ax.text(
            3,
            8,
            "boxed italics text in data coords",
            style="italic",
            bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
        )
        return (line,)

    def _init():
        line.set_data([], [])
        points.set_data([], [])
        return (line,)

    anim = animation.FuncAnimation(fig, _animate, init_func=_init, frames=a__.shape[0], interval=100, blit=True)

    return anim
