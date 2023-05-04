"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import torch as pt
from matplotlib import pyplot as plt
from torch.nn import functional as F

from providence.distributions import Weibull


def plot_weibull_growth_versus_parameters(x=pt.arange(-2, 3, step=0.01)):
    alpha = pt.exp(x)
    beta = F.softplus(x)
    _, ax = plt.subplots(ncols=2, figsize=(30, 15))

    ax[0].set_title("Weibull parameters compared of growth: Alpha vs Beta")
    ax[0].plot(x, alpha, label="alpha")
    ax[0].plot(x, beta, label="beta")
    ax[0].legend()

    ax[1].set_title("Summary statistic trends from those parameters")

    ax[1].plot(x, Weibull.mean(Weibull.Params(alpha, beta)), label="mean")
    ax[1].plot(x, Weibull.median(Weibull.Params(alpha, beta)), label="median")
    ax[1].plot(x, Weibull.mode(Weibull.Params(alpha, beta)), label="mode")
    ax[1].set_ylim(ax[0].get_ylim()[0], 100)
    ax[1].legend()


def plot_weibull_summary_stat_trends(x=pt.arange(-1.5, 10, step=0.01)):
    alpha = pt.exp(x)
    beta = F.softplus(x)

    _, ax = plt.subplots(ncols=2, figsize=(20, 15))

    ax[0].set_title("Weibull summary statistic trends with linearly increasing parameters")

    _mean = Weibull.mean(Weibull.Params(alpha, beta))
    _median = Weibull.median(Weibull.Params(alpha, beta))
    _mode = Weibull.mode(Weibull.Params(alpha, beta))

    ax[0].plot(x, _mean, label="mean")
    ax[0].plot(x, _median, label="median")
    ax[0].plot(x, _mode, label="mode")
    ax[0].legend()

    ax[1].set_title("Weibull summary statistic DIFFERENCE trends with linearly increasing parameters")
    meamed = pt.abs(_mean - _median)
    meamo = pt.abs(_mean - _mode)
    medmo = pt.abs(_median - _mode)

    ax[1].plot(x, meamed, label="mean - median")
    ax[1].plot(x, meamo, label="mean - mode")
    ax[1].plot(x, medmo, label="median - mode")
    ax[1].legend()


def plot_softplus_behavior(x=pt.arange(-10, 10, step=0.01)):
    _, ax = plt.subplots(figsize=(20, 10))

    ax.set_title(r"$Softplus(x; \beta) = \frac{1}{\beta} \log(1 + \exp(\beta \cdot x))$ behavior around 0 ")
    ax.plot(x, F.softplus(x), label="softplus")
    ax.legend()


def plot_weibull_summary_trends_in_3D():
    x = pt.arange(0, 5, step=0.01)
    y = pt.arange(0, 5, step=0.01)
    alpha = pt.exp(x)
    beta = F.softplus(y)

    # global state, yay.
    plt.figure(figsize=(20, 15))
    ax = plt.axes(projection="3d")
    ax.view_init(40, -30)
    x, y = pt.meshgrid(x, y)  # for the coordinates
    a, b = pt.meshgrid(alpha, beta)  # for the actual computation

    ax.set_title("Weibull summary statistic trends with SEPARATE linearly increasing parameters")

    _mean = Weibull.mean(Weibull.Params(a, b))
    _median = Weibull.median(Weibull.Params(a, b))
    _mode = Weibull.mode(Weibull.Params(a, b))

    ax.plot_surface(x, y, _mean, label="mean")
    ax.plot_surface(x, y, _median, label="median")
    ax.plot_surface(x, y, _mode, label="mode")
    # ax.legend()


plot_weibull_summary_trends_in_3D()


plt.show()
