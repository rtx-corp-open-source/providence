# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from typing import Tuple

"""
Placeholder for doc string

reference paper: https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf for more information
"""


def loglike_discrete(alpha: Tensor, beta: Tensor, y: Tensor, censor: Tensor, epsilon: float = 1e-7, clamp_min=1e-5, clamp_max=30.0):
    """
    Discrete weibull loglikelihood function. The idea here is we want to know the marginal hazard
    from time t to t+1. We constrain the values in order to avoid exploding gradients

    :param alpha: Tensor of Weibull alpha values
    :param beta: Tensor of Weibull beta values
    :param y: Tensor of time values
    :param censor: Tensor of boolean censor values
    :param epsilon: int value for epsilon
    :param clamp_min: minimum float value of hazard tensor
    :param clamp_max: maximum float value of hazard tensor
    :return: Tensor of logliklihood values

    """
    hazard0 = ((y + epsilon) / alpha) ** beta
    hazard1 = ((y + 1.0) / alpha) ** beta
    hazards = torch.clamp(hazard1 - hazard0, min=1e-5, max=30)  # constrain the values or else this going to get ugly
    loglikelihoods = censor * torch.log(torch.exp(hazards) - 1) - hazard1
    return loglikelihoods


def loglike_continuous(y: Tensor, censor: Tensor, alpha: Tensor, beta: Tensor, epsilon: float = 1e-7) -> Tensor:
    """
    Continuous weibull loglikelihood function.
    :param alpha: Tensor of Weibull alpha values
    :param beta: Tensor of Weibull beta values
    :param y: Tensor of time values
    :param censor: Tensor of boolean censor values
    :param epsilon: int value for epsilon
    :return: Tensor of logliklihood values
    """
    epsilon = Tensor([epsilon])
    ya = (y + epsilon) / alpha
    loglikelihoods = censor * (torch.log(beta) + beta * torch.log(ya)) - torch.pow(ya, beta)
    return loglikelihoods


def cumulative_hazard(y: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """
    Reduced cumulative hazard function for the weibull distribution

    :param alpha: Tensor of Weibull alpha values
    :param beta: Tensor of Weibull beta values
    :param y: Tensor of time values

    :return: Tensor of cumulative hazards (failures) at time y
    """
    return torch.pow(y / alpha, beta)


def median(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Median function for a two parameter Weibull

    Reference: https://www.itl.nist.gov/div898/handbook/apr/section1/apr162.htm
    https://web.archive.org/web/20201103014419/https://www.itl.nist.gov/div898/handbook/apr/section1/apr162.htm

    :param alpha: Weibull alpha parameter
    :param beta: Weibull beta parameter

    :return: Tensor of Weibull median values

    """
    return alpha * torch.pow(-torch.log(torch.tensor(0.5)), (1 / beta))


def mode(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Mode function for a two parameter Wiebull

    :param alpha: Weibull alpha parameter
    :param beta: Weibull beta parameter

    :return: Tensor of Weibull mode values
    """
    return alpha * ((beta - 1) / beta) ** (1 / beta)


def mean(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Mean function for a two parameter Wiebull

    :param alpha: Weibull alpha parameter
    :param beta: Weibull beta parameter

    :return: Tensor of Weibull mean values
    """
    return alpha * (1 + 1 / beta).lgamma().exp()


def pdf(alpha: torch.Tensor, beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Probability density for a two parameter Wiebull at time t

    :param alpha: Weibull alpha parameter
    :param beta: Weibull beta parameter
    :param t: Tensor of time values

    :return: Tensor of Weibull mode values
    """
    return (beta / alpha) * (t / alpha) ** (beta - 1.0) * torch.exp(-1.0 * (t / alpha) ** beta)


def compute_distribution_params(model: torch.nn.Module, features: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    Helper function to extract Weibull parameters alpha and beta

    :param model: Providence-trained model
    :param features: tensor of features for device

    :return: Tuple of alpha and beta values
    """

    if model.training:
        model.eval()

    with torch.no_grad():
        a, b = model(features.unsqueeze(1), torch.Tensor([features.shape[0]]))
    alpha, beta = a.detach()[:, 0, 0], b.detach()[:, 0, 0]

    return alpha, beta
