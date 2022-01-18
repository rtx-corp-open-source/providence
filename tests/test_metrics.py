# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from lifelines.utils import concordance_index
from typing import List, Optional, Tuple, Union, Dict
import torch
from torch import Tensor, nn
from providence import metrics
from providence.distributions import weibull
from providence.model.lstm import ProvidenceLSTM


@pytest.mark.parametrize(
    "metric_function,y_true,y_pred,want",
    [
        (metrics.mse, pd.Series(range(1, 11)), pd.Series(range(2, 12)), 1),
        (metrics.mse, np.array(range(1, 11)), np.array(range(2, 12)), 1),
        (metrics.mse, torch.Tensor(range(1, 11)), torch.Tensor(range(2, 12)), 1),
        (metrics.smape, pd.Series(range(1, 11)), pd.Series(np.arange(1, 11)), 0),
        (metrics.smape, np.array(range(1, 11)), np.array(range(1, 11)), 0),
        (metrics.smape, torch.Tensor(range(1, 11)), torch.Tensor(range(1, 11)), 0),
        (metrics.mfe, pd.Series(range(1, 11)), pd.Series(range(2, 12)), -1),
        (metrics.mfe, np.array(range(1, 11)), np.array(range(2, 12)), -1),
        (metrics.mfe, torch.Tensor(range(1, 11)), torch.Tensor(range(2, 12)), -1),
        (metrics.smpe, pd.Series(range(1, 11)), pd.Series(np.arange(1, 11)), 0),
        (metrics.smpe, np.array(range(1, 11)), np.array(range(1, 11)), 0),
        (metrics.smpe, torch.Tensor(range(1, 11)), torch.Tensor(range(1, 11)), 0),
        (metrics.smape, np.array([0]), np.array([0]), 0),
        (metrics.smpe, np.array([0]), np.array([0]), 0),
    ],
)
def test_metric(metric_function, y_true, y_pred, want):

    got = metric_function(y_true, y_pred)

    assert got == want


@pytest.fixture
def alpha() -> torch.Tensor:
    return torch.Tensor([2.0])


@pytest.fixture
def beta() -> torch.Tensor:
    return torch.Tensor([2.0])


@pytest.fixture
def tte_target():
    return torch.Tensor([[5, 1]])


class TestGenerateDistributionMeasures:
    def test_cols(self, alpha, beta, tte_target):
        dist_args = {"alpha": alpha, "beta": beta}

        got = metrics.generate_distribution_measures(weibull, dist_args, tte_target)
        want = ["alpha", "beta", "tte", "censor", "mean", "median", "mode"]

        assert list(got.columns) == want

    def test_size(self, alpha, beta, tte_target):
        dist_args = {"alpha": alpha, "beta": beta}

        got = metrics.generate_distribution_measures(weibull, dist_args, tte_target).shape
        want = (1, 7)

        assert got == want


class TestComputeWeibullParams:
    def test_model_is_eval(self):
        model = ProvidenceLSTM(10, 5)
        random_input = torch.rand((2, 10))
        _, _ = weibull.compute_distribution_params(model, random_input)

        assert model.training == False

    def test_no_grad(self):

        model = ProvidenceLSTM(10, 5)
        random_input = torch.rand((2, 10))
        alpha, beta = weibull.compute_distribution_params(model, random_input)

        assert alpha.requires_grad == False
        assert beta.requires_grad == False

    def test_output_shape(self):

        model = ProvidenceLSTM(10, 5)
        random_input = torch.rand((2, 10))
        alpha, beta = weibull.compute_distribution_params(model, random_input)

        assert len(alpha.shape) == 1
        assert len(beta.shape) == 1
