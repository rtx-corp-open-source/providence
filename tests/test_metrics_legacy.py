# -*- coding: utf-8 -*-
"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import pytest
import numpy as np
import pandas as pd
import torch as pt
from providence import metrics
from providence.distributions import Weibull
from providence.nn.rnn import ProvidenceLSTM


@pytest.mark.parametrize(
    "metric_function,y_true,y_pred,want",
    [
        (metrics.mse, pd.Series(range(1, 11)), pd.Series(range(2, 12)), 1),
        (metrics.mse, np.array(range(1, 11)), np.array(range(2, 12)), 1),
        # NOTE see comment in metrics.py
        # (metrics.mse, torch.Tensor(range(1, 11)), torch.Tensor(range(2, 12)), 1),
        (metrics.smape, pd.Series(range(1, 11)), pd.Series(np.arange(1, 11)), 0),
        (metrics.smape, np.array(range(1, 11)), np.array(range(1, 11)), 0),
        (metrics.smape, pt.Tensor(range(1, 11)), pt.Tensor(range(1, 11)), 0),
        (metrics.mfe, pd.Series(range(1, 11)), pd.Series(range(2, 12)), -1),
        (metrics.mfe, np.array(range(1, 11)), np.array(range(2, 12)), -1),
        (metrics.mfe, pt.Tensor(range(1, 11)).numpy(), pt.Tensor(range(2, 12)).numpy(), -1),
        (metrics.smpe, pd.Series(range(1, 11)), pd.Series(np.arange(1, 11)), 0),
        (metrics.smpe, np.array(range(1, 11)), np.array(range(1, 11)), 0),
        (metrics.smpe, pt.Tensor(range(1, 11)), pt.Tensor(range(1, 11)), 0),
        (metrics.smape, np.array([0]), np.array([0]), 0),
        (metrics.smpe, np.array([0]), np.array([0]), 0),
    ],
)
def test_metric(metric_function, y_true, y_pred, want):

    got = metric_function(y_true, y_pred)

    assert got == want


@pytest.fixture
def alpha() -> pt.Tensor:
    return pt.Tensor([2.0])


@pytest.fixture
def beta() -> pt.Tensor:
    return pt.Tensor([2.0])


@pytest.fixture
def tte_target():
    return pt.Tensor([[5, 1]])



class TestComputeWeibullParams:
    def test_model_is_eval(self):
        model = ProvidenceLSTM(10, 5)
        random_input = pt.rand((2, 10))
        _, _ = Weibull.compute_distribution_parameters(model, random_input)

        assert model.training == False, "Testing internal state mutation"

    def test_no_grad(self):

        model = ProvidenceLSTM(10, 5)
        random_input = pt.rand((2, 10))
        alpha, beta = Weibull.compute_distribution_parameters(model, random_input)

        assert alpha.requires_grad == False
        assert beta.requires_grad == False

    def test_output_shape(self):

        model = ProvidenceLSTM(10, 5)
        random_input = pt.rand((2, 10))
        alpha, beta = Weibull.compute_distribution_parameters(model, random_input)

        assert len(alpha.shape) == 1
        assert len(beta.shape) == 1


class TestNasaScore:
    def test_equal(self):

        y_true = pd.Series([10, 20, 30, 40, 50])
        y_pred = pd.Series([10, 20, 30, 40, 50])
        score_equal = metrics.score_phm08(y_true, y_pred)
        score_equal_expected = 0

        assert np.isclose(score_equal, score_equal_expected)
    
    def test_over(self):

        y_true = pd.Series([33, 20])
        y_pred = y_true + 10
        score_over = metrics.score_phm08(y_true, y_pred)
        score_over_expected = 2*(np.exp(1) - 1)

        assert np.isclose(score_over, score_over_expected)
    
    def test_under(self):

        y_true = pd.Series([33, 40])
        y_pred = y_true - 13
        score_under = metrics.score_phm08(y_true, y_pred)
        score_under_expected = 2*(np.exp(1) - 1)

        assert np.isclose(score_under, score_under_expected)
    
    def test_under_less_over_same(self):

        y_true = pd.Series([52, 55, 49, 103, 72])
        y_diff = pd.Series([12, 2, 7, 33, 16])
        y_pred_over = y_true + y_diff
        y_pred_under = y_true - y_diff
        score_over_comp = metrics.score_phm08(y_true, y_pred_over)
        score_under_comp = metrics.score_phm08(y_true, y_pred_under)

        assert np.less(score_under_comp, score_over_comp)