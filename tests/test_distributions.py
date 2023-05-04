# -*- coding: utf-8 -*-
"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import pytest
import torch as pt

from providence.distributions import Weibull


@pytest.fixture
def y() -> pt.Tensor:
    return pt.Tensor([10.0])


@pytest.fixture
def censor() -> pt.Tensor:
    return pt.Tensor([1.0])


@pytest.fixture
def weibull_params() -> Weibull.Params:
    return Weibull.Params(alpha=pt.Tensor([2.0]), beta=pt.Tensor([2.0]))


class TestWeibull:
    def test_cumulative_hazard(self, y, weibull_params):
        got = Weibull.cumulative_hazard(weibull_params, y)
        want = pt.Tensor([25.0])

        assert got.equal(want)

    def test_loglike_discrete_uncensored(self, y, weibull_params, censor):
        got = Weibull.loglikelihood_discrete(weibull_params, y, censor)
        want = pt.Tensor([-25.0053])

        assert pt.allclose(got, want)

    def test_loglikelihood_discrete_censored(self, y, weibull_params):
        got = Weibull.loglikelihood_discrete(weibull_params, y, pt.Tensor([0.0]))
        want = pt.Tensor([-30.2500])

        assert pt.allclose(got, want)

    def test_loglikelihood_continuous(self, y, weibull_params, censor):
        got = Weibull.loglikelihood_continuous(weibull_params, y, censor)
        want = pt.Tensor([-21.0880])

        assert pt.allclose(got, want)

    def test_mode(self, weibull_params):
        got = Weibull.mode(weibull_params)
        want = pt.tensor([1.4142])

        assert pt.allclose(got, want)

    def test_mean(self, weibull_params):
        got = Weibull.mean(weibull_params)
        want = pt.tensor([1.772454])

        assert pt.allclose(got, want)

    def test_pdf(self, weibull_params):
        got = Weibull.pdf(weibull_params, pt.tensor([1]))
        want = pt.tensor([0.3894])

        assert pt.allclose(got, want)

    def test_likelihood_sequence_mixed_censor(self, y, weibull_params):
        test_seq_length = 5
        new_params = Weibull.Params(
            weibull_params.alpha.tile(test_seq_length),
            weibull_params.beta.tile(test_seq_length),
        )
        censor_indicator = pt.randint(0, 1, (5,), dtype=pt.float)
        got = Weibull.loglikelihood_discrete(new_params, y, censor_indicator)
        want = pt.where(censor_indicator.to(pt.bool), -25.0053, -30.25)

        assert pt.allclose(got, want)
