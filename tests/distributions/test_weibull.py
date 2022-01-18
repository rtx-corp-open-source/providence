# -*- coding: utf-8 -*-
import pytest
import torch

from providence.distributions import weibull


@pytest.fixture
def y() -> torch.Tensor:
    return torch.Tensor([10.0])


@pytest.fixture
def censor() -> torch.Tensor:
    return torch.Tensor([1.0])


@pytest.fixture
def alpha() -> torch.Tensor:
    return torch.Tensor([2.0])


@pytest.fixture
def beta() -> torch.Tensor:
    return torch.Tensor([2.0])


class TestWeibull:
    def test_cumulative_hazard(self, y, alpha, beta):
        got = weibull.cumulative_hazard(y, alpha, beta)
        want = torch.Tensor([25.0])

        assert got.equal(want)

    def test_loglike_discrete_uncensored(self, y, alpha, beta, censor):
        got = weibull.loglike_discrete(alpha, beta, y, censor)
        want = torch.Tensor([-25.0053])

        assert torch.allclose(got, want)

    def test_loglike_discrete_censored(self, y, alpha, beta):
        got = weibull.loglike_discrete(alpha, beta, y, torch.Tensor([0.0]))
        want = torch.Tensor([-30.2500])

        assert torch.allclose(got, want)

    def test_loglike_continuous(self, y, alpha, beta, censor):
        got = weibull.loglike_continuous(y, censor, alpha, beta)
        want = torch.Tensor([-21.0880])

        assert torch.allclose(got, want)

    def test_mode(self, alpha, beta):
        got = weibull.mode(alpha, beta)
        want = torch.tensor([1.4142])

        assert torch.allclose(got, want)

    def test_mean(self, alpha, beta):
        got = weibull.mean(alpha, beta)
        want = torch.tensor([1.772454])

        assert torch.allclose(got, want)

    def test_pdf(self, alpha, beta):
        got = weibull.pdf(alpha, beta, torch.tensor([1]))
        want = torch.tensor([0.3894])

        assert torch.allclose(got, want)
