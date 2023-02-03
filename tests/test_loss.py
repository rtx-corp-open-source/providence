"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import pytest
import torch as pt
from providence.dataloaders import ProvidenceDataLoader
from torchtyping import TensorType

from providence.loss import DiscreteWeibullLoss, discrete_weibull3_loss_fn, discrete_weibull_loss_fn
from providence.training import unpack_label_and_censor
from providence.distributions import Weibull, Weibull3
from providence.nn.weibull import Weibull3Head
from providence.loss import DiscreteWeibullMSELoss, discrete_weibull_mse


@pytest.fixture
def y() -> pt.Tensor:
    return pt.Tensor([10.0])


@pytest.fixture
def censor() -> pt.Tensor:
    return pt.Tensor([1.0])


@pytest.fixture
def weibull2_params() -> Weibull.Params:
    return Weibull.Params(alpha=pt.Tensor([2.0]), beta=pt.Tensor([2.0]))


class TestDiscreteWeibull:

    def test_rmse_loss(self, y : TensorType["steps"], weibull2_params: Weibull.Params):
        _rmse = discrete_weibull_mse(weibull2_params, y)
        print(f"{_rmse = }")
    
    def test_rmse_loss_objectified(self, simple_weibull_model, simple_providence_ds):
        dl = ProvidenceDataLoader(simple_providence_ds, batch_size=1)
        loss_fn = DiscreteWeibullMSELoss() # NOTE: if Weibull.mean is used, there's no nan
        features, lengths, targets = next(iter(dl))
        model_output = simple_weibull_model.to(pt.float64)(features.to(pt.float64), lengths)
        print(f"{model_output = }")

        y_true, censor_ = unpack_label_and_censor(targets)
        _weibull_rmse_loss = loss_fn(Weibull.Params(*model_output), y_true, censor_, lengths)
        print(f"{_weibull_rmse_loss = }")

    def test_weibull_loss(self, simple_weibull_model, simple_providence_ds):
        dl = ProvidenceDataLoader(simple_providence_ds, batch_size=2)
        features, lengths, targets = next(iter(dl))
        model_output = simple_weibull_model.to(pt.float64)(features.to(pt.float64), lengths)

        y_true, censor_ = unpack_label_and_censor(targets)
        _weibull_loss = discrete_weibull_loss_fn(Weibull.Params(*model_output), y_true, censor_, lengths)
        print(f"{_weibull_loss = }")

    
    def test_weibull_loss_objectified(self, simple_weibull_model, simple_providence_ds):
        dl = ProvidenceDataLoader(simple_providence_ds, batch_size=1)
        loss_fn = DiscreteWeibullLoss()
        features, lengths, targets = next(iter(dl))
        model_output = simple_weibull_model.to(pt.float64)(features.to(pt.float64), lengths)

        y_true, censor_ = unpack_label_and_censor(targets)
        _weibull_loss = loss_fn(Weibull.Params(*model_output), y_true, censor_, lengths)
        print(f"{_weibull_loss = }")

    def test_weibull3_loss(self, simple_weibull_model, simple_providence_ds):
        dl = ProvidenceDataLoader(simple_providence_ds, batch_size=2)
        features, lengths, targets = next(iter(dl))
        simple_weibull_model.out = Weibull3Head()
        model_output = simple_weibull_model.to(pt.float64)(features.to(pt.float64), lengths)

        y_true, censor_ = unpack_label_and_censor(targets)
        _weibull_loss = discrete_weibull3_loss_fn(Weibull3.Params(*model_output), y_true, censor_, lengths)
        print(f"{_weibull_loss = }")

    