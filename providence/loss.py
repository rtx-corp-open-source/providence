"""
Strong typing around the Providence loss function.
Outlines the interface such a function would follow (and creates an abstract base class if you would rather follow the inheritance route)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from abc import ABC, abstractmethod
from typing import List, Protocol, Tuple, Union

import torch as pt
from torchtyping import TensorType

from providence.distributions import SurvivalAnalysisDistribution, Weibull, Weibull3


class ProvidenceLossInterface(Protocol):
    """The interface for functions that which to be invoked for a Providence loss"""
    def __call__(
        self, params: SurvivalAnalysisDistribution.Params, y: TensorType["time"], censor: TensorType["time"],
        x_lengths: Union[List[int], TensorType["time"]]
    ) -> Union[float, Tuple[float, ...]]:
        ...


class ProvidenceLoss(pt.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, params: SurvivalAnalysisDistribution.Params, y: TensorType["time"], censor: TensorType["time"],
        x_lengths: Union[List[int], TensorType["time"]]
    ) -> Union[float, Tuple[float, ...]]:
        ...


def discrete_weibull_mse(params: Weibull.Params, tte: TensorType["time"]) -> float:
    "Compute the RMSE of the Weibull predictions, treating the tte column as the correct prediction"
    assert params.alpha.shape == tte.shape

    tte_preds = Weibull.mode(params)
    mse = pt.nanmean((tte_preds - tte) ** 2)
    return mse


class DiscreteWeibullMSELoss(ProvidenceLoss):
    """Objectified wrapper around `discrete_weibull_rmse` (see that for more)."""
    def forward(
        self, params: SurvivalAnalysisDistribution.Params, y: TensorType["time"], _censor: TensorType["time"],
        _x_lengths: Union[List[int], TensorType["time"]]
    ) -> Union[float, Tuple[float, ...]]:
        return discrete_weibull_mse(params, y)


def discrete_weibull_loss_fn(
    params: Weibull.Params, y: TensorType["time"], censor: TensorType["time"], x_lengths: TensorType["batch"], epsilon=1e-7
) -> float:
    """Discrete loglikelihood loss for a weibull curve prediction"""
    # consider an update:  weight by tte (normalized or inverse log) so predictions are heavily weighted for TTE -> 0
    # (This requires parameters (sequence length, censor bool, tte))
    # follow-up / branch work: prefer undershooting to overshooting
    if not isinstance(x_lengths, pt.Tensor): x_lengths = pt.tensor(x_lengths, dtype=pt.long)

    loglikelihoods = Weibull.loglikelihood_discrete(params, y, censor, epsilon=epsilon)
    # TODO: (future work) weigh the loglikelihoods

    max_length, batch_size, *trailing_dims = loglikelihoods.size()

    ranges = pt.arange(max_length, dtype=pt.long, device=loglikelihoods.device)
    ranges = ranges.unsqueeze_(1).expand(-1, batch_size)

    lengths = x_lengths.detach().clone().to(loglikelihoods.device, dtype=pt.long)
    lengths = lengths.unsqueeze_(0).expand_as(ranges)
    # print(f"{loglikelihoods.device = } {ranges.device = } {lengths.device = }")

    mask = ranges < lengths
    mask = mask.unsqueeze_(-1).expand_as(loglikelihoods)

    return -1 * pt.mean(loglikelihoods * mask)


class DiscreteWeibullLoss(ProvidenceLoss):
    """Objectified wrapper around `discrete_weibull_loss`"""
    def __init__(self, *, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, params: SurvivalAnalysisDistribution.Params, y: TensorType["time"], censor: TensorType["time"],
        x_lengths: Union[List[int], TensorType["time"]]
    ) -> Union[float, Tuple[float, ...]]:
        return discrete_weibull_loss_fn(params, y, censor, x_lengths, epsilon=self.epsilon)


def discrete_weibull3_loss_fn(
    params: Weibull3.Params, y: TensorType["time"], censor: TensorType["time"], x_lengths: TensorType["batch"], epsilon=1e-7
) -> float:
    """Discrete loglikelihood loss for a weibull curve prediction"""
    if not isinstance(x_lengths, pt.Tensor): x_lengths = pt.tensor(x_lengths, dtype=pt.long)

    loglikelihoods = Weibull3.loglikelihood_discrete(params, y, censor, epsilon=epsilon)

    max_length, batch_size, *trailing_dims = loglikelihoods.size()

    ranges = pt.arange(max_length, dtype=pt.long, device=loglikelihoods.device)
    ranges = ranges.unsqueeze_(1).expand(-1, batch_size)

    lengths = x_lengths.detach().clone().to(loglikelihoods.device, dtype=pt.long)
    lengths = lengths.unsqueeze_(0).expand_as(ranges)

    mask = ranges < lengths
    mask = mask.unsqueeze_(-1).expand_as(loglikelihoods)

    return -1 * pt.mean(loglikelihoods * mask)


class DiscreteWeibull3Loss(ProvidenceLoss):
    """Objectified wrapper around `discrete_weibull_loss`"""
    def __init__(self, *, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self, params: SurvivalAnalysisDistribution.Params, y: TensorType["time"], censor: TensorType["time"],
        x_lengths: Union[List[int], TensorType["time"]]
    ) -> Union[float, Tuple[float, ...]]:
        return discrete_weibull3_loss_fn(params, y, censor, x_lengths, epsilon=self.epsilon)
