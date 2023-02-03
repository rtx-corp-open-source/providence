"""
This module captures the spirit of a distribution we would consider for survival analysis and time-to-event modeling,
and prescribes a method that make it readily usable without constructing hundreds of objects (which is slow).

If you follow after `SurvivalAnalysisDistribution` (no leading underscore, `_`), you shoud be able to properly implement
a new distribution.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from abc import ABC, abstractmethod
from typing import List, NamedTuple, Tuple

from torch import Tensor, BoolTensor
import torch as pt
from torch.nn import Module as TorchModule

from providence.utils import TODO


class _SurvivalAnalysisDistribution(ABC):
    DistributionParams = Tuple[Tensor, ...]

    @classmethod
    @abstractmethod
    def cumulative_hazard(cls, params: DistributionParams, t: Tensor) -> Tensor:
        """Computes the cumulative hazard from a tensor of time steps t"""

    @classmethod
    @abstractmethod
    def loglikelihood_discrete(cls, params: DistributionParams, t: Tensor, censor: BoolTensor):
        ...

    @classmethod
    @abstractmethod
    def loglikelihood_continuous(cls, params: DistributionParams, t: Tensor, censor: BoolTensor):
        ...

    @classmethod
    @abstractmethod
    def mean(cls, params: DistributionParams) -> Tensor:
        ...

    @classmethod
    @abstractmethod
    def median(cls, params: DistributionParams) -> Tensor:
        ...

    @classmethod
    @abstractmethod
    def mode(cls, params: DistributionParams) -> Tensor:
        ...

    @classmethod
    @abstractmethod
    def pdf(cls, params: DistributionParams, t: Tensor) -> Tensor:
        """Computes the PDF i.e. the highest point in the probability curve at times `t`"""

    @classmethod
    @abstractmethod
    def compute_distribution_parameters(cls, model: TorchModule, features: Tensor) -> DistributionParams:
        """
        Extract the parameters that characterize this distribution.

        Arguments:
            model: Providence-trained Model. Could/should also be a ProvidenceModule
            features: tensor of features for some given device

        Returns: DistributionParams, a consistent-length tuple for this distribution, though can differ between
        child classes.
        """
        ...


class SurvivalAnalysisDistribution(_SurvivalAnalysisDistribution, ABC):
    """
    Abstract root, encapsulating the notion of contract of functions that a given distribution supports.
    This gets weird with the typer checkers of today, because while the following works

    def func(d: SurvivalAnalysisDistribution) -> None:
        assert isinstance(d, SurvivalAnalysisDistribution)
    """
    class Params(NamedTuple):
        """
        All SurvivalAnalysisDistribution child types are expected to have some child Parameters-like named tuple,
        to comply with `DistributionParams`
        """
        ...

    @classmethod
    def parameter_names(cls) -> List[str]:
        "Returns the names of the statistical parameters, which are the parameters of this type's NamedTuple"
        return list(cls.Params._fields)


class Weibull(SurvivalAnalysisDistribution):
    """
    The 2-parameter Weibull distribution, with the motivating usage of facilitating Providence's learning.

    Herein we use beta ** -1 == (1 / beta) for beta's multiplicative inverse because it's strictly faster on CPU as
    measured by a test battery akin to scipy's guantlet. However, this is only for the 1 / x case, not the general
    x_1 / x_0 case. For that, it is faster to do the division, rather than introduce a second operation
    """
    class Params(NamedTuple):
        """local version of DistributionParams, specified for the Weibull distribution leveraged by Providence,
        Arguments:
            alpha: Tensor of the alpha values
            beta: Tensor of the beta values
        There is an implicit assumption that alpha.size(0) == beta.size(0)
        """
        alpha: Tensor
        beta: Tensor

    @classmethod
    def cumulative_hazard(cls, params: Params, t: Tensor) -> Tensor:
        return pt.pow(t / params.alpha, params.beta)

    @classmethod
    def loglikelihood_discrete(
        cls,
        params: Params,
        t: Tensor,
        censor: BoolTensor,
        epsilon: float = 1e-7,
        clamp_min: float = 1e-5,
        clamp_max: float = 30.0
    ):
        """
        This is the loglikelihood for the discretized survival function for the Weibull distribution.
        """
        #   hazard = (beta / eta) * ((x - k) / eta)**(beta-1)
        hazard0 = Weibull.cumulative_hazard(params, t + epsilon)  # ((t + epsilon) / alpha)**beta
        hazard1 = Weibull.cumulative_hazard(params, t + 1.0)  # ((t + 1.0) / alpha)**beta
        # constrain the values or else this going to get ugly
        hazards = pt.clamp(hazard1 - hazard0, min=clamp_min, max=clamp_max)
        loglikelihoods = pt.xlogy(censor, pt.exp(hazards) - 1) - hazard1  # faster than the original
        # loglikelihoods = censor * pt.log(pt.exp(hazards) - 1) - hazard1
        return loglikelihoods

    @classmethod
    def loglikelihood_continuous(cls, params: Params, t: Tensor, censor: BoolTensor, epsilon: float = 1e-7):
        alpha, beta = params
        ya = (t + epsilon) / alpha
        loglikelihoods = censor * (pt.log(beta) + pt.xlogy(beta, ya)) - pt.pow(ya, beta)  # faster
        # loglikelihoods = censor * (pt.log(beta) + beta * pt.log(ya)) - pt.pow(ya, beta)
        # loglikelihoods = censor * pt.log(beta) + beta * pt.log(t / alpha) - (t / alpha) ** beta
        return loglikelihoods

    @classmethod
    def mean(cls, params: Params) -> Tensor:
        alpha, beta = params
        return alpha * (1 + (beta**-1)).lgamma().exp()

    @classmethod
    def median(cls, params: Params) -> Tensor:
        alpha, beta = params
        return alpha * pt.pow(-pt.log(pt.tensor(0.5)), (beta**-1))

    @classmethod
    def mode(cls, params: Params) -> Tensor:
        alpha, beta = params
        # use beta ** -1 == (1 / beta) because it's strictly faster on CPU.
        # This is only for the 1 / x case, not x_1 / x_0 case
        return alpha * ((beta - 1) / beta)**(beta**-1)

    @classmethod
    def pdf(cls, params: Params, t: Tensor) -> Tensor:
        alpha, beta = params
        ch = cls.cumulative_hazard(cls.Params(alpha, beta - 1), t)  # (t / alpha)**(beta - 1.0)
        exp_ch = cls.cumulative_hazard(params, t)  # (t / alpha)**beta
        return (beta / alpha) * ch * pt.exp(-1.0 * exp_ch)

    @classmethod
    @pt.no_grad()
    def compute_distribution_parameters(cls, model: TorchModule, features: Tensor) -> Params:
        model.eval()
        # features go from [time, datum] to [time, batch of 1, datum]
        a, b = model(features.unsqueeze(1), pt.tensor([features.shape[0]], dtype=pt.long))
        # because model output size = 1: for all time, for that batch of 1, take out that data
        alpha, beta = a.detach()[:, 0, 0], b.detach()[:, 0, 0]
        return Weibull.Params(alpha, beta)


# NOTE: the design of the above is oriented around a DiscreteWeibull, even though we have means to support another
def _weibull3_cumulative_hazard(x, k, beta, eta):
    """This is the cumulative hazard for the 3-parameter Weibull, which includes the translation parameter k.
    Though the previous implementations have a likelihood_continuous, it's not supported in the cumulative hazard framing.
    """
    # Compute the cumulative hazard function
    cumulative_hazard = (beta / (beta - 1)) * ((x - k) / eta)**(beta - 1)

    return cumulative_hazard


class Weibull3(SurvivalAnalysisDistribution):
    """Discretization of the Weibull distribution, introducing a location / translation / shift parameter: k.
    Using this distribution with k=0 should be functionally equivalent to using the """
    class Params(NamedTuple):
        """local version of DistributionParams, holding the parameters for scale (alpha), shape (beta), and location (k).
        In the literature, you may find that alpha is (lowercase) theta or omega, and k is (lowercase) gamma or 'a'.
        """
        alpha: Tensor
        beta: Tensor
        k: Tensor

    @classmethod
    def cumulative_hazard(cls, params: Params, t: Tensor) -> Tensor:
        return pt.pow((t - params.k) / params.alpha, params.beta)

    @classmethod
    def loglikelihood_discrete(
        cls,
        params: Params,
        t: Tensor,
        censor: BoolTensor,
        epsilon: float = 1e-7,
        clamp_min: float = 1e-5,
        clamp_max: float = 30.0
    ):
        """
        This is the loglikelihood for the discretized survival function for the Weibull distribution.
        """
        hazard0 = Weibull.cumulative_hazard(params, t + epsilon)  # ((t + epsilon) / alpha)**beta
        hazard1 = Weibull.cumulative_hazard(params, t + 1.0)  # ((t + 1.0) / alpha)**beta
        # constrain the values or else this going to get ugly
        hazards = pt.clamp(hazard1 - hazard0, min=clamp_min, max=clamp_max)
        loglikelihoods = pt.xlogy(censor, pt.exp(hazards) - 1) - hazard1  # faster than the original
        return loglikelihoods

    @classmethod
    def loglikelihood_continuous(cls, params: Params, t: Tensor, censor: BoolTensor, epsilon: float = 1e-7):
        """
        Does not leverage the (beta / (beta-1)) term because that is not what Martinnson uses in his paper.
        Further survival analysis research is necessary.
        """
        alpha, beta, k = params
        ya = (t + epsilon - k) / alpha
        loglikelihoods = censor * (pt.xlogy(beta, ya) + pt.log(beta)) - pt.pow(ya, beta)  # faster
        return loglikelihoods

    @classmethod
    def mean(cls, params: Params) -> Tensor:
        alpha, beta, k = params
        return k + alpha * (1 + (beta**-1)).lgamma().exp()

    @classmethod
    def median(cls, params: Params) -> Tensor:
        """Median of the Weibull.
        This implementation is uncertainty, but extrapolated from the implementation of mode and mean, which sum-in the translational shift
        """
        alpha, beta, k = params
        return k + alpha * pt.pow(pt.log(pt.tensor(2)), (beta**-1))

    @classmethod
    def mode(cls, params: Params) -> Tensor:
        alpha, beta, k = params
        _mode = k + alpha * ((beta - 1) / beta)**(beta**-1)
        return pt.where(beta <= 0, _mode, k)

    @classmethod
    def pdf(cls, params: Params, t: Tensor) -> Tensor:
        alpha, beta, k = params
        ch = cls.cumulative_hazard(params._replace(beta=beta - 1), t)  # ((t - k)/ alpha)**(beta - 1.0)
        exp_ch = cls.cumulative_hazard(params, t)  # ((t - k )/ alpha)**beta
        return (beta / alpha) * ch * pt.exp(-1.0 * exp_ch)

    @classmethod
    @pt.no_grad()
    def compute_distribution_parameters(cls, model: TorchModule, features: Tensor) -> Params:
        model.eval()
        # features go from [time, datum] to [time, batch of 1, datum]
        a, b, k_ = model(features.unsqueeze(1), pt.tensor([features.shape[0]], dtype=pt.long))
        # because model output size = 1: for all time, for that batch of 1, take out that data
        alpha, beta, k = a.detach()[:, 0, 0], b.detach()[:, 0, 0], k_.detach()[:, 0, 0]
        return Weibull3.Params(alpha, beta, k)


class GeneralizedWeibull(SurvivalAnalysisDistribution):
    """
    The Generalized Weibull (aka Anchored Amorosso [Amorosso ≡ Generalized Gamma]) extends the representational power of the
    continuous and discrete Weibull.
    """
    class Params(NamedTuple):
        """local version of DistributionParams, specified for the Weibull distribution leveraged by Providence,
        Arguments:
            alpha: Tensor of the alpha values
            beta: Tensor of the beta values
            eta: Tensor of the eta values
            omega: Tensor of the omega values
        There is an implicit assumption that alpha.size(0) == beta.size(0)
        """
        alpha: Tensor
        beta: Tensor
        eta: Tensor
        omega: Tensor


GeneralizedWeibull.__init__ = lambda: TODO("Don't use this type or suffer this consequence. REPEATEDLY")
