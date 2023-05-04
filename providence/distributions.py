"""
This module captures the spirit of a distribution we would consider for survival analysis and time-to-event modeling,
and prescribes a method that make it readily usable without constructing hundreds of objects (which is slow).

If you follow after ``SurvivalAnalysisDistribution`` (no leading underscore, `_`), you shoud be able to properly implement
a new distribution.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from abc import ABC
from abc import abstractmethod
from typing import Generic, List, Protocol, TypeVar
from typing import NamedTuple

import torch as pt
from torch import BoolTensor
from torch import Tensor
from torch.nn import Module as TorchModule

from providence.types import TypeAlias
from providence.utils import TODO


SurvivalDistributionParams: TypeAlias = NamedTuple
T_DistributionParams = TypeVar("T_DistributionParams", bound=SurvivalDistributionParams, contravariant=True)
"""The type of a ``{distribution}.Params`` should be a child of NamedTuple"""


class SurvivalAnalysisDistribution(Generic[T_DistributionParams]):
    @abstractmethod
    def cumulative_hazard(_, params: T_DistributionParams, t: Tensor) -> Tensor:
        """Computes the cumulative hazard from a tensor of time steps t."""

    @abstractmethod
    def loglikelihood_discrete(_, params: T_DistributionParams, t: Tensor, censor: BoolTensor):
        """Compute discrete loglikelihood of this survival analysis distribution.

        For more on the necessity of differentiating the discrete and continuous loglikelihood per distribution,
        we recommend reading Section 2 of Martinsson's 2017 Thesis: "Weibull Time-to-Event Recurrent Neural Network"
        out of the University of Gothenburg.

        Args:
            params (DistributionParams): parameters for this distribution. Packing as a tuple, NamedTuple, or dataclass
                is recommended.
            t (Tensor): timesteps, a sequence of ``t``s, used in place of ``x`` in a probability distribution function
            censor (BoolTensor): sequence indicating which time steps occur before or after the corresponding timestep.
                That is, this should be a binary stream where 0 indicates no event occurs, and 1 indicates that an
                event occurs at TTE=0.

        Returns:
            float: (or FloatTensor, 1-item or broadcasting based on ``t.shape``)
        """
        ...

    @abstractmethod
    def loglikelihood_continuous(_, params: T_DistributionParams, t: Tensor, censor: BoolTensor):
        """Compute continuous loglikelihood of this survival analysis distribution.

        For more on the necessity of differentiating the discrete and continuous loglikelihood per distribution,
        we recommend reading Section 2 of Martinsson's 2017 Thesis: "Weibull Time-to-Event Recurrent Neural Network"
        out of the University of Gothenburg.

        Args:
            params (T_DistributionParams): parameters for this distribution. Packing as a tuple, NamedTuple, or dataclass
                is recommended.
            t (Tensor): time steps, a sequence of ``t``s, used in place of ``x`` in a probability distribution function
            censor (BoolTensor): sequence indicating which time steps occur before or after the corresponding timestep.
                That is, this should be a binary stream where 0 indicates no event occurs, and 1 indicates that an
                event occurs at TTE=0.

        Returns:
            float: (or 1-item FloatTensor)
        """
        ...

    @abstractmethod
    def mean(_, params: T_DistributionParams) -> Tensor:
        """Compute the mean of this survivaly analysis distribution based on the ``params``.

        Args:
            params (T_DistributionParams): parameters for this distribution. Packaging as a tuple, NamedTuple, or
                dataclass is recommended.

        Returns:
            Tensor: a mean result for each set of parameters represented in ``params``, such that the means
                correspond to each grouping (e.g. pair, triple) of values that could compute a mean for this type.
        """
        ...

    @abstractmethod
    def median(_, params: T_DistributionParams) -> Tensor:
        """Compute the median of this survivaly analysis distribution based on the ``params``.

        Args:
            params (T_DistributionParams): parameters for this distribution. Packaging as a tuple, NamedTuple, or
                dataclass is recommended.

        Returns:
            Tensor: a median result for each set of parameters represented in ``params``, such that the medians
                correspond to each grouping (e.g. pair, triple) of values that could compute a median for this type.
        """
        ...

    @abstractmethod
    def mode(_, params: T_DistributionParams) -> Tensor:
        """Compute the mode of this survivaly analysis distribution based on the ``params``.

        Args:
            params (T_DistributionParams): parameters for this distribution. Packaging as a tuple, NamedTuple, or
                dataclass is recommended.

        Returns:
            Tensor: a mode result for each set of parameters represented in ``params``, such that the modes
                correspond to each grouping (e.g. pair, triple) of values that could compute a mode for this type.
        """
        ...

    @abstractmethod
    def pdf(_, params: T_DistributionParams, t: Tensor) -> Tensor:
        """Computes the PDF i.e. the highest point in the probability curve at times ``t``.

        Args:
            params (T_DistributionParams): parameters for this distribution. Packaging as a tuple, NamedTuple, or
                dataclass is recommended.

        Returns:
            Tensor: the values of the probability distribution at values x=``t`` for all parameter groupings
        """
        ...

    @abstractmethod
    def compute_distribution_parameters(_, model: TorchModule, features: Tensor) -> SurvivalDistributionParams:
        """Extract the parameters that characterize this distribution.

        Arguments:
            model: Providence-trained Model. Could/should also be a ProvidenceModule
            features: tensor of features for some given device

        Returns:
            T_DistributionParams, a consistent-length tuple for this distribution, though can differ between
                child classes.
        """
        ...

    @classmethod
    def parameter_names(cls) -> List[str]:
        """Returns the names of the statistical parameters, which are the parameters of this type's NamedTuple."""
        return list(cls.Params._fields) if hasattr(cls, "Params") else []


class SurvivalAnalysisDistribution_Old(SurvivalAnalysisDistribution, ABC):
    """Abstract root, encapsulating the notion of contract of functions that a given distribution supports.

    This gets weird with the typer checkers of today, because while the following works
        >>> def func(d: SurvivalAnalysisDistribution) -> None:
                assert issubclass(d, SurvivalAnalysisDistribution)
        >>> func(Weibull)

    Tools like ``mypy`` reject the current type hierarchy because of their inversion of the mantra
    "child types specialize behavior", by requiring the child types only broaden parameter types instead of supporting
    covariant child types in parameters.
    """

    Params: TypeAlias = T_DistributionParams
    """Parameters for a some child type of this kind.

    Child types should implement their own ``Params`` child type, such that it can be accessed as ``MyDist.Params``.
    """

    @classmethod
    def parameter_names(cls) -> List[str]:
        """Returns the names of the statistical parameters, which are the parameters of this type's NamedTuple."""
        return list(cls.Params._fields)


class Weibull(SurvivalAnalysisDistribution["Weibull.Params"]):
    """The 2-parameter Weibull distribution, with the motivating usage of facilitating Providence's learning.

    Herein we use beta ** -1 == (1 / beta) for beta's multiplicative inverse because it's strictly faster on CPU as
    measured by a test battery akin to scipy's guantlet. However, this is only for the 1 / x case, not the general
    x_1 / x_0 case. For that, it is faster to do the division, rather than introduce a second operation
    """

    class Params(NamedTuple):
        """Distribution parameters for the Weibull distribution leveraged by Providence.

        There is an implicit assumption that alpha.size(0) == beta.size(0)

        Args:
            alpha (Tensor): shape (N,) representing the alpha parameter of the Weibull distribution.
            beta (Tensor): shape (N,) representing the beta parameter of the Weibull distribution.
        """

        alpha: Tensor
        beta: Tensor

    @classmethod
    def cumulative_hazard(cls, params: Params, t: Tensor) -> Tensor:
        """Compute the translated cumulative hazard.

        Args:
            params (Params): parameters tuple, containing alpha, and beta of equal pt.Size or pt.Tensor.shape
            t (Tensor): time steps to evaluate the cumulative hazard

        Returns:
            Tensor: of a shape that matches broadcasting semantics i.e.
                0-order tensor: if all shapes of alpha, beta, and t are of 0-order
                1-order tensor: if all shapes of alpha, beta are 1st order (all must be equal), ``t`` is of 0-order
                    or 1st order if ``t``'s shape is equal to the others.

        Raises:
            RuntimeError: if any of the shapes of the pt.Tensors in ``params`` aren't broadcastable
        """
        return pt.pow(t / params.alpha, params.beta)

    @classmethod
    def loglikelihood_discrete(
        cls,
        params: Params,
        t: Tensor,
        censor: BoolTensor,
        epsilon: float = 1e-7,
        clamp_min: float = 1e-5,
        clamp_max: float = 30.0,
    ):
        """Compute the loglikelihood for the discretized survival function of the Weibull distribution.

        For more on the necessity of differentiating the discrete and continuous loglikelihood per distribution,
        we recommend reading Section 2.5.2 (and the whole of section 2, really) of Martinsson's 2017 Thesis:
        "Weibull Time-to-Event Recurrent Neural Network" out of the University of Gothenburg.
        The computation herein is the implementation of the equation on the second half of page 35 i.e. Section 2.5.2.

        Args:
            params (Params): parameters for this distribution.
            t (Tensor): timesteps, a sequence of ``t``s, used in place of ``x`` in a probability distribution function
            censor (BoolTensor): sequence indicating which time steps occur before or after the corresponding timestep.
                That is, this should be a binary stream where 0 indicates no event occurs, and 1 indicates that an
                event occurs at TTE=0.
            epsilon (float, optional): epsilon added to the denominator of the cumulative hazard to avoid division by 0
                Defaults to 1e-7.
            clamp_min (float, optional): minimum for the difference between hazards. Defaults to 1e-5.
            clamp_max (float, optional): maximum for the difference between hazards. Defaults to 30.0.

        Returns:
            float: (or FloatTensor, 1-item or broadcasting based on ``t.shape``)
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
        """Compute the loglikelihood for the continuous survival function of the Weibull distribution.

        For more on the necessity of differentiating the discrete and continuous loglikelihood per distribution,
        we recommend reading Section 2.5.2 (and the whole of section 2, really) of Martinsson's 2017 Thesis:
        "Weibull Time-to-Event Recurrent Neural Network" out of the University of Gothenburg.
        The computation herein is the implementation of the equation on the first half of page 35 i.e. Section 2.5.2.

        Args:
            params (Params): parameters for this distribution.
            t (Tensor): timesteps, a sequence of ``t``s, used in place of ``x`` in a probability distribution function
            censor (BoolTensor): sequence indicating which time steps occur before or after the corresponding timestep.
                That is, this should be a binary stream where 0 indicates no event occurs, and 1 indicates that an
                event occurs at TTE=0.
            epsilon (float, optional): epsilon added to the denominator of the cumulative hazard to avoid division by 0
                Defaults to 1e-7.

        Returns:
            float: (or FloatTensor, 1-item or broadcasting based on ``t.shape``)
        """
        alpha, beta = params
        ya = (t + epsilon) / alpha
        loglikelihoods = censor * (pt.log(beta) + pt.xlogy(beta, ya)) - pt.pow(ya, beta)  # faster
        # loglikelihoods = censor * (pt.log(beta) + beta * pt.log(ya)) - pt.pow(ya, beta)
        # loglikelihoods = censor * pt.log(beta) + beta * pt.log(t / alpha) - (t / alpha) ** beta
        return loglikelihoods

    @classmethod
    def mean(cls, params: Params) -> Tensor:
        """Compute the mean of a two-parameter Weibull distribution.

        Args:
            params (Params): parameters representing one or more Weibull distributions

        Returns:
            Tensor: shape (N,) containing the mean of the Weibull distributions from the given alpha and beta pairs.
        """
        alpha, beta = params
        return alpha * (1 + (beta**-1)).lgamma().exp()

    @classmethod
    def median(cls, params: Params) -> Tensor:
        """Compute the median of a two-parameter Weibull distribution.

        Args:
            params (Params): parameters representing one or more Weibull distributions

        Returns:
            Tensor: shape (N,) containing the median of the Weibull distributions from the given alpha and beta pairs.
        """
        alpha, beta = params
        return alpha * pt.pow(-pt.log(pt.tensor(0.5)), (beta**-1))

    @classmethod
    def mode(cls, params: Params) -> Tensor:
        """Compute the mode of a two-parameter Weibull distribution.

        Args:
            params (Params): parameters representing one or more Weibull distributions

        Returns:
            Tensor: shape (N,) containing the mode of the Weibull distributions from the given alpha and beta pairs.
        """
        alpha, beta = params
        # use beta ** -1 == (1 / beta) because it's strictly faster on CPU.
        # This is only for the 1 / x case, not x_1 / x_0 case
        return alpha * ((beta - 1) / beta) ** (beta**-1)

    @classmethod
    def pdf(cls, params: Params, t: Tensor) -> Tensor:
        """Probability at points ``t`` in distributions for pairn of ``params``.

        Probability distribution function for a Weibull is
        f(x) = (β/|α|) * ((x/α)^(β - 1)) * exp( -1 * ((x / α)^β))

        Args:
            params (Params): parameters representing one or more Weibull distributions
            t (Tensor): time steps, a sequence of ``t``s, used in place of ``x`` in a probability distribution function

        Returns:
            Tensor: shape (N,) containing the mode of the Weibull distributions from the given alpha and beta pairs.
        """
        alpha, beta = params
        ch = cls.cumulative_hazard(cls.Params(alpha, beta - 1), t)  # (t / alpha)**(beta - 1.0)
        exp_ch = cls.cumulative_hazard(params, t)  # (t / alpha)**beta
        return (beta / alpha) * ch * pt.exp(-1.0 * exp_ch)

    @classmethod
    @pt.no_grad()
    def compute_distribution_parameters(cls, model: TorchModule, features: Tensor) -> Params:
        """Invoke ``model`` on ``features`` in a Providence fashion, packaging in a ``Params`` instance.

        This is primarily a convenience function

        Args:
            model (TorchModule): Providence-compatible nn.Module implementation
            features (Tensor): Providence-prepared features Tensor, representing just a single entity / subject

        Returns:
            Params: alpha, beta for each time step of the ``features`` Tensor
        """
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
    cumulative_hazard = (beta / (beta - 1)) * ((x - k) / eta) ** (beta - 1)

    return cumulative_hazard


class Weibull3(SurvivalAnalysisDistribution):
    """The 3-parameter Weibull distribution, introducing a location / translation / shift parameter: k.

    Using this distribution with k=0 should be functionally equivalent to using the ``Weibull`` above.
    """

    class Params(NamedTuple):
        """``Weibull3``'s DistributionParams, holding the parameters for scale (alpha), shape (beta), and location (k).

        In the literature, you may find that alpha is eta, theta or omega, and k is eta, gamma or 'a'.

        Args:
            alpha (Tensor): shape (N,) representing the alpha parameter of the Weibull distribution.
            beta (Tensor): shape (N,) representing the beta parameter of the Weibull distribution.
            k (Tensor): shape (N,) representing the beta parameter of the Weibull distribution.
        """

        alpha: Tensor
        beta: Tensor
        k: Tensor

    @classmethod
    def cumulative_hazard(cls, params: Params, t: Tensor) -> Tensor:
        """Compute the translated cumulative hazard.

        Args:
            params (Params): parameters tuple, containing alpha, beta, and k of equal pt.Size or pt.Tensor.shape
            t (Tensor): time steps to evaluate the cumulative hazard

        Returns:
            Tensor: of a shape that matches broadcasting semantics i.e.
                0-order tensor: if all shapes of alpha, beta, k, and t are of 0-order
                1-order tensor: if all shapes of alpha, beta, k are 1st order (all must be equal), ``t`` is of 0-order
                    or 1st order if ``t``'s shape is equal to the others.

        Raises:
            RuntimeError: if any of the shapes of the pt.Tensors in ``params`` aren't broadcastable
        """
        return pt.pow((t - params.k) / params.alpha, params.beta)

    @classmethod
    def loglikelihood_discrete(
        cls,
        params: Params,
        t: Tensor,
        censor: BoolTensor,
        epsilon: float = 1e-7,
        clamp_min: float = 1e-5,
        clamp_max: float = 30.0,
    ):
        """Compute the loglikelihood for the discretized survival function of the Weibull.

        This implementation is effectively identical to that of the ``Weibull.loglikelihood_discrete``, but references
        ``Weibull3.cumulative_hazard`` instead, to utilize the translation parameter ``params.k``

        Args:
            params (DistributionParams): parameters for this distribution.
            t (Tensor): timesteps, a sequence of ``t``s, used in place of ``x`` in a probability distribution function
            censor (BoolTensor): sequence indicating which time steps occur before or after the corresponding timestep.
                That is, this should be a binary stream where 0 indicates no event occurs, and 1 indicates that an
                event occurs at TTE=0.
            epsilon (float, optional): epsilon added to the denominator of the cumulative hazard to avoid division by 0
                Defaults to 1e-7.
            clamp_min (float, optional): minimum for the difference between hazards. Defaults to 1e-5.
            clamp_max (float, optional): maximum for the difference between hazards. Defaults to 30.0.

        Returns:
            float: (or FloatTensor, 1-item or broadcasting based on ``t.shape``)
        """
        # ((t + epsilon) / alpha)**beta
        hazard0 = cls.cumulative_hazard(params, t + epsilon)
        # ((t + 1.0) / alpha)**beta
        hazard1 = cls.cumulative_hazard(params, t + 1.0)
        # constrain the values or else this going to get ugly
        hazards = pt.clamp(hazard1 - hazard0, min=clamp_min, max=clamp_max)
        loglikelihoods = pt.xlogy(censor, pt.exp(hazards) - 1) - hazard1  # faster than the original
        return loglikelihoods

    @classmethod
    def loglikelihood_continuous(cls, params: Params, t: Tensor, censor: BoolTensor, epsilon: float = 1e-7):
        """EXPERIMENTAL: Compute the loglikelihood of the continuous survival function the 3-parameter Weibull.

        The survival function does not leverage the ``(beta / (beta-1))`` term found in the PDF because that is not
        what Martinnson uses in his paper.
        Further survival analysis research is necessary.

        Args:
            params (DistributionParams): parameters for this distribution.
            t (Tensor): timesteps, a sequence of ``t``s, used in place of ``x`` in a probability distribution function
            censor (BoolTensor): sequence indicating which time steps occur before or after the corresponding timestep.
                That is, this should be a binary stream where 0 indicates no event occurs, and 1 indicates that an
                event occurs at TTE=0.
            epsilon (float, optional): epsilon added to the denominator of the cumulative hazard to avoid division by 0
                Defaults to 1e-7.

        Returns:
            float: (or FloatTensor, 1-item or broadcasting based on ``t.shape``)
        """
        alpha, beta, k = params
        ya = (t + epsilon - k) / alpha
        loglikelihoods = censor * (pt.xlogy(beta, ya) + pt.log(beta)) - pt.pow(ya, beta)  # faster
        return loglikelihoods

    @classmethod
    def mean(cls, params: Params) -> Tensor:
        """Compute the mean of a three-parameter Weibull distribution.

        Args:
            params (Params): parameters representing one or more Weibull distributions

        Returns:
            Tensor: shape (N,) containing the mean of the Weibull(alpha, beta, k) <- params(alpha, beta, k).
        """
        alpha, beta, k = params
        return k + alpha * (1 + (beta**-1)).lgamma().exp()

    @classmethod
    def median(cls, params: Params) -> Tensor:
        """EXPERIMENTAL: Compute the median of a three-parameter Weibull distribution.

        This implementation is uncertain, but extrapolated from the implementation of mode and mean, which sum-in the
        translational shift.

        Args:
            params (Params): parameters representing one or more Weibull distributions

        Returns:
            Tensor: shape (N,) containing the median of the Weibull(alpha, beta, k) <- params(alpha, beta, k).
        """
        alpha, beta, k = params
        return k + alpha * pt.pow(pt.log(pt.tensor(2)), (beta**-1))

    @classmethod
    def mode(cls, params: Params) -> Tensor:
        """Compute the mode of a three-parameter Weibull distribution.

        Args:
            params (Params): parameters representing one or more Weibull distributions

        Returns:
            Tensor: shape (N,) containing the mode of the Weibull(alpha, beta, k) <- params(alpha, beta, k).
        """
        alpha, beta, k = params
        _mode = k + alpha * ((beta - 1) / beta) ** (beta**-1)
        return pt.where(beta <= 0, _mode, k)

    @classmethod
    def pdf(cls, params: Params, t: Tensor) -> Tensor:
        """Probability at points ``t`` in distributions for pairn of ``params``.

        Probability distribution function for a Weibull is
        f(x; α, β, k) = (β/α) * (((x-k)/α)^(β - 1)) * exp( -1 * (((x-k) / α)^β))

        Args:
            params (Params): parameters representing one or more Weibull distributions
            t (Tensor): time steps, a sequence of ``t``s, used in place of ``x`` in a probability distribution function

        Returns:
            Tensor: shape (N,) containing the mode of the Weibull distributions from the given alpha and beta pairs.
        """
        alpha, beta, _ = params
        ch = cls.cumulative_hazard(params._replace(beta=beta - 1), t)  # ((t - k)/ alpha)**(beta - 1.0)
        exp_ch = cls.cumulative_hazard(params, t)  # ((t - k )/ alpha)**beta
        return (beta / alpha) * ch * pt.exp(-1.0 * exp_ch)

    @classmethod
    @pt.no_grad()
    def compute_distribution_parameters(cls, model: TorchModule, features: Tensor) -> Params:
        """Invoke ``model`` on ``features`` in a Providence fashion, packaging in a ``Params`` instance.

        This is primarily a convenience function

        Args:
            model (TorchModule): Providence-compatible nn.Module implementation
            features (Tensor): Providence-prepared features Tensor, representing just a single entity / subject

        Returns:
            Params: alpha, beta, k for each time step of the ``features`` Tensor
        """
        model.eval()
        # features: [time, feature] -> [time, batch of 1, feature]
        a, b, k_ = model(features.unsqueeze(1), pt.tensor([features.shape[0]], dtype=pt.long))
        # because model output size = 1: for all time, for that batch of 1, take out that data
        alpha, beta, k = a.detach()[:, 0, 0], b.detach()[:, 0, 0], k_.detach()[:, 0, 0]
        return Weibull3.Params(alpha, beta, k)


class GeneralizedWeibull(SurvivalAnalysisDistribution):
    """The Generalized Weibull (aka Anchored Amorosso aka Anchored Generalized Gamma) extends the representational power of the Weibull."""

    class Params(NamedTuple):
        """The GeneralizedWeibull version of Params, specified for the Weibull distribution leveraged by Providence.

        There is an implicit assumption that alpha.size(0) == beta.size(0)

        Arguments:
            alpha: Tensor of the alpha values
            beta: Tensor of the beta values
            eta: Tensor of the eta values
            omega: Tensor of the omega values
        """

        alpha: Tensor
        beta: Tensor
        eta: Tensor
        omega: Tensor


GeneralizedWeibull.__init__ = lambda: TODO("Don't use this type or suffer this consequence. REPEATEDLY")  # type: ignore[assignment, misc]
