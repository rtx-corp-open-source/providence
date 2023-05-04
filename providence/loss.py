"""
Strong typing around the Providence loss function.
Outlines the interface such a function would follow (and creates an abstract base class if you would rather follow the inheritance route)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import List
from typing import Protocol
from typing import Tuple
from typing import Union

import torch as pt
from jaxtyping import Float

from providence.distributions import T_DistributionParams
from providence.distributions import Weibull
from providence.distributions import Weibull3
from providence.types import LengthsTensor
from providence.types import TimeTensor
from providence.types import TypeAlias
from providence.utils import validate


T_Loss: TypeAlias = Float[pt.Tensor, "1"]


class ProvidenceLossInterface(Generic[T_DistributionParams], Protocol):
    """The interface for functions that which to be invoked for a Providence loss.

    A Python function can respect this interface. See ``ProvidenceLoss`` for details, ignoring aspects of inherintance.
    """

    def __call__(
        self,
        params: T_DistributionParams,
        y: TimeTensor,
        censor: TimeTensor,
        x_lengths: Union[List[int], LengthsTensor],
    ) -> T_Loss:
        ...


class ProvidenceLoss(Generic[T_DistributionParams], pt.nn.Module, ABC):
    """Parent class for a loss in Providence training.

    Including everything necessary for the most complex loss computation, point-estimates should be derived from the
    predicted distribution (parameters) and measured against the ground-truth ``y`` in some way.

    Args:
        params (SurvivalDistributionParams): predicted parameters for a given survival analytic distribution.
            Expected length to match the first dimension of ``y`` or ``censor``.
        y (Tensor): shape (N,) of the true time-to-event at even given timestep.
        censor (BoolTensor): shape (N,) sequence indicating which time steps occur before or after the corresponding timestep.
            That is, this should be a binary stream where 0 indicates no event occurs at the end of the sequence, and a
            1 indicates that an event occurs at TTE=0 (even if the dataset doesn't have that event i.e. is right-censored)
        x_lengths (IntTensor): lengths of entities in the dataset i.e.
            ``x_lengths = [len(entity) for entity in dataset]``
        epsilon (float, optional): epsilon added to the denominator of the cumulative hazard to avoid division by 0
            Defaults to 1e-7.
    """

    @abstractmethod
    def forward(
        self,
        params: T_DistributionParams,
        y: TimeTensor,
        censor: TimeTensor,
        x_lengths: Union[List[int], LengthsTensor],
    ) -> T_Loss:
        ...


def discrete_weibull_mse(params: Weibull.Params, tte: TimeTensor) -> float:
    """Compute the RMSE of the Weibull predictions, treating the tte column as the correct prediction.

    Traditional MSE is something like
        >>> loss = mse(y_pred, y_true)

    This function is analogous
        >>> loss = discrete_weibull_mse(pred_params, tte)

    Args:
        params (Weibull.Params): predicted parameters for the two-parameter Weibull.
        y (Tensor): the true time-to-event at even given timestep

    Returns:
        FloatTensor: the mean (ignoring NaNs) squared error of predicted distribution mode's vs tte
    """
    validate(
        params.alpha.shape == tte.shape,
        f"Params and tte tensors are of different lengths: {params.alpha.shape=} {params.beta.shape=} {tte.shape=}",
    )

    tte_preds = Weibull.mode(params)
    mse = pt.nanmean((tte_preds - tte) ** 2)
    return mse


class DiscreteWeibullMSELoss(ProvidenceLoss):
    """Mean-squared error loss for the Discrete two-parameter Weibull."""

    def forward(
        self,
        params: Weibull.Params,
        y: TimeTensor,
        censor: TimeTensor,
        x_lengths: Union[List[int], LengthsTensor],
    ) -> Union[float, Tuple[float, ...]]:
        """Compute the RMSE of the Weibull predictions, treating the tte column as the correct prediction.

        Traditional MSE is something like
            >>> loss = mse(y_pred, y_true)

        This function is analogous
            >>> loss = discrete_weibull_mse(pred_params, tte)

        Args:
            params (Weibull.Params): predicted parameters for the two-parameter Weibull.
            y (Tensor): the true time-to-event at even given timestep
            censor (Tensor): ignored
            x_lengths (Tensor): ignored

        Returns:
            FloatTensor: the mean (ignoring NaNs) squared error of predicted distribution mode's vs tte
        """
        return discrete_weibull_mse(params, y)


def discrete_weibull_loss_fn(
    params: Weibull.Params,
    y: TimeTensor,
    censor: TimeTensor,
    x_lengths: Union[List[int], LengthsTensor],
    epsilon=1e-7,
) -> float:
    """Discrete loglikelihood loss for a weibull curve prediction.

    Args:
        params (Weibull.Params): predicted parameters for the two-parameter Weibull.
        y (Tensor): the true time-to-event at even given timestep
        censor (BoolTensor): sequence indicating which time steps occur before or after the corresponding timestep.
            That is, this should be a binary stream where 0 indicates no event occurs, and 1 indicates that an
            event occurs at TTE=0.
        x_lengths (IntTensor): lengths of entities in the dataset i.e.
            ``x_lengths = [len(entity) for entity in dataset]``
        epsilon (float, optional): epsilon added to the denominator of the cumulative hazard to avoid division by 0
            Defaults to 1e-7.

    Returns:
        FloatTensor: shape (1, ) of the negative (mean) loglikelihood of the distribution predicting correctly given
            ``y``.
    """
    # consider an update:  weight by tte (normalized or inverse log) so predictions are heavily weighted for TTE -> 0
    # (This requires parameters (sequence length, censor bool, tte))
    # follow-up / branch work: prefer undershooting to overshooting
    if not isinstance(x_lengths, pt.Tensor):
        x_lengths = pt.tensor(x_lengths, dtype=pt.long)

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
    """Discrete Weibull loss.

    See ``discrete_weibull_loss_fn`` for details

    Args:
        epsilon (float, optional): epsilon added to the denominator of the cumulative hazard to avoid division by 0.
            Using this version over the purely functional ``discrete_weibull_loss_fn`` may help the C-runtime retain
            this value and avoid the (slow) Python lookup. Defaults to 1e-7.
    """

    def __init__(self, *, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        params: Weibull.Params,
        y: TimeTensor,
        censor: TimeTensor,
        x_lengths: Union[List[int], LengthsTensor],
    ) -> Union[float, Tuple[float, ...]]:
        """Compute the loss.

        Args:
            params (Weibull.Params): predicted parameters for the three-parameter Weibull.
            y (Tensor): the true time-to-event at even given timestep
            censor (BoolTensor): sequence indicating which time steps occur before or after the corresponding timestep.
                That is, this should be a binary stream where 0 indicates no event occurs, and 1 indicates that an
                event occurs at TTE=0.
            x_lengths (IntTensor): lengths of entities in the dataset i.e.
                ``x_lengths = [len(entity) for entity in dataset]``

        Returns:
            FloatTensor: shape (1, ) of the negative (mean) loglikelihood of the distribution predicting correctly
            given ``y``.
        """
        return discrete_weibull_loss_fn(params, y, censor, x_lengths, epsilon=self.epsilon)


def discrete_weibull3_loss_fn(
    params: Weibull3.Params,
    y: TimeTensor,
    censor: TimeTensor,
    x_lengths: Union[List[int], LengthsTensor],
    epsilon=1e-7,
) -> float:
    """Discrete loglikelihood loss for a Weibull3 curve prediction.

    Args:
        params (Weibull.Params): predicted parameters for the three-parameter Weibull.
        y (Tensor): the true time-to-event at even given timestep
        censor (BoolTensor): sequence indicating which time steps occur before or after the corresponding timestep.
            That is, this should be a binary stream where 0 indicates no event occurs, and 1 indicates that an
            event occurs at TTE=0.
        x_lengths (IntTensor): lengths of entities in the dataset i.e.
            ``x_lengths = [len(entity) for entity in dataset]``
        epsilon (float, optional): epsilon added to the denominator of the cumulative hazard to avoid division by 0
            Defaults to 1e-7.

    Returns:
        FloatTensor: shape (1, ) of the negative (mean) loglikelihood of the distribution predicting correctly given
            ``y``.
    """
    if not isinstance(x_lengths, pt.Tensor):
        x_lengths = pt.tensor(x_lengths, dtype=pt.long)

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
    """Discrete Weibull3 loss.

    See ``discrete_weibull3_loss_fn`` for details.

    Args:
        epsilon (float, optional): epsilon added to the denominator of the cumulative hazard to avoid division by 0.
            Using this version over the purely functional ``discrete_weibull3_loss_fn`` may help the C-runtime retain
            this value and avoid the (slow) Python lookup. Defaults to 1e-7.
    """

    def __init__(self, *, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        params: Weibull3.Params,
        y: TimeTensor,
        censor: TimeTensor,
        x_lengths: Union[List[int], LengthsTensor],
    ) -> Union[float, Tuple[float, ...]]:
        """Compute the loss.

        Args:
            params (Weibull.Params): predicted parameters for the three-parameter Weibull.
            y (Tensor): the true time-to-event at even given timestep
            censor (BoolTensor): sequence indicating which time steps occur before or after the corresponding timestep.
                That is, this should be a binary stream where 0 indicates no event occurs, and 1 indicates that an
                event occurs at TTE=0.
            x_lengths (IntTensor): lengths of entities in the dataset i.e.
                ``x_lengths = [len(entity) for entity in dataset]``

        Returns:
            FloatTensor: shape (1, ) of the negative (mean) loglikelihood of the distribution predicting correctly
            given ``y``.
        """
        return discrete_weibull3_loss_fn(params, y, censor, x_lengths, epsilon=self.epsilon)
