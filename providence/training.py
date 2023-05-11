"""
This module contains several functions that make our lives easier when it comes to conducting model training.
Most importantly, it allows us to stop focusing on the training loop semantics and devote energies to more
important things.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Union

from jaxtyping import Float
import torch as pt
from torch import no_grad
from torch import zeros
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from providence.distributions import Weibull
from providence.loss import discrete_weibull_loss_fn
from providence.loss import ProvidenceLossInterface
from providence.nn.module import ProvidenceModule
from providence.types import DataLoaders


time_device_2 = "time device 2"
ProvidenceLossInterfaceType = Union[ProvidenceLossInterface, Callable]


class EpochLosses(NamedTuple):
    """Losses collected during a given epoch of model training.

    Exists to impose semantic meaning on the standard tuple.

    Args:
        training (float): loss from training during this epoch
        validation (float): loss from the validation/evaluation phase of this epoch
    """

    training: float
    validation: float

    @property
    def train(self) -> float:
        """Alias for ``training``.

        Returns:
            float: training loss
        """
        return self.training

    @property
    def val(self) -> float:
        """Alias for ``validation``.

        Returns:
            float: validation loss
        """
        return self.validation


class LossAggregates(NamedTuple):
    """Aggregate losses for model training.

    Args:
        training_losses (List[float]): losses accumulated or to accumulate during training epochs
        validation_losses (List[float]): losses accumulated during validation portions of training epochs
    """

    training_losses: List[float]
    validation_losses: List[float]

    def append_losses(self, losses: EpochLosses):
        """Unpack and append ``losses`` to their corresponding lists within this instance.

        Effectively perform
        loss_train, loss_valid = losses
        self.training_losses.append(losses.training)
        self.validation_losses.append(loss.validation)

        Args:
            losses (EpochLosses): tuple of training and validation losses
        """
        self.training_losses.append(losses.training)
        self.validation_losses.append(losses.validation)


class OptimizerWrapper(NamedTuple):
    """A PyTorch Optimizer and relevant hyperparameters for its use.

    We find these three parameters are more tightly coupled together than others, so we package them together.

    Args:
        opt (Optimizer): PyTorch optimizer
        batch_size (int): batch size to be used when training with the optimizer ``opt``
        num_epochs (int): number of epochs to be used during tarining
    """

    opt: Optimizer
    batch_size: int
    num_epochs: int


def unpack_label_and_censor(targets: Float[pt.Tensor, time_device_2]):
    """Separate and flatten the true ``tte`` and ``censor`` columns.

    This function may be replaced by torch.(Tensor.)split in a later version.
    We keep the named version to

    Args:
        targets (Tensor): shape [time, entity, 2] to be split in two.

    Returns:
        Tuple[Tensor, Tensor]: (labels, censor) two tensors of shape [time, entity, 1]
    """
    y_true = targets[:, :, 0]
    y_true.unsqueeze_(-1)
    censor_ = targets[:, :, 1]
    censor_.unsqueeze_(-1)
    return y_true, censor_


def training_pass(
    train_dataloader: DataLoader,
    model: ProvidenceModule,
    optimizer: Optimizer,
    *,
    loss_criterion: ProvidenceLossInterfaceType = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params,
    clip_gradients=True,
):
    """Perform the training pass on the dataloader i.e. running it to exhaustion on the ``model`` with ``optimizer``.

    We follow best practices of
    1. setting optimizer gradients to ``None`` at the top of each epoch
    2. clipping gradients after back passes
    3. only accumulating the loss, rather than returning the mean - as there may be other statistics to concern with
        the total loss outside this function.

    The ``loss_criterion`` and ``model_output_type`` are coupled, as the model inference will be packaged into the
    latter and then pass to the former as the first argument.

    Args:
        train_dataloader (DataLoader): data to use during the model training
        model (ProvidenceModule): the model to fit
        optimizer (Optimizer): optimizer for tunig the objective, either minimization or maximization
        loss_criterion (ProvidenceLossInterface, optional): a loss function that is compatible with the Providence
            training regime. Defaults to discrete_weibull_loss_fn.
        model_ouput_type (type, optional): The type of the parameters output by the model, which will be fed to the
            ``loss_criterion``. Defaults to Weibull.Params.
        clip_gradients (bool, optional): Whether to clip the gradients. Defaults to True.

    Returns:
        float: cumulative loss during the training, **not** averaged
    """
    train_loss = zeros(1, device=model.device)
    # NOTE: ProvidenceModule methods not recognized by mypy.
    model.train()  # type: ignore[attr-defined]
    for data in train_dataloader:
        feats, lengths, targets = data
        # print(f"{feats.dtype = }, {type(lengths[0]) = }, {targets.dtype = }")
        optimizer.zero_grad(set_to_none=True)  # should be faster

        outputs = model(feats.to(model.device), lengths)  # type: ignore[operator]
        distribution_params = model_ouput_type(*outputs)

        y_true, censor_ = unpack_label_and_censor(targets.to(model.device))
        loss = loss_criterion(distribution_params, y_true, censor_, lengths)
        loss.backward()  # type: ignore[union-attr]
        if clip_gradients:
            clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)  # type: ignore [attr-defined]
        optimizer.step()
        train_loss += loss
    return train_loss.item()


@no_grad()
def validation_pass(
    validation_dataloader: DataLoader,
    model: ProvidenceModule,
    *,
    loss_criterion: ProvidenceLossInterfaceType = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params,
):
    """Perform the validation pass of an epoch, not optimizing any weights but only accumulating the loss.

    Args:
        validation_dataloader (DataLoader): data to use during the model validation
        model (ProvidenceModule): the model to evaluate
        loss_criterion (ProvidenceLossInterface, optional): a loss function that is compatible with the Providence
            training regime. Defaults to discrete_weibull_loss_fn.
        model_ouput_type (type, optional): The type of the parameters output by the model, which will be fed to the
            ``loss_criterion``. Defaults to Weibull.Params.
        clip_gradients (bool, optional): Whether to clip the gradients. Defaults to True.

    Returns:
        float: cumulative loss during the validation, **not** averaged
    """
    val_loss = zeros(1, device=model.device)
    # NOTE: ProvidenceModule methods not recognized by mypy.
    # Setting model to eval mode stops training behavior such as dropout
    model.eval()  # type: ignore[attr-defined]
    for data in validation_dataloader:
        feats, lengths, targets = data

        outputs = model(feats.to(model.device), lengths)  # type: ignore[operator]
        distribution_params = model_ouput_type(*outputs)

        y_true, censor_ = unpack_label_and_censor(targets.to(model.device))
        loss = loss_criterion(distribution_params, y_true, censor_, lengths)
        val_loss += loss

    return val_loss.item()


def training_epoch(
    dataloaders: DataLoaders,
    model: ProvidenceModule,
    optimizer: Optimizer,
    *,
    loss_criterion: ProvidenceLossInterfaceType = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params,
) -> EpochLosses:
    """Perform a standard training epoch: first train, then validate against corresponding DataLoaders in ``dataloaders``.

    It is not lost on the authors that this may not be standard, and validation may be much less frequent than in parallel
    with the training epoch. The user is free to using only one portion or the other to ease their time.

    Args:
        dataloaders (DataLoaders): data to use during the model training and validation
        model (ProvidenceModule): the model to fit and evaluated
        optimizer (Optimizer): optimizer for tunig the objective, either minimization or maximization
        loss_criterion (ProvidenceLossInterface, optional): a loss function that is compatible with the Providence
            training regime. Defaults to discrete_weibull_loss_fn.
        model_ouput_type (type, optional): The type of the parameters output by the model, which will be fed to the
            ``loss_criterion``. Defaults to Weibull.Params.
        clip_gradients (bool, optional): Whether to clip the gradients. Defaults to True.

    Returns:
        EpochLosses: cumulative loss during the training and validation, **neither** being averaged
    """
    training_loss = training_pass(
        dataloaders.train,
        model,
        optimizer,
        loss_criterion=loss_criterion,
        model_ouput_type=model_ouput_type,
    )
    validation_loss = validation_pass(
        dataloaders.validation,
        model,
        loss_criterion=loss_criterion,
        model_ouput_type=model_ouput_type,
    )

    return EpochLosses(training_loss, validation_loss)


def generic_training(
    model: ProvidenceModule,
    optimizer: OptimizerWrapper,
    dataloaders: DataLoaders,
    *,
    loss_fn: ProvidenceLossInterfaceType = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params,
) -> LossAggregates:
    """Train a ``model`` for ``optimizer.num_epochs`` against all data in ``dataloaders``

    We take the liberty of moving the ``model`` to its (training) ``device``, and then back to the CPU once
    ``optimizer.num_epochs`` have elapsed and results have been aggregated.

    Args:
        model (ProvidenceModule): model to train
        optimizer (OptimizerWrapper): Optimizer and associated hyperparameters to use in training
        dataloaders (DataLoaders): training and validation DataLoaders to be used in training and validation phases, resp.
        loss_fn (ProvidenceLossInterface, optional): a loss function that is compatible with the Providence
            training regime. Defaults to discrete_weibull_loss_fn.
        model_ouput_type (type, optional): The type of the parameters output by the model, which will be fed to the
            ``loss_criterion``. Defaults to Weibull.Params.

    Returns:
        LossAggregates: All training- and validation-phase losses from training ``model`` on the given ``dataloaders``
    """
    model.to(model.device)  # type: ignore[attr-defined]
    # dataloaders.to_device(model.device)
    loss_agg = LossAggregates([], [])
    for _ in range(optimizer.num_epochs):
        losses = training_epoch(
            dataloaders,
            model,
            optimizer.opt,
            loss_criterion=loss_fn,
            model_ouput_type=model_ouput_type,
        )
        loss_agg.append_losses(losses)

    # dataloaders.to_device('cpu')
    model.to("cpu")  # type: ignore[attr-defined]
    return loss_agg


################################################################################
#
# Super utilities. CAUTION: Use with care ;)
#
################################################################################


def use_gpu_if_available() -> pt.device:
    """Returns the ``pt.device`` that corresponds to an NVIDIA GPU if there is one to use; otherwise returns the CPU."""
    return pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")


def set_torch_default_dtypes(chosen_type: pt.dtype):
    """Set the default dtype and default tensor type in PyTorch to ``chosen_type``.

    Each dtype has a corresponding tensor type, so this convenience is written to shorten preamble "setup" code.

    Args:
        chosen_type (pt.dtype): some floating point PyTorch dtype
    """
    if chosen_type == pt.float16:
        dtype, tensor_type = pt.float16, pt.HalfTensor
    elif chosen_type == pt.float32:
        dtype, tensor_type = pt.float32, pt.FloatTensor
    elif chosen_type == pt.float64:
        dtype, tensor_type = pt.float64, pt.DoubleTensor
    elif chosen_type == pt.bfloat16:
        dtype, tensor_type = (
            pt.bfloat16,
            pt.BFloat16Tensor,
        )  # doesn't IntelliSense, but should be there JIT

    pt.set_default_dtype(dtype)
    pt.set_default_tensor_type(tensor_type)


def minimize_torch_runtime_overhead():
    """Attempt to minimize the runtime in training deep learning models with Pytorch."""
    pt.set_anomaly_enabled(False)
    pt.autograd.profiler.profile(enabled=False)

    # autocast-ing can be slower for big models (https://discuss.pytorch.org/t/amp-autocast-not-faster-than-fp32/111757)
    if pt.is_autocast_enabled():
        pt.set_autocast_enabled(False)

    # run a pre-check on cuDNN + cuBLAS stuff, warming the cache for all ops, making training times consistent.
    pt.backends.cudnn.benchmark = True
