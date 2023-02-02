"""
This module contains several functions that make our lives easier when it comes to conducting model training.
Most importantly, it allows us to stop focusing on the training loop semantics and devote energies to more
important things.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from typing import List, NamedTuple

import torch as pt
from torch import no_grad, zeros
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchtyping import TensorType

from providence.distributions import Weibull
from providence.loss import ProvidenceLossInterface, discrete_weibull_loss_fn
from providence.nn.module import ProvidenceModule
from providence.types import DataLoaders


class EpochLosses(NamedTuple):
    training: float
    validation: float

    @property
    def train(self) -> float:
        return self.training

    @property
    def val(self) -> float:
        return self.validation


class LossAggregates(NamedTuple):
    training_losses: List[float]
    validation_losses: List[float]

    def append_losses(self, losses: EpochLosses):
        self.training_losses.append(losses.training)
        self.validation_losses.append(losses.validation)


class OptimizerWrapper(NamedTuple):
    opt: Optimizer
    batch_size: int
    num_epochs: int


def unpack_label_and_censor(targets: TensorType["time":..., "device":..., "label_and_censor":2]):
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
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params,
    clip_gradients=True
):
    train_loss = zeros(1, device=model.device)
    model.train()
    for data in train_dataloader:
        feats, lengths, targets = data
        # print(f"{feats.dtype = }, {type(lengths[0]) = }, {targets.dtype = }")
        optimizer.zero_grad(set_to_none=True)  # should be faster

        outputs = model(feats.to(model.device), lengths)
        distribution_params = model_ouput_type(*outputs)

        y_true, censor_ = unpack_label_and_censor(targets.to(model.device))
        loss = loss_criterion(distribution_params, y_true, censor_, lengths)
        loss.backward()
        if clip_gradients:
            clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
        train_loss += loss
    return train_loss.item()


@no_grad()
def validation_pass(
    validation_dataloader: DataLoader,
    model: ProvidenceModule,
    *,
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params
):
    val_loss = zeros(1, device=model.device)
    model.eval()  # Setting model to eval mode stops training behavior such as dropout
    for data in validation_dataloader:
        feats, lengths, targets = data

        outputs = model(feats.to(model.device), lengths)
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
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params
) -> EpochLosses:
    training_loss = training_pass(
        dataloaders.train, model, optimizer, loss_criterion=loss_criterion, model_ouput_type=model_ouput_type
    )
    validation_loss = validation_pass(dataloaders.validation, model, loss_criterion=loss_criterion, model_ouput_type=model_ouput_type)

    return EpochLosses(training_loss, validation_loss)


def generic_training(
    model: ProvidenceModule,
    optimizer: OptimizerWrapper,
    dataloaders: DataLoaders,
    *,
    loss_fn: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params,
) -> LossAggregates:
    model.to(model.device)
    # dataloaders.to_device(model.device)
    loss_agg = LossAggregates([], [])
    for _ in range(optimizer.num_epochs):
        losses = training_epoch(dataloaders, model, optimizer.opt, loss_criterion=loss_fn, model_ouput_type=model_ouput_type)
        loss_agg.append_losses(losses)

    # dataloaders.to_device('cpu')
    model.to('cpu')
    return loss_agg


################################################################################
#
# Super utilities. CAUTION: Use with care ;)
#
################################################################################

def use_gpu_if_available() -> pt.device:
    return pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')


def set_torch_default_dtypes(chosen_type: pt.dtype):
    if chosen_type == pt.float16:
        dtype, tensor_type = pt.float16, pt.HalfTensor
    elif chosen_type == pt.float32:
        dtype, tensor_type = pt.float32, pt.FloatTensor
    elif chosen_type == pt.float64:
        dtype, tensor_type = pt.float64, pt.DoubleTensor
    elif chosen_type == pt.bfloat16:
        dtype, tensor_type = pt.bfloat16, pt.BFloat16Tensor # doesn't IntelliSense, but should be there JIT

    pt.set_default_dtype(dtype)
    pt.set_default_tensor_type(tensor_type)

def minimize_torch_runtime_overhead():
    pt.set_anomaly_enabled(False)
    pt.autograd.profiler.profile(enabled=False)

    # autocast-ing can be slower for big models (https://discuss.pytorch.org/t/amp-autocast-not-faster-than-fp32/111757)
    if pt.is_autocast_enabled():
        pt.set_autocast_enabled(False)

    # run a pre-check on cuDNN + cuBLAS stuff, warming the cache for all ops, making training times consistent.
    pt.backends.cudnn.benchmark = True