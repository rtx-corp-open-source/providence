"""
Trainer abstraction to make it easier to do model training (especially with our custom Callbacks)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from abc import abstractmethod
from typing import List, NamedTuple, Protocol
from click import progressbar

import torch as pt
from torch import no_grad, zeros
from torch.nn.utils.clip_grad import clip_grad_norm_

from providence.types import TorchDataLoader

from providence.distributions import SurvivalAnalysisDistribution, Weibull
from providence.loss import ProvidenceLossInterface, discrete_weibull_loss_fn
from providence.nn import ProvidenceModule
from providence.training import EpochLosses, LossAggregates, OptimizerWrapper, unpack_label_and_censor
from providence.types import DataLoaders

from torch.optim import Optimizer

from providence_utils.callbacks import Callback, check_before_epoch


class EpochInterface(Protocol):
    @abstractmethod
    def __call__(
        self,
        dls: DataLoaders,
        model: ProvidenceModule,
        optimizer: Optimizer,
        *,
        loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
        model_ouput_type=Weibull.Params
    ) -> EpochLosses:
        """
        This function represents the contract of an epoch of training of the supplied `model` on the given `dls` wrt the `optimizer`.
        Whether you utilize `dls.test` is up to your usage.

        `dls` are first in the parameter list because they are the primary discriminating factor in the implementation.
        While it matches the `epoch` functions to have `model` first, semantically it's more cognitive load to go looking
        even one or two parameters to see where two function call sites differ. Either `transformer_epoch` or `training_epoch`
        demonstrate this well.
        """
        ...


class Trainer:
    """A wrapper around training, so that I don't have to keep defining `callback_training` and needing a
    new function / param for slight variations in training
    """
    def __init__(self, epoch_func: EpochInterface, verbose: bool = False):
        self.epoch_func = epoch_func
        self.verbose = verbose
        self.training_termination_reason = None

    def callback_training(
        self,
        model: ProvidenceModule,
        optimizer: OptimizerWrapper,
        dataloaders: DataLoaders,
        cbs: List[Callback] = None
    ) -> LossAggregates:
        "Training with callbacks"
        if cbs is None:
            cbs = []

        model.train()
        model.to(model.device)

        loss_agg = LossAggregates([], [])
        epochs = range(1, optimizer.num_epochs + 1)
        if self.verbose:
            epochs = progressbar(epochs)

        for current_epoch in epochs:
            terminate_training, termination_message = check_before_epoch(callbacks=cbs)
            if terminate_training:
                self.terminated_early = True
                self.training_termination_reason = termination_message
                break

            losses = self.epoch_func(dataloaders, model, optimizer.opt)
            loss_agg.append_losses(losses)

            for cb in cbs:
                cb.after_epoch(current_epoch, model, optimizer.opt, losses, dataloaders)
        else:
            self.training_termination_reason = "Completed all epochs"
            self.terminated_early = False

        for cb in cbs:
            if self.verbose: print(f"{type(cb).__name__}.after_training(...)")
            cb.after_training(current_epoch, model, optimizer.opt, losses, dataloaders)

        model.to('cpu')
        model.eval()
        return loss_agg

    def basic_training(
        self, model: ProvidenceModule, optimizer: OptimizerWrapper, dataloaders: DataLoaders
    ) -> LossAggregates:
        "Training "
        model.to(model.device)
        # dataloaders.to_device(model.device)

        loss_agg = LossAggregates([], [])
        for _ in range(1, optimizer.num_epochs + 1):
            losses = self.epoch_func(dataloaders, model, optimizer.opt)
            loss_agg.append_losses(losses)

        # dataloaders.to_device('cpu')
        model.to('cpu')
        return loss_agg


class ProvidenceLossHandle(NamedTuple):
    loss_criterion: ProvidenceLossInterface
    model_output_t: SurvivalAnalysisDistribution.Params

class ProvidenceTrainer:
    """
    A wrapper around training to make it easier to edit the parts that needs to change e.g. how you infer on a given batch
    (say, because you're passing kwargs to a model.forward) while keeping the benefits of the old design in making
    callback training easier

    Args:
    - model: the model to train
    - optimizer: instantiated optimizer, probably pointed at the model
    - loss_handle: holds the loss function and the `model_output_t` to be constructed from the model's forward pass outputs
    - should_clip_gradients: if we should perform gradient clipping
    """

    def __init__(self,model: ProvidenceModule, optimizer: Optimizer, loss_handle: ProvidenceLossHandle, should_clip_gradients: bool = True) -> None:
        self.loss_handle = loss_handle
        self.clip_gradients = should_clip_gradients
        self.model = model
        self.optimizer = optimizer
        self.cbs = [] # just add to this directly.

    def prepare_batch(self, batch):
        # default thanks to the Providence Dataloader
        # features, lengths, targets, *rest = args
        # return features, lengths, targets
        return batch

    def batch_train_step(self, batch) -> pt.Tensor:
        self.optimizer.zero_grad(set_to_none=True)  # should be faster
        feats, lengths, targets = batch

        distribution_params = self.batch_inference(self.model, feats, lengths)

        y_true, censor_ = unpack_label_and_censor(targets.to(self.model.device))
        loss = self.loss_handle.loss_criterion(distribution_params, y_true, censor_, lengths)
        loss.backward()

        if self.clip_gradients: clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
        self.optimizer.step()

        return loss

    def batch_inference(self, model, feats, lengths) -> SurvivalAnalysisDistribution.Params:
        """Perform inference on a training batch. Override this for a different behavior"""
        outputs = model(feats.to(model.device), lengths, encoder_mask=True)
        distribution_params = self.loss_handle.model_output_t(*outputs)
        return distribution_params


    def batch_eval_step(self, batch) -> pt.Tensor:
        """Workings before and after the `batch_inference` for validation pass"""
        features, lengths, targets = batch
        distribution_params = self.batch_inference(self.model, features, lengths)
        y_true, censor_ = unpack_label_and_censor(targets.to(self.model.device))
        loss = self.loss_handle.loss_criterion(distribution_params, y_true, censor_, lengths)
        return loss

    def epoch_pass(self, dls: DataLoaders) -> EpochLosses:
        """Train and validation passes in this epoch, flattening on the structure from the original, allowing
        all of the magic to be visible at this level"""
        self.model.train()
        train_loss = zeros(1, device=self.model.device)
        for i, batch in enumerate(dls.train): train_loss += self.batch_train_step(batch)
        epoch_train_loss = train_loss.mean()

        self.model.eval()
        validation_loss = zeros(1, device=self.model.device)
        for i, batch in enumerate(dls.validation): validation_loss += self.batch_eval_step(batch)
        epoch_validation_loss = validation_loss.mean()
        return EpochLosses(epoch_train_loss.cpu(), epoch_validation_loss.cpu())
    
    def _should_terminate(self) -> bool:
        terminate_training, termination_message = check_before_epoch(callbacks=self.cbs)
        if terminate_training:
            self.terminated_early = True
            self.training_termination_reason = termination_message
            return True
        return False


    def train(self, num_epochs: int, dls: DataLoaders) -> LossAggregates:
        agg = LossAggregates([], [])
        self.model.to(self.model.device)
        for current_epoch in range(num_epochs):
            if self._should_terminate(): break

            epoch_losses = self.epoch_pass(dls)

            agg.append_losses(epoch_losses)
            for cb in self.cbs:
                cb.after_training(current_epoch, self.model, self.optimizer, epoch_losses, dls)

        return LossAggregates



################################################################################
#
# Standard Implementations
#
################################################################################


def transformer_training_pass(
    train_dataloader: TorchDataLoader,
    model: ProvidenceModule,
    optimizer: Optimizer,
    *,
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params,
    clip_gradients=True
):
    train_loss = zeros(1, device=model.device)
    model.train()
    for feats, lengths, targets in train_dataloader:
        optimizer.zero_grad(set_to_none=True)  # should be faster

        outputs = model(feats.to(model.device), lengths, encoder_mask=True)
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
def transformer_validation_pass(
    validation_dataloader: TorchDataLoader,
    model: ProvidenceModule,
    *,
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params
):
    val_loss = zeros(1, device=model.device)
    model.eval()  # Setting model to eval mode stops training behavior such as dropout
    for feats, lengths, targets in validation_dataloader:
        outputs = model(feats.to(model.device), lengths, encoder_mask=True)
        distribution_params = model_ouput_type(*outputs)

        y_true, censor_ = unpack_label_and_censor(targets.to(model.device))
        loss = loss_criterion(distribution_params, y_true, censor_, lengths)
        val_loss += loss

    return val_loss.item()


def transformer_epoch(
    dls: DataLoaders,
    model: ProvidenceModule,
    optimizer: Optimizer,
    *,
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params
) -> EpochLosses:
    training_loss = transformer_training_pass(
        dls.train, model, optimizer, loss_criterion=loss_criterion, model_ouput_type=model_ouput_type
    )
    validation_loss = transformer_validation_pass(
        dls.validation, model, loss_criterion=loss_criterion, model_ouput_type=model_ouput_type
    )

    return EpochLosses(training_loss, validation_loss)
