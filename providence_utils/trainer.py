"""
Trainer abstraction to make it easier to do model training (especially with our custom Callbacks)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from abc import abstractmethod
from typing import Generic, List, Tuple
from typing import NamedTuple
from typing import Optional
from typing import Protocol

import torch as pt
from click import progressbar
from torch import no_grad
from torch import zeros
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer

from providence.distributions import SurvivalAnalysisDistribution, T_DistributionParams
from providence.distributions import Weibull
from providence.loss import discrete_weibull_loss_fn
from providence.loss import ProvidenceLossInterface
from providence.nn import ProvidenceModule
from providence.training import EpochLosses
from providence.training import LossAggregates
from providence.training import OptimizerWrapper
from providence.training import unpack_label_and_censor
from providence.types import DataLoaders
from providence.types import TorchDataLoader
from providence.types import TypeAlias
from providence_utils.callbacks import Callback
from providence_utils.callbacks import check_before_epoch


class EpochInterface(Protocol):
    @abstractmethod
    def __call__(
        self,
        dls: DataLoaders,
        model: ProvidenceModule,
        optimizer: Optimizer,
        *,
        step: Optional[int] = None,
        loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
        model_ouput_type=Weibull.Params,
    ) -> EpochLosses:
        """The contract of an epoch of training of the supplied ``model`` on the given ``dls`` wrt the ``optimizer``.


        Historical note:
        ``dls`` are first in the parameter list because they are the primary discriminating factor in the implementation.
        Whether you utilize ``dls.test`` is up to your usage.
        While it matches the ``epoch`` functions to have ``model`` first, semantically it's more cognitive load to go
        looking even one or two parameters to see where two function call sites differ. Either ``transformer_epoch`` or
        ``training_epoch`` demonstrate this well.

        NOTE: (dls, model, optimizer) is the preferred triple. Having to switch back and forth between this is a burden
        """
        ...


class Trainer:
    """A training orchestrator, to DRY usage of ``callback_training`` and supporting *slight* variations in training."""

    def __init__(self, epoch_func: EpochInterface, verbose: bool = False):
        self.epoch_func = epoch_func
        self.verbose = verbose
        self.training_termination_reason: Optional[str] = None

    def callback_training(
        self,
        model: ProvidenceModule,
        optimizer: OptimizerWrapper,
        dataloaders: DataLoaders,
        cbs: List[Callback] = None,
    ) -> LossAggregates:
        """Train generically, with callback support.

        Args:
            model (ProvidenceModule): TorchModule to train
            optimizer (OptimizerWrapper): Optimizer and hyperpameters
            dataloaders (DataLoaders): training and validation ``DataLoader`` s
            cbs (List[Callback], optional): A list of callbacks, which will be checked against every epoch.
                Defaults to None.

        Returns:
            LossAggregates: All training- and validation-phase losses from training ``model`` on the given ``dataloaders``
        """
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
            if self.verbose:
                print(f"{type(cb).__name__}.after_training(...)")
            cb.after_training(current_epoch, model, optimizer.opt, losses, dataloaders)

        model.to("cpu")
        model.eval()
        return loss_agg

    def basic_training(
        self,
        model: ProvidenceModule,
        optimizer: OptimizerWrapper,
        dataloaders: DataLoaders,
    ) -> LossAggregates:
        """Train with basic configurable epoch support for a custom epoch function (which can produce a custom loss).

        We take the liberty of moving the ``model`` to its (training) ``device``, and then back to the CPU once
        ``optimizer.num_epochs`` have elapsed and results have been aggregated.
        We do not concern ourselves (as in ``generic_training``) with the ``model_output_type``, as all of that would
        be handled in ``epoch_func``

        Args:
            model (ProvidenceModule): model to train
            optimizer (OptimizerWrapper): Optimizer and associated hyperparameters to use in training
            dataloaders (DataLoaders): training and validation DataLoaders to be used in training and validation phases, resp.

        Returns:
            LossAggregates: All training- and validation-phase losses from training ``model`` on the given ``dataloaders``
        """

        model.to(model.device)
        # dataloaders.to_device(model.device)

        loss_agg = LossAggregates([], [])
        for _ in range(1, optimizer.num_epochs + 1):
            losses = self.epoch_func(dataloaders, model, optimizer.opt)
            loss_agg.append_losses(losses)

        # dataloaders.to_device('cpu')
        model.to("cpu")
        return loss_agg


class ProvidenceLossHandle(Generic[T_DistributionParams]):
    """Pair for handling the loss and model output which should feed into said loss."""

    def __init__(self, loss_criterion: ProvidenceLossInterface, model_output_t: T_DistributionParams) -> None:
        super().__init__()
        self.loss_criterion = loss_criterion
        self.model_output_t = model_output_t


ProvidenceBatch: TypeAlias = Tuple[pt.Tensor, pt.Tensor, pt.Tensor]
"""Type of a ProvidenceBatch: features, length, and labels - the values returned by ``providence_collate_fn()``."""


class ProvidenceTrainer(Generic[T_DistributionParams]):
    """Training to make it easier to edit the parts that needs to change e.g. how you infer on a given batch.

    This is an improvement over ``Trainer`` while keeping the benefits of the old design in making
    callback training easier.
    This is meant to be inherited from and partially overridden, letting inheritance do the rest of the work of the
    training / evaluation loop.

    Args:
        model: the model to train
        optimizer: instantiated optimizer, probably pointed at the model
        loss_handle: holds the loss function and the ``model_output_t`` to be constructed from the model's forward pass outputs
        should_clip_gradients: if we should perform gradient clipping during training. See ``providence.training`` for more.
    """

    def __init__(
        self,
        model: ProvidenceModule,
        optimizer: Optimizer,
        loss_handle: ProvidenceLossHandle,
        should_clip_gradients: bool = True,
    ) -> None:
        self.loss_handle = loss_handle
        self.clip_gradients = should_clip_gradients
        self.model = model
        self.optimizer = optimizer
        self.cbs: List = []  # just add to this directly.

    def prepare_batch(self, batch: ProvidenceBatch):
        """Prepare the batch for inference.

        If any changes need to be made to the batch = (features, lengths, labels, ), the user should override and denote such changes here.
        If you make a modification here, you need to make changes to ``batch_train_step`` and ``batch_eval_step`` at least.

        Args:
            batch (Tuple): tuple of (features, lengths, labels)

        Returns:
            Tuple: batch, exactly as passed in
        """
        # default thanks to the Providence Dataloader
        # features, lengths, targets, *rest = args
        # return features, lengths, targets
        return batch

    def batch_train_step(self, batch: ProvidenceBatch) -> pt.Tensor:
        """Train on ``batch``; used for every ``batch`` in some training ``DataLoader``.

        Override this function is you need different behavior on the inner loop of the training epoch

        Args:
            batch (ProvidenceBatch): tuple of (features, lengths, labels)
            If you override ``prepare_batch``, this wil be whatever you return therein.

        Returns:
            pt.FloatTensor: training loss for ``batch``
        """
        self.optimizer.zero_grad(set_to_none=True)  # should be faster
        feats, lengths, targets = batch

        distribution_params = self.batch_inference(self.model, feats, lengths)

        y_true, censor_ = unpack_label_and_censor(targets.to(self.model.device))
        loss = self.loss_handle.loss_criterion(distribution_params, y_true, censor_, lengths)
        loss.backward()

        if self.clip_gradients:
            clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
        self.optimizer.step()

        return loss

    def batch_inference(self, model: ProvidenceModule, feats: pt.Tensor, lengths: pt.Tensor) -> NamedTuple:
        """Perform inference on a training batch. Override this for a different behavior.

        NOTE: we return ``NamedTuple`` for the sake of MyPy

        Args:
            model (ProvidenceModule): should also be a TorchModule
            feats (pt.Tensor): features to predict, aligned in a ``model``-compliant way.
            lengths (pt.Tensor): length of each sequence in ``feats``

        Returns:
            T <: SurvivalAnalysisDistribution.Params: Some parameters for some survival analysis distribution
        """
        outputs = model(feats.to(model.device), lengths, encoder_mask=True)
        distribution_params = self.loss_handle.model_output_t(*outputs)
        return distribution_params

    def batch_eval_step(self, batch) -> pt.Tensor:
        """Evaluate on ``batch``; used for every ``batch`` in some validation ``DataLoader``.

        Override this function is you need different behavior on the inner loop of the training epoch

        Args:
            batch (ProvidenceBatch): tuple of (features, lengths, labels)
            If you override ``prepare_batch``, this wil be whatever you return therein.

        Returns:
            pt.FloatTensor: validation loss for ``batch``
        """
        features, lengths, targets = batch
        distribution_params = self.batch_inference(self.model, features, lengths)
        y_true, censor_ = unpack_label_and_censor(targets.to(self.model.device))
        loss = self.loss_handle.loss_criterion(distribution_params, y_true, censor_, lengths)
        return loss

    def epoch_pass(self, dls: DataLoaders) -> EpochLosses:
        """Run training and validation in this epoch.

        Args:
            dls (DataLoaders): training and validation ``DataLoader`` s

        Returns:
            EpochLosses: mean training and mean validaton losses
        """
        self.model.train()
        train_loss = zeros(1, device=self.model.device)
        for i, batch in enumerate(dls.train):
            train_loss += self.batch_train_step(batch)
        epoch_train_loss = train_loss.mean()

        self.model.eval()
        validation_loss = zeros(1, device=self.model.device)
        for i, batch in enumerate(dls.validation):
            validation_loss += self.batch_eval_step(batch)
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
        """Train ``self.model`` on ``dls`` for ``num_epochs``.

        Args:
            num_epochs (int): number of epochs
            dls (DataLoaders): training and validation ``DataLoader`` s

        Returns:
            LossAggregates: All training- and validation-phase mean losses from training
                ``model`` on the given ``dataloaders``
        """
        agg = LossAggregates([], [])
        self.model.to(self.model.device)
        for current_epoch in range(num_epochs):
            if self._should_terminate():
                break

            epoch_losses = self.epoch_pass(dls)

            agg.append_losses(epoch_losses)
            for cb in self.cbs:
                cb.after_training(current_epoch, self.model, self.optimizer, epoch_losses, dls)

        return agg


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
    clip_gradients=True,
):
    """Perform the training pass on the dataloader i.e. running it to exhaustion on the ``model`` with ``optimizer``.

    The difference between this and ``training.training_pass`` is that this triggers the Transformre model masking, to
    guarantee the model doesn't use the future to predict the past.

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
    model_ouput_type=Weibull.Params,
):
    """Perform the validation pass of an epoch, not optimizing any weights but only accumulating the loss.

    The difference between this and ``training.validation_pass`` is that this triggers the Transformre model masking,
    to guarantee the model doesn't use the future to predict the past.

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
    model_ouput_type=Weibull.Params,
) -> EpochLosses:
    """Perform a standard training epoch: first train, then validate against corresponding DataLoaders in ``dataloaders``.

    The only difference between this and ``training.training_epoch`` is the usage of the above Transformer-specific
    functions.

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
    training_loss = transformer_training_pass(
        dls.train,
        model,
        optimizer,
        loss_criterion=loss_criterion,
        model_ouput_type=model_ouput_type,
    )
    validation_loss = transformer_validation_pass(
        dls.validation,
        model,
        loss_criterion=loss_criterion,
        model_ouput_type=model_ouput_type,
    )

    return EpochLosses(training_loss, validation_loss)
