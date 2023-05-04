"""
Utilities and helpers for doing model training on MLFlow

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from functools import wraps
from pathlib import Path
from typing import List, Optional

import mlflow
from progressbar import progressbar
from torch.optim import Optimizer

from providence.distributions import Weibull
from providence.loss import ProvidenceLossInterface, discrete_weibull_loss_fn
from providence.nn.module import ProvidenceModule
from providence.training import EpochLosses
from providence.training import LossAggregates
from providence.training import OptimizerWrapper
from providence.type_utils import patch
from providence.types import DataLoaders
from providence_utils.callbacks import Callback
from providence_utils.callbacks import check_before_epoch
from providence_utils.callbacks import type_name
from providence_utils.trainer import EpochInterface
from providence_utils.trainer import Trainer


def create_or_set_experiment(experiment_name) -> str:
    "Return experiment id after creating or setting the experiment"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:  # noqa E722
        experiment_id = mlflow.set_experiment(experiment_name).experiment_id

    return experiment_id


@patch
def mlflow_training(
    self: Trainer,
    model: ProvidenceModule,
    optimizer: OptimizerWrapper,
    dataloaders: DataLoaders,
    cbs: List[Callback] = None,
) -> LossAggregates:
    """Mlflow-specific training function for a Trainer.

    If you would like to use with method syntax, you must still import the function.
    For example:
        >>> from providence_utils.mlflow import mlflow_training
    Both of the following will work
        >>> mlflow_training(Trainer, model, optimizer, dls)
        >>> Trainer.mlflow_training(model, optimizer, dls)
    FastAI prefer the latter.

    Args:
        model (ProvidenceModule): model to train
        optimizer (OptimizerWrapper): Optimizer and associated hyperparameters to use in training
        dataloaders (DataLoaders): training and validation DataLoaders to be used in training and validation phases, resp.
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

        losses = self.epoch_func(dataloaders, model, optimizer.opt, step=current_epoch)
        loss_agg.append_losses(losses)

        for cb in cbs:
            cb.after_epoch(current_epoch, model, optimizer.opt, losses, dataloaders)
    else:
        self.training_termination_reason = "Completed all epochs"
        self.terminated_early = False

    for cb in cbs:
        if self.verbose:
            print(f"{type_name(cb)}.after_training(...)")
        cb.after_training(current_epoch, model, optimizer.opt, losses, dataloaders)

    model.to("cpu")
    model.eval()
    return loss_agg


def wrap_epoch_func(epoch_func: EpochInterface) -> EpochInterface:
    """Wrap ``epoch_func`` to support logging losses to MLFlow metrics.

    Args:
        epoch_fun (EpochInterface): a valid epoch definition

    Returns:
        EpochInterface: a function that will log mlflow metrics ``train_loss`` and ``val_loss`` after the epoch
    """

    @wraps(epoch_func)
    def f(
        dls: DataLoaders,
        model: ProvidenceModule,
        optimizer: Optimizer,
        *,
        step: Optional[int] = None,
        loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
        model_ouput_type=Weibull.Params,
        **log_metrics_kwargs,
    ) -> EpochLosses:
        losses = epoch_func(
            dls, model, optimizer, step=step, loss_criterion=loss_criterion, model_ouput_type=model_ouput_type
        )
        mlflow.log_metric("train_loss", losses.training, **log_metrics_kwargs)
        mlflow.log_metric("val_loss", losses.validation, **log_metrics_kwargs)
        return losses

    return f


def try_log_artifacts(output_dir: Path):
    """Attempt to log artifacts to MLFlow.

    Called "try" because this would occasionally fail - and then correct itself - withuot any changes to source
    code. So we attempt to do the logging, because we can't make any guarantees that it will work.

    Args:
        output_dir (Path): Path in databricks storage (or environs)
    """
    try:
        mlflow.log_artifacts(str(output_dir))
    except Exception as e:
        print("MLFlow failed to log run artifacts:", e)
