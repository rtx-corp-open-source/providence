"""
Utilities and helpers for doing model training on MLFlow

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from pathlib import Path
from typing import List

from progressbar import progressbar
from providence.nn.module import ProvidenceModule
from providence.training import EpochLosses, LossAggregates, OptimizerWrapper
from providence.type_utils import patch
from providence.types import DataLoaders

import mlflow
from providence_utils.callbacks import Callback, check_before_epoch, type_name
from providence_utils.trainer import EpochInterface, Trainer


def create_or_set_experiment(experiment_name) -> str:
    "Return experiment id after creating or setting the experiment"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.set_experiment(experiment_name).experiment_id

    return experiment_id


@patch
def mlflow_training(
    self: Trainer, model: ProvidenceModule, optimizer: OptimizerWrapper, dataloaders: DataLoaders, cbs: List[Callback]=None
):
    """
    Give an mlflow-specific training function for the Trainer. This is only applied if you import this package.
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

    model.to('cpu')
    model.eval()
    return loss_agg




def wrap_epoch_func(epoch_func: EpochInterface) -> EpochInterface:
    "Wraps a function for logging losses to MLFlow metrics."
    def f(dls: DataLoaders, model: ProvidenceModule, optimizer: OptimizerWrapper, **log_metrics_kwargs) -> EpochLosses:
        losses = epoch_func(dls, model, optimizer)
        mlflow.log_metric("train_loss", losses.training, **log_metrics_kwargs)
        mlflow.log_metric("val_loss", losses.validation, **log_metrics_kwargs)
        return losses

    return f


def try_log_artifacts(output_dir: Path):
    try:
        mlflow.log_artifacts(str(output_dir))
    except Exception as e:
        print("MLFlow failed to log run artifacts:", e)
