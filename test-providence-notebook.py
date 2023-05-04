"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
# flake8: noqa
# Databricks notebook source
# export http_proxy=http://devproxy.utc.com:80
import os

os.environ["http_proxy"] = "http://devproxy.utc.com:80"
os.environ["https_proxy"] = "http://devproxy.utc.com:80"

# COMMAND ----------

import providence.datasets as ds_lib

dir(ds_lib)

# COMMAND ----------

from providence.datasets import NasaFD00XDatasets, NasaTurbofanTest

nasa_fd001_train, nasa_fd001_test = NasaFD00XDatasets(
    NasaTurbofanTest.FD001, data_root="/dbfs/FileStore/datasets/providence/"
)

# COMMAND ----------

from providence.dataloaders import NasaFD00XDataLoaders, NasaDataLoaders

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training setup
# MAGIC Want the option for callbacks, if there's a need

# COMMAND ----------

from typing import List

from providence_utils.callbacks import (
    CachedIntervalMetricsVisualizer,
    Callback,
    ModelCheckpointer,
    WriteModelOutputs,
    check_before_epoch,
)
from providence.training import (
    DataLoaders,
    OptimizerWrapper,
    LossAggregates,
    training_epoch,
)

from progressbar import progressbar


def callback_training(
    model,
    optimizer: OptimizerWrapper,
    dataloaders: DataLoaders,
    cbs: List[Callback] = None,
):
    """Generic training + callbacks + progressbar"""
    if cbs is None:
        cbs = []

    model.to(model.device)
    # dataloaders.to_device(model.device)

    loss_agg = LossAggregates([], [])
    for current_epoch in progressbar(range(1, optimizer.num_epochs + 1)):
        terminate_training, termination_message = check_before_epoch(callbacks=cbs)
        if terminate_training:
            print(termination_message)
            break
        losses = training_epoch(dataloaders, model, optimizer.opt)
        loss_agg.append_losses(losses)

        for cb in cbs:
            cb.after_epoch(current_epoch, model, optimizer.opt, losses, dataloaders)
        else:
            # done to make sure everything is managed correctly. Callbacks shouldn't leave a mess but...
            model.to(model.device)
            # dls.to_device(model.device)

    for cb in cbs:
        cb.after_training(current_epoch, model, optimizer.opt, losses, dataloaders)

    # dataloaders.to_device('cpu')
    model.to("cpu")
    return loss_agg


# COMMAND ----------

# MAGIC %md
# MAGIC # Providence reproductions
# MAGIC One step at a time, first the NASA models

# COMMAND ----------

import mlflow

# COMMAND ----------

from pathlib import Path

from providence.paper_reproductions import (
    NasaRNN,
    NasaTraining,
    NasaRnnOptimizer,
    compute_loss_metrics,
)
from providence.nn import ProvidenceRNN
from providence.training import (
    OptimizerWrapper,
    minimize_torch_runtime_overhead,
    use_gpu_if_available,
)
from providence.utils import configure_logger_in_dir, now_dt_string

from torch import device as torch_device, initial_seed
import numpy as np

experiment_id = mlflow.set_experiment("/Users/40000889@azg.utccgl.com/Providence Reproductions").experiment_id

minimize_torch_runtime_overhead()

with mlflow.start_run(experiment_id=experiment_id):
    model_init = NasaRNN
    model_name = model_init.__name__
    model = model_init()
    model.device = torch_device(use_gpu_if_available())
    optimizer = NasaRnnOptimizer(model)
    dls = NasaDataLoaders(batch_size=optimizer.batch_size, data_root="/dbfs/FileStore/datasets/providence")

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("seed", initial_seed())
    mlflow.log_param("optimizer", type(optimizer.opt).__name__)

    # writing directly to the dbfs directories works just fine
    run_output_dir = Path(
        f"/dbfs/FileStore/manual-experiment-outputs/providence-{model_name}-NasaFull-{now_dt_string()}"
    )
    logger = configure_logger_in_dir(run_output_dir, logger_name="first-training-logger")

    losses = callback_training(
        model,
        optimizer,
        dls,
        [
            WriteModelOutputs(100, run_output_dir / "model-outputs", logger),
            CachedIntervalMetricsVisualizer(100, run_output_dir / "visualizations", logger),
            ModelCheckpointer(
                run_output_dir / "checkpoints", "val_loss", logger, keep_old=True
            ),  # fill up on models and model history
        ],
    )

    loss_metrics = compute_loss_metrics(losses)

    mlflow.log_metrics(loss_metrics)
    mlflow.log_artifacts(str(run_output_dir))  # copy everything that was generated from this run.


# COMMAND ----------

from providence_utils.visualization import plot_loss_curves

plot_loss_curves(losses.training_losses, losses.validation_losses, fname=None)
