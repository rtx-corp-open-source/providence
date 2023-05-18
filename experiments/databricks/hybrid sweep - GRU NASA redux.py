# Databricks notebook source
# MAGIC %pip install --force /dbfs/FileStore/binaries/providence/providence-1.0.0_rc7e-py3-none-any.whl
"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
# COMMAND ----------
# !nvidia-smi
# COMMAND ----------
from collections import defaultdict
from pathlib import Path
from typing import Callable
from typing import Type

import mlflow
import torch as pt

from providence.dataloaders import CustomProvidenceDataloaders
from providence.dataloaders import ProvidenceDataLoader
from providence.datasets import NasaDatasets
from providence.datasets.adapters import BACKBLAZE_FEATURE_NAMES
from providence.datasets.adapters import NASA_FEATURE_NAMES
from providence.distributions import Weibull3
from providence.loss import discrete_weibull3_loss_fn
from providence.metrics import SDist
from providence.nn.module import ProvidenceModule
from providence.paper_reproductions import GeneralMetrics
from providence.training import EpochLosses
from providence.training import LossAggregates
from providence.training import OptimizerWrapper
from providence.training import set_torch_default_dtypes
from providence.training import training_epoch
from providence.training import use_gpu_if_available
from providence.type_utils import type_name
from providence.types import DataLoaders
from providence.utils import clear_memory
from providence.utils import configure_logger_in_dir
from providence.utils import now_dt_string
from providence.utils import set_seed
from providence_utils.callbacks import DeferrableEarlyStopping
from providence_utils.callbacks import EmergencyBrake
from providence_utils.callbacks import IntervalMetricsVisualizer
from providence_utils.callbacks import LearningCurveTracker
from providence_utils.callbacks import LearningRateScheduler
from providence_utils.callbacks import ModelCheckpointer
from providence_utils.callbacks import WriteModelOutputs
from providence_utils.hyperparameter_sweeper import nest_keys
from providence_utils.mlflow import create_or_set_experiment
from providence_utils.mlflow import mlflow_training
from providence_utils.mlflow import try_log_artifacts
from providence_utils.mlflow import wrap_epoch_func
from providence_utils.trainer import EpochInterface
from providence_utils.trainer import Trainer

# COMMAND ----------

general_config = {
    # don't use float64, because it's bigger than we need
    "dtype": pt.float32,
    "batch_size": 8,
}

# COMMAND ----------

DS_NAME_TO_FEATURE_COUNT = {
    "backblaze": len(BACKBLAZE_FEATURE_NAMES),
    "bbe": len(BACKBLAZE_FEATURE_NAMES),
    "nasa": len(NASA_FEATURE_NAMES),
}

# COMMAND ----------

# MAGIC %md ##

# COMMAND ----------

metadata_to_log = defaultdict(dict)
metadata_to_log["general"]["dtype"] = general_config["dtype"]

set_torch_default_dtypes(metadata_to_log["general"]["dtype"])

# COMMAND ----------


def callback_default_config(num_epochs: int = 500) -> dict:
    callback_config = dict()
    callback_config["epoch_logger_epochs"] = 10
    callback_config["visualizer"] = IntervalMetricsVisualizer.__name__
    callback_config["visualization_freq"] = min(200, num_epochs // 5)
    callback_config["model_output_freq"] = min(100, num_epochs // 2)
    callback_config["n_kept_checkpoints"] = 5
    callback_config["early_stopping.patience"] = 5
    callback_config["early_stopping.defer"] = 40  # typically want this matching the ebrake
    callback_config["ebrake_epoch"] = 40
    callback_config["ebrake_requisite_loss"] = 12.0
    callback_config["tracking_metric"] = "val_loss"
    return callback_config


def optimizer_default_config() -> dict:
    # Why isn't this passed in? These are driven heuristically (from earlier analysis).
    optim_config = dict()
    optim_config["schedule_min"] = 1e-5
    optim_config["schedule_T_mult"] = 2
    return optim_config


# COMMAND ----------


# COMMAND ----------
def do_mlflow_run(
    model_type: Type[ProvidenceModule],
    get_dls: Callable[[], DataLoaders],
    experiment_id: str,
    random_seed: int,
    optim_config: dict,
    instantiation_params: dict,
    *,
    root_dir: Path,
    epoch_func: EpochInterface,
    dist_t: SDist,
):
    with mlflow.start_run(experiment_id=experiment_id):
        set_seed(random_seed)
        print(f"{random_seed=} {pt.initial_seed()=}")
        model = model_type(**instantiation_params, device=use_gpu_if_available())
        optimizer = OptimizerWrapper(
            optim_config["type"](model.parameters(), lr=optim_config["learning_rate"]),
            batch_size=optim_config["batch_size"],
            num_epochs=optim_config["num_epochs"],
        )
        run_config = defaultdict(dict)
        run_config["model"].update(instantiation_params)  # copy key-values
        run_config["model"]["type"] = type_name(model_type)
        run_config["model"]["seed"] = random_seed
        run_config["callbacks"].update(callback_default_config())
        run_config["optimizer"].update(optim_config)

        callback_config = run_config["callbacks"]
        output_dir: Path = root_dir / now_dt_string()
        logger = configure_logger_in_dir(output_dir)
        cbs = [
            LearningRateScheduler(
                optimizer.opt,
                T_mult=optim_config["schedule_T_mult"],
                eta_min=optim_config["schedule_min"],
            ),
            IntervalMetricsVisualizer(
                callback_config["visualization_freq"],
                output_dir,
                logger=logger,
                dist_t=dist_t,
            ),
            WriteModelOutputs(
                callback_config["model_output_freq"],
                output_dir,
                logger=logger,
                verbose=False,
                dist_t=dist_t,
            ),
            ModelCheckpointer(
                output_dir=(output_dir / "model-checkpoints"),
                track=callback_config["tracking_metric"],
                logger=logger,
                verbose=True,
                keep_old=5,
            ),
            DeferrableEarlyStopping(
                track=callback_config["tracking_metric"],
                patience=callback_config["early_stopping.patience"],
                defer_until=callback_config["early_stopping.defer"],
            ),
            # our weibull loss takes a bit longer to bit below 1.0, but still begets strong results
            EmergencyBrake(
                callback_config["ebrake_epoch"],
                callback_config["ebrake_requisite_loss"],
            ),
            LearningCurveTracker(1000, output_dir=(output_dir / "learning_curve")),
        ]
        trainer = Trainer(epoch_func)
        # run_config to something we can log to mlflow
        merged_config = dict(**run_config, **metadata_to_log)
        nested_config = {k: (type_name(v) if isinstance(v, type) else v) for (k, v) in nest_keys(merged_config).items()}

        mlflow.log_params(nested_config)
        dls = get_dls()
        print("Starting training")
        losses = trainer.mlflow_training(model, optimizer, dls, cbs)
        print("Training completed")

        logger.info(f"{trainer.terminated_early=} {trainer.training_termination_reason=}")
        print("Generating metrics")
        model_metrics = GeneralMetrics(model, dls.test_ds, losses, dist_t=dist_t)
        print("Generating metrics completed")
        mlflow.log_metrics(model_metrics.iloc[0].to_dict())
        try_log_artifacts(output_dir)
    return model, model_metrics


# COMMAND ----------

# !nvidia-smi

# COMMAND ----------

# MAGIC %md Data access

# COMMAND ----------

metadata_to_log["data"]["global_seed_at_init"] = pt.initial_seed()

GLOBAL_train_ds, GLOBAL_val_ds = NasaDatasets(data_root="/dbfs/FileStore/datasets/providence")
GLOBAL_test_ds = GLOBAL_val_ds
metadata_to_log["data"]["name"] = "NASA"

# COMMAND ----------


def global_dls(
    bs=general_config["batch_size"],
):  # replicate what was done above. Just do it again
    return CustomProvidenceDataloaders(
        GLOBAL_train_ds,
        GLOBAL_val_ds,
        batch_size=bs,
        num_workers=1,
        pin_memory=True,
    )._replace(test=ProvidenceDataLoader(GLOBAL_test_ds, batch_size=1, num_workers=1))


# COMMAND ----------

# MAGIC %md Well we better get started

# COMMAND ----------

# MAGIC %md
# MAGIC # Reproducibility proof

# COMMAND ----------

EXPERIMENT_NAME = "/Users/{UserNameHere}/Providence-Investigative-Research/NASA GRU (Hybrid, Redux)"

EXPERIMENT_ID = create_or_set_experiment(EXPERIMENT_NAME)

ROOT_DIR = Path("/dbfs/FileStore/providence-benchmarking/nasa-gru-hybrid-redux")
ROOT_DIR.mkdir(exist_ok=True, parents=True)

# COMMAND ----------


def weibull3_training_epoch(
    dataloaders: DataLoaders,
    model: ProvidenceModule,
    optimizer: pt.optim.Optimizer,
) -> EpochLosses:
    losses = training_epoch(
        dataloaders,
        model,
        optimizer,
        loss_criterion=discrete_weibull3_loss_fn,
        model_ouput_type=Weibull3.Params,
    )

    # calculate the mean loss per batch
    averaged_losses = losses._replace(
        training=losses.training / len(dataloaders.train),
        validation=losses.validation / len(dataloaders.validation),
    )
    return averaged_losses


# COMMAND ----------
from providence.nn import ProvidenceGRU


def reproducibility_proof__seed_check():
    # this run started after the set_seed changed to return to setting the rng state
    experiment_id = create_or_set_experiment(EXPERIMENT_NAME.replace("GRU", "GRU Reproducibility Check"))
    n_seeds_to_check = 1
    for _ in range(n_seeds_to_check):
        fixed_seed = pt.seed()
        for i in range(2):
            model, _ = do_mlflow_run(
                ProvidenceGRU,
                global_dls,
                experiment_id,
                random_seed=fixed_seed,
                optim_config={
                    **optimizer_default_config(),
                    "type": pt.optim.Adam,
                    "learning_rate": 3e-4,
                    "batch_size": 8,
                    "num_epochs": 700,
                },
                instantiation_params={
                    "input_size": DS_NAME_TO_FEATURE_COUNT["nasa"],
                    "hidden_size": 128,
                    "num_layers": 2,
                    "dropout": 0.7,
                    "activation": "weibull3",
                },
                epoch_func=wrap_epoch_func(weibull3_training_epoch),
                root_dir=ROOT_DIR,
                dist_t=Weibull3,
            )
        clear_memory(model)


reproducibility_proof__seed_check()

# MAGIC %md
# MAGIC # Model runs

# COMMAND ----------
