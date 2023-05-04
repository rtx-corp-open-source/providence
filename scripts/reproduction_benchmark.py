"""
This script captures the key-value correspondence between a reference model and their best case performance
given all of the parameters we have available to configure the space.

Include everything for the best performance on the NASA FD00X subsets
-   Hyperparameters
-   Seeds
-   Every line of initialization

For engineering:
- If you want to use these models, you can just import these functions and have at it.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from collections import defaultdict
from pathlib import Path
from typing import Callable
from typing import Type

import pandas as pd
import torch as pt

from providence.dataloaders import CustomProvidenceDataloaders
from providence.dataloaders import ProvidenceDataLoader
from providence.datasets.adapters import BACKBLAZE_FEATURE_NAMES
from providence.datasets.adapters import NASA_FEATURE_NAMES
from providence.datasets.nasa import NasaDatasets
from providence.nn.module import ProvidenceModule
from providence.nn.rnn import ProvidenceGRU
from providence.nn.rnn import ProvidenceLSTM
from providence.nn.rnn import ProvidenceVanillaRNN
from providence.nn.transformer import ProvidenceTransformer
from providence.paper_reproductions import GeneralMetrics
from providence.training import OptimizerWrapper
from providence.training import set_torch_default_dtypes
from providence.training import training_epoch
from providence.training import use_gpu_if_available
from providence.types import DataLoaders
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
from providence_utils.trainer import EpochInterface
from providence_utils.trainer import Trainer
from providence_utils.trainer import transformer_epoch

# NOTE: IMPORTANT. this is what we use, and is pretty standard in research
set_torch_default_dtypes(pt.float32)


def callback_default_config(num_epochs: int = 500) -> dict:
    callback_config = dict()
    callback_config["epoch_logger_epochs"] = 10
    callback_config["visualizer"] = IntervalMetricsVisualizer.__name__
    callback_config["visualization_freq"] = min(200, num_epochs // 5)
    callback_config["model_output_freq"] = min(100, num_epochs // 2)
    callback_config["n_kept_checkpoints"] = 5
    callback_config["early_stopping.patience"] = 5
    callback_config["early_stopping.defer"] = 100  # typically want this matching the ebrake
    callback_config["ebrake_epoch"] = 80
    callback_config["ebrake_requisite_loss"] = 12.0
    callback_config["tracking_metric"] = "val_loss"
    return callback_config


def optimizer_default_config() -> dict:
    optim_config = dict()
    optim_config["schedule_min"] = 1e-5
    optim_config["schedule_T_mult"] = 2
    return optim_config


def do_model_run(
    model_type: Type[ProvidenceModule],
    get_dls: Callable[[], DataLoaders],
    random_seed: int,
    optim_params: dict,
    model_params: dict,
    callback_config: dict,
    outputs_root_dir: Path,
    epoch_definition: EpochInterface,
):
    set_seed(random_seed)
    model = model_type(**model_params, device=use_gpu_if_available())
    optimizer = OptimizerWrapper(
        optim_params["type"](model.parameters(), lr=optim_params["learning_rate"]),
        batch_size=optim_params["batch_size"],
        num_epochs=optim_params["num_epochs"],
    )
    run_config = defaultdict(dict)
    run_config["model"].update(model_params)  # copy key-values
    run_config["model"]["type"] = model_type.__name__
    run_config["model"]["seed"] = random_seed
    run_config["callbacks"].update(callback_config)
    run_config["optimizer"].update(optim_params)
    run_config["optimizer"]["type"] = optim_params["type"].__name__

    callback_config = run_config["callbacks"]
    output_dir: Path = outputs_root_dir / now_dt_string()
    logger = configure_logger_in_dir(output_dir)
    cbs = [
        LearningRateScheduler(
            optimizer.opt,
            T_mult=optim_params["schedule_T_mult"],
            eta_min=optim_params["schedule_min"],
        ),
        IntervalMetricsVisualizer(callback_config["visualization_freq"], output_dir, logger=logger),
        WriteModelOutputs(
            callback_config["model_output_freq"],
            output_dir,
            logger=logger,
            verbose=False,
        ),
        ModelCheckpointer(
            output_dir=(output_dir / "model-checkpoints"),
            track=callback_config["tracking_metric"],
            logger=logger,
            verbose=True,
            keep_old=5,
        ),
        DeferrableEarlyStopping(
            patience=callback_config["early_stopping.patience"],
            track=callback_config["tracking_metric"],
            defer_until=callback_config["early_stopping.defer"],
        ),
        EmergencyBrake(callback_config["ebrake_epoch"], callback_config["ebrake_requisite_loss"]),
        LearningCurveTracker(1000, output_dir=(output_dir / "learning_curve")),
    ]
    trainer = Trainer(epoch_definition)
    # run_config to something we can log to mlflow

    dls = get_dls()
    print("Starting training with config\n{}".format(nest_keys(run_config)))
    losses = trainer.callback_training(model, optimizer, dls, cbs)
    print("Training completed")

    logger.info(f"{trainer.terminated_early=} {trainer.training_termination_reason=}")
    print("Generating metrics")
    model_metrics = GeneralMetrics(model, dls.test_ds, losses)
    print("Generating metrics completed")

    if outputs_cb := next(filter(lambda cb: isinstance(cb, ModelCheckpointer), cbs), None):
        outputs_cb: ModelCheckpointer
        best_model: pt.nn.Module = pt.load(outputs_cb.best_model)
        best_metrics = GeneralMetrics(best_model, dls.test_ds, losses)
        model_metrics = pd.concat([model_metrics, best_metrics])

    return model, model_metrics


feature_counts = {
    "nasa": len(NASA_FEATURE_NAMES),
    "backblaze": len(BACKBLAZE_FEATURE_NAMES),
}


# NOTE one of these is redundant, as its covered in the paper_reproductions module
# NOTE: these statefully set the global seed.
def NASA_Aggregate_VanillRNN(data_root_path: str, outputs_root: str):
    """
    Arguments:
        data_root_path: the root for the download of the dataset. Needs write permissions
        outputs_root: the root for the outputs emitted by the callbacks, as well as the metrics CSV. Needs write permissions

    Return:
        model: the trained model. There may be a model checkpoint that was better than this one, so we encourage you to compare
            with the file model checkpoint output by `ModelCheckpointer`
        metrics: produced by `GeneralMetrics`
    """
    train_ds, val_ds = NasaDatasets(data_root=data_root_path)
    test_ds = val_ds

    def fresh_dls(bs: int):  # replicate what was done above. Just do it again
        return CustomProvidenceDataloaders(
            train_ds,
            val_ds,
            batch_size=bs,
            num_workers=1,
            pin_memory=True,
        )._replace(test=ProvidenceDataLoader(test_ds, batch_size=1, num_workers=1))

    optim_params = {
        **optimizer_default_config(),
        "batch_size": 4,
        "learning_rate": 0.003,
        "num_epochs": 200,
        "schedule_T_mult": 2,
        "schedule_min": 1e-05,
        "type": pt.optim.Adam,
    }

    model, metrics = do_model_run(
        ProvidenceVanillaRNN,
        fresh_dls,
        random_seed=11068621650300516211,
        optim_params=optim_params,
        model_params=dict(
            input_size=feature_counts["nasa"],
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
        ),
        callback_config=callback_default_config(),
        outputs_root_dir=outputs_root,
        epoch_definition=training_epoch,
    )
    return model, metrics


NASA_Aggregate_VanillRNN.best_metrics = pd.DataFrame(
    {
        "MSE": [2008.5],
        "MFE": [-0.215],
        "SMAPE": [0.13],
        "SMPE": [0.003],
    }
)  # yapf: disable


def NASA_Aggregate_GRU(data_root_path: str, outputs_root: str):
    """
    Arguments:
        data_root_path: the root for the download of the dataset. Needs write permissions
        outputs_root: the root for the outputs emitted by the callbacks, as well as the metrics CSV. Needs write permissions

    Return:
        model: the trained model. There may be a model checkpoint that was better than this one, so we encourage you to compare
            with the file model checkpoint output by `ModelCheckpointer`
        metrics: produced by `GeneralMetrics`
    """
    train_ds, val_ds = NasaDatasets(data_root=data_root_path)
    test_ds = val_ds

    def fresh_dls(bs: int):  # replicate what was done above. Just do it again
        return CustomProvidenceDataloaders(
            train_ds,
            val_ds,
            batch_size=bs,
            num_workers=1,
            pin_memory=True,
        )._replace(test=ProvidenceDataLoader(test_ds, batch_size=1, num_workers=1))

    optim_params = {
        **optimizer_default_config(),
        "batch_size": 2,
        "learning_rate": 1e-3,
        "num_epochs": 50,
        "schedule_T_mult": 2,
        "schedule_min": 1e-05,
        "type": pt.optim.Adam,
    }

    model, metrics = do_model_run(
        ProvidenceGRU,
        fresh_dls,
        random_seed=11068621650300516211,
        optim_params=optim_params,
        model_params=dict(
            input_size=feature_counts["nasa"],
            hidden_size=512,
            num_layers=3,
            dropout=0.75,
        ),
        callback_config=callback_default_config(),
        outputs_root_dir=outputs_root,
        epoch_definition=training_epoch,
    )
    return model, metrics


NASA_Aggregate_GRU.best_metrics = pd.DataFrame(
    {
        "MSE": [2008.5],
        "MFE": [-0.215],
        "SMAPE": [0.13],
        "SMPE": [0.003],
    }
)  # yapf: disable


def NASA_Aggregate_LSTM(data_root_path: str, outputs_root: str):
    """
    Arguments:
        data_root_path: the root for the download of the dataset. Needs write permissions
        outputs_root: the root for the outputs emitted by the callbacks, as well as the metrics CSV. Needs write permissions

    Return:
        model: the trained model. There may be a model checkpoint that was better than this one, so we encourage you to compare
            with the file model checkpoint output by `ModelCheckpointer`
        metrics: produced by `GeneralMetrics`
    """
    train_ds, val_ds = NasaDatasets(data_root=data_root_path)
    test_ds = val_ds

    def fresh_dls(bs: int):  # replicate what was done above. Just do it again
        return CustomProvidenceDataloaders(
            train_ds,
            val_ds,
            batch_size=bs,
            num_workers=1,
            pin_memory=True,
        )._replace(test=ProvidenceDataLoader(test_ds, batch_size=1, num_workers=1))

    optim_params = {
        **optimizer_default_config(),
        "batch_size": 4,
        "learning_rate": 3e-3,
        "num_epochs": 40,
        "schedule_T_mult": 2,
        "schedule_min": 1e-05,
        "type": pt.optim.Adam,
    }

    model, metrics = do_model_run(
        ProvidenceLSTM,
        fresh_dls,
        random_seed=11068621650300516211,
        optim_params=optim_params,
        model_params=dict(
            input_size=feature_counts["nasa"],
            hidden_size=1024,
            num_layers=3,
            dropout=0.25,
        ),
        callback_config=callback_default_config(),
        outputs_root_dir=outputs_root,
        epoch_definition=training_epoch,
    )
    return model, metrics


NASA_Aggregate_LSTM.best_metrics = pd.DataFrame(
    {
        "MSE": [2008.5],
        "MFE": [-0.215],
        "SMAPE": [0.13],
        "SMPE": [0.003],
    }
)  # yapf: disable

# TODO: fill in from the Confluence page (https://confluence.utc.com/display/NAPD/Providence+Reproducibility)


def Backblaze_VanillaRNN():
    ...


Backblaze_VanillaRNN.best_metrics = pd.DataFrame(...)


def Backblaze_LSTM():
    ...


Backblaze_LSTM.best_metrics = pd.DataFrame(...)


def Backblaze_GRU():
    ...


Backblaze_GRU.best_metrics = pd.DataFrame(...)


def BackblazeExtended_VanillaRNN():
    ...


BackblazeExtended_VanillaRNN.best_metrics = pd.DataFrame(...)


def BackblazeExtended_LSTM():
    ...


BackblazeExtended_LSTM.best_metrics = pd.DataFrame(...)


def BackblazeExtended_GRU():
    ...


BackblazeExtended_GRU.best_metrics = pd.DataFrame(...)
