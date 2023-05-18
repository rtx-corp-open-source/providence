# Databricks notebook source
# MAGIC %pip install --force /dbfs/FileStore/binaries/providence-1.0.0_rc7c-py3-none-any.whl
"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
# COMMAND ----------
# !nvidia-smi
# COMMAND ----------
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import Final
from typing import Tuple
from typing import Type

import mlflow
import torch

from providence.dataloaders import CustomProvidenceDataloaders
from providence.dataloaders import ProvidenceDataLoader
from providence.datasets.adapters import BACKBLAZE_FEATURE_NAMES
from providence.datasets.adapters import BackblazeQuarter
from providence.datasets.adapters import NASA_FEATURE_NAMES
from providence.datasets.backblaze import BackblazeDatasets
from providence.nn.module import ProvidenceModule
from providence.paper_reproductions import GeneralMetrics
from providence.training import OptimizerWrapper
from providence.training import set_torch_default_dtypes
from providence.training import training_epoch
from providence.training import use_gpu_if_available
from providence.types import DataLoaders
from providence.utils import configure_logger_in_dir
from providence.utils import multilevel_sort
from providence.utils import now_dt_string
from providence.utils import set_seed
from providence_utils.callbacks import EarlyStopping
from providence_utils.callbacks import EmergencyBrake
from providence_utils.callbacks import IntervalMetricsVisualizer
from providence_utils.callbacks import LearningCurveTracker
from providence_utils.callbacks import LearningRateScheduler
from providence_utils.callbacks import ModelCheckpointer
from providence_utils.callbacks import WriteModelOutputs
from providence_utils.hyperparameter_sweeper import Hyperparameter
from providence_utils.hyperparameter_sweeper import HyperparameterList
from providence_utils.hyperparameter_sweeper import HyperparameterSweeper
from providence_utils.hyperparameter_sweeper import nest_keys
from providence_utils.trainer import Trainer

# COMMAND ----------
_SWEEPS_DEFAULT_HIDDEN_DIM: Final[int] = 64
_SWEEPS_DEFAULT_NUM_LAYERS: Final[int] = 2
_SWEEPS_DEFAULT_DROPOUT: Final[int] = 0.3
_SWEEPS_DEFAULT_DATA_RANDOM_SEED: Final[int] = 1234

# COMMAND ----------
general_config = {
    # don't use float64, because it's bigger than we need
    "dtype": torch.float32,
    "batch_size": 128,
}

# COMMAND ----------
DS_NAME_TO_FEATURE_COUNT = {
    "backblaze": len(BACKBLAZE_FEATURE_NAMES),
    "bbe": len(BACKBLAZE_FEATURE_NAMES),
    "nasa": len(NASA_FEATURE_NAMES),
}

# COMMAND ----------
"""
# MLFlow Pre-work

"""

# COMMAND ----------


def mlflow_epoch(dls, model, optimizer):
    losses = training_epoch(dls, model, optimizer)
    mlflow.log_metric("train_loss", losses.training)
    mlflow.log_metric("val_loss", losses.validation)
    return losses


# COMMAND ----------

sum(1 for _ in Path("/dbfs/FileStore/providence-legacy-recovery/backblaze-vanilla-rnn").glob("**/*.*"))

# COMMAND ----------

# MAGIC %md
# MAGIC 637 of the above are all from older runs, in a "sweeps" fashion.

# COMMAND ----------


def recursive_delete(directory: Path) -> None:
    for child in directory.glob("**/*.*"):
        if child.is_dir():
            recursive_delete(child)
            child.rmdir()
        else:
            child.unlink()


# recursive_delete(Path("/dbfs/FileStore/providence-legacy-recovery/backblaze-gru"))

# COMMAND ----------

experiment_name = "/Users/{UserNameHere}/ProvidenceReproductionNotebooks/Backblaze GRU (Linear)"

try:
    EXPERIMENT_ID = mlflow.create_experiment(experiment_name)
except:
    EXPERIMENT_ID = mlflow.set_experiment(experiment_name).experiment_id

ROOT_DIR = Path("/dbfs/FileStore/providence-legacy-recovery/backblaze-gru")

# COMMAND ----------

metadata_to_log = defaultdict(dict)
metadata_to_log["general"]["dtype"] = general_config["dtype"]

set_torch_default_dtypes(metadata_to_log["general"]["dtype"])

# COMMAND ----------

DS_SEED = _SWEEPS_DEFAULT_DATA_RANDOM_SEED
metadata_to_log["data"]["seed"] = DS_SEED
metadata_to_log["data"]["global_seed_at_init"] = torch.initial_seed()
metadata_to_log["data"]["use_test_set"] = True
metadata_to_log["data"]["split_percentage"] = 0.8

train_ds, val_ds, test_ds = BackblazeDatasets(
    quarter=BackblazeQuarter._2019_Q4,
    include_validation=metadata_to_log["data"]["use_test_set"],
    split_percentage=metadata_to_log["data"]["split_percentage"],
    data_root="/dbfs/FileStore/datasets/providence",
    random_seed=DS_SEED,
)
metadata_to_log["data"]["name"] = "Backblaze"


# COMMAND ----------
def dataset_fiddling(num_workers: int = 1):
    """Scratching out how to do a Stratified KFold... It's a lot... And it means another nested loop..."""
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import SubsetRandomSampler

    labels = []
    for _, df in train_ds.data + val_ds.data + test_ds.data:
        is_eventful = df[train_ds.event_indicator_column].sum()
        labels.append(is_eventful)

    print(f"{labels = }")
    n_devices = len(train_ds) + len(val_ds) + len(test_ds)
    print(f"{len(labels) == n_devices = }")
    # I already shuffled / split things. I don't want numpy to double dip and ruin things
    strat = StratifiedKFold(n_splits=4)
    indices = list(range(n_devices))
    for i_split, (train_index, test_index) in enumerate(strat.split(indices, labels)):
        print(
            f"""{i_split = }
            {test_index = }"""
        )

        train_subsampler = SubsetRandomSampler(train_index)
        validation_subsampler = SubsetRandomSampler(test_index)
        full_ds = train_ds + val_ds + test_ds
        train_dl = ProvidenceDataLoader(full_ds, pin_memory=True, num_workers=num_workers, sampler=train_subsampler)
        validation_dl = ProvidenceDataLoader(
            full_ds,
            pin_memory=True,
            num_workers=num_workers,
            sampler=validation_subsampler,
        )
        dls = DataLoaders(train_dl, validation_dl, test=validation_dl)


# COMMAND ----------
dls = CustomProvidenceDataloaders(
    train_ds,
    val_ds,
    batch_size=general_config["batch_size"],
    pin_memory=True,
    num_workers=1,
)._replace(test=ProvidenceDataLoader(test_ds, batch_size=1))


# COMMAND ----------
def try_log_artifacts(output_dir: Path):
    try:
        mlflow.log_artifacts(str(output_dir))
    except:
        print("MLFlow sucks at its own stuff and is inconsistent in documentation")


# COMMAND ----------
def write_callback_default_config(d: defaultdict, num_epochs: int = 500) -> None:
    callback_config = d["callbacks"]
    callback_config["epoch_logger_epochs"] = 10
    callback_config["visualizer"] = IntervalMetricsVisualizer.__name__
    callback_config["visualization_freq"] = min(200, num_epochs // 5)
    callback_config["model_output_freq"] = min(100, num_epochs // 2)
    callback_config["n_kept_checkpoints"] = 5
    callback_config["early_stopping_patience"] = 20
    callback_config["ebrake_epoch"] = 20
    callback_config["ebrake_requisite_loss"] = 4.0
    callback_config["tracking_metric"] = "val_loss"


def optimizer_default_config() -> dict:
    # Why isn't this passed in? These are driven heuristically (from earlier analysis).
    optim_config = dict()
    optim_config["schedule_min"] = 1e-5
    optim_config["schedule_T_mult"] = 2
    return optim_config


# COMMAND ----------
def do_mlflow_run(
    model_type: Type[ProvidenceModule],
    dls: DataLoaders,
    experiment_id: str,
    random_seed: int,
    optim_config: dict,
    instantiation_params: dict,
):
    with mlflow.start_run(experiment_id=experiment_id):
        set_seed(random_seed)
        model = model_type(**instantiation_params, device=use_gpu_if_available())
        optimizer = OptimizerWrapper(
            optim_config["type"](model.parameters(), lr=optim_config["learning_rate"]),
            batch_size=optim_config["batch_size"],
            num_epochs=optim_config["num_epochs"],
        )
        run_config = defaultdict(dict)
        run_config["model"].update(instantiation_params)  # copy key-values
        run_config["model"]["type"] = model_type.__name__
        run_config["model"]["seed"] = random_seed
        write_callback_default_config(run_config)
        run_config["optimizer"].update(optim_config)

        callback_config = run_config["callbacks"]
        output_dir: Path = ROOT_DIR / now_dt_string()
        logger = configure_logger_in_dir(output_dir)
        cbs = [
            LearningRateScheduler(
                optimizer.opt,
                T_mult=optim_config["schedule_T_mult"],
                eta_min=optim_config["schedule_min"],
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
            EarlyStopping(
                patience=callback_config["early_stopping_patience"],
                track=callback_config["tracking_metric"],
            ),
            # our weibull loss takes a bit longer to bit below 1.0, but still begets strong results
            EmergencyBrake(
                callback_config["ebrake_epoch"],
                callback_config["ebrake_requisite_loss"],
            ),
            LearningCurveTracker(1000, output_dir=(output_dir / "learning_curve")),
        ]
        trainer = Trainer(mlflow_epoch)
        # run_config to something we can log to mlflow
        merged_config = dict(**run_config, **metadata_to_log)
        mlflow.log_params(nest_keys(merged_config))
        losses = trainer.callback_training(model, optimizer, dls, cbs)
        logger.info(f"{trainer.terminated_early=} {trainer.training_termination_reason=}")
        model_metrics = GeneralMetrics(model, dls.test_ds, losses)
        mlflow.log_metrics(model_metrics.iloc[0].to_dict())
        try_log_artifacts(output_dir)
    return model, model_metrics


# COMMAND ----------
def hybrid_parameter_sweep(
    model_type: Type[ProvidenceModule],
    hyperparameters: Dict[str, HyperparameterList],
    dls: DataLoaders,
    *,
    model_defaults: Dict[str, Hyperparameter],
    optimizer_default: Dict[str, Hyperparameter],
    choice_metrics=("NaN_count", "RMSE", "SMAPE"),
    experiment_id=EXPERIMENT_ID,
    random_seed: int,
) -> Tuple[Dict[str, Hyperparameter], Dict[str, Hyperparameter]]:
    """Grid search over the model hyperparemeters, which is a product space over the 'network' key, in the supplied ``hyperparameters``.
    Iterate over the hyperparameters of the optimizer, linear in the sum length of the value lists under the dict keys
    - 'optimizer', for the optimizer of the above.

    Notes:
    Configurations are called 'default', when they're more like the bases upon which changes are iterated against.
    - If that's what you think of when you see 'default', then remove this comment.
    """
    optimizer_default_parameters = {**optimizer_default, **optimizer_default_config()}
    sweeper = HyperparameterSweeper(**hyperparameters["network"])

    sortable_models = []
    for model_params in sweeper.poll_sweeps():
        instantiation_parameters = {**model_defaults, **model_params}
        print(f"After update {instantiation_parameters = }")
        model, model_metrics = do_mlflow_run(
            model_type,
            dls,
            experiment_id,
            random_seed,
            optimizer_default_parameters,
            instantiation_parameters,
        )
        model_metrics = model_metrics.iloc[0].to_dict()
        model_metrics["model"] = model
        model_metrics["params"] = model_params
        sortable_models.append(model_metrics)

    best_models_and_metrics = multilevel_sort(sortable_models, keys=choice_metrics)
    best_model_with_metrics = best_models_and_metrics[0]
    best_model_hparams = best_model_with_metrics["params"]

    # NOTE: what follows is the aforementioned linear search over the optimizer parameters.
    best_optimizer_hparams = {}
    invariant_network_parameters = {**model_defaults, **best_model_hparams}
    print("Best parameters for this model: {}".format(invariant_network_parameters))

    for hparam, param_values in hyperparameters["optimizer"].items():
        print(f"Checking OPTIMIZER hyperparameter {hparam} over range of values {param_values}")
        sortable_models = []
        optimizer_parameters_base = {**optimizer_default, **optimizer_default_config()}
        for hparam_instance in param_values:
            optimizer_parameters = {
                **optimizer_parameters_base,
                **best_optimizer_hparams,
                hparam: hparam_instance,
            }
            print(f"{hparam_instance = }")
            print(f"After update {optimizer_parameters = }")
            model, model_metrics = do_mlflow_run(
                model_type,
                dls,
                experiment_id,
                random_seed,
                optimizer_parameters,
                invariant_network_parameters,
            )
            model_metrics = model_metrics.iloc[0].to_dict()
            model_metrics["model"] = model
            model_metrics["param"] = hparam_instance
            sortable_models.append(model_metrics)

        best_models_and_metrics = multilevel_sort(sortable_models, keys=choice_metrics)
        best_model_with_metrics = best_models_and_metrics[0]
        best_optimizer_hparams[hparam] = best_model_with_metrics["param"]

    return best_model_hparams, best_optimizer_hparams


# COMMAND ----------
def sweep_linearly_in_parameters(
    model_type: Type[ProvidenceModule],
    hyperparameters: Dict[str, HyperparameterList],
    dls: DataLoaders,
    *,
    model_defaults: Dict[str, Hyperparameter],
    optimizer_default: Dict[str, Hyperparameter],
    choice_metrics=("NaN_count", "RMSE", "SMAPE"),
    experiment_id=EXPERIMENT_ID,
    random_seed: int,
) -> Tuple[Dict[str, Hyperparameter], Dict[str, Hyperparameter]]:
    """
    Iterate over the hyperparameters of the network and optimizer, linear in the sum length of the value lists under the dict keys
    - 'network', for the neural network architecture
    - 'optimizer', for the optimizer of the above.

    Notes:
    Configurations are called 'default', when they're more like the bases upon which changes are iterated against.
    - If that's what you think of when you see 'default', then remove this comment.
    """
    best_model_hparams = {}
    optimizer_default_parameters = {**optimizer_default, **optimizer_default_config()}
    for hparam, param_values in hyperparameters["network"].items():
        print(f"Checking NETWORK hyperparameter {hparam} over range of values {param_values}")
        print(f"{hparam = }")
        sortable_models = []
        for hparam_instance in param_values:
            instantiation_parameters = {
                **model_defaults,
                **best_model_hparams,
                hparam: hparam_instance,
            }
            print(f"{hparam_instance = }")
            print(f"After update {instantiation_parameters = }")
            model, model_metrics = do_mlflow_run(
                model_type,
                dls,
                experiment_id,
                random_seed,
                optimizer_default_parameters,
                instantiation_parameters,
            )
            model_metrics = model_metrics.iloc[0].to_dict()
            model_metrics["model"] = model
            model_metrics["param"] = hparam_instance
            sortable_models.append(model_metrics)

        best_models_and_metrics = multilevel_sort(sortable_models, keys=choice_metrics)
        best_model_with_metrics = best_models_and_metrics[0]
        best_model_hparams[hparam] = best_model_with_metrics["param"]

    best_optimizer_hparams = {}
    invariant_network_parameters = {**model_defaults, **best_model_hparams}
    print("Best parameters for this model: {}".format(invariant_network_parameters))

    for hparam, param_values in hyperparameters["optimizer"].items():
        print(f"Checking OPTIMIZER hyperparameter {hparam} over range of values {param_values}")
        sortable_models = []
        optimizer_parameters_base = {**optimizer_default, **optimizer_default_config()}
        for hparam_instance in param_values:
            optimizer_parameters = {
                **optimizer_parameters_base,
                **best_optimizer_hparams,
                hparam: hparam_instance,
            }
            print(f"{hparam_instance = }")
            print(f"After update {optimizer_parameters = }")
            model, model_metrics = do_mlflow_run(
                model_type,
                dls,
                experiment_id,
                random_seed,
                optimizer_parameters,
                invariant_network_parameters,
            )
            model_metrics = model_metrics.iloc[0].to_dict()
            model_metrics["model"] = model
            model_metrics["param"] = hparam_instance
            sortable_models.append(model_metrics)

        best_models_and_metrics = multilevel_sort(sortable_models, keys=choice_metrics)
        best_model_with_metrics = best_models_and_metrics[0]
        best_optimizer_hparams[hparam] = best_model_with_metrics["param"]

    return best_model_hparams, best_optimizer_hparams


# COMMAND ----------

# !nvidia-smi

# COMMAND ----------
from providence.nn.rnn import ProvidenceGRU

model_search_space = {
    "type": ProvidenceGRU,
    "hyperparameters": {
        "hidden_size": [128, 200, 256, 512, 1024],
        "num_layers": [1, 2, 3, 4],
        "dropout": [0.0, 0.1, 0.3, 0.4, 0.6, 0.9],
    },
}

# COMMAND ----------
hyper_parameter_search_space = {
    "network": model_search_space["hyperparameters"],
    "optimizer": {
        "learning_rate": [3e-2, 1e-3, 3e-3, 1e-4],
        "batch_size": [32, 64, 128, 256],
    },
}

# COMMAND ----------
best_model_params, best_optim_params = sweep_linearly_in_parameters(
    model_search_space["type"],
    hyper_parameter_search_space,
    dls,
    model_defaults={
        "input_size": DS_NAME_TO_FEATURE_COUNT["backblaze"],
        "hidden_size": _SWEEPS_DEFAULT_HIDDEN_DIM,
        "num_layers": _SWEEPS_DEFAULT_NUM_LAYERS,
        "dropout": _SWEEPS_DEFAULT_DROPOUT,
    },
    optimizer_default={
        "type": torch.optim.Adam,
        "learning_rate": 3e-3,
        "batch_size": 64,
        "num_epochs": 700,
    },
    # use this seed for each run, reset anew at the top of each model configuration construction.
    random_seed=11068621650300516211,
)

# COMMAND ----------

best_model_params
