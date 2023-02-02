# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Train.py on Databricks
# MAGIC Purpose is to reproduce the research with the exact training setup as it was when we were doing the paper.
# MAGIC - As the dataset code is initialized before the random seed is set, we've transplanted that code from the new codebase
# MAGIC   - Also, that code was unit tests, side-by-side, and parity was found. So we have confidence that our Datasets are intact
# MAGIC - The model code wasn't really touched between the old and new codebase. It was the fixing around them that changed.
# MAGIC - Really it's just the machinery around the instantiations that changed. As this is the old machinery, we can trust it somewhat
# MAGIC
# MAGIC If this doesn't reproduce what we expect, there are two avenues
# MAGIC 1. We confess that our old work wasn't highly reproducible, and we work to re-achieve the same results with the new code.
# MAGIC 2. We scrap the project and scramble to make something else happen.
# MAGIC
# MAGIC Alternatives would likely be derivatives of these
# MAGIC 
# MAGIC
# MAGIC **Raytheon Technologies proprietary**
# MAGIC Export controlled - see license file

# COMMAND ----------

# MAGIC %pip install --force /dbfs/FileStore/binaries/providence-1.0.0rc7-py3-none-any.whl

# COMMAND ----------

# export http_proxy=http://devproxy.utc.com:80

import os

from experiments.databricks.axial_attention import transformer_training_epoch
from providence_utils.trainer import Trainer

os.environ["http_proxy"] = 'http://devproxy.utc.com:80'
os.environ["https_proxy"] = 'http://devproxy.utc.com:80'

# COMMAND ----------

# import the universe
import enum
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn, optim

from providence.datasets import (
    BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER, BackblazeDatasets, BackblazeExtendedDatasets, NasaDatasets,
    NasaFD00XDatasets, NasaTurbofanTest, ProvidenceDataset
)
from providence.datasets.adapters import BackblazeQuarter
from providence.nn import ProvidenceGRU, ProvidenceLSTM, ProvidenceRNN, ProvidenceVanillaRNN
from providence.nn.transformer import ProvidenceTransformer
from providence.training import (
    OptimizerWrapper, minimize_torch_runtime_overhead, set_torch_default_dtypes, training_pass, use_gpu_if_available,
    validation_pass
)
from providence.utils import (configure_logger_in_dir, name_and_args, now_dt_string, remove_keys_from_dict, set_seed)
from providence_utils.callbacks import (
    CachedIntervalMetricsVisualizer, EarlyStopping, EmergencyBrake, LearningCurveTracker, ModelCheckpointer,
    WriteModelOutputs
)
from providence_utils.hyperparameter_sweeper import (HyperparameterSweeper, Metrics)

# COMMAND ----------


class TrainingDataset(str, enum.Enum):
    nasa = "NASA"
    nasa_sub = "NASA-Seg"
    backblaze = "Backblaze"
    bbe = "BackblazeExtended"


def construct_datasets(dataset_name: TrainingDataset,
                       easy_mode: bool = False,
                       **kwargs) -> Tuple[ProvidenceDataset, ProvidenceDataset]:
    def equals_ignorecase(s1: str, s2: str) -> bool:
        return s1.casefold() == s2.casefold()

    if equals_ignorecase(dataset_name, TrainingDataset.backblaze):
        return BackblazeDatasets(quarter=BackblazeQuarter._2019_Q4, include_validation=False)
    elif equals_ignorecase(dataset_name, TrainingDataset.bbe):
        quarters = BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER
        return BackblazeExtendedDatasets(*quarters, include_validation=False, censoring_proportion=int(not easy_mode))
    elif equals_ignorecase(dataset_name, TrainingDataset.nasa_sub):
        test_num = NasaTurbofanTest(int(kwargs.get("nasa_test_num")))
        return NasaFD00XDatasets(test_num)
    else:
        return NasaDatasets()


def parameterized_run_name(network_type, optimizer_type, num_layers, layer_width, learning_rate, dropout):
    "Converts the parameter-value pairs into a clustered key name"
    params = [
        (network_type, ""),
        (optimizer_type, "opt"),
        (num_layers, "layers"),
        (layer_width, "hidden"),
        (learning_rate, "lr"),
        (dropout, "dropout"),
    ]

    return "run-" + "-".join([f"{short_name}{value}" for value, short_name in params]) + f"-{now_dt_string()}"


# COMMAND ----------


def run_suffix_from_training_config(cfg: Dict[str, float]) -> str:
    return "-".join([f"{key}{value}" for key, value in cfg.items()])


# COMMAND ----------


def search_hyperparameter_space(
    sweep_conf_json: dict,
    output_root: str = "outputs",
    dataset_choice: TrainingDataset = TrainingDataset.nasa,
    model_mode: str = "transformer",
    easy_mode: bool = False,
    early_stopping: int = -1,
    early_stopping_metric: str = "train_loss",
    nasa_test_num: int = 1,
    visualization_frequency: int = 50,
    ebrake_epoch: int = -1,
    ebrake_requisite_loss: float = 0,
):
    """
    Arguments:
        - output_root: This is the root directory, under which all outputs will be written
        - dataset_choice: Set this to use one of the provided datasets, identified by enum-shortname
        - model_made: Either 'transformer' or 'rnn'.
        - early_stopping: Set a non-negative integer if you desire to use this callback, or (default) -1 if you want to abstain
        - early_stopping_metric: Either 'train_loss' or 'val_loss'. Only enforced if early stopping is enabled
        - nasa_test_num: FD00X where X is the supplied argument
        - visualization_frequency: Frequency with which to draw metrics visualizations. Only takes effect is num_epochs >= 100.
                                    All visualizations are done after the last epoch
        - ebrake_epoch: Assign to a positive number if you want to terminate training early, programmatically
        - ebrake_requisite_loss: If neither train- nor validation-loss descends below this level, training is terminated early. Only takes effect is ebrake-epoch > 0. 
    """
    logger = logging.getLogger(__name__)
    device = use_gpu_if_available()

    logger.info(f"Using device: {device}")
    program_args = name_and_args()
    logger.info(f"Training args: {program_args}")  # reproducibility
    logger.info(f"Init seed: {torch.initial_seed()}")

    from pprint import pprint

    logger.info("Got JSON for sweep")
    pprint(sweep_conf_json)

    def transformer_init(
        n_features, *, hidden_size: int, n_layers: int, n_attention_heads: int, dropout: float, layer_norm_eps: float,
        **kwargs
    ) -> ProvidenceTransformer:
        return ProvidenceTransformer(
            n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_attention_heads=n_attention_heads,
            dropout=dropout,
            layer_norm_epsilon=layer_norm_eps,
            positional_encoding_dimension=kwargs.get("positional_encoding", 500)
        )

    def rnn_init(n_features, *, model_type: str, hidden_size: int, n_layers: int, dropout: float) -> ProvidenceRNN:
        ctor = {"gru": ProvidenceGRU, "lstm": ProvidenceLSTM, "rnn": ProvidenceVanillaRNN}[model_type]
        return ctor(
            n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        )

    model_init = rnn_init if model_mode == "rnn" else transformer_init
    if model_mode == "rnn":
        model_init = rnn_init
        unused_keys = ["n_attention_heads", "layer_norm_eps"]
    else:
        model_init = transformer_init
        unused_keys = ["model_type"]
    remove_keys_from_dict(sweep_conf_json["model"], unused_keys)

    logger.info("After cleaning up JSON")
    pprint(sweep_conf_json)
    del pprint

    sweeper = HyperparameterSweeper(**sweep_conf_json)
    logger.info(f"Number of configurations: {sweeper._count_configurations}")

    # data stuff that isn't going to change between runs.
    # everything is a double until I say otherwise
    set_torch_default_dtypes(torch.float64)

    minimize_torch_runtime_overhead()

    train_ds, test_ds = construct_datasets(dataset_choice, easy_mode=easy_mode, nasa_test_num=nasa_test_num)

    # variable shadowing... ewwwww
    num_features = train_ds.n_features

    # short utility methods

    def train_model(training_config: Dict) -> Metrics:
        global logger

        if isinstance(training_config["general"]["seed"], int):
            seed = training_config["general"]["seed"]
            torch.manual_seed(seed)
        else:
            # seed randomly
            seed = torch.seed()
            set_seed(seed)

        # overwrite the config with the actual, so seed is visible in the directory name when
        # seed is initially "random"
        training_config["general"]["seed"] = seed

        rundir = Path(
            output_root,
            dataset_choice.value,
            "-".join(
                [
                    model_mode,
                    "-".join(
                        [
                            run_suffix_from_training_config(training_config["model"]),
                            run_suffix_from_training_config(training_config["optimizer"]),
                            run_suffix_from_training_config(training_config["general"]),
                        ]
                    ),
                    now_dt_string(),
                ]
            ),
        )
        rundir.mkdir(parents=True, exist_ok=True)
        logger = configure_logger_in_dir(rundir, logger_name=__name__)
        logger.info(f"program args: {program_args}")
        logger.info(f"run config: {training_config}")
        logger.info(f"Run {seed = }")

        num_epochs = training_config["general"]["num_epochs"]

        model = model_init(num_features, **training_config["model"])
        opt = optim.SGD(model.parameters(), lr=training_config["optimizer"]["lr"], nesterov=True, momentum=0.9)
        optim_wrapper = OptimizerWrapper(
            opt, batch_size=training_config["general"]["bs"], num_epochs=training_config["general"]["num_epochs"]
        )

        # we don't want to visualize too often, because it's an expensive operation.
        _interval_viz_fraction = 4
        metrics_viz_frequency = num_epochs // _interval_viz_fraction if num_epochs < 100 else visualization_frequency
        callbacks = [
            # NOTE: inner call to plt.close('all') saves us in the long haul.
            LearningCurveTracker(every=2, output_dir=rundir),
            ModelCheckpointer(rundir, track=early_stopping_metric, logger=logger),
            CachedIntervalMetricsVisualizer(
                every=metrics_viz_frequency, output_dir=rundir / "metrics_plots", logger=logger
            ),
            WriteModelOutputs(last_epoch_number=num_epochs, output_dir=rundir / "outputs", logger=logger),
        ]

        if early_stopping > -1:
            callbacks.append(EarlyStopping(early_stopping, track=early_stopping_metric))

        if ebrake_epoch > 0 and ebrake_requisite_loss > 0:
            callbacks.append(EmergencyBrake(ebrake_epoch, ebrake_requisite_loss))

        losses = Trainer(transformer_training_epoch).callback_training(model, optim_wrapper, cbs=callbacks)

        torch.save(model, rundir / "model.pt")

        return losses
    # end train_model()
    print()


# COMMAND ----------

# MAGIC %md
# MAGIC # Sweep definitions

# COMMAND ----------

sweeps_json = {
    "model":
        {
            "model_type": ["gru", "lstm"],
            "hidden_size": [32, 64, 128, 256, 512],
            "n_layers": [4],
            "n_attention_heads": [4],
            "dropout": [0.1, 0.9],
            "layer_norm_eps": [1e-5, 1e-4, 3e-4, 1e-3]
        },
    "optimizer": {
        "lr": [3e-2, 1e-2]
    },
    "general": {
        "num_epochs": [100],
        "bs": [64, 128],
        "seed": ["random", 14682599077633313808, 1123705791327780546]
    }
}

# COMMAND ----------

output_path = '/dbfs/Filestore/providence-legacy/'

super_sweeps_result = search_hyperparameter_space(sweeps_json, output_root=output_path)
