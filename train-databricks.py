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

# COMMAND ----------

# MAGIC %pip install --force /dbfs/FileStore/binaries/providence-1.0.0rc7-py3-none-any.whl

# COMMAND ----------

# export http_proxy=http://devproxy.utc.com:80

import os

os.environ["http_proxy"] = 'http://devproxy.utc.com:80'
os.environ["https_proxy"] = 'http://devproxy.utc.com:80'

# COMMAND ----------

# import the universe
import enum
import logging
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import typer
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import ConcatDataset, Dataset

from providence.datasets import (
    BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER, BackblazeDatasets, BackblazeExtendedDatasets, NasaDatasets,
    NasaFD00XDatasets, NasaTurbofanTest, ProvidenceDataset
)
from providence.datasets.adapters import BackblazeQuarter
from providence.dataloaders import ProvidenceDataLoader
from providence.nn import ProvidenceGRU, ProvidenceLSTM, ProvidenceRNN, ProvidenceVanillaRNN
from providence.nn.transformer import ProvidenceTransformer
from providence.training import (
    minimize_torch_runtime_overhead,
    set_torch_default_dtypes, training_pass, use_gpu_if_available, validation_pass
)
from providence.utils import (configure_logger_in_dir, name_and_args, now_dt_string, remove_keys_from_dict)
from providence_utils.callbacks import (
    CachedIntervalMetricsVisualizer, Callback, EarlyStopping, EmergencyBrake, EpochLoggerCallback,
    IntervalMetricsVisualizer, LearningCurveTracker, ModelCheckpointer, NthEpochLoggerCallback, WriteModelOutputs,
    check_before_epoch
)
from providence_utils.hyperparameter_sweeper import (HyperparameterSweeper, Metrics)
from providence.types import DataLoaders
from providence.training import EpochLosses as Losses

# COMMAND ----------

class TrainingDataset(str, enum.Enum):
    nasa = "NASA"
    nasa_sub = "NASA-Seg"
    backblaze = "Backblaze"
    bbe = "BackblazeExtended"


def new_train(
    outdir: Path,
    train_data: Dataset,
    test_data: Dataset,
    model: nn.Module,
    optimizer: Optimizer,
    num_epochs: int = 200,
    callbacks: List[Callback] = [],
    run_suffix: str = None,
    *,
    train_batch_size: int = 128,
    val_batch_size: int = 512,
    save_full_model: bool = False,
    multiple_passes: int = 1,
    num_workers: int = 1,
    device: torch.device
) -> Dict[str, Any]:
    logger.info(model)
    model_name = model._get_name()

    model = model.to(device, dtype=torch.get_default_dtype())
    model.device = device

    if multiple_passes > 1:
        train_data = ConcatDataset([train_data] * int(multiple_passes))
    else:
        train_data = train_data

    
    train_dataloader, validation_dataloader = (
        ProvidenceDataLoader(train_data, batch_size=train_batch_size, num_workers=0),
        ProvidenceDataLoader(test_data, batch_size=val_batch_size, num_workers=0)
    )
    dls = DataLoaders(train_dataloader, validation_dataloader)
    dls.to_device(model.device)
    logger.info("Starting training loop")

    torch.cuda.empty_cache()
    losses = Losses(0, 0)

    for epoch in range(1, num_epochs + 1):

        terminate_training, termination_message = check_before_epoch(callbacks)
        if terminate_training:
            print(termination_message)
            break  # exiting training early

        losses = Losses(
            train=training_pass(train_dataloader, model, optimizer),
            validation=validation_pass(validation_dataloader, model),
        )

        for cb in callbacks:
            cb.after_epoch(epoch, model, optimizer, losses, dls)

    for cb in callbacks:
        cb.after_training(epoch, model, optimizer, losses, dls)

    run_name = model_name + (f"-{run_suffix}" if run_suffix else "")
    final_output_path = str(outdir / f"{run_name}-{'full' if save_full_model else 'weights'}.pt")
    torch.save(model if save_full_model else model.state_dict(), final_output_path)
    return {"final_epoch_losses": losses, "final_output_path": final_output_path}


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


def initialize_simple(model_ctor, optim_ctor, num_features, num_layers, hidden_size, dropout, learning_rate):
    "Initialize a model from its model and optimizer constructor, heavily leveraging named parameters"
    model = model_ctor(num_features, n_layers=num_layers, hidden_size=hidden_size, dropout=dropout)
    optimizer = optim_ctor(model.parameters(), lr=learning_rate)
    return model, optimizer


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

def main(
    network_type = typer.Option(
        'GRU', case_sensitive=False, help="Type of neural network to run 💻"
    ),
    optimizer_type = typer.Option(
        'Adam', case_sensitive=False, help="Type of optimizer to use. 👩‍🏫"
    ),
    anomaly_detection: bool = typer.Option(
        False, help="Enable anomaly detection for debugging. Serious performance hit if enabled. 🔎"
    ),
    num_epochs: int = typer.Option(100, min=1, help="Number of training epochs. ⌛"),
    num_layers: int = typer.Option(4, min=1, help="Number of recurrent layers. 🔢"),
    layer_width: int = typer.Option(32, min=1, help="Number of neurons in a given internal layer. 🔢"),
    learning_rate: float = typer.Option(1e-4, max=1e-1, help="Optimizer learning rate 👩‍🎓"),
    dropout: float = typer.Option(0.2, min=0.0, max=1.0, help="Probability of dropping outputs between layers 🍺"),
    dataset_choice: TrainingDataset = typer.Option(
        TrainingDataset.nasa, case_sensitive=False, help="Set this to use one of the provided datasets"
    ),
):
    outdir = Path(
        "outputs",
        dataset_choice.value,
        parameterized_run_name(network_type, optimizer_type, num_layers, layer_width, learning_rate, dropout),
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # configure global logger(?)
    global logger
    logger = configure_logger_in_dir(outdir, logger_name=__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")
    logger.info(f"Training args: {name_and_args()}")  # reproducibility
    logger.info(f"Init seed: {torch.initial_seed()}")

    if anomaly_detection:
        torch.autograd.set_detect_anomaly(True)

    train_ds, test_ds = construct_datasets(dataset_choice)

    # variable shadowing... ewwwww
    model, optimizer = nn_factory(network_type, optimizer_type)
    num_features = train_ds[0][0].size(-1)  # ds.features.first.num_columns

    model, optimizer = initialize_simple(
        model, optimizer, num_features, num_layers, layer_width, dropout, learning_rate
    )

    new_train(
        outdir=outdir,
        train_data=train_ds,
        test_data=test_ds,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        callbacks=[
            EpochLoggerCallback(logger),
            IntervalMetricsVisualizer(every=ceil(num_epochs // 5), output_dir=outdir / "metrics_plots", logger=logger),
            WriteModelOutputs(last_epoch_number=num_epochs, output_dir=outdir / "outputs", logger=logger),
        ],
        device=device
    )


# COMMAND ----------

def run_suffix_from_training_config(cfg: Dict[str, float]) -> str:
    return "-".join([f"{key}{value}" for key, value in cfg.items()])


# COMMAND ----------

def search_hyperparameter_space(
    sweep_conf_json: dict,
    output_root: str = "outputs",# help="This is the root directory, under which all outputs will be written."
    dataset_choice: TrainingDataset = TrainingDataset.nasa, #case_sensitive=False, help="Set this to use one of the provided datasets"
    model_mode: str = "transformer",# case_sensitive=False, help="Either 'transformer' or 'rnn', to hasten experimentation"
    save_full_model: bool = True,
    easy_mode: bool = False,
    early_stopping: int = -1, #help="Set a non-negative integer if you desire to use this callback, or -1 if you want to abstain (Defaut).\n"
    early_stopping_metric: str = "train_loss",# help="Either 'train_loss' or 'val_loss'. Only enforced if --early-stopping is set"
    n_passes: int = 1,# min=1, help="To use multi-pass training per epoch, set this to a quantity greater than 1"
    nasa_test_num: int = 1, #max=4,  help="FD00X where X is the supplied argument"),
    visualization_frequency: int = 50, #help="Frequency with which to draw metrics visualizations. Only takes effect is num_epochs >= 100. Regardless, visualizations are done after the last epoch"
    ebrake_epoch: int = -1, # min=-1, help="Assign to a positive number if you want to terminate training early if losses aren't looking good"
    ebrake_requisite_loss: float = 0, #min=0, help="Only takes effect is ebrake-epoch > 0. If neither train- nor validation-loss descends below this level, training is terminated early"
):
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
            torch.manual_seed(seed)

        # overwrite the config with the actual, so seed is visible in the directory name when
        # seed is initially "random"
        training_config["general"]["seed"] = seed

        outdir = Path(
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
        outdir.mkdir(parents=True, exist_ok=True)
        logger = configure_logger_in_dir(outdir, logger_name=__name__)
        logger.info(f"program args: {program_args}")
        logger.info(f"run config: {training_config}")
        logger.info(f"Run {seed = }")

        num_epochs = training_config["general"]["num_epochs"]

        model = model_init(num_features, **training_config["model"])
        optimizer = optim.SGD(model.parameters(), lr=training_config["optimizer"]["lr"], nesterov=True, momentum=0.9)

        # we don't want to visualize too often, because it's an expensive operation.
        _interval_viz_fraction = 3
        metrics_viz_frequency = num_epochs // _interval_viz_fraction if num_epochs < 100 else visualization_frequency
        callbacks = [
            LearningCurveTracker(every=2,
                                 output_dir=outdir),  # NOTE: inner call to plt.close('all') saves us in the long haul.
            EpochLoggerCallback(logger) if num_epochs < 100 else NthEpochLoggerCallback(metrics_viz_frequency, logger),  # TODO(speed-up): log less frequently.
            ModelCheckpointer(outdir, track=early_stopping_metric, logger=logger),
            CachedIntervalMetricsVisualizer(
                every=metrics_viz_frequency, output_dir=outdir / "metrics_plots", logger=logger
            ),
            WriteModelOutputs(last_epoch_number=num_epochs, output_dir=outdir / "outputs", logger=logger),
        ]

        if early_stopping > -1:
            callbacks.append(EarlyStopping(early_stopping, track=early_stopping_metric))

        if ebrake_epoch > 0 and ebrake_requisite_loss > 0:
            callbacks.append(EmergencyBrake(ebrake_epoch, ebrake_requisite_loss))

        train_outputs = new_train(
            outdir=outdir,
            train_data=train_ds,
            test_data=test_ds,
            model=model,
            optimizer=optimizer,
            num_epochs=num_epochs,
            callbacks=callbacks,
            save_full_model=save_full_model,
            train_batch_size=training_config["general"]["bs"],
            val_batch_size=training_config["general"]["bs"],
            multiple_passes=n_passes,
            device=device,
            num_workers=0
        )
        logger.info(f"{train_outputs =}")
        training_config["run_name_nested"] = run_suffix_from_training_config(training_config)

        return train_outputs

    # end train_model()

    from functools import cmp_to_key

    def metric_to_val(m1: Dict[str, Any], m2: Dict[str, Any]) -> float:
        """Metric key to those with the lowest lost, then the least difference with validation loss, first"""

        if early_stopping_metric.startswith("train"):
            m1_loss = m1["final_epoch_losses"].train
            m2_loss = m2["final_epoch_losses"].train
        else:
            m1_loss = m1["final_epoch_losses"].validation
            m2_loss = m2["final_epoch_losses"].validation

        return m1_loss - m2_loss

    metrics = sweeper.sweep(train_model)
    logger = configure_logger_in_dir("outputs", __name__)

    top_5_best = sorted(metrics, key=cmp_to_key(metric_to_val))[:5]
    logger.info(f"Best metrics: {top_5_best}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Sweep definitions

# COMMAND ----------

sweeps_json = {
    "model": {
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
