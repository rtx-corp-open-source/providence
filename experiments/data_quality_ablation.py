"""
The data-quality requirements ablation study experiment

Purpose: This document is for the outline and execution of the data-quality requirements study on Providence-trained models
We know that the conjunction of architecture and framework can train on fairly small data (e.g. NASA turbofan) and fairly large
data (e.g. four quarters of Backblaze, downsampled to only include only a 1-1 censored-uncensored dataset), with less than
1-to-1, censored-to-uncensored data. ('Less than 1-to-1' in this context means there were more uncensored than censored sequences).
As shown in `censored_subset()`, the censoring strategy has an upsampling perspective.

Yet, we don't know to what extent the loss function, our models, or the framework as a whole depends on these.
The following experiment is designed to test the censoring dimension, with eventualities including:
- architectures
- optimizers

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import json
from enum import Enum
from logging import getLogger
from time import perf_counter
from typing import Union

import torch
import typer
from pandas import concat as concat_dataframes
from pandas import DataFrame

from providence import paper_reproductions
from providence.dataloaders import BasicDataloaders
from providence.dataloaders import CustomProvidenceDataloaders
from providence.datasets import BackblazeDataset
from providence.datasets import BackblazeDatasets_from_split
from providence.datasets import BackblazePreprocessing
from providence.datasets import censor_backblaze_splits
from providence.datasets import DataSubsetId
from providence.datasets import df_train_test_split
from providence.datasets import downsample_to_event_portion
from providence.datasets import NasaDataset
from providence.datasets.adapters import BackblazeQuarter
from providence.datasets.adapters import load_backblaze_csv
from providence.training import minimize_torch_runtime_overhead
from providence.training import use_gpu_if_available
from providence.utils import cached_dataframe
from providence.utils import name_and_args
from providence.utils import now_dt_string
from providence.utils import save_df

logger = getLogger(__name__)


def censoring_experiment_round(
    df_all: DataFrame,
    censoring_proportion: Union[float, int],
    *,
    seed: int = 1234,
    n_laps: int = 3,
) -> DataFrame:
    """
    Run something between a mid-level and low-level experiment, chiefly from all the interaction with
    the Dataset (which is the point of this experiment).
    """
    df_split = df_train_test_split(df_all, "serial_number", split_percentage=0.8, seed=seed)

    # 2. censoring proportion calculation, after this point, the test is device-fixed
    # NOTE: upsampling the censored devices across both samples maintains the proportionality with all seeds under test
    # NOTE: as a result, the test set will grow over the course of training.
    df_split = censor_backblaze_splits(df_split, censoring_proportion)

    # 3. selection of actual training set
    df_split = BackblazePreprocessing(df_split)

    # 6. convert to Providence Dataset
    # TODO: rename to _from_splits
    train_ds, test_ds = BackblazeDatasets_from_split(df_split)

    model = paper_reproductions.BackblazeTransformer()
    model.device = use_gpu_if_available()

    # find best of three
    training_run_metrics = []
    for training_lap in range(n_laps):
        logger.info(f"{training_lap = }")
        model.reset_parameters()
        model_optim = paper_reproductions.BackblazeTransformerOptimizer(model)
        dataloaders = BasicDataloaders(train_ds, test_ds, batch_size=model_optim.batch_size)

        start_time = perf_counter()
        losses = paper_reproductions.BackblazeTraining(model, model_optim, dataloaders)
        training_time = perf_counter() - start_time
        completion_time = now_dt_string()

        metrics = paper_reproductions.GeneralMetrics(model, test_ds, losses)
        metrics["completion_time"] = completion_time
        metrics["training_seconds"] = training_time
        metrics["censoring_proportion"] = str(censoring_proportion)
        metrics["losses_json"] = json.dumps(losses._asdict())
        training_run_metrics.append(metrics)

    results_df = concat_dataframes(training_run_metrics, ignore_index=True)

    return results_df


def run_data_censoring_experiment(random_seed: int = 1234, n_laps: int = 3):
    """As previously expressed, the following runs the censoring control experiment"""
    logger.info("Experiment start")
    logger.info("Filtering out nans...")
    experiment_start_timestamp = now_dt_string()
    start_time = perf_counter()
    # caching this because I'm tired of waiting 110 seconds for filtering, and 45 seconds for file load
    # That's knocking on 3 minutes just to do the thing that I'm actually interested in
    df_all = cached_dataframe(
        lambda: (
            load_backblaze_csv(BackblazeQuarter._2019_Q4, data_root="./.data")
            .groupby("serial_number")
            .filter(lambda df: df.isna().sum().sum() == 0)
        ),
        f"./.data/backblaze-download/{BackblazeQuarter._2019_Q4.value}/filtered.csv",
    )
    duration = perf_counter() - start_time
    logger.info(f"Filtering out nans took {duration} seconds")

    experiment_metrics = []
    # - DOE(censoring_proportion): the main knob that we're going to turn
    for censoring_proportion in [0, 0.5, 1, 2, 3]:
        logger.info(f"censoring proportion (censored:uncensored) = {censoring_proportion}:1")
        metrics = censoring_experiment_round(df_all, censoring_proportion, seed=random_seed, n_laps=n_laps)
        metrics["censoring_proportion"] = censoring_proportion
        experiment_metrics.append(metrics)

    metrics_all = concat_dataframes(experiment_metrics)
    save_df(
        metrics_all,
        f"censoring_experiment-{experiment_start_timestamp}.csv",
        root="./outputs",
    )


def get_model_and_optimizer_initializer(dataset_name: str):
    if dataset_name == "backblaze":
        return (
            paper_reproductions.BackblazeTransformer,
            paper_reproductions.BackblazeTransformerOptimizer,
        )
    elif dataset_name == "nasa" or dataset_name == "nasa-old":
        return (
            paper_reproductions.NasaTransformer,
            paper_reproductions.NasaTransformerOptimizer,
        )
    else:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'")


def get_dataset_initializers(dataset_name: str, *, random_seed: int = 1234):
    if dataset_name == "backblaze":

        def train_init():
            return BackblazeDataset(DataSubsetId.Train, random_seed=random_seed, normalization_by="device")

        def test_init():
            return BackblazeDataset(DataSubsetId.Test, normalization_by="fleet", random_seed=random_seed)

    elif dataset_name == "nasa":

        def train_init():
            return NasaDataset(DataSubsetId.Train)

        def test_init():
            return NasaDataset(DataSubsetId.Test)

    else:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'")

    return train_init, test_init


def run_data_failure_portion_experiment(random_seed: int = 1234, n_laps: int = 3, dataset_name: str = "backblaze"):
    """Run an experiment that reduces the number of failure events available in the training set and reports the impact on performance"""
    assert dataset_name in {"backblaze", "nasa"}, "Should use a vaild dataset name"
    logger.info("Starting failure portioning experiment")

    experiment_start_timestamp = now_dt_string()

    logger.info("Generating reference test set")
    init_train_ds, init_test_ds = get_dataset_initializers(dataset_name, random_seed=random_seed)
    test_ds = init_test_ds()
    logger.info(f"{len(test_ds) = }")

    model_init, model_optim_init = get_model_and_optimizer_initializer(dataset_name)

    experiment_cohort_metrics = []
    for failure_portion in [0.25, 0.5, 0.75, 1]:
        logger.info(f"Failure portion (failures:non-failures) = {failure_portion}:1")
        # Get the dataset from the paper
        logger.info("Loading training set")
        train_ds = init_train_ds()
        logger.info(f"Before downsampling of event portion: {len(train_ds) = }")
        # Take out the failures using the function you wrote the test for (test_portioning.py)
        downsample_to_event_portion(train_ds, failure_portion)  # done in-place

        logger.info(f"After downsampling of event portion: {len(train_ds) = }")

        # run training.
        model = model_init()
        model.device = use_gpu_if_available()
        training_run_metrics = []
        for training_lap in range(n_laps):
            logger.info(f"{training_lap = }")
            model.reset_parameters()
            model_optim = model_optim_init(model)
            dataloaders = CustomProvidenceDataloaders(
                train_ds, test_ds, batch_size=model_optim.batch_size, num_workers=0
            )

            start_time = perf_counter()
            losses = paper_reproductions.BackblazeTraining(model, model_optim, dataloaders)
            training_time = perf_counter() - start_time
            completion_time = now_dt_string()

            metrics = paper_reproductions.GeneralMetrics(model, test_ds, losses)
            metrics["completion_time"] = completion_time
            metrics["training_seconds"] = training_time
            metrics["failure_portion"] = str(failure_portion)
            metrics["losses_json"] = json.dumps(losses._asdict())
            training_run_metrics.append(metrics)

        run_metrics_df = concat_dataframes(training_run_metrics)
        experiment_cohort_metrics.append(run_metrics_df)

    experiment_metrics = concat_dataframes(experiment_cohort_metrics)
    # log general metrics
    save_df(
        experiment_metrics,
        f"failure_portioning_experiment-{experiment_start_timestamp}-{dataset_name}.csv",
        root="./outputs",
    )


class ExperimentToRun(Enum):
    censoring = "censoring"
    failure_volume = "failure"


def main(
    minimize_overhead: bool = typer.Option(
        True,
        help="Disables all extraneous checks that PyTorch enables to ease debugging. 🔎",
    ),
    laps: int = typer.Option(
        3,
        min=1,
        max=10,
        help="Number of times to re-run a model through epochs. For collecting statistics",
    ),
    random_seed: int = typer.Option(1234, min=0, help="Integral-valued random seed for the experiment run"),
    experiment_prefix: ExperimentToRun = typer.Option(
        ExperimentToRun.censoring.value,
        help="The experiment that you would like to run",
    ),
    dataset_name: str = typer.Option(None, help="Only used for failure-experiment"),
):
    logger.info(f"Training args: {name_and_args()}")  # reproducibility
    logger.info(f"Init seed: {torch.initial_seed()}")
    torch.manual_seed(random_seed)
    logger.info(f"Initial seed after torch.manual_seed(): {torch.initial_seed()}")

    if minimize_overhead:
        minimize_torch_runtime_overhead()

    if experiment_prefix == ExperimentToRun.censoring:
        run_data_censoring_experiment(random_seed, n_laps=laps)
    elif experiment_prefix == ExperimentToRun.failure_volume:
        run_data_failure_portion_experiment(random_seed, n_laps=laps, dataset_name=dataset_name)
    else:
        raise ValueError("Got past Typer's type checking")


if __name__ == "__main__":
    typer.run(main)
