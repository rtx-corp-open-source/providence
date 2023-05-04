"""
Experiment to assess which attention axis is most effective on our metrics: temporal or feature-wise attention?
The latter is the norm in the field, but we have seen good results on longer time horizons with temporal - though anecdotally.
These experiments will emit the data sufficient for us to compare and make the call

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import itertools
import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypedDict

import mlflow
import progressbar
import torch
import typer
from pandas import concat
from pandas import Series
from torch.optim import SGD

from providence.dataloaders import CustomProvidenceDataloaders
from providence.datasets import BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER
from providence.datasets import BackblazeDataset
from providence.datasets import BackblazeDatasets
from providence.datasets import BackblazeExtendedDataset
from providence.datasets import BackblazeExtendedDatasets
from providence.datasets import DataSubsetId
from providence.datasets import NasaDataset
from providence.datasets import NasaDatasets
from providence.datasets import NasaFD00XDataset
from providence.datasets import NasaFD00XDatasets
from providence.datasets import ProvidenceDataset
from providence.datasets.adapters import BackblazeQuarter
from providence.datasets.adapters import NasaTurbofanTest
from providence.distributions import Weibull
from providence.loss import discrete_weibull_loss_fn
from providence.metrics import fleet_metrics
from providence.nn import ProvidenceTransformer
from providence.nn.transformer.transformer import set_attention_axis
from providence.paper_reproductions import GeneralMetrics
from providence.paper_reproductions import GranularMetrics
from providence.training import clip_grad_norm_
from providence.training import DataLoader
from providence.training import EpochLosses
from providence.training import LossAggregates
from providence.training import minimize_torch_runtime_overhead
from providence.training import no_grad
from providence.training import Optimizer
from providence.training import OptimizerWrapper
from providence.training import ProvidenceLossInterface
from providence.training import ProvidenceModule
from providence.training import unpack_label_and_censor
from providence.training import use_gpu_if_available
from providence.training import zeros
from providence.types import DataLoaders
from providence.utils import configure_logger_in_dir
from providence.utils import now_dt_string
from providence.utils import set_seed
from providence_utils.callbacks import CachedIntervalMetricsVisualizer
from providence_utils.callbacks import Callback
from providence_utils.callbacks import check_before_epoch
from providence_utils.callbacks import EarlyStopping
from providence_utils.callbacks import ModelCheckpointer
from providence_utils.callbacks import WriteModelOutputs
from providence_utils.hyperparameter_sweeper import Hyperparameter
from providence_utils.mlflow import create_or_set_experiment

# inconventional import (pulling deps through the training module) just to appease Python's weak type annotations


def transformer_training_pass(
    train_dataloader: DataLoader,
    model: ProvidenceModule,
    optimizer: Optimizer,
    *,
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params,
    clip_gradients=True,
):
    train_loss = zeros(1, device=model.device)
    model.train()
    for data in train_dataloader:
        feats, lengths, targets = data
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
    validation_dataloader: DataLoader,
    model: ProvidenceModule,
    *,
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params,
):
    val_loss = zeros(1, device=model.device)
    model.eval()  # Setting model to eval mode stops training behavior such as dropout
    for data in validation_dataloader:
        feats, lengths, targets = data

        outputs = model(feats.to(model.device), lengths, encoder_mask=True)
        distribution_params = model_ouput_type(*outputs)

        y_true, censor_ = unpack_label_and_censor(targets.to(model.device))
        loss = loss_criterion(distribution_params, y_true, censor_, lengths)
        val_loss += loss

    return val_loss.item()


def transformer_training_epoch(
    dls: DataLoaders,
    model: ProvidenceModule,
    optimizer: Optimizer,
    *,
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_ouput_type=Weibull.Params,
) -> EpochLosses:
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


def callback_training(
    model,
    optimizer: OptimizerWrapper,
    dataloaders: DataLoaders,
    cbs: List[Callback] = None,
):
    """Generic training + callbacks"""
    if cbs is None:
        cbs = []

    model.to(model.device)
    # dataloaders.to_device(model.device)

    loss_agg = LossAggregates([], [])
    for current_epoch in range(1, optimizer.num_epochs + 1):
        terminate_training, termination_message = check_before_epoch(callbacks=cbs)
        if terminate_training:
            # print(termination_message)
            break

        losses = transformer_training_epoch(dataloaders, model, optimizer.opt)
        loss_agg.append_losses(losses)

        for cb in cbs:
            cb.after_epoch(current_epoch, model, optimizer.opt, losses, dataloaders)

    for cb in cbs:
        cb.after_training(current_epoch, model, optimizer.opt, losses, dataloaders)

    # dataloaders.to_device('cpu')
    model.to("cpu")
    return loss_agg


def compute_loss_metrics(losses: LossAggregates) -> Series:
    import numpy as np

    return Series(
        {
            "loss_train_total": np.nansum(losses.training_losses),
            "loss_train_max": np.nanmax(losses.training_losses),
            "loss_train_min": np.nanmin(losses.training_losses),
            "loss_train_final": losses.training_losses[-1],
            "loss_val_total": np.nansum(losses.validation_losses),
            "loss_val_max": np.nanmax(losses.validation_losses),
            "loss_val_min": np.nanmin(losses.validation_losses),
            "loss_val_final": losses.validation_losses[-1],
        },
        name="loss_metric",
    )


def datasets_for_experiment_1_config(
    data_config: dict, *, seed: int = 1234, data_root: str = "./.data"
) -> Tuple[ProvidenceDataset, ProvidenceDataset]:
    dataset_name = data_config["name"]
    if equals_ignorecase(dataset_name, "NasaSub"):
        return NasaFD00XDatasets(NasaTurbofanTest(data_config["nasa_test_num"]), data_root=data_root)
    elif equals_ignorecase(dataset_name, "NASA"):
        return NasaDatasets(data_root=data_root)
    elif equals_ignorecase(dataset_name, "Backblaze"):
        return BackblazeDatasets(quarter=BackblazeQuarter._2019_Q4, random_seed=seed, data_root=data_root)
    elif equals_ignorecase(dataset_name, "BackblazeExtended"):
        return BackblazeExtendedDatasets(
            *BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER,
            include_validation=False,
            random_seed=seed,
            data_root=data_root,
        )
    else:
        raise ValueError(f"{dataset_name=} not supported")


def equals_ignorecase(s1: str, s2: str) -> bool:
    return s1.casefold() == s2.casefold()


class TrainingConfig(TypedDict):
    batch_size: int
    num_epochs: int


class ExperimentTaskDefinition(TypedDict):
    """This type encapsulate the sub run for which a plurality of completions constitue a full experiment.
    In other words, an Experiment is composed of many tasks and this outlines one of those task
    """

    name: Optional[str]
    dataset: Dict[str, Hyperparameter]
    model: Dict[str, Hyperparameter]
    optimizer: Dict[str, Hyperparameter]
    training: TrainingConfig


DBFS_CONFIG_HOME = "/dbfs/FileStore/AIML/scratch/Providence-Attention-Axis/configs"

# _EXPERIMENT_1_CONFIGS = json.load(open(f"{DBFS_CONFIG_HOME}/experiment-001.json"))["configurations"]

_EXPERIMENT_1_CONFIGS = list(
    filter(
        lambda cfg: cfg["name"].startswith("0th"),
        json.load(open(f"{DBFS_CONFIG_HOME}/experiment-001.json"))["configurations"],
    )
)


def experiment_1__top_configurations_per_model(
    mlflow_experiment_id: str,
    *,
    output_root: Path,
    data_root: str = "./.data",
    experiment_configurations=_EXPERIMENT_1_CONFIGS,
    with_early_stopping=False,
):
    """Top 3 configurations for each model, for each dataset, with a fixed seeds per architecture, retrain with each attentional axis"""
    output_root.mkdir(parents=True, exist_ok=True)

    minimize_torch_runtime_overhead()

    _ATTENTION_TYPES = ["temporal", "feature"]
    _OPTIMIZERS = [SGD, torch.optim.Adam]
    _N_SUBITERATES = len(_ATTENTION_TYPES) * len(_OPTIMIZERS)

    _progress_bar = progressbar.ProgressBar(max_value=(len(experiment_configurations) * _N_SUBITERATES))

    for configuration_index, experiment_config in enumerate(experiment_configurations):
        training_config = experiment_config["training"]
        if "seed" in training_config:
            seed = int(training_config["seed"])
        else:
            # generate new seed if the seed isn't defined in the config don't want to invoke this prematurely
            seed = torch.seed()

        train_ds, test_ds = datasets_for_experiment_1_config(experiment_config["data"], data_root=data_root)
        model_config = experiment_config["model"]
        with mlflow.start_run(experiment_id=mlflow_experiment_id):
            mlflow.log_params(
                {  # yapf: skip
                    "early_stopping": with_early_stopping,
                    "model_seed": seed,
                    "model_config_index": configuration_index,
                    "batch_size": training_config["batch_size"],
                    "run_name": experiment_config["name"],
                    "dataset_name": experiment_config["data"]["name"],
                }
            )

            for subiterate_cursor, (attention_type, optimizer_type) in enumerate(
                itertools.product(_ATTENTION_TYPES, _OPTIMIZERS), start=1
            ):
                model_config["attention_axis"] = attention_type

                with mlflow.start_run(nested=True):
                    axial_attention_inner_func(  # yapf: skip
                        output_root,
                        with_early_stopping,
                        training_config,
                        seed,
                        train_ds,
                        test_ds,
                        model_config,
                        attention_type,
                        optimizer_type,
                        run_name=experiment_config["name"],
                        dataset_name=experiment_config["data"]["name"],
                        learning_rate=experiment_config["optimizer"]["lr"],
                    )
                _progress_bar.update(configuration_index * _N_SUBITERATES + subiterate_cursor)


class MLFlowMetricTracker(Callback):
    def after_epoch(
        self,
        epoch: int,
        model: "nn.Module",
        optimizer,
        losses: EpochLosses,
        dls: DataLoaders,
    ):
        mlflow.log_metrics({"loss_train": losses.train, "loss_val": losses.val}, step=epoch)


def axial_attention_inner_func(
    output_root: Path,
    with_early_stopping: bool,
    training_config: TrainingConfig,
    seed: int,
    train_ds,
    test_ds,
    model_config: Dict[str, Hyperparameter],
    attention_type: str,
    optimizer_type: Type[torch.optim.Optimizer],
    run_name: str,
    dataset_name: str,
    learning_rate: float,
):
    mlflow.log_params(
        {  # yapf: skip
            "optimizer": optimizer_type.__name__,
            "optimizer.lr": learning_rate,
            "attention_axis": attention_type,
        }
    )

    set_seed(seed)
    model = ProvidenceTransformer(len(train_ds.feature_columns), **model_config)
    model.device = use_gpu_if_available()

    optimizer = OptimizerWrapper(
        opt=optimizer_type(model.parameters(), lr=learning_rate),
        batch_size=training_config["batch_size"],
        num_epochs=training_config["num_epochs"],
    )
    dls = CustomProvidenceDataloaders(train_ds, test_ds, batch_size=optimizer.batch_size, pin_memory=True)

    run_output_dir = Path(output_root, attention_type, dataset_name, run_name)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logger_in_dir(run_output_dir / "logs", logger_name="experiment-01")
    logger.info("Instantiating callbacks")

    callbacks = [
        WriteModelOutputs(100, run_output_dir / "model-outputs", logger),
        CachedIntervalMetricsVisualizer(100, run_output_dir / "visualizations", logger),
        ModelCheckpointer(run_output_dir / "checkpoints", "val_loss", logger, keep_old=5),
        MLFlowMetricTracker(),
    ]

    if with_early_stopping:
        logger.info("Using early stopping")
        callbacks.append(EarlyStopping(40, "val_loss"))

    logger.info("Starting training")
    losses = callback_training(model, optimizer, dls, callbacks)

    logger.info("Computing metrics")
    if "backblaze" in dataset_name:
        metric_func = GranularMetrics
    else:
        metric_func = GeneralMetrics
    m = metric_func(model, dls.validation_ds, losses).iloc[0]  # DataFrame -> Series

    logger.info("Persisting with MLFlow")
    mlflow.log_metrics(m.to_dict())
    try:
        mlflow.log_artifacts(run_output_dir)
    except:
        logger.info("mlflow tripping over itself.", exc_info=True)

    logger.info("Ticking progress bar")


def dataset_for_experiment_2(dataset_name: str, nasa_test_num: int = None, *, data_root: str) -> ProvidenceDataset:
    """This experiment only requires a test (or validation) set, so that's all we return"""
    if equals_ignorecase(dataset_name, "nasa"):
        return NasaDataset(subset_choice=DataSubsetId.Test, data_dir=data_root)
    if equals_ignorecase(dataset_name, "nasa-seg"):
        return NasaFD00XDataset(
            NasaTurbofanTest(nasa_test_num),
            subset_choice=DataSubsetId.Test,
            data_dir=data_root,
        )
    if equals_ignorecase(dataset_name, "Backblaze"):
        return BackblazeDataset(DataSubsetId.Test, data_dir=data_root)
    if equals_ignorecase(dataset_name, "BackblazeExtended"):
        return BackblazeExtendedDataset(
            *BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER,
            subset_choice=DataSubsetId.Test,
            data_root=data_root,
        )
    raise ValueError(f"Invalid arguments: {dataset_name=} {nasa_test_num=}")


class Experiment2_Definition(TypedDict):
    paths: List[str]


class Experiment2(TypedDict):
    backblaze: List[Experiment2_Definition]
    backblazeextended: List[Experiment2_Definition]
    nasa_seg: List[Experiment2_Definition]
    nasa: List[Experiment2_Definition]


_EXPERIMENT_2_CONFIGS: Experiment2 = []  # json.load(open(relative_file("../configs/experiment-002.json")))


def experiment_2__best_models_switching_attentional_axis(output_root: Path, *, data_root: str = "./.data"):
    output_root.mkdir(parents=True, exist_ok=True)

    minimize_torch_runtime_overhead()

    results = []

    for ds_name, configurations in _EXPERIMENT_2_CONFIGS.items():
        print("Dataset:", ds_name)

        assert isinstance(configurations, list)
        # We should only have plural configurations for the nasa-seg datasets. Everything else should be a singleton
        assert (len(configurations) > 1) == (ds_name == "nasa-seg"), f"{len(configurations) =} but {ds_name = }"

        for config_num, config in enumerate(configurations, 1):
            print("Config", config_num, f"{config}")
            best_models_on_ds = config.pop("paths")  # state change, so we don't pass "paths" to the following
            val_ds = dataset_for_experiment_2(ds_name, data_root=data_root, **config)

            for path_num, model_root in enumerate(
                progressbar.progressbar(best_models_on_ds, max_len=len(best_models_on_ds)),
                1,
            ):
                for model_path in Path(model_root).glob("*.pt"):
                    model = torch.load(model_path, map_location="cpu")
                    temporal_metrics = fleet_metrics(model, Weibull, val_ds).assign(
                        exec_id=f"{ds_name}_{config_num}_{path_num}",
                        attention_axis="temporal",
                    )

                    set_attention_axis(model.transformer, "feature")
                    feature_metrics = fleet_metrics(model, Weibull, val_ds).assign(
                        exec_id=f"{ds_name}_{config_num}_{path_num}",
                        attention_axis="feature",
                    )

                    metrics_to_compare = concat((temporal_metrics, feature_metrics), ignore_index=True)
                    metrics_to_compare["exact_model"] = model_path.name
                    results.append(metrics_to_compare)

    results_df = concat(results, ignore_index=True)

    output_csv_path = output_root / "results.csv"
    results_df.to_csv(output_csv_path, index=False)
    print(f"Wrote experiment results to {output_csv_path}")


def main(experiment_num: int = typer.Argument(1, help="experiment number", min=1, max=2)) -> None:
    run_output_dir = Path(
        "/dbfs/FileStore/AIML/scratch/Providence-Attention-Axis/outputs",
        f"experiment_{experiment_num}",
        now_dt_string(),
    )
    experiment_id = create_or_set_experiment(
        "/Users/40000889@azg.utccgl.com/Providence-Investigative-Research/Attention Axis Re-evaluation (legacy code)"
    )

    if experiment_num == 1:
        experiment_1__top_configurations_per_model(
            mlflow_experiment_id=experiment_id,
            data_root="/dbfs/FileStore/datasets/providence",
            output_root=run_output_dir,
            with_early_stopping=True,
        )
    elif experiment_num == 2:
        experiment_2__best_models_switching_attentional_axis(run_output_dir)


if __name__ == "__main__":
    typer.run(main)
