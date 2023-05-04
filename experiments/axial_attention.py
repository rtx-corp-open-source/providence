"""
Legacy version of the attention-axis experiment. See the new version under `/databricks` for more

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypedDict

import torch
import typer
from pandas import concat
from pandas import Series
from progressbar import progressbar
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
from providence.metrics import fleet_metrics
from providence.nn import ProvidenceTransformer
from providence.nn.transformer.transformer import set_attention_axis
from providence.training import LossAggregates
from providence.training import minimize_torch_runtime_overhead
from providence.training import OptimizerWrapper
from providence.training import use_gpu_if_available
from providence.types import DataLoaders
from providence.utils import configure_logger_in_dir
from providence.utils import now_dt_string
from providence.utils import set_seed
from providence_utils.callbacks import CachedIntervalMetricsVisualizer
from providence_utils.callbacks import Callback
from providence_utils.callbacks import EarlyStopping
from providence_utils.callbacks import ModelCheckpointer
from providence_utils.callbacks import WriteModelOutputs
from providence_utils.hyperparameter_sweeper import Hyperparameter
from providence_utils.trainer import Trainer
from providence_utils.trainer import transformer_epoch


def relative_file(fpath: str) -> str:
    return (Path(__file__).parent / fpath).as_posix()


def callback_training(
    model,
    optimizer: OptimizerWrapper,
    dataloaders: DataLoaders,
    cbs: List[Callback] = None,
):
    trainer = Trainer(transformer_epoch, verbose=True)
    loss_agg = trainer.callback_training(model, optimizer, dataloaders, cbs)
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


class ExperimentTaskDefinition(TypedDict):
    """This type encapsulate the sub run for which a plurality of completions constitue a full experiment.
    In other words, an Experiment is composed of many tasks and this outlines one of those task
    """

    name: Optional[str]
    dataset: Dict[str, Hyperparameter]
    model: Dict[str, Hyperparameter]
    optimizer: Dict[str, Hyperparameter]
    training: TypedDict("training_params", {"batch_size": int, "num_epochs": int})


_EXPERIMENT_1_CONFIGS: List[ExperimentTaskDefinition] = json.load(open(relative_file("configs/experiment-001.json")))[
    "configurations"
]


def experiment_1__top_configurations_per_model(
    output_root: Path,
    *,
    data_root: str = "./.data",
    experiment_configurations=_EXPERIMENT_1_CONFIGS,
    with_early_stopping=False,
):
    """Top 3 configurations for each model, for each dataset, retrain with each attentional axis"""
    output_root.mkdir(parents=True, exist_ok=True)

    minimize_torch_runtime_overhead()

    run_results = []
    for experiment_num, experiment_config in enumerate(experiment_configurations, start=1):
        training_config = experiment_config["training"]
        if "seed" in training_config:
            seed = int(training_config["seed"])
        else:
            # generate new seed if the seed isn't defined in the config don't want to invoke this prematurely
            seed = torch.seed()

        train_ds, test_ds = datasets_for_experiment_1_config(experiment_config["data"], data_root=data_root)
        model_config = experiment_config["model"]
        for attention_type in ["temporal", "feature"]:
            model_config["attention_axis"] = attention_type

            set_seed(seed)
            model = ProvidenceTransformer(len(train_ds.feature_columns), **model_config)
            model.device = use_gpu_if_available()
            optimizer = SGD(model.parameters(), lr=experiment_config["optimizer"]["lr"])
            optimizer = OptimizerWrapper(
                optimizer,
                batch_size=training_config["batch_size"],
                num_epochs=training_config["num_epochs"],
            )
            dls = CustomProvidenceDataloaders(train_ds, test_ds, batch_size=optimizer.batch_size)

            run_output_dir = Path(
                output_root,
                attention_type,
                experiment_config["data"]["name"],
                experiment_config["name"],
            )
            run_output_dir.mkdir(parents=True, exist_ok=True)

            logger = configure_logger_in_dir(run_output_dir, logger_name="experiment-01")
            logger.info(f"Run start time = {now_dt_string()}")

            callbacks = [
                WriteModelOutputs(100, run_output_dir / "model-outputs", logger),
                CachedIntervalMetricsVisualizer(100, run_output_dir / "visualizations", logger),
                ModelCheckpointer(run_output_dir / "checkpoints", "val_loss", logger),
            ]

            if with_early_stopping:
                callbacks.append(EarlyStopping(40, "val_loss"))

            losses = callback_training(model, optimizer, dls, callbacks)
            logger.info(f"Run end time = {now_dt_string()}")

            loss_metrics = compute_loss_metrics(losses)
            loss_metrics.to_csv(run_output_dir / "loss-metrics.csv", index=False)
            # TODO: use GeneralMetrics to shorten this up

            m = fleet_metrics(model, Weibull, dls.validation_ds).assign(
                attention_type=attention_type,
                experiment_num=experiment_num,
                run_name=experiment_config["name"],
                ds_name=experiment_config["data"]["name"],
            )
            for loss_metric, measure in loss_metrics.items():
                m[loss_metric] = measure

            m.to_csv(run_output_dir / "run-results.csv", index=False)

            run_results.append(m)
    concat(run_results).to_csv(output_root / "results.csv", index=False)


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


_EXPERIMENT_2_CONFIGS: Experiment2 = json.load(open(relative_file("configs/experiment-002.json")))


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

            for path_num, model_root in enumerate(progressbar(best_models_on_ds, max_len=len(best_models_on_ds)), 1):
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


def main(experiment_num: int = typer.Argument(..., help="experiment number", min=1, max=2)) -> None:
    run_output_dir = Path("outputs", f"experiment_{experiment_num}", now_dt_string())
    if experiment_num == 1:
        experiment_1__top_configurations_per_model(run_output_dir)
    elif experiment_num == 2:
        experiment_2__best_models_switching_attentional_axis(run_output_dir)


if __name__ == "__main__":
    typer.run(main)
