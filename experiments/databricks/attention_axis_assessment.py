"""
Title: Attention Axis Assessment
Author: Stephen Fox <stehen.fox@rtx.com>
Purpose: Herein lies an experiment for evaluating whether the Attention Head mechanism (first outline in Vaswani et al arxiv:1706.03762)
as employed for the research around the Providence framework for time-to-event prediction. 

We run a regression-based evaluation per either experiment because that is what matters at the end of the day. However, there are other
characteristics that allow us to sift between the best and worse models like the following (intended to be exhaustive and to be updated over time) list:
- Progression of the Weibull sequence: it should be relevatively smooth, not something that hops all over the place wrt the final time step of the sequence
  - For eventful entities, we would expect this to an smooth approach to TTE=1. The earlier this confidence is gained, the further out we can predict.
  - For uneventful entities, we would expect to see a shallow / short distribution until the end of sequence.
    - As we (in production-like scenarios) don't have the true / realized time-to-event for a given entity, a reasonably smooth curve can beget confidence
      in the model and predictions produced
- Overshoot / undershoot proclivity wrt the entities that experience an event. Similarly, the willingness to predict TTE=0 for entities that don't experience
  an event
  - The lacker case is generally one we want to avoid (see the previous set of bullets)
  - The former needs brief elaboration:
    - Undershooting means predicting failure occurs earlier than they would actually. A proclivity towards such is preferable to undershoot, meaning to miss
      the eventful timestep by predicting event occurrence further in the future than realized.
    - Furthermore, we have a figure which shows the predicted TTE per timestep per device in a dataset and color codes wrt the prediction was over- or undershot.
      (This is typically performed on the last 40 time steps.) By simple visual inspection, one can see which side of the terminal line the model favors, or
      if it even predicts correctly at all.

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
from typing import Union

import mlflow
import torch
import typer
from torch.optim import Adam
from torch.optim import Optimizer
from torch.optim import SGD

from providence.dataloaders import CustomProvidenceDataloaders
from providence.datasets import BACKBLAZE_EXTENDED_QUARTERS_FOR_PAPER
from providence.datasets import BackblazeDatasets
from providence.datasets import BackblazeExtendedDatasets
from providence.datasets import NasaDatasets
from providence.datasets import NasaFD00XDatasets
from providence.datasets import ProvidenceDataset
from providence.datasets.adapters import BackblazeQuarter
from providence.datasets.adapters import NasaTurbofanTest
from providence.distributions import Weibull
from providence.loss import discrete_weibull_loss_fn
from providence.nn import ProvidenceTransformer
from providence.nn.transformer.memory_efficient import MemoryEfficientMHA
from providence.nn.transformer.transformer import _AttentionalAxis
from providence.nn.transformer.transformer import MultiheadedAttention3
from providence.paper_reproductions import GeneralMetrics
from providence.training import EpochLosses
from providence.training import minimize_torch_runtime_overhead
from providence.training import set_torch_default_dtypes
from providence.training import use_gpu_if_available
from providence.type_utils import type_name
from providence.types import DataLoaders
from providence.utils import also
from providence.utils import clear_memory
from providence.utils import configure_logger_in_dir
from providence.utils import now_dt_string
from providence.utils import set_seed
from providence_utils.callbacks import CachedIntervalMetricsVisualizer
from providence_utils.callbacks import Callback
from providence_utils.callbacks import EarlyStopping
from providence_utils.callbacks import ModelCheckpointer
from providence_utils.callbacks import WriteModelOutputs
from providence_utils.hyperparameter_sweeper import Hyperparameter
from providence_utils.mlflow import create_or_set_experiment
from providence_utils.mlflow import try_log_artifacts
from providence_utils.trainer import ProvidenceLossHandle
from providence_utils.trainer import ProvidenceTrainer


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


class Experiment1_TaskDefinition(TypedDict):
    """This type encapsulate the sub run for which a plurality of completions constitue a full experiment.
    In other words, an Experiment is composed of many tasks and this outlines one of those task
    """

    name: Optional[str]
    data: Dict[str, Hyperparameter]
    model: Dict[str, Hyperparameter]
    optimizer: Dict[str, Hyperparameter]
    training: TrainingConfig


def experiment_1_top_configurations_per_model(
    mlflow_experiment_id: str,
    *,
    output_root: Path,
    data_root: str = "./.data",
    experiment_configs: List[Experiment1_TaskDefinition],
):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    minimize_torch_runtime_overhead()
    set_torch_default_dtypes(torch.float32)

    for configuration in experiment_configs:
        with mlflow.start_run(experiment_id=mlflow_experiment_id):
            clear_memory()
            mlflow.log_param("experiment_num", 1)

            # BEGIN: pre-work
            experiment1_exec = Experiment1Run(configuration)
            if (seed_s := experiment1_exec.training_config.get("seed", None)) is not None:
                seed = int(seed_s)  # probably not a string, but int(12) == 12, so it's fine
                experiment1_exec.training_config["fresh_seed"] = False  # should have a better name than this
            else:  # set a new seed, because these params were deemed general enough to not need the hard-coding.
                seed = torch.seed()
                experiment1_exec.training_config["fresh_seed"] = True
            set_seed(seed)
            mlflow.log_params({f"model.{key}": value for key, value in experiment1_exec.model_config.items()})
            mlflow.log_params({f"training.{key}": value for key, value in experiment1_exec.training_config.items()})
            mlflow.log_params({f"data.{key}": value for key, value in experiment1_exec.data_config.items()})
            mlflow.log_params({f"optimizer.{key}": value for key, value in experiment1_exec.optimizer_config.items()})
            # END: pre-work
            train_ds, test_ds = datasets_for_experiment_1_config(experiment1_exec.data_config, data_root=data_root)
            dls = CustomProvidenceDataloaders(
                train_ds,
                test_ds,
                batch_size=4,  # experiment1_exec.training_config["batch_size"],
                pin_memory=True,
            )

            for opt_type, attention_axis in itertools.product([SGD, Adam], ["temporal", "feature"]):
                with mlflow.start_run(nested=True):
                    mlflow.log_params(
                        {
                            "optimizer.type": type_name(opt_type),
                            "model.attention_axis": attention_axis,
                        }
                    )
                    run_output_dir = also(
                        # Path(output_root, attention_axis, experiment1_exec.data_config["name"], experiment1_exec.name),
                        Path(output_root, experiment1_exec.name),
                        # if we make it in the same directory, that means we have a bad experiment name
                        lambda p: p.mkdir(parents=True, exist_ok=False),
                    )

                    experiment1_exec.run_training(
                        opt_type,
                        attention_axis,
                        dls,
                        prepare_default_callbacks(
                            with_early_stopping=True,
                            run_output_dir=run_output_dir,
                            logger=configure_logger_in_dir(run_output_dir, logger_name="experiment-01-logger"),
                        ),
                    )
                    results = experiment1_exec.evaluate()
                    mlflow.log_metrics(results["metrics"])
                    try_log_artifacts(run_output_dir)
                clear_memory(experiment1_exec.model)
        # mlflow.end_run()


class Experiment1Run:
    """
    A sticky state management type for the execution of the 'experiment' (really, a re-evaluation) of the attention axis'
    power and contribution to the success of Providence models."""

    def __init__(self, configuration: Experiment1_TaskDefinition):
        self.name = configuration["name"]
        self.data_config = configuration["data"]
        self.model_config = configuration["model"]
        self.optimizer_config = configuration["optimizer"]
        self.training_config = configuration["training"]

    def build_model(self, model_dimension: int, attention_axis: _AttentionalAxis) -> ProvidenceTransformer:
        # NOTE: as this implementation doesn't respect the attention axis, is this experiment even valid?
        return ProvidenceTransformer(
            model_dimension,
            self.model_config["hidden_size"],
            self.model_config["n_layers"],
            self.model_config["n_attention_heads"],
            self.model_config["dropout"],
            self.model_config["layer_norm_epsilon"],
            self.model_config["positional_encoding_dimension"],
            attention_axis=attention_axis,
            t_attention=MemoryEfficientMHA,
            device=use_gpu_if_available(),
        )

    def load_data(self):
        return datasets_for_experiment_1_config(self.data_config)

    def run_training(
        self,
        optimizer_t: Type[Optimizer],
        attention_axis: _AttentionalAxis,
        dls: DataLoaders,
        cbs: List[Callback] = [],
    ):
        self.model = self.build_model(dls.train_ds.n_features, attention_axis)
        trainer = ProvidenceTrainer(
            self.model,
            optimizer_t(self.model.parameters(), lr=self.optimizer_config["lr"]),
            ProvidenceLossHandle(discrete_weibull_loss_fn, Weibull.Params),
            should_clip_gradients=True,
        )
        trainer.cbs.extend(cbs)
        self.losses = trainer.train(self.training_config["num_epochs"], dls)

    def evaluate(self, dls: DataLoaders):
        m = GeneralMetrics(self.model, dls.validation_ds, self.losses).iloc[0]  # DataFrame -> Series
        return {"metrics": m}


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


def prepare_default_callbacks(with_early_stopping: bool, run_output_dir: Union[str, Path], logger):
    callbacks = [
        WriteModelOutputs(100, run_output_dir / "model-outputs", logger),
        CachedIntervalMetricsVisualizer(100, run_output_dir / "visualizations", logger),
        ModelCheckpointer(run_output_dir / "checkpoints", "val_loss", logger),
        MLFlowMetricTracker(),
    ]

    if with_early_stopping:
        logger.info("Using early stopping")
        callbacks.append(EarlyStopping(40, "val_loss"))
    return callbacks


class Experiment2_Definition(TypedDict):
    paths: List[str]


class Experiment2(TypedDict):
    backblaze: List[Experiment2_Definition]
    backblazeextended: List[Experiment2_Definition]
    nasa_seg: List[Experiment2_Definition]
    nasa: List[Experiment2_Definition]


def experiment_2__hyperparameter_sweep_anew(output_root: Path, *, data_root: str = "./.data"):
    output_root.mkdir(parents=True, exist_ok=True)

    minimize_torch_runtime_overhead()


def main(
    experiment_num: int = typer.Argument(1, help="experiment number", min=1, max=2),
    dbfs_root: str = typer.Option(
        "/dbfs/FileStore/AIML/scratch/Providence-Attention-Axis/outputs",
        help="The root of the file tree where all Attention Axis stuff is being kept",
    ),
    config_home: str = typer.Option(
        "/dbfs/FileStore/AIML/scratch/Providence-Attention-Axis/configs",
        help="The directory to scan for configuration files",
    ),
) -> None:
    experiment_id = create_or_set_experiment(
        "/Users/{UserNameHere}/Providence-Investigative-Research/Attention Axis Re-evaluation"
    )
    for _ in range(10):
        mlflow.end_run("KILLED")  # make sure there's nothing else running
    # _EXPERIMENT_1_CONFIGS = json.load(open(f"{DBFS_CONFIG_HOME}/experiment-001.json"))["configurations"]
    experiment_1_configs = [
        cfg
        for cfg in json.load(open(f"{config_home}/experiment-001.json"))["configurations"]
        if cfg["name"].startswith("0th")  # only checking the best params
    ]
    experiment_output_root = Path(dbfs_root, f"experiment_{experiment_num:02}", now_dt_string())
    if experiment_num == 1:
        experiment_1_top_configurations_per_model(
            experiment_id,
            output_root=experiment_output_root,
            data_root="/dbfs/FileStore/datasets/providence",
            experiment_configs=experiment_1_configs,
        )
    elif experiment_num == 2:
        print("This is a test message")


if __name__ == "__main__":
    typer.run(main)
