"""
This script captures the key-value correspondence between a reference model and their best case performance
given all of the parameters we have available in configuration space (and budget to test).

If you follow this file, you should get *better* results than were published in Providence, as noted by the ``best_metrics``
bound to each instantiation.
(Providence is avaiable to read at https://ieeexplore.ieee.org/document/9843469 or by DOI: 10.1109/AERO53065.2022.9843469)

We don't share model weights simply because we've found these parameters to be robust, unless otherwise noted.

Includes everything for the best performance on the NASA, Backblaze, and BackblazeExtended datasets for the RNN-based models
-   Hyperparameters
-   Seeds
-   Every line of initialization

For engineering:
- If you want to use these models, you can just import these functions and have at it.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from collections import defaultdict
from ctypes import Union
from pathlib import Path
from typing import Any, Callable, NamedTuple, TypeAlias
from typing import TypedDict
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
from providence.training import OptimizerWrapper, ProvidenceLossInterfaceType
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
    """Default configuration for Callbacks used in training. See ``providence_utils.callbacks.py`` for more

    Args:
        num_epochs: Number of epochs used it training; used to derive upper bounds for metadata / evaluation output frequency. Defaults to 500.

    Returns:
        configuration settings
    """
    callback_config = dict()
    callback_config["epoch_logger_epochs"] = 10
    callback_config["visualizer"] = IntervalMetricsVisualizer.__name__

    # with our larger learning rate training, you generally don't need this many epochs
    # but this is the configuration we experimented with.
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
    """Returns default configuration for the optimizers used in these benchmarks.
    """
    optim_config = dict()
    optim_config["schedule_min"] = 1e-5
    optim_config["schedule_T_mult"] = 2
    # Adam works well. SGD needs more effort in the LR Schedule. The schedule can be done without, but is more useful when
    # using SGD optimizers. Omitted here, because it didn't beat out Adam in our experimentation
    optim_config["type"] = pt.optim.Adam
    return optim_config


class TransformerModelParams(TypedDict):
    ...


class RnnModelParams(TypedDict):
    """Parameters for the RNN-typed Providence model to be instantiated.

    Args:
        input_size: number of inputs received. Should be a value in ``feature_counts``
        hidden_size: number of hidden inputs in the recurrent cell
        num_layers: number of repetitions of the recurrent cell in the model. Tends not to exceed 5
        dropout: float within [0, 1] that determines the (approximate) number of nodes that are set to zero every forward pass
    """
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float


ModelParams: TypeAlias = Union[RnnModelParams, TransformerModelParams]


class OptimizerParams(TypedDict):
    """Parameters for the optimization algorithm that is used to update our model weights.

    Includes (hyper)parameters that aren't typically bundled with the optimizer, but serve no other purpose but the
    rate and/or likelihood of optimization (e.g. batch size), so they live here

    Args:
        batch_size: the number of entities per batch
        type: the type (not the instance) of an Optimizer for training your model
        learning_rate: multiplier for the weight update
        num_epochs: number of epochs to train for, at a maximum. Has obvious interactions with EarlyStopping and EmergencyBrake
        schedule_min: the lowest attainable learning rate with the learning rate schedule(r)
        schedule_T_mult: a multiplier within the scheduler learning rate-updater. See documentation of the LearningRateScheduler for more.
    """
    batch_size: int
    type: Type[pt.optim.Optimizer]
    learning_rate: float
    num_epochs: int

    # these are bounds for the learning rate schedule. See callabacks.LearningRateScheduler for more
    schedule_min: float
    schedule_T_mult: float


def train_providence_model(
    model_type: Type[ProvidenceModule],
    get_dls: Callable[[], DataLoaders],
    random_seed: int,
    optim_params: OptimizerParams,
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
            optimizer.opt, T_mult=optim_params["schedule_T_mult"], eta_min=optim_params["schedule_min"]
        ),
        IntervalMetricsVisualizer(callback_config["visualization_freq"], output_dir, logger=logger),
        WriteModelOutputs(callback_config["model_output_freq"], output_dir, logger=logger, verbose=False),
        ModelCheckpointer(
            output_dir=(output_dir / "model-checkpoints"),
            track=callback_config["tracking_metric"],
            logger=logger,
            verbose=True,
            keep_old=5
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
    # convert run_config to something we can log to mlflow
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


def _nasa_dls_func(data_root_path):
    train_ds, val_ds = NasaDatasets(data_root=data_root_path)
    # NOTE: this is not best practice, but it simplified things for the paper. Holdout sets aren't *required*
    test_ds = val_ds

    def fresh_dls(bs: int):  # replicate what was done above. Just do it again
        return CustomProvidenceDataloaders(
            train_ds,
            val_ds,
            batch_size=bs,
            num_workers=1,
            pin_memory=True,
        )._replace(test=ProvidenceDataLoader(test_ds, batch_size=1, num_workers=1))

    return fresh_dls


def _backblaze_dls_func(data_root_path: str):
    train_ds, val_ds = NasaDatasets(data_root=data_root_path)
    # NOTE: this is not best practice, but it simplified things for the paper. Holdout sets aren't *required*
    test_ds = val_ds

    def fresh_dls(bs: int):  # replicate what was done above. Just do it again
        return CustomProvidenceDataloaders(
            train_ds,
            val_ds,
            batch_size=bs,
            num_workers=1,
            pin_memory=True,
        )._replace(test=ProvidenceDataLoader(test_ds, batch_size=1, num_workers=1))

    return fresh_dls


class ProvidenceModelWithParams(NamedTuple):
    # not really a Callable[[ModelParams], ProvidenceModule], but we can't do varargs in callable, so this is what we have
    model_type: Union[Callable, Type[ProvidenceModule]]
    model_params: ModelParams
    opt_params: OptimizerParams
    seed: int
    dataloader_func: Callable[[str], DataLoaders]
    epoch_definition: ProvidenceLossInterfaceType = training_epoch

    def train(self, data_root_path: str, outputs_root: str) -> tuple[ProvidenceModule, dict]:
        """Train the model with the given parameters

        Args:
            data_root_path: the root for the download of the dataset. Accelerates data loading (i.e. caching) and debugging data integrity.
                Needs read-write permissions.
            outputs_root: the root for the outputs emitted by the callbacks, as well as the metrics CSV. Needs write permissions

        Return:
            model: the trained model. There may be a model checkpoint that was better than this one, so we encourage you to compare
                with the file model checkpoint output by ``ModelCheckpointer``
            metrics: output of ``GeneralMetrics``
        """
        model, metrics = train_providence_model(
            self.model_type,
            self.dataloader_func(data_root_path),
            random_seed=self.seed,
            optim_params=self.opt_params,
            model_params=self.model_params,
            callback_config=callback_default_config(self.opt_params["num_epochs"]),
            outputs_root_dir=outputs_root,
            epoch_definition=self.epoch_definition,
        )
        return model, metrics


################################################################################
#
# The best models themselves
#
################################################################################

NASA_Agg_VanillaRnn = ProvidenceModelWithParams(
    model_type=ProvidenceVanillaRNN,
    model_params=dict(input_size=feature_counts["nasa"], hidden_size=64, num_layers=2, dropout=0.3),
    opt_params=dict(**optimizer_default_config(), batch_size=4, learning_rate=3e-3, num_epochs=200),
    seed=11068621650300516211,
    dataloader_func=_nasa_dls_func,
    best_metrics=pd.DataFrame({
        "MSE": [2008.5],
        "MFE": [-0.215],
        "SMAPE": [0.13],
        "SMPE": [0.003],
    })
)

NASA_Agg_GRU = ProvidenceModelWithParams(
    model_type=ProvidenceGRU,
    model_params=dict(input_size=feature_counts["nasa"], hidden_size=512, num_layers=3, dropout=0.75),
    opt_params=dict(**optimizer_default_config(), batch_size=2, learning_rate=1e-3, num_epochs=50),
    seed=11068621650300516211,
    dataloader_func=_nasa_dls_func,
    best_metrics=pd.DataFrame({
        "MSE": [1524.7],
        "MFE": [-3.181],
        "SMAPE": [0.105],
        "SMPE": [0.013],
    })
)

NASA_Agg_LSTM = ProvidenceModelWithParams(
    model_type=ProvidenceGRU,
    opt_params=dict(**optimizer_default_config(), batch_size=4, learning_rate=3e-3, num_epochs=40),
    model_params=dict(input_size=feature_counts["nasa"], hidden_size=1024, num_layers=3, dropout=0.25),
    seed=11068621650300516211,
    dataloader_func=_nasa_dls_func,
    best_metrics=pd.DataFrame({
        "MSE": [2008.5],
        "MFE": [-0.215],
        "SMAPE": [0.13],
        "SMPE": [0.003],
    })
)

Backblaze_VanillaRNN = ProvidenceModelWithParams(
    model_type=ProvidenceVanillaRNN,
    opt_params=dict(**optimizer_default_config(), batch_size=1, learning_rate=3e-3, num_epochs=700),
    model_params=dict(input_size=feature_counts["backblaze"], hidden_size=128, num_layers=2, dropout=0.6),
    seed=15620825294243023828,
    dataloader_func=_backblaze_dls_func,
    best_metrics=pd.DataFrame({
        "MSE": [795.1],
        "MFE": [0.969],
        "SMAPE": [0.328],
        "SMPE": [0.094],
    })
)

Backblaze_LSTM = ProvidenceModelWithParams(
    model_type=ProvidenceLSTM,
    opt_params=dict(**optimizer_default_config(), batch_size=8, learning_rate=3e-3, num_epochs=700),
    model_params=dict(input_size=feature_counts["backblaze"], hidden_size=512, dropout=0.75, num_layers=3),
    seed=11068621650300516211,
    dataloader_func=_backblaze_dls_func,
    # paper: MSE 1113.22, MFE 22.36, SMAPE 0.43, SMPE -0.27
    best_metrics=pd.DataFrame({
        "MSE": [788.1],
        "MFE": [1.68],
        "SMAPE": [0.332],
        "SMPE": [0.07]
    })
)

Backblaze_GRU = ProvidenceModelWithParams(
    model_type=ProvidenceGRU,
    opt_params=dict(**optimizer_default_config(), batch_size=64, learning_rate=3e-3, num_epochs=700),
    model_params=dict(input_size=feature_counts["backblaze"], hidden_size=1024, dropout=0.3, num_layers=2),
    seed=11068621650300516211,
    dataloader_func=_backblaze_dls_func,
    # paper: MSE 834.77, MFE 17.46, SMAPE 0.38, SMPE -0.18
    best_metrics=pd.DataFrame({
        "MFE": [-0.154],
        "MSE": [678.2],
        "SMAPE": [0.309],
        "SMPE": [0.093],
    })
)

BackblazeExtended_VanillaRNN = ProvidenceModelWithParams(
    model_type=ProvidenceVanillaRNN,
    model_params=dict(input_size=feature_counts["backblaze"], hidden_size=512, num_layers=2, dropout=0.5),
    opt_params=dict(**optimizer_default_config(), batch_size=8, num_epochs=100, learning_rate=1e-3),
    seed=5002337303666687583,
    dataloader_func=_backblaze_dls_func,
    # paper: MSE 6316, MFE -73.33, SMAPE 0.58, 0.57
    best_metrics=pd.DataFrame({
        "MFE": [35.48],
        "MSE": [2411.3],
        "SMAPE": [0.534],
        "SMPE": [-0.427],
    })
)

BackblazeExtended_GRU = ProvidenceModelWithParams(
    model_type=ProvidenceGRU,
    model_params=dict(input_size=feature_counts["backblaze"], hidden_size=512, num_layers=2, dropout=0.5),
    opt_params=dict(**optimizer_default_config(), batch_size=32, num_epochs=700, learning_rate=3e-3),
    seed=11068621650300516211,
    dataloader_func=_backblaze_dls_func,
    # paper: MSE 924.9, MFE 14.11, SMAPE 0.37, SMPE -0.15
    best_metrics=pd.DataFrame({
        "MFE": [-2.312],
        "MSE": [520.1],
        "SMAPE": [0.276],
        "SMPE": [0.128],
    })
)

BackblazeExtended_LSTM = ProvidenceModelWithParams(
    model_type=ProvidenceLSTM,
    model_params=dict(input_size=feature_counts["backblaze"], hidden_size=512, num_layers=2, dropout=0.5),
    opt_params=dict(**optimizer_default_config(), batch_size=8, num_epochs=100, learning_rate=1e-3),
    seed=15620825294243023828,
    dataloader_func=_backblaze_dls_func,
    # paper: MSE 610.86, MFE 14.45, SMAPE 0.37, SMPE -0.16
    best_metrics=pd.DataFrame({
        "MFE": [0.809],
        "MSE": [515.8],
        "SMAPE": [0.276],
        "SMPE": [0.094],
    })
)
