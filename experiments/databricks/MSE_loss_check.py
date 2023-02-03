"""
An experiment to assess whether MSE would be a viable metric for Providence models.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from collections import defaultdict
from pathlib import Path
from types import FunctionType
from typing import List, Tuple, Union

import mlflow
import torch as pt
from providence.dataloaders import CustomProvidenceDataloaders, ProvidenceDataLoader
from providence.datasets.adapters import DS_NAME_TO_FEATURE_COUNT
from providence.datasets import NasaDatasets
from providence.distributions import SurvivalAnalysisDistribution, Weibull
from providence.loss import DiscreteWeibullLoss, DiscreteWeibullMSELoss, ProvidenceLoss, discrete_weibull_loss_fn
from providence.nn import ProvidenceGRU, ProvidenceModule, ProvidenceTransformer
from providence.paper_reproductions import GeneralMetrics
from providence.training import OptimizerWrapper, set_torch_default_dtypes, use_gpu_if_available, training_epoch
from providence.utils import now_dt_string, set_seed
from providence_utils.callbacks import CachedIntervalMetricsVisualizer
from providence_utils.hyperparameter_sweeper import nest_keys, HyperparameterSweeper, nest_values
from providence_utils.merge_dict import merge_dictionaries
from providence.type_utils import patch

from torchtyping import TensorType


################################################################################
#
# MLFLow Setup
#
################################################################################

from providence_utils.mlflow import create_or_set_experiment, try_log_artifacts
from providence.training import training_epoch


def mlflow_epoch(dls, model, optimizer, *, step: int = None, loss_criterion):
    losses = training_epoch(dls, model, optimizer, loss_criterion=loss_criterion)

    mlflow.log_metric("train_loss", losses.training, step=step)
    mlflow.log_metric("val_loss", losses.validation, step=step)
    return losses


################################################################################
#
# General Pre-work
#
################################################################################

EXPERIMENT_NAME = "/Users/40000889@azg.utccgl.com/Providence-Investigative-Research/{}"

ROOT_DIR = Path("/dbfs/FileStore/AIML/scratch/Providence-MSE-Loss-Checks")
ROOT_DIR.mkdir(exist_ok=True, parents=True)

metadata_to_log = defaultdict(dict)
metadata_to_log["general"]["dtype"] = pt.float32

set_torch_default_dtypes(metadata_to_log["general"]["dtype"])

import providence.nn.transformer.deepmind as dm

dm._DEBUG = False

################################################################################
#
# Data Access
#
################################################################################

metadata_to_log["data"]["global_seed_at_init"] = pt.initial_seed()

GLOBAL_train_ds, GLOBAL_val_ds = NasaDatasets(data_root="/dbfs/FileStore/datasets/providence", )
GLOBAL_test_ds = GLOBAL_val_ds
metadata_to_log["data"]["name"] = "NASA"


def global_dls(bs=128):  # replicate what was done above. Just do it again
    return CustomProvidenceDataloaders(
        GLOBAL_train_ds,
        GLOBAL_val_ds,
        batch_size=bs,
        num_workers=1,
        pin_memory=True,
    )._replace(test=ProvidenceDataLoader(GLOBAL_test_ds, batch_size=1, num_workers=1))


from providence.training import LossAggregates

# trainer = Trainer(mlflow_epoch)

RUN_CONFIG = {
    "model":
        {
            "n_heads": 2,
            "n_layers": 1,
            "n_features": DS_NAME_TO_FEATURE_COUNT["nasa"],
            "n_embedding": 256,  # down from 128
            "max_seq_len": 700,
            "dropout": 0.1,
        },
    "optimizer": {
        "learning_rate": 1e-4,
        "batch_size": 22,  # down from 128
        "num_epochs": 80,
    },
    "dataset": "nasa"
}

# NOTE: just two function to keep things straight fortard


def make_transformer_model() -> ProvidenceModule:
    metadata_to_log["model.type"] = ProvidenceTransformer
    model = ProvidenceTransformer(
        RUN_CONFIG["model"]["n_features"],
        hidden_size=RUN_CONFIG["model"]["n_embedding"],
        n_layers=RUN_CONFIG["model"]["n_layers"],
        n_attention_heads=RUN_CONFIG["model"]["n_heads"],
        dropout=RUN_CONFIG["model"]["dropout"],
        positional_encoding_dimension=710,
        device=use_gpu_if_available()
    )
    return model


def make_rnn_model() -> ProvidenceModule:
    metadata_to_log["model.type"] = ProvidenceGRU
    return ProvidenceGRU(
        RUN_CONFIG["model"]["n_features"],
        RUN_CONFIG["model"]["n_embedding"],
        RUN_CONFIG["model"]["n_layers"],
        dropout=RUN_CONFIG["model"]["dropout"],
        device=use_gpu_if_available()
    )


@patch
def cast(self: Weibull.Params, t_: pt.dtype) -> Weibull.Params:
    return Weibull.Params(self.alpha.to(t_), self.beta.to(t_))


@patch
def __getitem__(self: Weibull.Params, indices) -> Weibull.Params:
    return Weibull.Params(self.alpha[indices], self.beta[indices])


class DiscreteWeibullNanMeanNllFusion(ProvidenceLoss):
    def forward(
        self, params: SurvivalAnalysisDistribution.Params, y: TensorType["time"],
        _censor: TensorType["time"], _x_lengths: Union[List[int], TensorType["time"]]
    ) -> Union[float, Tuple[float, ...]]:

        means = Weibull.mean(params)
        weibull_loss = discrete_weibull_loss_fn(params, y, _censor, _x_lengths)
        # medians = Weibull.median(params)
        # tte_preds = Weibull.mode(params)

        # print(f"""{means.flatten().nansum() = } {medians.flatten().nansum() = }
        # {tte_preds_nansum = } {mse = } {rmse = }
        # {complex_tte_preds_nansum = } {complex_mse = }
        # """)
        nmean = pt.nanmean((means - y) ** 2)
        # if nmean < 0:
        #     nmean *= -1

        return nmean + weibull_loss

def post_run_metrics_evaluation(run_config, model, dls_to_use, output_dir, loss_agg):
    import logging
    print("Generating metrics", now_dt_string())
    dls_to_use.test_ds.device = model.device
    dls_to_use.test_ds.use_device_for_iteration(True)
    model_metrics = GeneralMetrics(model, dls_to_use.test_ds, loss_agg)

    metric_visualizer = CachedIntervalMetricsVisualizer(1, output_dir, logging.getLogger())
    metric_visualizer.callable.func(run_config["optimizer"]["num_epochs"], model, None, None, dls_to_use)
    
    print("Generating metrics completed", now_dt_string())

    merged_config = dict(**run_config, **metadata_to_log)
    nested_config = {k: (type_or_func_name(v) if isinstance(v, (type, FunctionType)) else v) for (k, v) in nest_keys(merged_config).items()}
    # nested_config["model.type"] = type_name(model)

    mlflow.log_params(nested_config) # NOTE: this should **not** live here, but I messed up the code flow and it's not worth fixing for the findings.

    mlflow.log_metrics(model_metrics.iloc[0].to_dict())
    try_log_artifacts(output_dir)


def dumb_run(dls_to_use, run_config, make_model, new_loss = None):

    if new_loss is None:
        new_loss = DiscreteWeibullMSELoss()
    
    config = nest_keys(run_config)
    set_seed(pt.seed())
    # model = dm.ProvidenceBertTransformer(2, 1, DS_NAME_TO_FEATURE_COUNT["nasa"], 128, max_seq_len=710, device=use_gpu_if_available())
    model = make_model()

    opt_wrapper = OptimizerWrapper(
        pt.optim.Adam(
            model.parameters(),
            lr=config["optimizer.learning_rate"],
        ),
        batch_size=config["optimizer.batch_size"],
        num_epochs=config["optimizer.num_epochs"]
    )

    # training-eval loop
    model.to(model.device)
    loss_agg = LossAggregates([], [])
    for epoch_i in range(1, opt_wrapper.num_epochs + 1):
        epoch_losses = mlflow_epoch(dls_to_use, model, opt_wrapper.opt, step=epoch_i, loss_criterion=new_loss)
        loss_agg.append_losses(epoch_losses)

    # model.to('cpu')
    return model, loss_agg


def smart_dict_upsert(base: dict, update: dict) -> dict:
    "The update should be nested to the desired level i.e. `top.mid.low: 12`"
    return nest_values(merge_dictionaries(update, nest_keys(base)))


def fix_mlflow_backcatalog():
    """Because MLFlow runs don't terminate when the code terminates,
    even though that's exactly what the context manager is designed to facilitate."""
    for _ in range(100):
        mlflow.end_run()


def sweeping_assessment():
    """Assess the MSE Loss function with a full sweep, allow the best model to shine and showing what parameters we don't need consider for this
    formulation of the Providence problem."""

    fix_mlflow_backcatalog()

    with mlflow.start_run(experiment_id=create_or_set_experiment(EXPERIMENT_NAME.format("MSE Check"))):

        for updates in HyperparameterSweeper.from_dict({
                "optimizer.num_epochs": [10, 20, 40, 80],
                "optimizer.learning_rate": [3e-6, 3e-4, 3e-3, 1e-3,],
                "optimizer.batch_size": [4, 6, 8, 10, 12, 14, 16],
        }).poll_sweeps():
            run_config = smart_dict_upsert(run_config, updates)
            for make_model in [
                make_transformer_model,
                make_rnn_model
            ]:
                with mlflow.start_run(nested=True):
                    dls_to_use = global_dls(run_config["optimizer"]["batch_size"])
                    output_dir = ROOT_DIR / now_dt_string()
                    model, loss_agg = dumb_run(dls_to_use, run_config, make_model, new_loss=DiscreteWeibullMSELoss())
                    post_run_metrics_evaluation(run_config, model, dls_to_use, output_dir, loss_agg)


def single_run(new_exp_name = "MSE Check"):
    fix_mlflow_backcatalog()

    with mlflow.start_run(experiment_id=create_or_set_experiment(EXPERIMENT_NAME.format(new_exp_name))):
        for updates in HyperparameterSweeper.from_dict({
            "optimizer.num_epochs": [40, 80],
            "optimizer.learning_rate": [1e-6, 3e-6, 3e-4, 1e-3],
            "optimizer.batch_size": [2, 4, 16, 128],
        }).poll_sweeps():
            run_config = smart_dict_upsert(RUN_CONFIG, updates)

            for make_model in [
                make_transformer_model,
                # make_rnn_model,
            ]:
                with mlflow.start_run(nested=True):
                    dls_to_use = global_dls(run_config["optimizer"]["batch_size"])
                    output_dir = ROOT_DIR / now_dt_string()
                    model, loss_agg = dumb_run(dls_to_use, run_config, make_model)
                    post_run_metrics_evaluation(run_config, model, dls_to_use, output_dir, loss_agg)


def type_or_func_name(f):
    if isinstance(f, (type, FunctionType)):
        return f.__name__
    return type(f).__name__


def single_run_comparing_loss_functions(new_exp_name = "MSE Check - comparing losses"):
    """Compare the three losses that we've generated against possible valid parameters that should work."""
    fix_mlflow_backcatalog()

    for loss_func in [DiscreteWeibullLoss(), DiscreteWeibullMSELoss(), DiscreteWeibullNanMeanNllFusion()]:
        with mlflow.start_run(experiment_id=create_or_set_experiment(EXPERIMENT_NAME.format(new_exp_name)), run_name=type_or_func_name(loss_func)):
            for updates in HyperparameterSweeper.from_dict({
                "optimizer.num_epochs": [80],
                "optimizer.batch_size": [2, 4, 16, 128],
                "optimizer.learning_rate": [3e-6, 3e-4, 1e-3],
            }).poll_sweeps():
                run_config = smart_dict_upsert(RUN_CONFIG, updates)

                set_seed(metadata_to_log["data"]["global_seed_at_init"]) # NOTE: SETTING SEEDS IS IMPORTANT!

                for make_model in [
                    make_transformer_model,
                    make_rnn_model,
                ]:
                    with mlflow.start_run(nested=True):
                        dls_to_use = global_dls(run_config["optimizer"]["batch_size"])
                        output_dir = ROOT_DIR / now_dt_string()
                        model, loss_agg = dumb_run(dls_to_use, run_config, make_model, new_loss=loss_func)
                        post_run_metrics_evaluation(run_config, model, dls_to_use, output_dir, loss_agg)



################################################################################
#
# Execution Point
#
################################################################################

single_run_comparing_loss_functions()