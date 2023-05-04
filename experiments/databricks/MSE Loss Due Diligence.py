# Databricks notebook source
# MAGIC %pip uninstall providence -y

# COMMAND ----------

# MAGIC %pip install /dbfs/FileStore/binaries/providence/providence-1.0.post1.dev7-py3-none-any.whl

# COMMAND ----------

"""
An experiment to assess whether MSE would be a viable metric for Providence models.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from argparse import ArgumentError
from collections import defaultdict
from operator import attrgetter
from pathlib import Path
from types import FunctionType, MethodType
from typing import Callable, List, Literal, Tuple, TypeVar, Union

from jaxtyping import Int
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

# COMMAND ----------

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

# COMMAND ----------

################################################################################
#
# General Pre-work
#
################################################################################

EXPERIMENT_NAME = "/Users/40000889@azg.utccgl.com/Providence-Investigative-Research/{}"
try:
    ROOT_DIR = Path("/dbfs/FileStore/AIML/scratch/Providence-MSE-Loss-Checks")
    ROOT_DIR.mkdir(exist_ok=True, parents=True)
except:
    print("Failed to access path for experiment artifacts")

metadata_to_log = defaultdict(dict)
metadata_to_log["general"]["dtype"] = pt.float32

set_torch_default_dtypes(metadata_to_log["general"]["dtype"])

import providence.nn.transformer.deepmind as dm

dm._DEBUG = False

# COMMAND ----------

################################################################################
#
# Data Access
#
################################################################################

metadata_to_log["data"]["global_seed_at_init"] = pt.initial_seed()


def global_dls(bs=128):  # replicate what was done above. Just do it again
    GLOBAL_train_ds, GLOBAL_val_ds = NasaDatasets(
        data_root="/dbfs/FileStore/datasets/providence",
    )
    GLOBAL_test_ds = GLOBAL_val_ds
    metadata_to_log["data"]["name"] = "NASA"
    return CustomProvidenceDataloaders(
        GLOBAL_train_ds,
        GLOBAL_val_ds,
        batch_size=bs,
        num_workers=1,
        pin_memory=True,
    )._replace(test=ProvidenceDataLoader(GLOBAL_test_ds, batch_size=1, num_workers=1))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Model

# COMMAND ----------

# trainer = Trainer(mlflow_epoch)

RUN_CONFIG = {
    "model": {
        "n_heads": 2,
        "n_layers": 2,
        "n_features": DS_NAME_TO_FEATURE_COUNT["nasa"],
        "n_embedding": 256,
        "max_seq_len": 700,
        "dropout": 0.1,
    },
    "dataset": "nasa",
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
        device=use_gpu_if_available(),
    )
    return model


def make_rnn_model() -> ProvidenceModule:
    metadata_to_log["model.type"] = ProvidenceGRU
    return ProvidenceGRU(
        RUN_CONFIG["model"]["n_features"],
        RUN_CONFIG["model"]["n_embedding"],
        RUN_CONFIG["model"]["n_layers"],
        dropout=RUN_CONFIG["model"]["dropout"],
        device=use_gpu_if_available(),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and Evaluation

# COMMAND ----------

from providence.training import LossAggregates

def post_run_metrics_evaluation(run_config, model, dls_to_use, output_dir, loss_agg):
    import logging

    print("Generating metrics", now_dt_string())
    dls_to_use.test_ds.device = model.device
    dls_to_use.test_ds.use_device_for_iteration(True)
    model_metrics = GeneralMetrics(model, dls_to_use.test_ds, loss_agg)

    metric_visualizer = CachedIntervalMetricsVisualizer(1, output_dir, logging.getLogger())
    metric_visualizer.callable.func(run_config["optimizer"]["num_epochs"], model, None, None, dls_to_use)

    print("Generating metrics completed", now_dt_string())


    mlflow.log_metrics(model_metrics.iloc[0].to_dict())
    try_log_artifacts(output_dir)


def dumb_run(dls_to_use, run_config, make_model, new_loss=None):

    if new_loss is None:
        new_loss = DiscreteWeibullMSELoss()

    config = nest_keys(run_config)
    model = make_model()

    opt_wrapper = OptimizerWrapper(
        pt.optim.Adam(
            model.parameters(),
            lr=config["optimizer.learning_rate"],
        ),
        batch_size=config["optimizer.batch_size"],
        num_epochs=config["optimizer.num_epochs"],
    )

    # log the parameters of the run
    merged_config = dict(**run_config, **metadata_to_log)
    nested_config = {
        k: (type_or_func_name(v) if isinstance(v, (type, FunctionType, MethodType)) else v)
        for (k, v) in nest_keys(merged_config).items()
    }

    mlflow.log_params(nested_config) 

    
    # training-eval loop
    model.to(model.device)
    loss_agg = LossAggregates([], [])
    for epoch_i in range(1, opt_wrapper.num_epochs + 1):
        epoch_losses = mlflow_epoch(dls_to_use, model, opt_wrapper.opt, step=epoch_i, loss_criterion=new_loss)
        loss_agg.append_losses(epoch_losses)

    # model.to('cpu')
    return model, loss_agg

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loss Definitions
# MAGIC AKA The heart of this experiment

# COMMAND ----------

def smart_dict_upsert(base: dict, update: dict) -> dict:
    "The update should be nested to the desired level i.e. `top.mid.low: 12`"
    return nest_values(merge_dictionaries(update, nest_keys(base)))


def fix_mlflow_backcatalog():
    """Because MLFlow runs don't terminate when the code terminates,
    even though that's exactly what the context manager is designed to facilitate."""
    for _ in range(100):
        mlflow.end_run()


def type_or_func_name(f):
    if isinstance(f, (type, FunctionType, MethodType)):
        return f.__name__
    return type(f).__name__



class TunableWeibullMXEFusion(ProvidenceLoss):
    def __init__(self, regression_stat: Literal["mode", "mean", "median"], error_func: Literal["mae", "mse"], λ: float):
        super().__init__()
        get_func = attrgetter(regression_stat)
        self.inference_func: Callable[[Weibull.Params], Union[float, pt.Tensor]] = get_func(Weibull)
        self.scale = self.λ = λ

        if error_func == "mse":

            def mse(preds, target):
                return pt.nanmean((preds - target) ** 2)  # used in place of nn.functional.mse_loss

            error_f = mse
        elif error_func == "mae":

            def mae(preds, target):
                return pt.nanmean(pt.abs(preds - target))  # torch doesn't ship with an MAE

            error_f = mae
        else:
            raise ArgumentError("error_func", "Invalid because it was not found in {'mae', 'mse'}")
        self.regression_error = error_f

    def __repr__(self):
        return (
            type_or_func_name(self)
            + f"({type_or_func_name(self.inference_func)}, {type_or_func_name(self.regression_error)}, {self.λ})"
        )


class TemperedWeibullMXEFusion(TunableWeibullMXEFusion):
    """
    The fusion of the standard NLL Weibull loss with a Weibull regression inference - one of mean, median, mode -
    error - one of MSE, MAE - tempered by factor λ.

    The desired effect follows after the following pseudo-code
    ```
    inference = Weibull_s, s ∈ {mean, median, mode}
    regression_error ∈ {MSE(inference, y), MAE(inference, y)}
    nll = ...
    λ ∈ {10^i for i ∈ {0, -1, ..., -7}}

    tempered_Weibull_loss = nll + λ*regression_error
    ```
    """

    def forward(
        self,
        params: SurvivalAnalysisDistribution.Params,
        y: Int[pt.Tensor, "time"],
        _censor: Int[pt.Tensor, "time"],
        _x_lengths: Union[List[int], Int[pt.Tensor, "time"]],
    ) -> Union[float, Tuple[float, ...]]:
        inference = self.inference_func(params)
        weibull_loss = discrete_weibull_loss_fn(params, y, _censor, _x_lengths)
        error_ = self.regression_error(inference, y)
        total_loss = weibull_loss + self.scale * error_

        return total_loss


class ComplementaryWeibullMXEFusion(TunableWeibullMXEFusion):
    """
    The fusion of the standard NLL Weibull loss with a Weibull regression inference - one of mean, median, mode -
    error - one of MSE, MAE - tradied off in a complementary fashion

    The desired effect follows after the following pseudo-code
    ```
    inference = Weibull_s, s ∈ {mean, median, mode}
    regression_error ∈ {MSE(inference, y), MAE(inference, y)}
    nll = ...
    λ ∈ (0, 1)

    complementary_Weibull_loss = (1-λ)*nll + (λ)*regression_error
    ```
    """

    def forward(
        self,
        params: SurvivalAnalysisDistribution.Params,
        y: Int[pt.Tensor, "time"],
        _censor: Int[pt.Tensor, "time"],
        _x_lengths: Union[List[int], Int[pt.Tensor, "time"]],
    ) -> Union[float, Tuple[float, ...]]:
        inference = self.inference_func(params)
        weibull_loss = discrete_weibull_loss_fn(params, y, _censor, _x_lengths)
        error_ = self.regression_error(inference, y)
        total_loss = (1 - self.scale) * weibull_loss + self.scale * error_

        return total_loss


# Like SDist in metrics, this is a type signature for a type variable, not an instance
T_TunableLoss = TypeVar("T_TunableLoss", ComplementaryWeibullMXEFusion, TemperedWeibullMXEFusion)

# COMMAND ----------

def single_run_comparing_scaled_loss_functions(loss_type: T_TunableLoss, new_exp_name: str):
    """Generate evidence for the MSE / MAE loss contribution being detrimental (or not)
    to the training of Providence models
    """
    from operator import itemgetter

    fix_mlflow_backcatalog()

    optimizer_sweeper = HyperparameterSweeper.from_dict(
        {
            "optimizer.num_epochs": [80, 110, 200, 500],
            "optimizer.batch_size": [2, 4, 16, 128],
            "optimizer.learning_rate": [3e-6, 3e-4, 1e-3],
        }
    )

    loss_sweep_options = {
        "weibull.regression_stat": ["mode", "mean"],
        "weibull.regression_error_method": ["mae", "mse"],
    }

    if loss_type == TemperedWeibullMXEFusion:
        loss_sweep_options["weibull.scale"] = pt.logspace(0, -7, base=10, steps=8)
    else:
        assert loss_type == ComplementaryWeibullMXEFusion
        loss_sweep_options["weibull.scale"] = pt.linspace(0, 1, 11)

    loss_sweeper = HyperparameterSweeper.from_dict(loss_sweep_options)

    # done this way to make a one-line tuple destructuring to capture the parameters in declared order (see below)
    loss_config_unpacker = itemgetter(*list(loss_sweeper.hyperparameters.keys()))

    
    for loss_configuration in map(nest_keys, loss_sweeper.poll_sweeps()):
    
        regression_stat, error_method, scale_factor = loss_config_unpacker(loss_configuration)
        loss_func = loss_type(regression_stat, error_method, scale_factor)
        with mlflow.start_run(
            experiment_id=create_or_set_experiment(EXPERIMENT_NAME.format(new_exp_name)), run_name=str(loss_func)
        ):
            for updates in optimizer_sweeper.poll_sweeps():
                run_config = smart_dict_upsert(RUN_CONFIG, updates)

                set_seed(metadata_to_log["data"]["global_seed_at_init"])  # NOTE: SETTING SEEDS IS IMPORTANT!

                for make_model in [
                    make_transformer_model,
                    make_rnn_model,
                ]:
                    with mlflow.start_run(nested=True):
                        dls_to_use = global_dls(run_config["optimizer"]["batch_size"])
                        output_dir = ROOT_DIR / now_dt_string()
                        model, loss_agg = dumb_run(dls_to_use, run_config, make_model, new_loss=loss_func)
                        post_run_metrics_evaluation(run_config, model, dls_to_use, output_dir, loss_agg)

# COMMAND ----------

################################################################################
#
# Execution Point
#
################################################################################

if __name__ == "__main__":
    experiment_name_ = "Weibull MSE ({}) - Due Diligence"

    single_run_comparing_scaled_loss_functions(TemperedWeibullMXEFusion, experiment_name_.format("Tempered"))
    # single_run_comparing_scaled_loss_functions(ComplementaryWeibullMXEFusion, experiment_name_.format("Complementary"))
