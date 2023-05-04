"""
WIP: An iteration on the MSE experiment, but now with the DeepMind models on display

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from collections import defaultdict
from pathlib import Path

import mlflow
import torch as pt

from providence.dataloaders import CustomProvidenceDataloaders
from providence.dataloaders import ProvidenceDataLoader
from providence.datasets import NasaDatasets
from providence.datasets.adapters import DS_NAME_TO_FEATURE_COUNT
from providence.loss import DiscreteWeibullMSELoss
from providence.nn import ProvidenceGRU
from providence.nn import ProvidenceModule
from providence.nn import ProvidenceTransformer
from providence.paper_reproductions import GeneralMetrics
from providence.training import OptimizerWrapper
from providence.training import set_torch_default_dtypes
from providence.training import training_epoch
from providence.training import use_gpu_if_available
from providence.type_utils import type_name
from providence.utils import now_dt_string
from providence.utils import set_seed
from providence_utils.hyperparameter_sweeper import HyperparameterSweeper
from providence_utils.hyperparameter_sweeper import nest_keys
from providence_utils.mlflow import create_or_set_experiment
from providence_utils.mlflow import try_log_artifacts

################################################################################
#
# MLFLow Setup
#
################################################################################


def mlflow_epoch(dls, model, optimizer, *, step: int = None, loss_criterion=None):
    if loss_criterion is not None:
        losses = training_epoch(dls, model, optimizer, loss_criterion=loss_criterion)
    else:
        losses = training_epoch(dls, model, optimizer)

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

GLOBAL_train_ds, GLOBAL_val_ds = NasaDatasets(
    data_root="/dbfs/FileStore/datasets/providence",
)
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


def reproducibility_proof_():
    # this run started after the set_seed changed to return to setting the rng state
    experiment_id = create_or_set_experiment()

    fixed_seed = 11545829402875347079
    for cfg in HyperparameterSweeper(
        n_heads=[2, 4, 6, 8],
        n_layers=[1, 2],
    ).poll_sweeps():
        for i in range(2):
            ...


from providence.training import LossAggregates

# trainer = Trainer(mlflow_epoch)

run_config = {
    "model": {
        "n_heads": 2,
        "n_layers": 1,
        "n_features": DS_NAME_TO_FEATURE_COUNT["nasa"],
        "n_embedding": 64,  # down from 128
        "max_seq_len": 700,
    },
    "optimizer": {
        "learning_rate": 3e-2,
        "batch_size": 64,  # down from 128
        "num_epochs": 80,
    },
    "dataset": "nasa",
}

# NOTE: just two function to keep things straight fortard


def make_transformer_model() -> ProvidenceModule:
    model = ProvidenceTransformer(
        DS_NAME_TO_FEATURE_COUNT["nasa"],
        hidden_size=run_config["model"]["n_embedding"],
        n_layers=run_config["model"]["n_layers"],
        n_attention_heads=run_config["model"]["n_heads"],
        dropout=0,
        positional_encoding_dimension=710,
        device=use_gpu_if_available(),
    )
    return model


def make_rnn_model() -> ProvidenceModule:
    return ProvidenceGRU(
        DS_NAME_TO_FEATURE_COUNT["nasa"],
        run_config["model"]["n_embedding"],
        run_config["model"]["n_layers"],
        dropout=0,
        device=use_gpu_if_available(),
    )


for make_model in [make_transformer_model, make_rnn_model]:
    new_loss = DiscreteWeibullMSELoss()

    with mlflow.start_run(experiment_id=create_or_set_experiment(EXPERIMENT_NAME.format("MSE Check"))):
        set_seed(pt.seed())
        # model = dm.ProvidenceBertTransformer(2, 1, DS_NAME_TO_FEATURE_COUNT["nasa"], 128, max_seq_len=710, device=use_gpu_if_available())
        model = make_model()

        opt_wrapper = OptimizerWrapper(
            pt.optim.Adam(
                model.parameters(),
                lr=run_config["optimizer"]["learning_rate"],
            ),
            batch_size=run_config["optimizer"]["batch_size"],
            num_epochs=run_config["optimizer"]["num_epochs"],
        )
        dls_to_use = global_dls()
        output_dir = ROOT_DIR / now_dt_string()

        # training-eval loop
        model.to(model.device)
        loss_agg = LossAggregates([], [])
        for epoch_i in range(1, opt_wrapper.num_epochs + 1):
            epoch_losses = mlflow_epoch(
                dls_to_use,
                model,
                opt_wrapper.opt,
                step=epoch_i,
                loss_criterion=new_loss,
            )
            loss_agg.append_losses(epoch_losses)

        # model.to('cpu')

        print("Generating metrics", now_dt_string())
        dls_to_use.test_ds.device = model.device
        dls_to_use.test_ds.use_device_for_iteration(True)
        model_metrics = GeneralMetrics(model, dls_to_use.test_ds, loss_agg)
        print("Generating metrics completed", now_dt_string())

        merged_config = dict(**run_config, **metadata_to_log)
        nested_config = {k: (type_name(v) if isinstance(v, type) else v) for (k, v) in nest_keys(merged_config).items()}
        nested_config["model.type"] = type_name(model)

        mlflow.log_params(nested_config)

        mlflow.log_metrics(model_metrics.iloc[0].to_dict())
        try_log_artifacts(output_dir)
