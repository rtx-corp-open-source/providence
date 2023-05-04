# Databricks notebook source
# MAGIC %pip install --force /dbfs/FileStore/binaries/providence-1.0.0_rc7b-py3-none-any.whl
"""
# Backblaze LSTM Recovery + Reproduction

We're just iterating to find a few good seeds.
I'm going to track this manually, just because we can get carried away with automation if the basic thing doesn't *first* work.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
# COMMAND ----------
import providence
from providence.paper_reproductions import GeneralMetrics

providence.__version__

# COMMAND ----------

from pathlib import Path
from typing import Any, DefaultDict, Dict

import torch
from providence.nn.rnn import ProvidenceLSTM
from providence_utils.callbacks import (
    CachedIntervalMetricsVisualizer,
    EarlyStopping,
    EmergencyBrake,
    LearningCurveTracker,
    ModelCheckpointer,
    NthEpochLoggerCallback,
    WriteModelOutputs,
)
from providence_utils.hyperparameter_sweeper import nest_keys
from providence_utils.trainer import Trainer
from providence.datasets.adapters import BACKBLAZE_FEATURE_NAMES, BackblazeQuarter
from providence.dataloaders import BackblazeDataLoaders
from providence.training import (
    LossAggregates,
    OptimizerWrapper,
    training_epoch,
    use_gpu_if_available,
)
from providence.utils import now_dt_string, set_seed


import mlflow

# COMMAND ----------

"""
# MLFlow Pre-work

"""

# COMMAND ----------

experiment_name = "/Users/40000889@azg.utccgl.com/ProvidenceReproductionNotebooks/Backblaze LSTM"

try:
    EXPERIMENT_ID = mlflow.create_experiment(experiment_name)
except:
    EXPERIMENT_ID = mlflow.set_experiment(experiment_name).experiment_id


# COMMAND ----------

"""
# Training Setup
"""
# COMMAND ----------

"""
## Constants
"""

# COMMAND ----------

BATCH_SIZE = 128
NUM_EPOCHS = 500

DATA_ROOT = "/dbfs/FileStore/providence-legacy-recovery"

# COMMAND ----------

# just wait. This is relevant, I swear
metadata_to_log = DefaultDict[str, Dict[str, Any]](dict)


# COMMAND ----------

"""
## Data Init

Only doing this once because the main thing is to iterate the model.
Because the experiment setup actually works, we can just stick the thing in a for-loop and let it rip for while
"""
DS_SEED = 1234
backblaze_dls = BackblazeDataLoaders(
    quarter=BackblazeQuarter._2019_Q4,
    batch_size=BATCH_SIZE,
    data_root="/dbfs/FileStore/datasets/providence",
    random_seed=DS_SEED,
)
metadata_to_log["data"]["name"] = "Backblaze"
metadata_to_log["data"]["global_seed_at_init"] = torch.initial_seed()
metadata_to_log["data"]["seed"] = DS_SEED

# COMMAND ----------

"""
## Experiment Init
"""

# COMMAND ----------

mlflow.start_run(experiment_id=EXPERIMENT_ID)

# COMMAND ----------

# torch.seed() gives a sufficiently long int, and set_seed() fixes the random state. Super seeded
set_seed(torch.seed())
metadata_to_log["general"]["seed"] = torch.initial_seed()

# COMMAND ----------

"""
## Model Init
"""

# COMMAND ----------

# run_name = "rnn-n_layers4-model_typelstm-hidden_size256-dropout0.1-lr0.01-num_epochs300-bs128-seedrandom-2021-10-08T05:15:43"
model = ProvidenceLSTM(
    len(BACKBLAZE_FEATURE_NAMES),
    hidden_size=256,
    dropout=0.1,
    device=use_gpu_if_available(),
)
metadata_to_log["model"]["input_size"] = model.input_size
metadata_to_log["model"]["hidden_size"] = model.hidden_size
metadata_to_log["model"]["num_layers"] = model.num_layers
metadata_to_log["model"]["dropout"] = model.dropout

# COMMAND ----------

"""
## Optimizer Init
"""
optimizer = OptimizerWrapper(torch.optim.SGD(model.parameters(), lr=0.01), BATCH_SIZE, NUM_EPOCHS)
metadata_to_log["optimizer"]["type"] = type(optimizer.opt).__name__
metadata_to_log["optimizer"]["batch_size"] = BATCH_SIZE
metadata_to_log["optimizer"]["num_epochs"] = optimizer.num_epochs

# COMMAND ----------

from providence.utils import configure_logger_in_dir

output_dir = Path(f"{DATA_ROOT}/backblaze-lstm/{now_dt_string()}")
output_dir.mkdir(parents=True, exist_ok=True)

logger = configure_logger_in_dir(output_dir / "logger_home", logger_name="lstm-recovery-logger")

# COMMAND ----------

"""
## Callback Init

Callbacks do a lot of work for us, so we're setting those now
"""

# COMMAND ----------

tracking_metric = "val_loss"
callbacks = [
    NthEpochLoggerCallback(10, logger=logger),
    CachedIntervalMetricsVisualizer(optimizer.num_epochs // 4, output_dir, logger=logger),
    WriteModelOutputs(optimizer.num_epochs // 2, output_dir, logger=logger),
    ModelCheckpointer(
        output_dir=(output_dir / "model-checkpoints"),
        track=tracking_metric,
        logger=logger,
        keep_old=5,
    ),
    EarlyStopping(patience=20, track=tracking_metric),
    EmergencyBrake(20, 4.0),  # our weibull loss takes a bit longer to bit below 1.0, but still begets strong results
    LearningCurveTracker(1000, output_dir=(output_dir / "")),
]
metadata_to_log["callbacks"]["epoch_logger_epochs"] = 10
metadata_to_log["callbacks"]["visualizer"] = CachedIntervalMetricsVisualizer.__name__
metadata_to_log["callbacks"]["model_output_freq"] = optimizer.num_epochs // 2
metadata_to_log["callbacks"]["n_kept_checkpoints"] = 5
metadata_to_log["callbacks"]["early_stopping_interval"] = 20
metadata_to_log["callbacks"]["ebrake_epoch"] = 20
metadata_to_log["callbacks"]["ebrake_requisite_loss"] = 4.0

mlflow.log_params(
    nest_keys(metadata_to_log, sep="/")
)  # using nest_keys() so mlflow doesn't spazz out over having dicts for values.


# COMMAND ----------

"""
# Actual Training
"""

# COMMAND ----------

metadata_to_log

# COMMAND ----------
trainer = Trainer(training_epoch)

# COMMAND ----------

try:
    losses: LossAggregates = trainer.callback_training(model, optimizer, backblaze_dls, callbacks)
    metrics_df = GeneralMetrics(model, backblaze_dls.validation_ds, losses)
    mlflow.log_metrics(metrics_df.iloc[0].to_dict())
    success = True
except:
    mlflow.end_run(status="FAILED")
    success = False
    raise

# COMMAND ----------
if success:
    try:
        mlflow.log_artifacts(str(output_dir))
    except:
        print("MLFlow sucks at its own stuff and is inconsistent in documentation")

mlflow.end_run()

# COMMAND ----------
