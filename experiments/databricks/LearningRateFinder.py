# Databricks notebook source
# MAGIC %pip install --force /dbfs/FileStore/binaries/providence/providence-1.0.post1.dev6-py3-none-any.whl
"""
Runs an improved Learning Rate Finder on databricks

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
# COMMAND ----------
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Union

import torch

from providence.dataloaders import ProvidenceDataLoader
from providence.datasets import NasaDatasets
from providence.datasets.adapters import BACKBLAZE_FEATURE_NAMES
from providence.datasets.adapters import NASA_FEATURE_NAMES
from providence.training import set_torch_default_dtypes
from providence.training import use_gpu_if_available
from providence.type_utils import type_name
from providence.utils import now_dt_string
from providence.utils import set_seed

# COMMAND ----------

DS_NAME_TO_FEATURE_COUNT = {
    "backblaze": len(BACKBLAZE_FEATURE_NAMES),
    "bbe": len(BACKBLAZE_FEATURE_NAMES),
    "nasa": len(NASA_FEATURE_NAMES),
}

# COMMAND ----------
import providence.nn.transformer.deepmind as dm

dm._DEBUG = False

# COMMAND ----------

# MAGIC %md
# MAGIC # MLFlow Pre-work

# COMMAND ----------
# from providence_utils.mlflow import create_or_set_experiment, try_log_artifacts

# EXPERIMENT_NAME = "/Users/40000889@azg.utccgl.com/Providence-Investigative-Research/DeepMind Transformer Performance"

# EXPERIMENT_ID = create_or_set_experiment(EXPERIMENT_NAME)

ROOT_DIR = Path("/dbfs/FileStore/AIML/scratch/Providence-LR-Finding")
ROOT_DIR.mkdir(exist_ok=True, parents=True)

# COMMAND ----------

set_torch_default_dtypes(torch.float32)


# COMMAND ----------


# COMMAND ----------

# MAGIC %md
# MAGIC # Learning Rate Finder Definition

# COMMAND ----------

from math import log10

from providence.distributions import Weibull
from providence.training import unpack_label_and_censor
from matplotlib import pyplot as plt

from tqdm import tqdm  # comes in with the Databricks environment
from torch import nn, optim
from torch.utils.data import DataLoader, dataset


class ProvidenceLearningRateFinder:
    def __init__(
        self,
        net_factory: Callable[[], nn.Module],
        opt_factory: Callable[[nn.Module], optim.Optimizer],
        loss_func,
        train_dataloader: DataLoader,
        model_name: str = None,
        *,
        verbose: bool = False,
    ):
        self.net_factory = net_factory
        self.opt_factory = opt_factory
        self.loss_func = loss_func
        self.trn_loader = train_dataloader
        self.model_name = model_name
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)

    def find_lr(self, init_value=1e-8, final_value=10.0, learning_beta=0.98):
        print("Finding learning rate")
        # get our objects
        maybe_gpu = use_gpu_if_available()
        net = self.net_factory().to(maybe_gpu)
        optimizer = self.opt_factory(net)

        self.model_name = getattr(self, "model_name", net._get_name())
        print("Model instantiated")

        num = len(self.trn_loader) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        optimizer.param_groups[0]["lr"] = lr
        avg_loss = 0.0
        best_loss = 0.0
        losses = []
        log_lrs = []
        self.log("Parameters initialized. Starting First pass")
        for batch_num, data in tqdm(enumerate(self.trn_loader, 1)):
            self.log(f"\nBatch: {batch_num}")
            # providence style, by Stephen.
            feats, lengths, targets = data
            feats = feats.to(maybe_gpu)
            targets = targets.to(maybe_gpu)

            optimizer.zero_grad()
            y_true, censor_ = unpack_label_and_censor(targets)

            output_tuple = net(feats, lengths)
            wrapped = Weibull.Params(*output_tuple)
            loss = self.loss_func(wrapped, y_true, censor_, lengths)
            self.log(f"{loss.item() = }")

            # Compute the smoothed loss
            avg_loss = learning_beta * avg_loss + ((1 - learning_beta) * loss.item())
            self.log(f"{avg_loss = }")

            smoothed_loss = avg_loss / (1 - learning_beta**batch_num)
            self.log(f"{smoothed_loss = }")
            should_exit = torch.isnan(loss) or torch.isnan(torch.tensor(smoothed_loss))

            # Stop if the loss is exploding
            if batch_num > 1 and ((smoothed_loss > 40 * best_loss) or should_exit):
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(log10(lr))
            # Do the SGD step
            loss.backward()
            optimizer.step()
            # Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]["lr"] = lr
        net.to("cpu")
        return log_lrs, losses

    def find_and_plot_lr(self, output_dir: Path, init_value=1e-8, final_value=10.0, beta=0.98):
        log_lrs, losses = self.find_lr(init_value, final_value, beta)
        fig, ax = plt.subplots(figsize=(14, 10))
        # ax.plot(log_lrs[10:-5], losses[10:-5])
        ax.plot(log_lrs, losses)

        now = now_dt_string()
        fig.suptitle(f"Learning Rate Finder for {self.model_name} @ {now}")
        fig_output_path = (output_dir / f"{self.model_name}-{now}.png").as_posix()
        fig.savefig(fig_output_path)
        print("Saved found plot to", fig_output_path)


# COMMAND ----------


# COMMAND ----------

from providence.nn import ProvidenceTransformer, ProvidenceGRU
import providence.loss as prov_loss

# just for reference and ease of documentation
# bs = 64, hidden = 256
# bs = 128, hidden = 512 both seem to work well.
# n_layers = 4 is genuinely more stable
from typing import TypedDict

class LR_Hyperparams(TypedDict):
    cycles: int
    bs: int
    hidden_size: int
    n_layers: int
    dropout: float


def run_lr_finder(dir_for_pictures: str, hyperparameters: LR_Hyperparams):
    train_data, _ = NasaDatasets(data_root="/dbfs/FileStore/datasets/providence")

    print(f"{len(train_data) = }")
    train_dataloader = ProvidenceDataLoader(
        dataset.ConcatDataset([train_data] * hyperparameters["cycles"]),
        shuffle=True,
        batch_size=hyperparameters["bs"],
        pin_memory=torch.cuda.is_available(),
    )

    print("Torch seed:", torch.initial_seed())
    # NOTE: can set seed here
    set_seed(torch.initial_seed())

    suffix = "-".join([f"{key}={value}" for key, value in hyperparameters.items()])

    def init_model() -> nn.Module:
        # model = ProvidenceTransformer(
        #     train_data.n_features,
        #     hidden_size=hyperparameters["hidden_size"],
        #     n_layers=hyperparameters["n_layers"],
        #     n_attention_heads=hyperparameters["n_attention_heads"],
        #     dropout=hyperparameters["dropout"],
        #     positional_encoding_dimension=700,
        # )
        model = ProvidenceGRU(
            train_data.n_features,
            hidden_size=hyperparameters["hidden_size"],
            num_layers=hyperparameters["n_layers"],
            dropout=hyperparameters["dropout"],
        )
        # return (
        #     dm.ProvidenceBertTransformer(
        #         hyperparameters["n_attention_heads"],
        #         hyperparameters["n_layers"],
        #         train_data.n_features,
        #         hyperparameters["hidden_size"],
        #         max_seq_len=700,
        #     )
        # )
        return model

    lr_finder = ProvidenceLearningRateFinder(
        init_model,
        # learning rate here doesn't matter because the first thing the loop does is reset it
        lambda net: optim.SGD(net.parameters(), 1),
        prov_loss.discrete_weibull_loss_fn,
        train_dataloader,
        f"{type_name(init_model())}-{suffix}",
    )

    output_dir = Path(dir_for_pictures, "lr-finder")
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_finder.find_and_plot_lr(output_dir)


# COMMAND ----------

# MAGIC %md
# MAGIC # Actually running the Learning Rate Finder

# COMMAND ----------

run_lr_finder(
    dir_for_pictures=ROOT_DIR,
    hyperparameters={
        "bs": 2,
        "hidden_size": 256,
        "n_layers": 1,
        "n_attention_heads": 2,
        "dropout": 0.0,
        "cycles": 3,
    },
)
