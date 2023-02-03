"""
Purpose:
    This *script* runs an analog to the FastAI learning rate finder and saves the plot to the desired output directory.
    This is straddling the line between hacky and classy, so judge carefully.

By: Stephen Fox
Adapted from work by Sylvain Gugger (https://github.com/sgugger/Deep-Learning/blob/master/Learning%20rate%20finder.ipynb)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import math
from pathlib import Path
from typing import Callable

import torch
from matplotlib import pyplot as plt
from progressbar.shortcuts import progressbar
from torch import nn, optim
from torch.utils.data import dataset
from torch.utils.data.dataloader import DataLoader

from providence.dataloaders import ProvidenceDataLoader
from providence.datasets import BackblazeDataset, BackblazeExtendedDataset
from providence.datasets.adapters import BackblazeQuarter
from providence.loss import discrete_weibull_loss_fn as providence_loss
from providence.nn import ProvidenceTransformer
from providence.training import unpack_label_and_censor
from providence.utils import now_dt_string


class ProvidenceLearningRateFinder:
    def __init__(
        self,
        net_factory: Callable[[], nn.Module],
        opt_factory: Callable[[nn.Module], optim.Optimizer],
        loss_func,
        train_dataloader: DataLoader,
        model_name: str = None,
        *,
        verbose: bool = False
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
        net = self.net_factory()
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
        for batch_num, data in progressbar(enumerate(self.trn_loader, 1)):
            self.log(f"\nBatch: {batch_num}")
            # As before, get the loss for this mini-batch of inputs/outputs
            # Sylvain's code for MNIST
            # inputs, labels = data
            # optimizer.zero_grad()
            # outputs = net(inputs)
            # loss = criterion(outputs, labels)

            # providence style, by Stephen.
            # Dropping gradient clipping in case that's restricting learning
            feats, lengths, targets = data
            optimizer.zero_grad()
            y_true, censor_ = unpack_label_and_censor(targets)

            alpha, beta = net(feats, lengths)
            loss = self.loss_func(alpha, beta, y_true, censor_, lengths)
            self.log(f"{loss.item() = }")

            # Compute the smoothed loss
            avg_loss = learning_beta * avg_loss + ((1 - learning_beta) * loss.item())
            self.log(f"{avg_loss = }")

            smoothed_loss = avg_loss / (1 - learning_beta ** batch_num)
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
            log_lrs.append(math.log10(lr))
            # Do the SGD step
            loss.backward()
            optimizer.step()
            # Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]["lr"] = lr
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


# just for reference and ease of documentation
# bs = 64, hidden = 256
# bs = 128, hidden = 512 both seem to work well.
# n_layers = 4 is genuinely more stable

if __name__ == "__main__":
    hyperparameters = {"bs": 64, "hidden_size": 64, "n_layers": 4, "n_attention_heads": 4, "dropout": 0.0, "cycles": 3, "easy-mode": True}
    # train_data = BackblazeExtendedDataset(quarters_for_download=BackblazeExtendedDataset.quarters_for_paper, train=True, failures_only=hyperparameters["easy-mode"])
    train_data = dataset.ConcatDataset([BackblazeDataset()] * hyperparameters["cycles"])

    # from scratch_ideas.BackblazeDatasetExpansion import BackblazeExtendedDataset

    # train_data = BackblazeExtendedDataset(
    #     quarters_for_download=[BackblazeQuarterName._2020_Q1, BackblazeQuarterName._2020_Q2],
    #     use_feather=True,
    #     train=True,
    #     failures_only=hyperparameters["easy-mode"],
    # )

    num_features = train_data[0][0].size(-1)  # ds.features.first.num_columns
    print(f"{len(train_data) = }")
    train_dataloader = ProvidenceDataLoader(
        dataset.ConcatDataset([train_data] * hyperparameters["cycles"]),
        shuffle=True,
        batch_size=hyperparameters["bs"],
        pin_memory=torch.cuda.is_available(),
    )

    print("Torch seed:", torch.initial_seed())

    suffix = "-".join([f"{key}={value}" for key, value in hyperparameters.items()])

    def init_learner() -> nn.Module:
        return (
            ProvidenceTransformer(
                num_features,
                hyperparameters["hidden_size"],
                hyperparameters["n_layers"],
                hyperparameters["n_attention_heads"],
                dropout=hyperparameters["dropout"],
            )
            # ProvidenceGRU(
            #     num_features,
            #     hidden_size=hyperparameters["hidden_size"],
            #     n_layers=hyperparameters["n_layers"],
            #     dropout=hyperparameters["dropout"],
            # )
        )

    lr_finder = ProvidenceLearningRateFinder(
        init_learner,
        # learning rate here doesn't matter because the first thing the loop does is reset it
        lambda net: optim.SGD(net.parameters(), 1),
        providence_loss,
        train_dataloader,
        f"{type(init_learner()).__name__}-{suffix}",
    )

    output_dir = Path("outputs", "lr-finder")
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_finder.find_and_plot_lr(output_dir)
