"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from collections import defaultdict
from logging import getLogger
from providence.datasets.backblaze import BackblazeDataset
from time import perf_counter
from typing import Type
from pandas import DataFrame

import torch
import typer
from torch import nn, optim

from providence.training import OptimizerType, nn_factory, train
from providence.utils import name_and_args

logger = getLogger(__name__)




def construct_transformer(
    model_ctor: Type[nn.Module],
    optimizer_ctor: Type[optim.Optimizer],
    hidden_size: int = 512,
    n_layers: int = 2,
    n_attention_heads: int = 4,
    learning_rate: float = 3e-3,
    dropout: float = 0.0,
    n_features: int = 24,
):

    model = model_ctor(
        model_dimension=n_features, hidden_size=hidden_size, n_layers=n_layers, n_attention_heads=n_attention_heads, dropout=dropout
    )
    optimizer = optimizer_ctor(model.parameters(), lr=learning_rate)
    return model, optimizer


def main(
    optimizer_type: OptimizerType = typer.Option(OptimizerType.adam, case_sensitive=False, help="Type of optimizer to use. 👩‍🏫"),
    anomaly_detection: bool = typer.Option(False, help="Enable anomaly detection for debugging. Serious performance hit if enabled. 🔎"),
    num_epochs: int = typer.Option(100, min=1, help="Number of training epochs. ⌛"),
    num_layers: int = typer.Option(4, min=1, help="Number of recurrent layers. 🔢"),
    feedforward_dim: int = typer.Option(64, min=1, help="Number of neurons in the transformer architectures. 🔢"),
    num_attention_heads: int = typer.Option(2, min=1, help="Number of attention heads in the transformer architectures. 🔢"),
    learning_rate: float = typer.Option(1e-4, max=1e-1, help="Optimizer learning rate 👩‍🎓"),
    dropout: float = typer.Option(0.2, min=0.0, max=1.0, help="Probability of dropping outputs between layers 🍺"),
    laps: int = typer.Option(3, min=2, max=10, help="Number of times to re-run a model through epochs. For collecting statistics"),
):
    logger.info(f"Training args: {name_and_args()}")  # reproducibility
    logger.info(f"Init seed: {torch.initial_seed()}")

    if anomaly_detection:
        torch.autograd.set_detect_anomaly(True)

    runs = defaultdict(list)

    # A note about out training and validation data:
    # The full training dataset consists of 709 unique engines ran until failire. The validation
    # datatset consists of a new set of 707 engines also ran until failure. However the validation
    # engines fail sometime after our observations end. This means that a validation engine may
    # have feature data for 100 time steps but actually fail at time step 150. Unlike typical
    # train // test // validation splits, we are not going to use a holdout from the training
    # data for validation

    # Instantiate these here so we can use them for graphing and debugging later
    # train_data_set, test_data_set = NasaDataSet(train=True), NasaDataSet(train=False)
    train_data_set, test_data_set = BackblazeDataset(train=True, use_feather=True), BackblazeDataset(train=False, use_feather=True)

    for network_type in ["transformer_old", "transformer"]:
        model_ctor, optimizer_ctor = nn_factory(network_type, optimizer_type)
        for lap in range(1, laps + 1):
            model, optimizer = construct_transformer(
                model_ctor,
                optimizer_ctor,
                n_features=len(BackblazeDataset.numerical_features),
                hidden_size=feedforward_dim,
                n_layers=num_layers,
                n_attention_heads=num_attention_heads,
                learning_rate=learning_rate,
                dropout=dropout,
            )
            start = perf_counter()
            train(model, optimizer, num_epochs, train_data_set, test_data_set, run_suffix=f"lap{lap:02}")
            run_time = perf_counter() - start

            runs["times"].append(run_time)
            runs["types"].append(network_type)
            runs["avg_epoch_time"].append(run_time / num_epochs)

    logger.info("Generated runs dictionary")
    logger.info(runs)
    DataFrame(runs).to_csv("benchmark.csv")


if __name__ == "__main__":
    typer.run(main)
