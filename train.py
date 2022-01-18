# -*- coding: utf-8 -*-
import inspect
import logging
from typing import Any, List, Tuple
import enum

import torch
import typer
from torch import nn, optim
from torch.utils.data import DataLoader


from providence.training import NetworkType, OptimizerType, nn_factory, loss_fn
from providence.utils import utils
from providence.utils.datasets import BackblazeDataset, NasaDataSet
from providence.utils import logging_utils

#### Logging ####
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] PROVIDENCE:%(levelname)s - %(name)s - %(message)s",
    handlers=[logging_utils.ch, logging_utils.fh],
)

logging.captureWarnings(True)
logger = logging.getLogger(__name__)
#################

class TrainingDataset(str, enum.Enum):
    nasa = "NASA"
    backblaze = "Backblaze"
    bbe = "BackblazeExtended"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def train(
    model_type: nn.Module, optimizer_type: optim.Optimizer, num_epochs: int, num_layers: int, learning_rate: float, dropout: float,
    dataset: TrainingDataset
) -> None:
    """
    Function for the training loop

    :param num_epochs: Number of training epochs
    :param num_layers: Number of GRU layers
    :param learning_rate: Optimizer learning rate
    :param droput: Probability of dropout between GRU layers except last
    :return: None
    """

    # A note about out training and validation data:
    # The full training dataset consists of 709 unique engines ran until failire. The validation
    # datatset consists of a new set of 707 engines also ran until failure. However the validation
    # engines fail sometime after our observations end. This means that a validation engine may
    # have feature data for 100 time steps but actually fail at time step 150. Unlike typical
    # train // test // validation splits, we are not going to use a holdout from the training
    # data for validation

    # TODO(stephen): add third path for Backblaze extended, once cleared for external release
    train_data = NasaDataSet(train=True) if dataset == TrainingDataset.nasa else BackblazeDataset(train=True)
    test_data = NasaDataSet(train=False) if dataset == TrainingDataset.nasa else BackblazeDataset(train=False)

    # One important thing to realize about using the dataloader is the collate_fn given here will
    # (1) sort the batches by decending length
    # (2) pad each sample in the batch to the longest sequence size
    # This is done in preparation for packing sequences before feeding them into our reucurrent
    # models. If you would like to have ragged batches set the value of `collate_fn` to None
    train_dataloader = DataLoader(
        train_data, shuffle=True, batch_size=100, pin_memory=torch.cuda.is_available(), collate_fn=utils.collate_fn
    )
    validation_dataloader = DataLoader(
        test_data, shuffle=True, batch_size=1000, pin_memory=torch.cuda.is_available(), collate_fn=utils.collate_fn
    )

    model = model_type(24, 120, n_layers=num_layers, dropout=dropout).to(device)
    optimizer = optimizer_type(model.parameters(), lr=learning_rate)
    logger.info(model)

    logger.info("Starting training loop")
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

        # Training
        model.train()
        for batch_idx, data in enumerate(train_dataloader, 0):
            feats, lengths, targets = data

            feats = feats.to(device)
            lengths = lengths
            targets = targets.to(device)

            optimizer.zero_grad()

            alpha, beta = model(feats, lengths)

            y_true = targets[:, :, 0]
            y_true.unsqueeze_(-1)

            censor_ = targets[:, :, 1]
            censor_.unsqueeze_(-1)

            loss = loss_fn(alpha, beta, y_true, censor_, lengths)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()  # Setting model to eval mode stops training behavior such as dropout
        for batch_idx, data in enumerate(validation_dataloader, 0):
            feats, lengths, targets = data

            feats = feats.to(device)
            lengths = lengths
            targets = targets.to(device)

            # No grad since we are do not need to comptue gradients for validation
            with torch.no_grad():
                alpha, beta = model(feats, lengths)

                y_true = targets[:, :, 0]
                y_true.unsqueeze_(-1)

                censor_ = targets[:, :, 1]
                censor_.unsqueeze_(-1)
                loss = loss_fn(alpha, beta, y_true, censor_, lengths)
                val_loss += loss.item()

        logger.info(f"[epoch: {epoch:03d}] training loss: {train_loss:.5f} ||| validation loss: {val_loss:.5f} ")
        train_loss, val_loss = 0.0, 0.0
    torch.save(model.state_dict(), "./outputs/model.pt")


def name_and_args() -> List[Tuple[str, Any]]:
    """
    Helper function to print args of the function it is called in.

    :return: Tuple of arg names and values
    """
    caller = inspect.stack()[1][0]
    args, _, _, values = inspect.getargvalues(caller)
    return [(i, values[i]) for i in args]


def main(
    network_type: NetworkType = typer.Option(NetworkType.gru, case_sensitive=False, help="Type of neural network to run 💻"),
    optimizer_type: OptimizerType = typer.Option(OptimizerType.adam, case_sensitive=False, help="Type of optimizer to use. 👩‍🏫"),
    anomaly_detection: bool = typer.Option(False, help="Enable anomaly detection for debugging. Serious performance hit if enabled. 🔎"),
    num_epochs: int = typer.Option(100, min=1, help="Number of training epochs. ⌛"),
    num_layers: int = typer.Option(4, min=1, help="Number of recurrent layers. 🔢"),
    learning_rate: float = typer.Option(1e-4, max=1e-1, help="Optimizer learning rate 👩‍🎓"),
    dropout: float = typer.Option(0.2, min=0.0, max=1.0, help="Probability of dropping outputs between layers 🍺"),
    dataset_choice: TrainingDataset = typer.Option(TrainingDataset.nasa, case_sensitive=False, help="Set this to use one of the provided datasets")
):
    logger.info(f"Training args: {name_and_args()}")  # reproducibility
    logger.info(f"Init seed: {torch.initial_seed()}")

    if anomaly_detection:
        torch.autograd.set_detect_anomaly(True)

    model, optimizer = nn_factory(network_type, optimizer_type)

    train(model, optimizer, num_epochs, num_layers, learning_rate, dropout, dataset_choice)


if __name__ == "__main__":
    typer.run(main)
