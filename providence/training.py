# -*- coding: utf-8 -*-
import logging
from typing import Any, List, Optional, Tuple
import enum

import torch
import typer
from torch import nn, optim
from torch.utils.data import DataLoader

from providence.distributions import weibull

from providence.model import gru, lstm, rnn, transformer

from providence.utils import utils
from providence.utils.datasets import NasaDataSet, ProvidenceDataLoader
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def loss_fn(alpha, beta, y, censor, x_lengths, epsilon=1e-7) -> float:
    "Need to move this into a nicer path"
    loglikelihoods = weibull.loglike_discrete(alpha, beta, y, censor, epsilon=epsilon)

    max_length, batch_size, *trailing_dims = loglikelihoods.size()

    ranges = loglikelihoods.data.new(max_length)
    ranges = torch.arange(max_length, out=ranges)
    ranges = ranges.unsqueeze_(1).expand(-1, batch_size)

    lengths = loglikelihoods.data.new(x_lengths)
    lengths = lengths.unsqueeze_(0).expand_as(ranges)

    mask = ranges < lengths
    mask = mask.unsqueeze_(-1).expand_as(loglikelihoods)

    return -1 * torch.mean(loglikelihoods * mask.float())


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    train_data: NasaDataSet,
    test_data: NasaDataSet,
    run_suffix: Optional[str] = None,
) -> None:
    """
    Function for the training loop

    :param num_epochs: Number of training epochs
    :return: None
    """

    # One important thing to realize about using the dataloader is the collate_fn given here will
    # (1) sort the batches by decending length
    # (2) pad each sample in the batch to the longest sequence size
    # This is done in preparation for packing sequences before feeding them into our reucurrent
    # models. If you would like to have ragged batches set the value of `collate_fn` to None
    train_dataloader = ProvidenceDataLoader(train_data, shuffle=True, batch_size=100, pin_memory=torch.cuda.is_available(),)
    validation_dataloader = ProvidenceDataLoader(test_data, shuffle=True, batch_size=1000, pin_memory=torch.cuda.is_available(),)

    model = model.to(device)
    model_name = model._get_name()
    logger.info(f"Model: {model_name}")

    logger.info("Starting training loop")
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

        # Training
        train_loss = training_pass(train_dataloader, model, optimizer)

        # Validation
        val_loss = validation_pass(validation_dataloader, model)

        logger.info(f"[epoch: {epoch:03d}] training loss: {train_loss:.5f} ||| validation loss: {val_loss:.5f} ")

    run_name = model_name + (f"-{run_suffix}" if run_suffix else "")
    torch.save(model.state_dict(), f"./outputs/model-{run_name}.pt")


def unpack_label_and_censor(targets):
    y_true = targets[:, :, 0]
    y_true.unsqueeze_(-1)
    censor_ = targets[:, :, 1]
    censor_.unsqueeze_(-1)
    return y_true, censor_


def training_pass(train_dataloader, model, optimizer):
    train_loss = 0.0
    model.train()
    for batch_idx, data in enumerate(train_dataloader, 0):
        feats, lengths, targets = data
        feats = feats.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        y_true, censor_ = unpack_label_and_censor(targets)

        alpha, beta = model(feats, lengths)
        loss = loss_fn(alpha, beta, y_true, censor_, lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()
        train_loss += loss.item()
    return train_loss


def validation_pass(validation_dataloader, model):
    val_loss = 0.0
    model.eval()  # Setting model to eval mode stops training behavior such as dropout
    for batch_idx, data in enumerate(validation_dataloader, 0):
        feats, lengths, targets = data

        feats = feats.to(device)
        lengths = lengths
        targets = targets.to(device)

        # No grad since we are do not need to comptue gradients for validation
        with torch.no_grad():
            alpha, beta = model(feats, lengths)

            y_true, censor_ = unpack_label_and_censor(targets)
            loss = loss_fn(alpha, beta, y_true, censor_, lengths)
            val_loss += loss.item()

    return val_loss


class NetworkType(str, enum.Enum):
    gru = "gru"
    lstm = "lstm"
    rnn = "vanilla_rnn"
    transformer = "transformer"
    transformer_old = "transformer_old"


class OptimizerType(str, enum.Enum):
    adam = "ADAM"
    rmsprop = "RMSprop"


def _str_equals_ignorecase(s1: str, s2: str) -> bool:
    return s1.casefold() == s2.casefold()


def nn_factory(network_type: NetworkType, optimizer_type: OptimizerType) -> Tuple[nn.Module, optim.Optimizer]:
    """
    Factory function to parse cli args and return corresponding objects
    We don't need the else block here since we are enforcing the Enum type
    """
    if network_type == NetworkType.gru:
        model = gru.ProvidenceGRU
    elif network_type == NetworkType.lstm:
        model = lstm.ProvidenceLSTM
    elif network_type == NetworkType.rnn:
        model = rnn.ProvidenceVanillaRNN
    elif network_type == NetworkType.transformer:
        model = transformer.ProvidenceTransformer
    elif network_type == NetworkType.transformer_old:
        model = transformer.ReferenceProvidenceTransformer

    if optimizer_type == OptimizerType.adam:
        optimizer = optim.Adam
    elif optimizer_type == OptimizerType.rmsprop:
        optimizer = optim.RMSprop

    return model, optimizer
