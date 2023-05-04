# -*- coding: utf-8 -*-
"""
This is an example of training a model on the NASA data, and how you might go about train-validation-test splitting by hand

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from os import path
from typing import Any
from typing import Callable
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler

from providence.dataloaders import ProvidenceDataLoader
from providence.datasets import NasaDataset
from providence.loss import discrete_weibull_loss_fn
from providence.training import use_gpu_if_available
from providence.utils import configure_logger_in_dir
from providence.utils import now_dt_string

# flake8: noqa


def train(
    model_type: Callable[[int, int, int, float], nn.Module],
    optimizer_type: Callable[[Iterable[nn.Parameter], float], optim.Optimizer],
    num_epochs: int,
    num_layers: int,
    learning_rate: float,
    dropout: float,
    data_set_number: int,
) -> None:
    """
    Function for the training loop

    :param num_epochs: Number of training epochs
    :param num_layers: Number of GRU layers
    :param learning_rate: Optimizer learning rate
    :param dropout: Probability of dropout between GRU layers except last
    :param data_set_number: Number of the dataset to specify 001, 002, 003, 004 subset
    :return: None
    """
    # A note about out training and validation data:
    # The FD001 training dataset consists of 100 unique engines ran until failire. The test
    # datatset consists of a new set of 100 engines also ran until failure. However the validation
    # engines fail sometime after our observations end. This means that a validation engine may
    # have feature data for 100 time steps but actually fail at time step 150.
    # Each engine has the same single fault mode and same set of flight conditions.

    # Instantiate these here so we can use them for graphing and debugging later
    logger = configure_logger_in_dir(f"./train_single-runs/{now_dt_string()}/")
    # TODO (mike): fix this up, follow the PyTorch image dataset example
    use_local_data = False  # some hacking here to expedite local debugging
    local_path = path.join(path.expanduser("~"), "public-datasets/CMAPSSData")
    if use_local_data:
        train_data = NasaDataset(train=True, data_path=local_path, data_set_number=data_set_number)
        # test_data = NasaDataset(train=False, data_path=local_path, data_set_number=data_set_number)
    else:
        train_data = NasaDataset(train=True, data_set_number=data_set_number)
        # test_data = NasaDataset(train=False, data_set_number=data_set_number)

    validation_split = 0.3
    shuffle_dataset = True
    random_seed = 42
    logger.info(f"validation split: {validation_split}")
    logger.info(f"numpy random seed for train/validation split: {random_seed}")

    dataset_size = len(train_data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]
    logger.info(f"validation set indices: {valid_indices}")

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    # One important thing to realize about using the dataloader is the collate_fn given here will
    # (1) sort the batches by decending length
    # (2) pad each sample in the batch to the longest sequence size
    # This is done in preparation for packing sequences before feeding them into our reucurrent
    # models. If you would like to have ragged batches set the value of `collate_fn` to None
    train_dataloader = ProvidenceDataLoader(
        train_data,
        batch_size=100,
        pin_memory=torch.cuda.is_available(),
        sampler=train_sampler,
    )
    validation_dataloader = ProvidenceDataLoader(
        train_data,
        batch_size=100,
        pin_memory=torch.cuda.is_available(),
        sampler=valid_sampler,
    )

    device = use_gpu_if_available()

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

            loss = discrete_weibull_loss_fn(alpha, beta, y_true, censor_, lengths)
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
                loss = discrete_weibull_loss_fn(alpha, beta, y_true, censor_, lengths)
                val_loss += loss.item()

        logger.info(f"[epoch: {epoch:03d}] training loss: {train_loss:.5f} ||| validation loss: {val_loss:.5f} ")
        train_loss, val_loss = 0.0, 0.0
    torch.save(model.state_dict(), "./outputs/model.pt")
