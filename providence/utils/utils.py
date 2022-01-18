# -*- coding: utf-8 -*-
import torch
from typing import List, Sequence, Type
import os

import pandas as pd


def is_list_of(maybe_list, T: Type) -> bool:
    is_T = lambda x: isinstance(x, T)
    return isinstance(maybe_list, (Sequence, list)) and all(map(is_T, maybe_list))

# TODO: see if we can move away from manipulating tensors directly and use numpy arrays instead
# we can get tensors out by using the ToTensor transform


def pad(data):
    """
    Padding function for variable length sequences
    This function concatenates a list of panels. The result
    will resemble something akin to the following:
    .. code-block::

            /     FEATURE2   /     FEATURE2   /     FEATURE3    /|
           /_______________ /________________/________________ / |
          /     FEATURE1   /     FEATURE1   /     FEATURE1    /| |
         /_______________ / _______________/_______________  / |/|
    T1   |   Subject1    |   Subject2    |   Subject3       | /| |
         |_______________|_______________|__________________|/ |/|
    T2   |   Subject1    |   Subject2    |   Subject3       | /| |
         |_______________|_______________|__________________|/ | |
         |               |               |                  |  | |
         ...
    :param data: List of NxM matricies
    :return: (Tensor, List[int])
    """
    lengths = [len(x) for x in data]
    num_features = data[0].shape[1:]
    dims = (max(lengths), len(data)) + num_features  # The resulting tensor will be TIME x SUBJECT X FEATURES
    padded = torch.zeros(*dims)  # initialize a zero tensor of with dimensions *dims

    if isinstance(data, torch.Tensor):
        data = data.clone().detach()
        # manual reshaping. Is there a better way to do this?
        for i, sequence in enumerate(data):
            padded[: lengths[i], i, :] = sequence
    elif is_list_of(data, torch.Tensor):
        for i, sequence in enumerate(data):
            padded[: lengths[i], i, :] = sequence.clone().detach()
    else:
        for i, sequence in enumerate(data):
            padded[: lengths[i], i, :] = torch.tensor(sequence)
    return padded, lengths


def collate_fn(batch: List[pd.DataFrame]) -> torch.Tensor:
    """
    The collate function is the callable the PyTorch dataloader uses to process the batch
    Datasets can have differing lengths, so we need to ensure each batch has a standard length

    :params batch: Dataset batch
    :returns: Padded inputs, the real sequence lengths, and padded targets

    """

    batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # we want to sort the largest array first
    inputs, targets = zip(*batch)
    inputs, inputs_lengths = pad(inputs)
    targets, _ = pad(targets)  # target sequence length is already captured by inputs_length

    return inputs, inputs_lengths, targets
