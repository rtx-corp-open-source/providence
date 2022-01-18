# -*- coding: utf-8 -*-
from typing import Any, Callable, List, Tuple

import torch
from torch.functional import Tensor


CensorMaskTensor = Tensor
LabelsTensor = Tensor
# There's a third argument that's a variadic of float tensors
LossFn = Callable[[LabelsTensor, CensorMaskTensor], Tensor]

def all_the_losses(loss_and_params: Tuple[LossFn, List[Any]], x_lengths, censor, y):
    loss, params = loss_and_params
    loglikelihoods = loss(y, censor, *params)

    max_length, batch_size, *trailing_dims = loglikelihoods.size()

    ranges = loglikelihoods.data.new(max_length)
    ranges = torch.arange(max_length, out=ranges)
    ranges = ranges.unsqueeze_(1).expand(-1, batch_size)

    lengths = loglikelihoods.data.new(x_lengths)
    lengths = lengths.unsqueeze_(0).expand_as(ranges)

    mask = ranges < lengths
    mask = mask.unsqueeze_(-1).expand_as(loglikelihoods)

    return -1 * torch.mean(loglikelihoods * mask.float())
