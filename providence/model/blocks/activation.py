# -*- coding: utf-8 -*-
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F


class WeibullActivation(nn.Module):
    """
    Activation layer for a two parameter Weibull

    :param input_size: The size of a network's last hidden layer

    :return: Alpha and Beta tensors for the Weibull distribution
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.input_size: int = input_size
        self.output_size: int = 1

        self.reset_parameters()

    def reset_parameters(self):
        self.alpha = nn.Linear(self.input_size, self.output_size)
        self.beta = nn.Linear(self.input_size, self.output_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # taken from https://github.com/ragulpr/wtte-rnn/blob/162f5c17f21db79a316d563b60835d178142fd69/python/wtte/wtte.py#L31

        alpha = self.alpha(x)
        alpha = torch.exp(alpha)

        beta = self.beta(x)
        beta = F.softplus(beta)

        return alpha, beta
