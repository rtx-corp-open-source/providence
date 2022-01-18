# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .blocks.activation import WeibullActivation


class ProvidenceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 20, n_layers: int = 1, dropout: float = 0):
        super().__init__()

        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.output_size = 1

        self.reset_parameters()

    def reset_parameters(self):
        """
        Builds the model layers
        """

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, dropout=self.dropout)
        self.activation = WeibullActivation(self.hidden_size)

    def forward(self, x, x_lengths):
        packed = pack_padded_sequence(x, x_lengths, batch_first=False)

        rnn_outputs, _ = self.rnn(packed)

        # Packing padded sequences is a trick to speed up computation
        # for more information check out https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=False)

        output = rnn_outputs  # doing this so we can add more layers later

        alpha, beta = self.activation(output)
        return alpha, beta
