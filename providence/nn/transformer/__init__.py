# -*- coding: utf-8 -*-
"""
Two implementations of a Transformer:
1. `ProvidenceTransformer`, our in-house implementation which allows arbitrary numbers of attention heads and temporal attention
2. `ReferenceProvidenceTransformer`, a thin wrapper around the PyTorch library's implementation of a Transformer, which struggles
    to support temporal attention.

In our work, we use the former.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from typing import List, Optional, Tuple, Union

from ..weibull import WeibullActivation
from .transformer import PositionalEncoding, Transformer
from .transformer import *
from .utils import make_bit_masking_tensor
from torch import device, Tensor
from torch.nn import Module


class ProvidenceTransformer(Module):
    """Custom Transformer implementation meant to work with Providence date of the shape (time, entity, feature).
    Supports changing the MultiheadedAttention implementation.

    Future versions will support different distribution activitations.
    """
    def __init__(
        self,
        model_dimension: int = 128,
        hidden_size: int = 512,
        n_layers: int = 2,
        n_attention_heads: int = 4,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        positional_encoding_dimension: int = 2000,
        *,
        attention_axis="temporal",
        t_attention: MhaInterface=MultiheadedAttention3,
        device = device('cpu')
    ):
        super().__init__()
        self.d_model = model_dimension
        self.ff_dimension = hidden_size
        self.n_layers = n_layers
        self.n_attention_heads = n_attention_heads
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.positional_encoding_dimension = positional_encoding_dimension
        self.attention_axis = attention_axis
        self.t_attention = t_attention
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        self.transformer = Transformer(
            model_dimension=self.d_model,
            feed_forward_internal_dimension=self.ff_dimension,
            num_layers=self.n_layers,
            num_heads=self.n_attention_heads,
            dropout=self.dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
            positional_encoding_dim=self.positional_encoding_dimension,
            t_attention=self.t_attention,
            attention_axis=self.attention_axis
        )
        self.activation = WeibullActivation(self.d_model)

    def forward(
        self,
        input: Tensor,
        input_lengths: Union[Tensor, List[int]],
        encoder_mask: Optional[Tensor] = None,
        decoder_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        if encoder_mask is not None:
            # encoder's aren't going to look at a bunch of nuls
            encoder_mask = None #  make_bit_masking_tensor(input_lengths, mask_offset=0).unsqueeze(2).to(input.device)
            # decoder's don't get to see the final time step
            decoder_mask = make_bit_masking_tensor(input_lengths, mask_offset=1).unsqueeze(2).to(input.device)
        # decoder_mask = make_bit_masking_tensor(input_lengths, mask_offset=1).unsqueeze(2).to(input.device)
        embedding = self.transformer(input, encoder_mask, decoder_mask)
        return self.activation(embedding)


class ReferenceProvidenceTransformer(Module):
    """A wrapper around the Pytorch-shipped Transformer to make it compatible with Providence's inference interface."""
    def __init__(
        self,
        model_dimension: int = 128,
        hidden_size: int = 512,
        n_layers: int = 2,
        n_attention_heads: int = 4,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        positional_encoding_dimension: int = 2000,
        *,
        device = device('cpu'),
    ):
        super().__init__()
        self.d_model = model_dimension
        self.ff_dimension = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_attention_heads
        self.dropout_p = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.positional_encoding_dimension = positional_encoding_dimension
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        from torch.nn import Transformer as PT_Transformer

        self.pe = PositionalEncoding(self.d_model, max_len=self.positional_encoding_dimension)
        self.transformer = PT_Transformer(
            d_model=self.d_model,
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            dim_feedforward=self.ff_dimension,
            dropout=self.dropout_p,
            layer_norm_eps=self.layer_norm_epsilon,
        )

        self.activation = WeibullActivation(self.d_model)

    def forward(
        self,
        input: Tensor,
        input_lengths: Union[Tensor, List[int]],
        encoder_mask: Optional[Tensor] = None,
        decoder_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        # we don't use the input_lengths
        if encoder_mask is not None or decoder_mask is not None:
            longest = max(input_lengths)
            decoder_mask = pt.triu(pt.ones(longest, longest), 1).to(input.device, dtype=pt.bool)
        embedding = self.transformer(input, input, src_mask=None, tgt_mask=decoder_mask)
        return self.activation(embedding)
