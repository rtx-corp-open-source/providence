# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Union

from providence.model.blocks.activation import WeibullActivation
from providence.model.blocks.transformer import Transformer
from torch import Tensor
from torch.nn import Module


class ProvidenceTransformer(Module):
    def __init__(
        self,
        model_dimension: int = 128,
        hidden_size: int = 512,
        n_layers: int = 2,
        n_attention_heads: int = 4,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.d_model = model_dimension
        self.ff_dimension = hidden_size
        self.n_layers = n_layers
        self.n_attention_heads = n_attention_heads
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.reset_parameters()

    def reset_parameters(self):
        self.transformer = Transformer(
            model_dimension=self.d_model,
            feed_forward_internal_dimension=self.ff_dimension,
            num_layers=self.n_layers,
            num_heads=self.n_attention_heads,
            dropout=self.dropout,
            layer_norm_epsilon=self.layer_norm_epsilon
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
        embedding = self.transformer(input, encoder_mask, decoder_mask)
        return self.activation(embedding)


class ReferenceProvidenceTransformer(Module):
    def __init__(
        self,
        model_dimension: int = 128,
        hidden_size: int = 512,
        n_layers: int = 2,
        n_attention_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = model_dimension
        self.ff_dimension = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_attention_heads
        self.dropout_p = dropout

        self.reset_parameters()

    def reset_parameters(self):
        from torch.nn import Transformer as PT_Transformer

        self.transformer = PT_Transformer(
            d_model=self.d_model,
            nhead=self.n_heads,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers,
            dim_feedforward=self.ff_dimension,
            dropout=self.dropout_p,
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
        embedding = self.transformer.forward(input, input, src_mask=encoder_mask, tgt_mask=decoder_mask)
        return self.activation(embedding)
