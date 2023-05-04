# -*- coding: utf-8 -*-
"""
Two implementations of a Transformer:
1. `ProvidenceTransformer`, our in-house implementation which allows arbitrary numbers of attention heads and temporal attention
2. `ReferenceProvidenceTransformer`, a thin wrapper around the PyTorch library's implementation of a Transformer, which struggles
    to support temporal attention.

In our work, we use the former.

# TODO(stephen): push down to non-__init__.py, declare __all__ for package privacy

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from torch import device
from torch import Tensor
from torch.nn import Module

from ..weibull import WeibullActivation
from .transformer import *  # noqa F403
from .transformer import PositionalEncoding
from .transformer import Transformer
from .utils import make_bit_masking_tensor


class ProvidenceTransformer(Module):
    """Custom Transformer implementation meant to work with Providence date of the shape (time, entity, feature).

    Supports changing the MultiheadedAttention implementation.
    Future versions will support different distribution activitations.

    Args:
        model_dimension (int, optional): number of input features necessary to infer on an example from the dataset.
            Defaults to 128.
        hidden_size (int, optional): size of all feedforward layers used throughout. Defaults to 512.
        n_layers (int, optional): number of layers in each (Encoder|Decoder)Block. Defaults to 2.
        n_attention_heads (int, optional): number of attention heads used in the multihead attention algorithm; thereby
            coupled to ``t_attention`` implementation. Defaults to 4.
        dropout (float, optional): percentage in [0.0, 1.0] to apply dropout. Best values depend on other parameters.
            Defaults to 0.0 (no dropout).
        layer_norm_epsilon (float, optional): epsilon fed to the LayerNorm internal to each (Encoder|Decoder)Block.
            Defaults to 1e-5.
        positional_encoding_dimension (int, optional): length of the positional encoding which decorates sequences
            Must be longer than the longest sequence in the dataset. Defaults to 2000.
        attention_axis (str, optional): defines whether attention is applied on features (like most transformers) or
            across the time axis (unique to this work and some attention-augmented RNNs). Defaults to "temporal".
        t_attention (MhaInterface, optional): the Multihead attention algorithm / implementation to use.
            Defaults to MultiheadedAttention3.
        device (_type_, optional): _description_. Defaults to device("cpu").

    Raises:
        ValueError: some implementations of ``MhaInferface`` may raise if ``n_attention_heads == 1``
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
        t_attention: MhaInterface = MultiheadedAttention3,
        device=device("cpu"),
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
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        self.transformer = Transformer(
            model_dimension=self.d_model,
            feed_forward_internal_dimension=self.ff_dimension,
            num_layers=self.n_layers,
            num_heads=self.n_attention_heads,
            dropout=self.dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
            positional_encoding_dim=self.positional_encoding_dimension,
            t_attention=self.t_attention,
            attention_axis=self.attention_axis,
        )
        self.activation = WeibullActivation(self.d_model)

    def forward(
        self,
        input: Tensor,
        input_lengths: Union[Tensor, List[int]],
        encoder_mask: Optional[Tensor] = None,
        decoder_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        """Perform a forward pass on ``input``, respecting data ``input_length``.

        Example:

            >>> alpha, beta = transformer(features, lengths, True)
            >>> # or
            >>> dist_params_tup = transformer(features, lengths, True)

        Args:
            input (Tensor): feature tensor of shape [time, entity, feature]
            input_lengths (Union[Tensor, List[int]]): a tensor / list of shape [entity]
            encoder_mask (Optional[Tensor], optional): Tensor behavior is deprecated.
                Supply a bool to utilize this feature. Defaults to None.
            decoder_mask (Optional[Tensor], optional): deprecated and ignored. Don't use. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: out = (alpha, beta) tensors for each time step i.e. ``len(out) == time``
        """
        if encoder_mask is not None:
            # encoder's aren't going to look at a bunch of nuls
            encoder_mask = None  # make_bit_masking_tensor(input_lengths, mask_offset=0).unsqueeze(2).to(input.device)
            # decoder's don't get to see the final time step
            decoder_mask = make_bit_masking_tensor(input_lengths, mask_offset=1).unsqueeze(2).to(input.device)
        # decoder_mask = make_bit_masking_tensor(input_lengths, mask_offset=1).unsqueeze(2).to(input.device)
        embedding = self.transformer(input, encoder_mask, decoder_mask)
        return self.activation(embedding)


class ReferenceProvidenceTransformer(Module):
    """The PyTorch-native Transformer implementation, made compatible with ``ProvidenceModule``.

    Exists for comparison with the ``ProvidenceTransformer``. In all our testing, this performs worse on our problem.

    Args:
        model_dimension (int, optional): number of input features necessary to infer on an example from the dataset.
            Defaults to 128.
        hidden_size (int, optional): size of all feedforward layers used throughout. Defaults to 512.
        n_layers (int, optional): number of layers in each (Encoder|Decoder)Block. Defaults to 2.
        n_attention_heads (int, optional): number of attention heads used in the multihead attention algorithm; thereby
            coupled to ``t_attention`` implementation. Defaults to 4.
        dropout (float, optional): percentage in [0.0, 1.0] to apply dropout. Best values depend on other parameters.
            Defaults to 0.0 (no dropout).
        layer_norm_epsilon (float, optional): epsilon fed to the LayerNorm internal to each (Encoder|Decoder)Block.
            Defaults to 1e-5.
        positional_encoding_dimension (int, optional): length of the positional encoding which decorates sequences
            Must be longer than the longest sequence in the dataset. Defaults to 2000.
        attention_axis (str, optional): defines whether attention is applied on features (like most transformers) or
            across the time axis (unique to this work and some attention-augmented RNNs). Defaults to "temporal".
        device (_type_, optional): _description_. Defaults to device("cpu").
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
        device=device("cpu"),
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
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
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
        """Perform a forward pass on ``input``, respecting data ``input_length``.

        Example:

            >>> alpha, beta = transformer(features, lengths, True)
            >>> # or
            >>> dist_params_tup = transformer(features, lengths, True)

        Args:
            input (Tensor): feature tensor of shape [time, entity, feature]
            input_lengths (Union[Tensor, List[int]]): a tensor / list of shape [entity]
            encoder_mask (Optional[Tensor], optional): if either this or ``decoder_mask`` is not None,
                will perform future-masking in the decoder for producing model outputs. Defaults to None.
            decoder_mask (Optional[Tensor], optional): see previous. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: out = (alpha, beta) tensors for each time step i.e. ``len(out) == time``
        """

        # we don't use the input_lengths
        if encoder_mask is not None or decoder_mask is not None:
            longest = max(input_lengths)
            decoder_mask = pt.triu(pt.ones(longest, longest), 1).to(input.device, dtype=pt.bool)
        embedding = self.transformer(input, input, src_mask=None, tgt_mask=decoder_mask)
        return self.activation(embedding)
