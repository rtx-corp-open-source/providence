"""
Still in testing. Do not use unless you're experimenting
An implementation of the memory-efficient attention mechanism shown in SELF-ATTENTION DOES NOT NEED O(n2) MEMORY (arxiv: 2112.05682)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Optional

from jaxtyping import Bool
import torch as pt
from torch import nn
from torch import Tensor
from torch.nn import Module

from providence.types import ProvidenceTensor

from .vendored_memory_effecient_attention import efficient_dot_product_attention


def _init_causal_mask(feature_dim: int, context_dim: int) -> Bool[pt.Tensor, "n_context n_feature"]:
    mask = pt.ones((context_dim, feature_dim))
    return pt.tril(mask).to(pt.bool)


class MemoryEfficientMHA(Module):
    """Memory-Efficient Attention made to comply with the ProvidenceModule interface

    Args:
        n_heads (int): number of heads in the efficient, multihead attention
        embed_dim (int): input dimension, to embed into the attention internal weights
        dropout (float, optional): percentage as a float in [0.0, 1.0]. Defaults to 0.0.
        attention_axis (str, optional): currently ignored. TODO(stephen): address this. Defaults to "temporal".
        with_bias (bool, optional): whether the linear layers use a bias term. Mild speed-ups can be had if set
            to False, but the field doesn't have a firm stance one way or the other. Defaults to True.
    """

    def __init__(
        self, n_heads: int, embed_dim: int, dropout: float = 0.0, *, attention_axis="temporal", with_bias: bool = True
    ):

        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.attention_concat_dim = n_heads * embed_dim
        self.dropout_p = dropout
        self.attention_axis = attention_axis
        self.with_bias = with_bias

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        self.q_proj = nn.Linear(self.embed_dim, self.attention_concat_dim, bias=self.with_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.attention_concat_dim, bias=self.with_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.attention_concat_dim, bias=self.with_bias)
        self.to_out = nn.Linear(self.attention_concat_dim, self.embed_dim, bias=self.with_bias)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(
        self,
        key: ProvidenceTensor,
        value: ProvidenceTensor,
        query: ProvidenceTensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Activates the encoder, applying a causal mask

        NOTE: because this receives key=value=query in Providence, we apply all projections on the forward pass

        Args:
            key (Tensor): some transformation of the input, shape expected to match query and value
            value (Tensor): some transformation of the input, shape expected to match key and query
            query (Tensor): some transformation of the input, shape expected to match key and value
            mask (Tensor, optional): square mask matching the last dimension of the input

        Returns:
            Tensor: attention, where shape should match input `key` exactly
        """
        T, E, F = key.shape
        k = self.k_proj(key)  # (T, E, F) -> (T, E, C) # C = nh * F
        q = self.q_proj(query)
        v = self.v_proj(value)

        # (T, E, C) -> (T, E, nh, F)
        k = k.view(T, E, self.n_heads, self.embed_dim)
        q = q.view(T, E, self.n_heads, self.embed_dim)
        v = v.view(T, E, self.n_heads, self.embed_dim)

        # shape: (1, E, E)
        # NOTE # .tile(self.n_heads, 1, 1) # tiling might make sense, but broadcasting should be similar in speed + scalable
        mask = _init_causal_mask(E, E).unsqueeze(0)
        mask = mask.to(key.device)

        # input: (T, E, nh, F) -> attention: (... (T), E, nh, F)
        attended = self.dropout(efficient_dot_product_attention(q, k, v, mask, key_chunk_size=1024))
        attended = attended.view(attended.size(0), attended.size(1), self.attention_concat_dim)  # (T, E, C)
        # print(f"{attended.size() = }")
        # print(f"{self.attention_concat_dim = }")

        return self.to_out(attended)  # (T, E, F)
