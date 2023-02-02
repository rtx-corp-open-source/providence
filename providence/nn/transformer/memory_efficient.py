"""
Still in testing. Do not use unless you're experimenting
An implementation of the memory-efficient attention mechanism shown in SELF-ATTENTION DOES NOT NEED O(n2) MEMORY (arxiv: 2112.05682)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Optional

import torch as pt
from torch import Tensor, nn
from torch.nn import Module
from torchtyping import TensorType


from .vendored_memory_effecient_attention import efficient_dot_product_attention


def _init_causal_mask(feature_dim: int, context_dim: int) -> TensorType["context_dim", "feature_dim"]:
    mask = pt.ones((context_dim, feature_dim))
    return pt.tril(mask).to(pt.bool)


class MemoryEfficientMHA(Module):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        dropout: float = 0.0,
        *,
        attention_axis="temporal",
        with_bias: bool = True
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
        self.q_proj = nn.Linear(self.embed_dim, self.attention_concat_dim, bias=self.with_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.attention_concat_dim, bias=self.with_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.attention_concat_dim, bias=self.with_bias)
        self.to_out = nn.Linear(self.attention_concat_dim, self.embed_dim, bias=self.with_bias)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(
        self,
        key: TensorType["time", "entitiy", "feature"],
        value: TensorType["time", "entitiy", "feature"],
        query: TensorType["time", "entitiy", "feature"],
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Activates the encoder, applying a causal mask

        NOTE: because this receives key=value=query in Providence, we apply all projections on the forward pass
        Output shape should match input `key` exactly
        """
        T, E, F = key.shape
        k = self.k_proj(key)   # (T, E, F) -> (T, E, C) # C = nh * F
        q = self.q_proj(query)
        v = self.v_proj(value)

        # (T, E, C) -> (T, E, nh, F)
        k = k.view(T, E, self.n_heads, self.embed_dim)
        q = q.view(T, E, self.n_heads, self.embed_dim)
        v = v.view(T, E, self.n_heads, self.embed_dim)

        # shape: (1, E, E)
        mask = _init_causal_mask(E, E).unsqueeze(0) #.tile(self.n_heads, 1, 1) # tiling might make sense, but broadcasting should be fast
        mask = mask.to(key.device)

        # input: (T, E, nh, F) -> attention: (... (T), E, nh, F)
        attended = self.dropout(efficient_dot_product_attention(q, k, v, mask, key_chunk_size=1024))
        attended = attended.view(attended.size(0), attended.size(1), self.attention_concat_dim) # (T, E, C)
        # print(f"{attended.size() = }")
        # print(f"{self.attention_concat_dim = }")

        return self.to_out(attended) # (T, E, F)
