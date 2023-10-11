"""
The Time Series Perceiver network, courtesy of TSAI, pulled September 21st, 2022

This code was pulled under an Apache 2.0 license. See the LICENSE in this directory.
(Originally, the Apache 2.0 license here - https://github.com/timeseriesAI/tsai/blob/b66bf0eeb32be9cc4e42b30456f68104d341f377/LICENSE#L1)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
__all__ = ["TSPerceiver", "Attention"]


from typing import Optional
from providence_utils.fastai_layers import GACP1d, GAP1d
from providence_utils.fastai_torch_core import Module

from providence_utils.fastai_utils import ifnone

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# Internal Cell
class ScaledDotProductAttention(Module):
    def __init__(self, d_k: int, res_attention: bool = False):
        self.d_k, self.res_attention = d_k, res_attention

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Input shape:
            q               : [bs x n_heads x q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_k]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [seq_len x seq_len]

        Output shape:
            context: [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
        """
        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        scores = torch.matmul(q, k)  # scores : [bs x n_heads x q_len x seq_len]

        # Scale
        scores = scores / (self.d_k**0.5)

        # Add previous scores (optional)
        if prev is not None:
            scores = scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                scores.masked_fill_(attn_mask, float("-inf"))
            else:
                scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # key_padding_mask with shape [bs x seq_len]
            scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # SoftMax
        attn = F.softmax(scores, dim=-1)  # attn   : [bs x n_heads x q_len x seq_len]

        # MatMul (attn, v)
        context = torch.matmul(attn, v)  # context: [bs x n_heads x q_len x d_v]

        if self.res_attention:
            return context, attn, scores
        else:
            return context, attn


class Attention(Module):
    def __init__(
        self,
        d_latent: int,
        d_context: Optional[int] = None,
        n_heads: int = 8,
        d_head: Optional[int] = None,
        attn_dropout: float = 0.0,
        res_attention: bool = False,
    ):
        d_context = ifnone(d_context, d_latent)
        n_heads = ifnone(n_heads, 1)
        d_head = ifnone(d_head, d_context // n_heads)

        self.scale = d_head**-0.5
        self.n_heads, self.d_head, self.res_attention = n_heads, d_head, res_attention

        self.to_q = nn.Linear(d_latent, d_head * n_heads, bias=False)
        self.to_kv = nn.Linear(d_context, d_head * n_heads * 2, bias=False)

        self.attn = ScaledDotProductAttention(d_k=d_head, res_attention=res_attention)

        self.to_out = nn.Sequential(nn.Linear(d_head * n_heads, d_latent), nn.Dropout(attn_dropout))

    def forward(self, x, context=None, mask=None):
        h, d = self.n_heads, self.d_head
        bs = x.shape[0]
        q = self.to_q(x).view(bs, -1, h, d).transpose(1, 2)
        context = ifnone(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.view(bs, -1, h, d).permute(0, 2, 3, 1)
        v = v.view(bs, -1, h, d).transpose(1, 2)

        if self.res_attention:
            x, _, scores = self.attn(q, k, v)
        else:
            x, _ = self.attn(q, k, v)
        x = x.permute(0, 2, 1, 3).reshape(bs, -1, h * d)

        x = self.to_out(x)
        if self.res_attention:
            return x, scores
        else:
            return x


class GEGLU(Module):
    def forward(self, x: Tensor):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Sequential):
    def __init__(self, dim, mult=2, dropout=0.0):
        layers = [nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim)]
        # layers = [nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim)]
        super().__init__(*layers)


class CrossAttention(Module):
    def __init__(self, d_latent, d_context=None, n_heads=8, d_head=None, attn_dropout=0.0, fc_dropout=0.0):
        d_context = ifnone(d_context, d_latent)
        self.norm_latent = nn.LayerNorm(d_latent)
        self.norm_context = nn.LayerNorm(d_context) if d_context is not None else None
        self.attn = Attention(d_latent, d_context=d_context, n_heads=n_heads, d_head=d_head, attn_dropout=attn_dropout)
        self.norm_ff = nn.LayerNorm(d_latent)
        self.ff = FeedForward(d_latent, dropout=fc_dropout)

    def forward(self, x, context=None, mask=None):
        x = self.norm_latent(x)
        if context is not None:
            context = self.norm_context(context)
        context = ifnone(context, x)
        x = self.attn(x, context)
        x = self.norm_ff(x)
        x = self.ff(x)
        return x


class LatentTransformer(Module):
    def __init__(self, d_latent, n_heads=8, d_head=None, attn_dropout=0.0, fc_dropout=0.0, self_per_cross_attn=1):
        self.layers = nn.ModuleList()
        for _ in range(self_per_cross_attn):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(d_latent),
                        Attention(d_latent, n_heads=n_heads, d_head=d_head, attn_dropout=attn_dropout),
                        nn.LayerNorm(d_latent),
                        FeedForward(d_latent, dropout=fc_dropout),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn_norm, att, ff_norm, ff in self.layers:
            x = attn_norm(x)
            x = att(x)
            x = ff_norm(x)
            x = ff(x)
        return x


# Cell
class TSPerceiver(Module):
    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        cat_szs=0,
        n_cont=0,
        n_latents=512,
        d_latent=128,
        d_context=None,
        n_layers=6,
        self_per_cross_attn=1,
        share_weights=True,
        cross_n_heads=1,
        self_n_heads=8,
        d_head=None,
        attn_dropout=0.0,
        fc_dropout=0.0,
        concat_pool=False,
    ):
        d_context = ifnone(d_context, d_latent)

        # Embedding
        self.to_ts_emb = nn.Linear(c_in, d_context)
        self.to_cat_emb = nn.ModuleList([nn.Embedding(s, d_context) for s in cat_szs]) if cat_szs else None
        self.to_cont_emb = nn.ModuleList([nn.Linear(1, d_context) for i in range(n_cont)]) if n_cont else None

        self.latent_array = nn.Parameter(torch.zeros(1, n_latents, d_context))  # N = q_len = indices = n_latents

        # Positional encoding
        # NOTE(stephen): it came with these comments
        # self.ts_pos_enc = nn.Parameter(torch.zeros(1, 1, d_context))
        # self.cat_pos_enc = nn.Parameter(torch.zeros(1, 1, d_context)) if cat_szs else None
        # self.cont_pos_enc = nn.Parameter(torch.zeros(1, 1, d_context)) if n_cont else None
        self.ts_pos_enc = nn.Parameter(torch.zeros(1, 1, 1))
        self.cat_pos_enc = nn.Parameter(torch.zeros(1, 1, 1)) if cat_szs else None
        self.cont_pos_enc = nn.Parameter(torch.zeros(1, 1, 1)) if n_cont else None
        # self.pos_enc = nn.Parameter(torch.zeros(1, seq_len + (len(cat_szs) if cat_szs else 0) + n_cont, d_context))
        pos_enc = (
            torch.linspace(-1, 1, seq_len + (len(cat_szs) if cat_szs else 0) + n_cont)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(1, 1, d_context)
        )
        self.pos_enc = nn.Parameter(pos_enc, requires_grad=False)

        # Cross-attention & Latent-transformer
        self.self_per_cross_attn = self_per_cross_attn
        self.attn = nn.ModuleList()
        for i in range(n_layers):
            if i < 2 or not share_weights:
                attn = [
                    CrossAttention(
                        d_latent,
                        d_context=d_context,
                        n_heads=cross_n_heads,
                        d_head=d_head,
                        attn_dropout=attn_dropout,
                        fc_dropout=fc_dropout,
                    )
                ]
                if self_per_cross_attn != 0:
                    attn += [
                        LatentTransformer(
                            d_latent,
                            n_heads=self_n_heads,
                            d_head=d_head,
                            attn_dropout=attn_dropout,
                            fc_dropout=fc_dropout,
                            self_per_cross_attn=self_per_cross_attn,
                        )
                    ]
            self.attn.append(nn.ModuleList(attn))

        self.head = nn.Sequential(
            GACP1d() if concat_pool else GAP1d(),
            nn.BatchNorm1d(d_latent * (1 + concat_pool)),
            nn.Linear(d_latent * (1 + concat_pool), c_out),
        )

    def forward(self, x):
        # Embedding
        # Time series
        if isinstance(x, tuple):
            x_ts, (x_cat, x_cont) = x
        else:
            x_ts, x_cat, x_cont = x, None, None
        context = self.to_ts_emb(x_ts.transpose(1, 2))
        context += self.ts_pos_enc
        # Categorical
        if self.to_cat_emb is not None:
            x_cat = torch.cat([e(x_cat[:, i]).unsqueeze(1) for i, e in enumerate(self.to_cat_emb)], 1)
            x_cat += self.cat_pos_enc
            context = torch.cat([context, x_cat], 1)
        # Continuous
        if self.to_cont_emb is not None:
            x_cont = torch.cat([e(x_cont[:, i].unsqueeze(1).unsqueeze(2)) for i, e in enumerate(self.to_cont_emb)], 1)
            x_cont += self.cont_pos_enc
            context = torch.cat([context, x_cont], 1)
        context += self.pos_enc

        # Latent array
        x = self.latent_array.repeat(context.shape[0], 1, 1)

        # Cross-attention & Latent transformer
        for i, attn in enumerate(self.attn):
            x = attn[0](x, context=context) + x  # cross-attention
            if self.self_per_cross_attn != 0:
                x = attn[1](x) + x  # latent transformer

        x = x.transpose(1, 2)

        # Head
        out = self.head(x)
        return out
