"""
This module is an attempt to a faithful reproduction of the DeepMind Transformer, per "Formal Algorithms for Transformers" (Phuong and Hutter, 2022).
ArXiv link https://arxiv.org/abs/2207.09238

This are not necessarily the optimal performance. More on that later.

**Raytheon Technologies proprietary**
Export controlled - see license file

Discussion: Is an (attention) encoder a stand-alone thing or is it a wrapper around a number of layers, like in the PyTorch implementation?

PyTorch only does that because the encoding doesn't change the dimensionality of the sequence, though this is afforded
by the generality of the algorithmic framework from DeepMind. Therefore it is reasonably inappropriate to add an additional
layer of abstraction: a design, grokking / understanding-to-interpret, and performance impediment.

The same applies to the Decoder portion, then.


Discussion: masking the feature for time series predictions
According to the DeepMind algorithms paper, using the upper right triangular matrix (and the subsequent negation)
precludes the final sequence/time step from being ingested by the model and quote:
> the output V[:, 1:t] only depends on X[:,1:t], hence can be used to predict X[:, t + 1]

As such, infering for the future - as a forecast of one time step - should really only involve the X[:, t+1]
(potentially wrapped on the outer dimension). Extended time series forecasting would involve treating the Transformer
effectively as a recurrent neural network: infer t + 1, use as input for t + 2, repeating until you've got the forecast distance you require.

While the language of the document refers to textual language processing, but it can be mapped directly to time series. Refering to the penultimate
paragraph of pg.5
1. token currently being predicted: an input feature (potentially embedded)
2. is mapped to a query vector
3. the token in the context (in time series the same sequence) are mapped to key and value vectors
4. the inner product q^{T} k is the degree to which the input feature is import for predicting the current query
- They are used to derive a distribution over the context (the source time series), which is then combined with the value vectors

As such, we see if the mapped attention over the time series generates attention for a given time step's relevance to predicting the current
feature input. Thus, we may be able to meaningfully predict the future from a retrospective, temporal attention.
"""
import enum
from dataclasses import dataclass
from math import sqrt
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

from jaxtyping import Bool
from jaxtyping import Float
import torch as pt
from torch.nn import Dropout
from torch.nn import functional as F
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import ModuleList
from typeguard import typechecked

from providence.nn.rnn import get_activation
from providence.types import LengthsTensor
from providence.types import ProvidenceTensor
from providence_utils.fastai_torch_core import Module
from providence_utils.fastai_utils import delegates

batch_time_embedding = "batch time embedding"
batch_time_features = "batch time features"
time_batch_1 = "... time batch 1"
time_embedding = "... time embedding"
time_n_context_in = "... time n_context_in"
time_n_features = "... time n_features"
time_value_dim = "... time value_dim"
context_dim_feature_dim = "context_dim feature_dim"
n_context_in_n_features_in = "n_context_in n_features_in"
_batch_time_1 = "*batch time 1"


class MaskMode(enum.Enum):
    """Mode or method of masking Tensors for attention applications.

    Because the attention mask must be matched to the features, and checking an enum `every` time ``Module.forward``
    is invoked is slow, we use this method to have clarity as to how to compute it directly and usually in-advance
    """

    unmasked = enum.auto()
    backward_only = enum.auto()
    bidirectional = unmasked
    unidirectional = backward_only
    cross_attention = unmasked


def _init_mask(feature_dim: int, context_dim: int, mode: MaskMode) -> Bool[pt.Tensor, context_dim_feature_dim]:
    mask = pt.ones((context_dim, feature_dim))
    if mode == MaskMode.unmasked:  # or bi-directional attention
        return mask
    else:
        return pt.triu(mask)


def _inverse_bit_mask(mask: pt.Tensor) -> pt.Tensor:
    return (~pt.tril(mask).to(pt.bool)).to(pt.int)


_DEBUG = True


def log_debug(*args):
    """Basic logging to stdout, silenced if debugging constant set to False."""
    if _DEBUG:
        print(*args)


def print_prop(x, prop: str) -> Any:
    """Prints property ``prop`` of ``x``.

    If you are familiar with Javascript (where all objects are dictionaries), behaves as ``x[prop]``

    Args:
        x (Any): non-None object
        prop (str): property to lookup from ``x``

    Returns:
        Any: the value of the property on ``x``

    Raises:
        AttributeError: if no such ``prop`` exists on ``x``
    """
    item = getattr(x, prop, None)
    if callable(item):
        item = item()
    log_debug(item)
    return x


class AttentionHead(Module):
    """Basic "soft attention" algorithm from the DeepMind paper.

    Has similar performance characteristics to the nn.transformer.BasicAttention module, but is more configurable
    as implied by the DeepMind specification.

    Args:
        n_features_in (int): number of features in input Tensor(s)
        n_context_in (int): number of context in input Tensor(s)
        att_out_dim (int): output dimension for the attention head. Often the same as ``n_features_in``.
        mask (MaskMode): the kind of mask to use.
        inner_att_dim (int, optional): _description_. Defaults to 32.
    """

    def __init__(
        self,
        n_features_in: int,
        n_context_in: int,
        att_out_dim: int,
        mask: Union[  # type: ignore[name-defined]
            MaskMode, Bool[pt.Tensor, n_context_in_n_features_in]
        ],  # NOTE(stephen): I don't think you can pass in a single mask at this level.
        *,
        inner_att_dim: int = 32,
    ):
        self.n_features_in, self.n_context_in, self.att_out_dim, self.inner_att_dim = (
            n_features_in,
            n_context_in,
            att_out_dim,
            inner_att_dim,
        )
        self._mask_check = mask
        self.reset_parameter()

    def reset_parameter(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        # NOTE(stephen): it's almost certain that this would be more quickly done with a linear model
        self.W_q = pt.nn.parameter.Parameter(pt.randn(self.inner_att_dim, self.n_features_in))
        self.b_q = pt.nn.parameter.Parameter(pt.randn(self.inner_att_dim, 1))
        self.W_k = pt.nn.parameter.Parameter(pt.randn(self.inner_att_dim, self.n_context_in))
        self.b_k = pt.nn.parameter.Parameter(pt.randn(self.inner_att_dim, 1))
        self.W_v = pt.nn.parameter.Parameter(pt.randn(self.att_out_dim, self.n_context_in))
        self.b_v = pt.nn.parameter.Parameter(pt.randn(self.att_out_dim, 1))

    def forward(
        self,
        X: Float[pt.Tensor, time_n_features],
        Z: Float[pt.Tensor, time_n_context_in],
    ) -> Float[pt.Tensor, time_value_dim]:
        """Apply single-head attention in a batched fashion.

        Internally we transpose to make the temporal dimension the axis of attention, leaning heavily into the
        Providence story. In a more generic usage, you can think that we attend over the penultimate dimension.
        NOTE: Should work on 2-D inputs, but has not been thoroughly tested.

        Args:
            X (pt.Tensor): input sequence of shape ``[batch, time, n_features_in]``
            Z (pt.Tensor): context sequence of shape ``[batch, time, n_context_in]``

        Returns:
            pt.Tensor: shape of [... (batch), time, value]
        """
        inverted_mask = _inverse_bit_mask(_init_mask(X.size(-2), Z.size(-2), mode=self._mask_check))
        # NOTE(stephen): need to be lined up on the same device, with the correct masking type.
        # to mask with bools is to say "yes / no" to a given index. to mask with longs is to say pt.cat([tens[i] for i in longs])
        inverted_mask = inverted_mask.to(X.device, dtype=pt.bool)

        # log_debug(inverted_mask)

        # change dimensionality to match the paper, also putting time on the last dimension.
        X = X.transpose(-1, -2)  # ..., F_in, time
        Z = Z.transpose(-1, -2)  # ..., C_in, time

        # produce the query, key, and value
        # query = print_prop(self.W_q @ X, 'shape') + self.b_q
        # [inner_att_dim, F_in] @ [F_in, time] = [inner_att_dim, time]
        query = self.W_q @ X + self.b_q

        # NOTE(stephen): this manually performed {xW^T + b} is probaly slower than it should be, but this is what's in core PyTorch...
        # [inner_att_dim, C_in] @ [C_in, time] = [inner_att_dim, time]
        key = self.W_k @ Z + self.b_k

        # [att_out_dim, C_in] @ [C_in, time] = [att_out_dim, time]
        value = self.W_v @ Z + self.b_v

        # scores = print_prop(key.transpose(-1, -2), "size") @ query  # similarities over the time access
        # key.transpose: [inner_att_dim, time] -> [time, inner_att_dim]
        # key.T @ query: [time, inner_att_dim] @ [inner_att_dim, time] -> [time, time]
        scores = key.transpose(-1, -2) @ query / sqrt(self.inner_att_dim)  # similarities over the time access
        # log_debug(f"{scores.size() = }")
        scores = scores.masked_fill(inverted_mask, -pt.inf)
        attended = F.softmax(scores, dim=-1)  # scores over the time dimension

        # log_debug(f"{attended = }")
        # log_debug(f"{attended.size() = }")

        # [att_out_dim, time] @ [time, time]
        att_weighted = value @ attended
        # att_weighted = print_prop(value @ attended, "size")
        return att_weighted.transpose(-1, -2)  # [att_out_dim, time] -> [time, att_out_dim]


class MHAttention(Module):
    """Multi-head attention literally implemented according to Formal Algorithms for Transformers.

    Performs a slow multi-headed attention, but is surprisingly faster than the implementation in
    providence.nn.transformer

    Args:
        n_heads (int): number of attention heads. Wall-clock time scales linearly with this feature.
        n_features_in (int): number of features in input Tensors
        n_context_in (int): number of features in context Tensor
        att_out_dim (int): the number of features in the output Tensor
        mask_mode (MaskMode): mode or method of masking, passed to each AttentionHead.
        att_inner_dim (int, optional): Inner dimension of the AttentionHead, for the keys, queries, values.
            Defaults to 128.
        dropout (float, optional): Percentage of dropout to apply to mapped-out attention values. Defaults to 0.1.
        output_dim (Optional[int], optional): number of output attention features. Often desirable to match to number
            of input features. Defaults to None.
    """

    def __init__(
        self,
        n_heads: int,
        n_features_in: int,
        n_context_in: int,
        att_out_dim: int,
        mask_mode: MaskMode,
        *,
        att_inner_dim: int = 128,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
    ):
        (
            self.n_heads,
            self.n_features_in,
            self.n_context_in,
            self.mask_mode,
            self.att_out_dim,  # TODO(stephen): swap this name with the att_out_dim of the AttentionHead
            self.att_inner_dim,
            self.dropout_p,
            self.output_dim,
        ) = (
            n_heads,
            n_features_in,
            n_context_in,
            mask_mode,
            att_out_dim,
            att_inner_dim,
            dropout,
            output_dim,
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        # self.mask = _init_mask(self.n_features_in, self.n_context_in, self.mask_mode)  # dimensionality of the mask
        self.attention_heads = ModuleList(
            [
                AttentionHead(
                    self.n_features_in,
                    self.n_context_in,
                    self.att_out_dim,
                    self.mask_mode,
                    inner_att_dim=self.att_inner_dim,
                )
                for _ in range(self.n_heads)
            ]
        )
        self.output_dim = self.output_dim or self.att_out_dim
        self.cat_to_out = Linear(self.n_heads * self.att_out_dim, self.output_dim)
        self.dropout = Dropout(self.dropout_p)

    def forward(
        self,
        X: Float[pt.Tensor, time_n_features],
        Z: Float[pt.Tensor, time_n_context_in],
    ) -> Float[pt.Tensor, time_value_dim]:
        """Perform multi-head attention pass on input ``X`` and context ``Z``.

        Args:
            X (pt.Tensor): input sequence of shape ``[batch, time, n_features_in]``
            Z (pt.Tensor): context sequence of shape ``[batch, time, n_context_in]``

        Returns:
            pt.Tensor: output size is [batch, time, out] where out = {
                att_inter_output_dim, if output_dim is None
                output_dim          , if otherwise
            }
        """
        attentions = [ah(X, Z) for ah in self.attention_heads]
        ys = pt.cat(attentions, dim=-1)
        out = self.cat_to_out(ys)
        # TODO(stephen): apply dropout
        return out


@dataclass
class CSAConfig:
    """Configuration for initializing Causal Self Attention.

    Limited reach in terms of parameters because Karpathy's code is littered with global variables.

    Args:
        block_size (int, optional): block size, typically the max sequence length to be ingested. Defaults to 1024
        n_head (int, optional): number of attention heads. Even numbers are better. Defaults to 12
        n_embed (int, optional): size of the internal attention matrix. Even numbers are best. Defaults to 768
        dropout (float, optional): dropout rate ∈ [0.0, 1.0] Defaults to 0.1
        mask_mode (MaskMode): masking strategy for the causal self-attention instance.
            Defaults to MaskMode.backward_only, to discourage cheating across time series data.
    """

    block_size: int = 1024
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    mask_mode: MaskMode = MaskMode.backward_only


class CausalSelfAttention(Module):
    """Causal, Soft- Self-Attention.

    Adapted from nanoGPT, Andrej Karpathy's tutorial for a GPT-2 (XL)-compatible implementation towards GPT.
    Source: https://github.com/karpathy/nanoGPT/blob/e0c689cf38478eea9416757cec5f834620983862/model.py#L102

    Args:
        CSAConfig: containing parameter settings for this instance
    """

    def __init__(self, config: CSAConfig):
        assert (
            div_check := config.n_embd % config.n_head
        ) == 0, f"({config.n_embd=}) % {(config.n_head)} == {div_check}"
        # key, query, value projections for all heads, but in a batch
        # NOTE: 3 is hard-coded to represent one chunk for each of - 'query', 'key', and 'value'.
        # The coupling of the number of heads and the embedding dimension is to support a simultaneous computation of the
        # query, key, and value vector , ather than doing three smaller matrix multiplies, you do them all at once
        # If you go to the q = k "Performer", switch it 2... This feels like bad programming, but whatever. < 300 lines for a GPT is epic.
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        # NOTE: needs to be the maximum sequence length. Not a big deal, just need to know it.
        con_bs = config.block_size
        _mask = pt.ones(con_bs, con_bs)
        self.register_buffer(
            "bias",
            (pt.tril(_mask) if config.mask_mode == MaskMode.backward_only else _mask).view(1, 1, con_bs, con_bs),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    @typechecked
    def forward(self, x: Float[pt.Tensor, batch_time_embedding]) -> Float[pt.Tensor, batch_time_embedding]:
        """Apply causal self-attention on input ``x``.

        Args:
            x (pt.Tensor): input Tensor of shape ``[batch, time, embedding]``

        Returns:
            Tensor: attention on ``x`` with shape ``[batch, time, embedding]``
        """
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        embedded_in = self.c_attn(x)
        # print(f"{embedded_in.shape = }")
        splitup = embedded_in.split(self.n_embd, dim=2)
        # print(f"{len(splitup) = }")
        # print(f"Split shapes: {[(t.size(i)) for t in splitup for i in range(len(t))]}")
        q, k, v = splitup
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    @property
    def output_dim(self) -> int:
        """The output dimension of this model.

        Returns:
            int: output dimension
        """
        return self.n_embd


class Decoder(Module):
    """Standard decoder of the primary (i.e. input) sequence. See Algorithm 8 in the paper for more.

    Standard decoder uses back-only masking for the encoder portion and full-attention on the self-attention on the decoder.

    Args:
        n_heads (int): number of attention heads
        n_features_in (int): number of features in the input Tensors
        att_inner_dim (int, optional): Inner dimension for the attention matrix. Defaults to 256.
        ff_dim (int, optional): shape of the feedforward (pt.nn.Linear) layer's output. Defaults to 256.
        activation_func (Union[Callable[[pt.Tensor], pt.Tensor], pt.nn.Module], optional): desired nonlinearity
            between the penultimate and ultimate linear layers. Defaults to F.relu.
        dropout (float, optional): Percentage of dropout to apply to mapped-out attention values. Defaults to 0.1.
        max_seq_len (int, optional): The maximum length of a sequence in the dataset this instance will decode.
            In other words, the length of the longest sequence in the dataset. Defaults to 1024.
    """

    def __init__(
        self,
        n_heads: int,
        n_features_in: int,
        *,
        att_inner_dim: int = 256,
        ff_dim: int = 256,
        activation_func=F.relu,
        dropout=0.1,
        max_seq_len: int = 1024,
    ):

        self.n_heads, self.n_features_in = n_heads, n_features_in
        self.att_inner_dim, self.ff_dim, self.max_seq_len = (
            att_inner_dim,
            ff_dim,
            max_seq_len,
        )
        self.dropout_p = dropout
        self.mha_inner_dim = n_heads * att_inner_dim
        self.activation_func = activation_func
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        embedding_dim = self.n_features_in
        config = CSAConfig(
            n_head=self.n_heads,
            n_embd=self.mha_inner_dim,
            dropout=self.dropout_p,
            block_size=self.max_seq_len,
        )

        self.mha_self = pt.nn.Sequential(
            Linear(self.n_features_in, self.mha_inner_dim),
            CausalSelfAttention(config),
            Linear(self.mha_inner_dim, self.n_features_in),
        )
        self.mha_encoder_mem = MHAttention(
            self.n_heads,
            self.n_features_in,
            n_context_in=self.n_features_in,
            att_out_dim=self.n_features_in,
            mask_mode=MaskMode.unidirectional,
            att_inner_dim=self.mha_inner_dim,
        )
        self.layer_norm1, self.layer_norm2, self.layer_norm3 = (
            LayerNorm(embedding_dim),
            LayerNorm(embedding_dim),
            LayerNorm(embedding_dim),
        )
        self.inner_mlp = Linear(embedding_dim, self.ff_dim)
        self.outer_mlp = Linear(self.ff_dim, embedding_dim)

    def forward(
        self,
        X: Float[pt.Tensor, batch_time_embedding],
        Z: Float[pt.Tensor, batch_time_embedding],
    ) -> Float[pt.Tensor, batch_time_embedding]:
        """Perform the decoding forward pass as explained in Algorithm 8 of the paper.

        Args:
            X (pt.Tensor): input sequence of shape ``[batch, time, embedding]``
            Z (pt.Tensor): context sequence of shape ``[batch, time, embedding]``

        Returns:
            pt.Tensor: tensor of shape exactly matching ``X`` and ``Z``
        """
        ln1 = self.layer_norm1(X)
        X = X + self.mha_self(ln1)
        ln2 = self.layer_norm2(X)
        X = X + self.mha_encoder_mem(ln2, Z)
        ln3 = self.layer_norm3(X)
        X = X + self.outer_mlp(self.activation_func(self.inner_mlp(ln3)))
        return ln3


class Encoder(Module):
    """Standard uni- or bi-directional encoder (controlled the ``mask_mode``).

    If you want to have differing primary and context sequences, use another implementation in this file.

    Args:
        n_heads (int): number of attention heads
        n_features_in (int): number of features in the input Tensors
        mask_mode (MaskMode): method of masking the sequence encoded. Variable (unlike the Decoder) because the best
            masking strategy changes per problem you're trying to solve.
        att_inner_dim (int, optional): Inner dimension for the attention matrix. Defaults to 256.
        ff_dim (int, optional): shape of the feedforward (pt.nn.Linear) layer's output. Defaults to 256.
        activation_func (Union[Callable[[pt.Tensor], pt.Tensor], pt.nn.Module], optional): desired nonlinearity
            between the penultimate and ultimate linear layers. Defaults to F.relu.
        dropout (float, optional): Percentage of dropout to apply to mapped-out attention values. Defaults to 0.1.
        max_seq_len (int, optional): The maximum length of a sequence in the dataset this instance will decode.
            In other words, the length of the longest sequence in the dataset. Defaults to 1024.
    """

    def __init__(
        self,
        n_heads: int,
        n_features_in: int,
        mask_mode: MaskMode,
        *,
        att_inner_dim: int = 256,
        ff_dim: int = 256,
        activation_func=F.gelu,
        dropout=0.1,
        max_seq_len: int = 1024,
    ):
        self.n_heads, self.n_features_in, self.mask_mode = (
            n_heads,
            n_features_in,
            mask_mode,
        )
        self.ff_dim = ff_dim
        # att_dim_per_head = input_dim // n_heads
        self.mha_inner_dim = n_heads * att_inner_dim
        self.max_seq_len = max_seq_len

        self.activation_func = activation_func
        self.dropout_p = dropout
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        # FIXME: if the transformer is going to map to something that we're compatible with, why all the consternation with the dimension shapes???
        config = CSAConfig(
            n_head=self.n_heads,
            n_embd=self.mha_inner_dim,
            mask_mode=self.mask_mode,
            dropout=self.dropout_p,
            block_size=self.max_seq_len,
        )

        self.mha = pt.nn.Sequential(
            Linear(self.n_features_in, self.mha_inner_dim),
            CausalSelfAttention(config),
            Linear(self.mha_inner_dim, self.n_features_in),
        )
        pseudo_embedding_dim = self.n_features_in
        self.layer_norm1, self.layer_norm2 = LayerNorm(pseudo_embedding_dim), LayerNorm(pseudo_embedding_dim)
        self.inner_mlp = Linear(pseudo_embedding_dim, self.ff_dim)
        self.outer_mlp = Linear(self.ff_dim, pseudo_embedding_dim)

    @typechecked
    def forward(self, X: Float[pt.Tensor, time_embedding]) -> Float[pt.Tensor, time_embedding]:
        """Perform encoding forwmard pass per algorithm 9 of the paper.

        Args:
            X (pt.Tensor): input tensor of shape ``[batch, time, embedding]`` for encoding.

        Returns:
            pt.Tensor: shape exactly matching ``X``
        """
        ln1 = self.layer_norm1(X)
        X = X + self.mha(ln1)
        ln2 = self.layer_norm2(X)
        X = X + self.outer_mlp(self.activation_func(self.inner_mlp(ln2)))
        return X


class EDTransformer(Module):
    """Standard encoder-decoder Transformer, per Algorithm 8 in the Formal Algorithms for Transformers (arXiv:2207.09238).

    In practice, X = Z and X_lengths = Z_lengths, so that is what we recommend you supply when you invoke the model.
    (It's what we do.)
    We also have an embedding into and out of the Transformer's internal state represented by two pt.nn.Linear layers.

    Args:
        n_heads (int): number of attention heads in both the encoder and decoder.
            NOTE: Currently coupled at parity, but some work with Transformers-as-GANs suggest using different n_heads
        n_layers (int): number of encoders applied to the context before passing to the same number of decoders
            NOTE: like the number of heads, there's reason to split this parameter. More testing is to be done.
        n_features_in (int): number of features in the input Tensors to the model
        n_embedding (int): dimension into- and out of the internal state of the Transformer. Features will be mapped
            through this embedding.
        mask_mode (int): mask mode for the encoder portion of the network
        max_seq_len (int): The maximum length of a sequence in the dataset this instance will decode.
            In other words, the length of the longest sequence in the dataset.
            Must be set by the user because the datasets vary by problem.
        dropout (float, optional): Percentage of dropout to apply to mapped-out attention values. Defaults to 0.1.
        att_inner_dim (int, optional): dimension of the inner-most attention matrix. Defaults to 256.
        ff_dim (int, optional): shape of the feedforward (pt.nn.Linear) layer's output. Defaults to 256.
    """

    def __init__(
        self,
        n_heads: int,
        n_layers: int,
        n_features_in: int,
        n_embedding: int,
        mask_mode: MaskMode,
        *,
        max_seq_len: int,
        dropout: float = 0.1,
        att_inner_dim: int = 256,
        ff_dim: int = 256,
    ):
        (
            self.n_heads,
            self.n_layers,
            self.n_features_in,
            self.embedding_in,
            self.mask_mode,
        ) = (n_heads, n_layers, n_features_in, n_embedding, mask_mode)
        self.max_seq_len, self.att_inner_dim, self.ff_dim, self.dropout_p = (
            max_seq_len,
            att_inner_dim,
            ff_dim,
            dropout,
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        # NOTE: for speed, we change the init of the positional_encoding: swap the indexing, add the 1. Same effect, less ops per forward()
        self.positional_encoding = pt.nn.parameter.Parameter(
            pt.randn(
                self.max_seq_len + 1,  # we'll key into this with lengths that are effectively 1-based. Hence, (+1)
                1,
                self.embedding_in,
            )
        )
        print(f"{self.positional_encoding.size() = }")
        self.embedder = Linear(self.n_features_in, self.embedding_in)
        self.unembedder = Linear(self.embedding_in, self.n_features_in, bias=False)
        self.encoders = pt.nn.ModuleList(
            [
                Encoder(
                    self.n_heads,
                    self.embedding_in,
                    self.mask_mode,
                    att_inner_dim=self.att_inner_dim,
                    dropout=self.dropout_p,
                    ff_dim=self.ff_dim,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.decoders = pt.nn.ModuleList(
            [
                Decoder(
                    self.n_heads,
                    self.embedding_in,
                    att_inner_dim=self.att_inner_dim,
                    ff_dim=self.ff_dim,
                    dropout=self.dropout_p,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(
        self,
        X: Float[pt.Tensor, batch_time_features],
        Z: Float[pt.Tensor, batch_time_features],
        X_lengths: LengthsTensor,
        Z_lengths: LengthsTensor,
    ) -> Float[pt.Tensor, batch_time_features]:
        """Encoder-Decoder transformer inference, using ``lengths`` to select out the positional encoding

        Args:
            X (pt.Tensor): input sequence, which will be condition on the context sequence
            Z (pt.Tensor): context sequence,
            X_lengths (pt.Tensor): length of each entity in ``X`` (pre-computed so we don't have to, each time)
            Z_lengths (pt.Tensor): lengths for each entity in ``Y``

        Returns:
            pt.Tensor: Attention across X on the features dimension, conditioned on Z
        """
        X_, Z_ = self.embedder(X), self.embedder(Z)

        Z_ += self.positional_encoding[Z_lengths, :]
        for enc in self.encoders:
            Z_ = enc(Z_)

        X_ += self.positional_encoding[X_lengths, :]
        for dec in self.decoders:
            X_ = dec(X_, Z_)

        return F.softmax(self.unembedder(X_), dim=-1)


class ETransformer(Module):
    """BERT-esque, encoder-only Transformer.

    The only real differences between this and the EDTransformer are
    1. the positional encoding for the Transformer,
    2. a plurality of Encoders,

    Note: an encoder-only Transformer was used for the TST work, both original implementation and TSAI's port.

    Args:
        n_heads (int): number of attention heads in the encoder
        n_layers (int): number of encoders applied to the input
        n_features_in (int): number of features in the input Tensors to the model
        n_embedding (int): dimension into the internal state of the Transformer. Features will be mapped
            through this embedding.
        mask_mode (int): mask mode for the encoder portion of the network
        max_seq_len (int): The maximum length of a sequence in the dataset this instance will decode.
            In other words, the length of the longest sequence in the dataset.
            Must be set by the user because the datasets vary by problem.
        dropout (float, optional): Percentage of dropout to apply to mapped-out attention values. Defaults to 0.1.
        att_inner_dim (int, optional): dimension of the inner-most attention matrix. Defaults to 256.
        ff_dim (int, optional): shape of the feedforward (pt.nn.Linear) layer's output. Defaults to 256.
        final_dim (int, optional): dimension of the output of the internal state of the Transformer, before returning
            to the ``n_features_in`` as the last dimension of the output. Useful for bottlenecking
    """

    def __init__(
        self,
        n_heads: int,
        n_layers: int,
        n_features_in: int,
        n_embedding: int,
        mask_mode: MaskMode,
        *,
        max_seq_len: int,
        dropout: float = 0.1,
        att_inner_dim: int = 256,
        ff_dim: int = 256,
        final_dim: int = 128,
    ):
        (
            self.n_heads,
            self.n_layers,
            self.n_features_in,
            self.embedding_in,
            self.mask_mode,
        ) = (n_heads, n_layers, n_features_in, n_embedding, mask_mode)
        self.max_seq_len, self.att_inner_dim, self.ff_dim, self.final_dim = (
            max_seq_len,
            att_inner_dim,
            ff_dim,
            final_dim,
        )
        self.dropout_p = dropout
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        # NOTE: for speed, we change the init of the positional_encoding: swap the indexing, add the 1. Same effect, less ops per forward()
        self.positional_encoding = pt.nn.parameter.Parameter(
            pt.randn(
                self.max_seq_len + 1,  # we'll key into this with lengths that are effectively 1-based. Hence, (+1)
                1,
                self.embedding_in,
            )
        )
        print(f"{self.positional_encoding.size() = }")
        self.embedder = Linear(self.n_features_in, self.embedding_in)
        self.final_proj = Linear(self.embedding_in, self.final_dim)
        self.final_norm = LayerNorm(self.final_dim)
        self.unembedder = Linear(self.final_dim, self.n_features_in, bias=False)
        self.encoders = pt.nn.ModuleList(
            [
                Encoder(
                    self.n_heads,
                    self.embedding_in,
                    self.mask_mode,
                    att_inner_dim=self.att_inner_dim,
                    dropout=self.dropout_p,
                    ff_dim=self.ff_dim,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(
        self, X: Float[pt.Tensor, batch_time_features], lengths: LengthsTensor
    ) -> Float[pt.Tensor, _batch_time_1]:
        """Encoder-only transformer inference, using ``lengths`` to select out the positional encoding.

        Follows Algorithm 9 from the paper

        Args:
            X (pt.Tensor): input tensor of shape ``[batch, time, features]`` for encoding.

        Returns:
            pt.Tensor: shape exactly matching ``X``
        """
        X_in = self.embedder(X)
        # log_debug(f"{X_in.size() = }")
        pos_enc = self.positional_encoding[lengths, :]
        # log_debug(f"{self.positional_encoding.size() = }")
        # log_debug(f"{pos_enc.size() = }")
        out = X_in + pos_enc
        # TODO: dropout around out at this point.
        for encoder in self.encoders:
            out = encoder(out)
        out = F.gelu(self.final_proj(out))
        out = self.final_norm(out)
        out = F.softmax(self.unembedder(out), dim=-1)
        return out


@delegates(ETransformer.__init__)
class ProvidenceBertTransformer(Module):
    """BERT-style Transformer that is adheres to the ProvidenceModule Protocol.

    Args:
        n_heads (int): number of attention heads in the encoder
        n_layers (int): number of encoders applied to the input
        n_features_in (int): number of features in the input Tensors to the model
        n_embedding (int): dimension into the internal state of the Transformer. Features will be mapped
            through this embedding.
        mask_mode (int, optional): mask mode for the encoder portion of the network. Defaults to ``backward_only``.
        max_seq_len (int): The maximum length of a sequence in the dataset this instance will decode.
            In other words, the length of the longest sequence in the dataset.
            Must be set by the user because the datasets vary by problem.
        device (pt.device): Torch device to use for training this model. Defaults to pt.device("cpu").
        activation (str): Name of the Providence activation to use with this model. Defaults to "weibull".
        dropout (float, optional): Percentage of dropout to apply to mapped-out attention values. Defaults to 0.1.
        att_inner_dim (int, optional): dimension of the inner-most attention matrix. Defaults to 256.
        ff_dim (int, optional): shape of the feedforward (pt.nn.Linear) layer's output. Defaults to 256.
        final_dim (int, optional): dimension of the output of the internal state of the Transformer, before returning
            to the ``n_features_in`` as the last dimension of the output. Useful for bottlenecking
    """

    def __init__(
        self,
        n_heads: int,
        n_layers: int,
        n_features_in: int,
        n_embedding: int,
        *,
        max_seq_len: int,
        mask_mode: MaskMode = MaskMode.backward_only,
        device=pt.device("cpu"),
        activation="weibull",
        **kwargs,
    ):
        self._initial_args = (
            n_heads,
            n_layers,
            n_features_in,
            n_embedding,
            mask_mode,
            device,
            kwargs,
        )
        (
            self.n_heads,
            self.n_layers,
            self.n_features_in,
            self.n_embedding,
            self.max_sequence_length,
            self.mask_mode,
            self.dist_name,
        ) = (
            n_heads,
            n_layers,
            n_features_in,
            n_embedding,
            max_seq_len,
            mask_mode,
            activation,
        )
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        self._initial_args[-1]["max_seq_len"] = self.max_sequence_length
        self.transformer = self.default = ETransformer(
            self.n_heads,
            self.n_layers,
            self.n_features_in,
            self.n_embedding,
            self.mask_mode,
            **self._initial_args[-1],
        )
        self.activation = get_activation(self.dist_name)

    def forward(self, X: ProvidenceTensor, lengths: LengthsTensor) -> Tuple[Float[pt.Tensor, time_batch_1], ...]:
        """Perform the full forward pass of Algorithm 9., extended to support Providence's inference paradigm.

        Args:
            X (pt.Tensor): input Tensor of shape ``[time, batch, features]``
            lengths (pt.Tensor): shape ``[batch]``

        Returns:
            Tuple[pt.Tensor, ...]: n-tensors with a final dimension of 1
        """
        embedded_and_attended = self.transformer(X.transpose(0, 1), lengths)
        return self.activation(embedded_and_attended.transpose(0, 1))


@delegates(EDTransformer.__init__)
class ProvidenceDeepMindTransformer(Module):
    """Transformer implemented according to Algorithm 8 in the paper.

    Args:
        n_heads (int): number of attention heads in both the encoder and decoder.
        n_layers (int): number of encoders applied to the context before passing to the same number of decoders
        n_features_in (int): number of features in the input Tensors to the model
        n_embedding (int): dimension into- and out of the internal state of the Transformer. Features will be mapped
            through this embedding.
        mask_mode (int): mask mode for the encoder portion of the network
        max_seq_len (int): The maximum length of a sequence in the dataset this instance will decode.
            In other words, the length of the longest sequence in the dataset.
            Must be set by the user because the datasets vary by problem.
        device (pt.device): Torch device to use for training this model. Defaults to pt.device("cpu").
        activation (str): Name of the Providence activation to use with this model. Defaults to "weibull".
        dropout (float, optional): Percentage of dropout to apply to mapped-out attention values. Defaults to 0.1.
        att_inner_dim (int, optional): dimension of the inner-most attention matrix. Defaults to 256.
        ff_dim (int, optional): shape of the feedforward (pt.nn.Linear) layer's output. Defaults to 256.

    """

    def __init__(
        self,
        n_heads: int,
        n_layers: int,
        n_features_in: int,
        n_embedding: int,
        *,
        mask_mode: MaskMode = MaskMode.backward_only,
        device=pt.device("cpu"),
        activation="weibull",
        **kwargs,
    ):
        self._initial_args = (
            n_heads,
            n_layers,
            n_features_in,
            n_embedding,
            mask_mode,
            device,
            kwargs,
        )
        (
            self.n_heads,
            self.n_layers,
            self.n_features_in,
            self.n_embedding,
            self.mask_mode,
            self.dist_name,
        ) = (n_heads, n_layers, n_features_in, n_embedding, mask_mode, activation)
        self._xtra = ["max_seq_len", "att_inner_dim", "ff_dim", "final_dim"]
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        self.default = self.transformer = EDTransformer(
            self.n_heads,
            self.n_layers,
            self.n_features_in,
            self.n_embedding,
            self.mask_mode,
            **self._initial_args[-1],
        )
        self.activation = get_activation(self.dist_name)

    def forward(self, X: ProvidenceTensor, lengths: LengthsTensor) -> Tuple[Float[pt.Tensor, time_batch_1], ...]:
        """Perform the full forward pass of Algorithm 8, extended to support Providence's inference paradigm.

        Args:
            X (pt.Tensor): input Tensor of shape ``[time, batch, features]``
            lengths (pt.Tensor): shape ``[batch]``

        Returns:
            Tuple[pt.Tensor, ...]: n-tensors with a final dimension of 1
        """

        X_T = X.transpose(0, 1)
        embedded_and_attended = self.transformer(X_T, X_T, lengths, lengths)
        return self.activation(embedded_and_attended.transpose(0, 1))
