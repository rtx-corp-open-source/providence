"""
The main goal of this implementation of Transformer is
1. reasonable code that's not terribly slow.
2. a Transformer architecture / attention head that readily support our work's "temporal attention".

So we're still using PyTorch, but we're OOP at the problem, to make the code more readable. I aimed for simplistic
forward() functions, that read like the paper(s) suggest. The production implementation may use some of these atoms,
but "flattened" for performance.
- For instance, the implementation is intrinsically more memory-heavy because of the number of objects created.
  - These aren't all explicitly necessary, but for educational purposes it will do.
- Again, for production/performance matters, only part of the below implementation would be kept: that which allows
  us to divorce the feature (and model) dimension from the parity of the number of transformer heads. That's a bad
  coupling to make and we should have a way around it.

To extend this system to the Time-to-event prediction paradigm, take the Transformer as is, allow it to attend to
  the batch or sequence axis - rather than just the feature axis - and then make sure your attention mask is compatible.

Alternative (though not simpler) implementations can be found at the following
[0]: Timeseries transformer in Keras, from Google Research
    - https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py
[1]: The Annotated Transformer
    - http://nlp.seas.harvard.edu/2018/04/01/attention.html
[2]: Google Research's Trax implementation of the DecoderBlock
    - https://github.com/google/trax/blob/4a519156f2d1ad091e93303b443c8d6c8ba8a474/trax/models/transformer.py#L103


All the parts and components that can be assembled to a complete transformer model.
Final implementation is at the bottom.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import copy
from typing import Literal
from typing import Optional
from typing import Type
from typing import Union

import torch as pt
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Identity
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import Sequential
from torch.nn.init import xavier_normal_
from torch.nn.init import xavier_uniform_
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout
from torch.nn.parameter import Parameter

from providence.utils import validate

_AttentionalAxis = Literal["temporal", "feature"]


class Residual(Module):
    """A Residual connection, which adds the inputs to the activation of every layer downstream.


    Treats the first argument of the variadic list as a identity element to add to all subsequent activations.
    Ownness is on the user to ensure the dimensionality of the propogated value matches all intermediary outputs.

    Args:
        layers: modules of a neural network
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = ModuleList(layers)

    def forward(self, *x, **kwargs) -> Tensor:
        """Applies a residual connection to all layers

        Args:
            x: variadic inputs, Tensors or otherwise, so long as ``layers`` understand the input
            kwargs: any keyword arguments for the first layer of the network

        Returns:
            Tensor: entirely dependent on supplied ``layers``
        """
        identity = x[0]
        if self.layers:
            x = self.layers[0](*x, **kwargs) + identity

            for layer in self.layers[1:]:
                x = layer(x) + identity
        return x


class SkipConnected(Sequential):
    """The skip connection, over an arbitrary number of module i.e. a block.

    Unlike Residual, the initial input is not shared between each of the layers
    Owness is on the user to make sure dimensions match for the addition.
    """

    def forward(self, input: Tensor) -> Tensor:
        """Perform the skip-connected forward pass on ``input``

        We add the input to the output of the inner layers

        Args:
            input (Tensor): non-None Tensor

        Returns:
            Tensor: shape matching ``input`` exactly
        """
        out = super().forward(input)
        return out + input


class MultiArgSequence(Sequential):
    """Extension to Sequential, supporting multiple arguments for the initial input

    I wanted to maintain the clean abstraction, without adding a bunch of noise to the code.
    Implementing this allows one to pass multiple arguments through a Sequential.
    Inherited rather than re-implented to pick up the JIT wrapper niceties already done for us.
    """

    def forward(self, *input, **kwargs):
        """Perform a forward pass, supporting variadic ``input`` and ``kwargs``

        Returns:
            Tensor: final output of the final layer, be it a single or tuple of Tensors
        """
        output = self[0](*input, **kwargs)
        for module in self[1:]:
            output = module(output)
        return output


class BasicAttention(Module):
    """Simple implementation of the Scaled Dot-Product Attention as shown in Attention is All You Need

    This implementatation is simplified by using more PyTorch objects than primitives (e.g. Parameter)
    to perform the attention computation.
    We have done very little testing varying the *_dim parameters, but it workable. The field has moved
    away from fiddling with these particular parameters, finding that scaling and more ineresting attention
    algorithms - in general - are more useful, more readily parallelizable, and more performant on a given task.
    You have been warned.

    Args:
        embed_dim (int): input dimension
        dropout (float, optional): _description_. Defaults to 0.0.
        bias (bool, optional): whether to use bias in the Linear layers. Defaults to True.
        key_dim (Optional[int], optional): configurable dimension for the keys. Defaults to None.
        value_dim (Optional[int], optional): configurable dimension for the values. Defaults to None.
        query_dim (Optional[int], optional): configurable dimension for the queries. Defaults to None.
        attention_axis (_AttentionalAxis, optional): axis to attend to either "feature" or "temporal".
            Defaults to "temporal".
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
        bias=True,
        *,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        query_dim: Optional[int] = None,
        attention_axis: _AttentionalAxis = "temporal",
    ):
        super().__init__()
        validate(isinstance(embed_dim, int), "Embedding dimension should be an integer")
        validate(embed_dim > 1, "Embedding dimension should be positive")
        self.embed_dim = embed_dim
        self.key_dim = key_dim or embed_dim
        self.value_dim = value_dim or embed_dim
        self.query_dim = query_dim or embed_dim
        self.dropout_p = dropout
        self.use_bias = bias
        self.attention_axis = attention_axis

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        # When would we NOT have all of these be (embed_dim x ___dim)?
        # When we're finally connecting the encoder layer with the decoder: k = v = memory output
        # before that (and maybe even afterwards) it's sensible that these be the same.
        self.k_proj = Linear(self.embed_dim, self.key_dim, bias=self.use_bias)
        self.v_proj = Linear(self.embed_dim, self.value_dim, bias=self.use_bias)
        self.q_proj = Linear(self.embed_dim, self.query_dim, bias=self.use_bias)
        self.dropout = Dropout(self.dropout_p)

        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)

        if self.use_bias:
            # HACK-lite: to perform Xavier init, need to expand the dimensions, then reshape for linear activation
            q_bias = pt.reshape(self.q_proj.bias, (1, self.q_proj.bias.shape[0]))
            k_bias = pt.reshape(self.k_proj.bias, (1, self.k_proj.bias.shape[0]))
            v_bias = pt.reshape(self.v_proj.bias, (1, self.v_proj.bias.shape[0]))
            xavier_normal_(q_bias)
            xavier_normal_(k_bias)
            xavier_normal_(v_bias)

            # Parameter ctor pulls everything through.
            self.q_proj.bias = Parameter(q_bias.reshape((q_bias.shape[-1])))
            self.k_proj.bias = Parameter(k_bias.reshape((k_bias.shape[-1])))
            self.v_proj.bias = Parameter(v_bias.reshape((v_bias.shape[-1])))

        if self.attention_axis == "temporal":
            att_dim = 0
        elif self.attention_axis == "feature":
            att_dim = -1
        else:
            raise ValueError(f"Supplied invalid attention_axis = {self.attention_axis}")
        self.attention_dim = att_dim

    def forward(self, key: Tensor, value: Tensor, query: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Apply (optionally masked) attention over ``key``, ``value``, and ``query``

        Args:
            key (Tensor): some transformation of the input, shape expected to match query and value
            value (Tensor): some transformation of the input, shape expected to match key and query
            query (Tensor): some transformation of the input, shape expected to match key and value
            mask (Tensor, optional): either our sequence-length mask or a square matrix matching the attention_axis dimension.
                Defaults to None.

        Returns:
            Tensor: attended values mapped back to the input space. Shape should exactly match ``key``
        """

        validate(
            len(key.shape) == 3,
            f"Expected key of shape (time, batch, features). Got shape {key.shape}",
        )
        validate(
            len(value.shape) == 3,
            f"Expected value of shape (time, batch, features). Got shape {value.shape}",
        )
        validate(
            len(query.shape) == 3,
            f"Expected query of shape (time, batch, features). Got shape {query.shape}",
        )

        q, k = self.q_proj(query), self.k_proj(key)
        prod = q @ k.transpose(1, 2)

        scaled = prod * (self.key_dim**-0.5)  # qk^T / sqrt(d_key)

        # mask should be a square matrix with the lower triangle lit with the sequences to keep
        # masked_fill is the most straightforward implementation. Skips the padding with infinity (pulled from annotated Transformer [1])
        masked = scaled if mask is None else scaled.masked_fill(mask == 0, -1e6)
        # NOTE: I set the axis to 0 for the sake of re-running the hyperparameter sweeps and seeeing what the paths are for one of the models
        # This needs to go back to being the parameterized version, or I need to fix the pickle loading. One of those is higher priority
        masked = F.softmax(masked.contiguous(), dim=self.attention_dim)  # attended probabilities
        masked = self.dropout(masked)
        # print(f"{masked = }")

        # value projection, followed by selection
        v = self.v_proj(value)
        # print(f"{masked.shape = }")
        # print(f"projected{v.shape = }")
        out = masked @ v
        return out

    def __setstate__(self, state):
        self.__dict__ = state
        # temporal axis for the models that are loaded after the implementation
        self.attention_axis = state.get("attention_axis", "temporal")
        self.attention_dim = 0 if self.attention_axis == "temporal" else -1


class MultiheadedAttention(Module):
    """Multiheaded Attention that applies the algorithm straightforwardly, but slowly.

    This is the implementation that was used to produce the paper Providence: A Neural Network Framework...
    Ideally, a fast version of this would be used. MultiheadedAttention3 is the next best thing, showing
    similar performance, but not slowing linearly with the number of attention aheads

    Args:
        n_heads (int): number of attention heads, must be positive integer. Computational time scale linear in
        number of heads
        embed_dim (int): dimension of the inner Linear layers internal to BasicAttention
        dropout (float, optional): percentage in [0.0, 1.0] to apply dropout. Best values depend on other parameters.
            Defaults to 0.0 (no dropout).
        attention_axis (_AttentionalAxis, optional): axis to attend to either "feature" or "temporal".
            Defaults to "temporal".
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        dropout: float = 0.0,
        *,
        attention_axis: _AttentionalAxis = "temporal",
    ):
        super().__init__()
        validate(
            isinstance(n_heads, int),
            "Number of heads must be an integer. You can't have half a head paying attention",
        )
        validate(n_heads > 1, "Must supply more than one head for MULTI-headed attention")
        validate(isinstance(embed_dim, int), "Embedding dimension should be an integer")
        validate(embed_dim > 0, "Embedding dimension should be positive")
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.attention_concat_dim = n_heads * embed_dim
        self.dropout = dropout
        self.attention_axis = attention_axis

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        self.attention_heads = _get_clones(
            BasicAttention(
                self.embed_dim,
                dropout=self.dropout,
                bias=False,
                attention_axis=self.attention_axis,
            ),
            self.n_heads,
        )
        self.linear_out = Linear(self.attention_concat_dim, self.embed_dim)

    def forward(self, key: Tensor, value: Tensor, query: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Apply (optionally masked) causal, multi-head attention over ``key``, ``value``, and ``query``

        This forward pass literally attends with n_heads many heads, one at a time. It is highly serial and performs
        similar to a RNN on the same data. See the other implementations of Multiheaded throughout this repository for
        more parallelizable implementations.

        Args:
            key (Tensor): some transformation of the input, shape expected to match query and value
            value (Tensor): some transformation of the input, shape expected to match key and query
            query (Tensor): some transformation of the input, shape expected to match key and value
            mask (Tensor, optional): either our sequence-length mask or a square matrix matching the attention_axis dimension.
                Defaults to None.

        Returns:
            Tensor: attended values mapped back to the input space. Shape should exactly match ``key``
        """

        # This is surprisingly not slow. Tensor.contiguous() makes it even faster. :)
        attended = pt.cat(
            [head(key, value, query, mask=mask) for head in self.attention_heads],
            dim=-1,
        ).contiguous()
        out = self.linear_out(attended)

        return out


class MultiheadedAttention2(Module):
    """Multiheaded Attention that attempts to follow the template of BasicAttention, but across multiple heads.

    This implementation does not perform well on any tasks. Our investigation suggests it is because the attention
    is too deliberately across all data, rather than per device.
    In other words, this implementation is technically defective and should be avoided until further experimentation.

    Args:
        n_heads (int): number of attention heads, must be positive integer. Computational time scales linear in
        number of heads
        embed_dim (int): dimension of the inner Linear layers internal to BasicAttention
        dropout (float, optional): percentage in [0.0, 1.0] to apply dropout. Best values depend on other parameters.
            Defaults to 0.0 (no dropout).
        attention_axis (_AttentionalAxis, optional): axis to attend to either "feature" or "temporal".
            Defaults to "temporal".
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        dropout: float = 0.0,
        *,
        attention_axis: _AttentionalAxis = "temporal",
    ):
        super().__init__()
        validate(
            isinstance(n_heads, int),
            "Number of heads must be an integer. You can't have half a head paying attention",
        )
        validate(isinstance(embed_dim, int), "Embedding dimension should be an integer")
        validate(embed_dim > 0, "Embedding dimension should be positive")
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.attention_concat_dim = n_heads * embed_dim
        self.dropout_p = dropout
        self.attention_axis = attention_axis

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        self.dropout = Dropout(self.dropout_p)
        self.key_projection = Linear(self.embed_dim, self.attention_concat_dim)
        self.value_projection = Linear(self.embed_dim, self.attention_concat_dim)
        self.query_projection = Linear(self.embed_dim, self.attention_concat_dim)
        self.linear_out = Linear(self.attention_concat_dim, self.embed_dim) if self.n_heads != 1 else Identity()

        if self.attention_axis == "temporal":
            att_dim = 0
        elif self.attention_axis == "feature":
            att_dim = -1
        else:
            raise ValueError(f"Supplied invalid attention_axis = {self.attention_axis}")
        self.attention_axis_index = att_dim

    def forward(self, key: Tensor, value: Tensor, query: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Apply (optionally masked) causal, multi-head attention over ``key``, ``value``, and ``query``

        Args:
            key (Tensor): some transformation of the input, shape expected to match query and value
            value (Tensor): some transformation of the input, shape expected to match key and query
            query (Tensor): some transformation of the input, shape expected to match key and value
            mask (Tensor, optional): either our sequence-length mask or a square matrix matching the attention_axis dimension.
                Defaults to None.

        Returns:
            Tensor: attended values mapped back to the input space. Shape should exactly match ``key``
        """

        # This is surprisingly not slow. Tensor.contiguous() makes it even faster. :)
        # length, batch, n_features = key.shape

        k = self.key_projection(key)
        q = self.query_projection(query)

        # attended = self.attention_head(key, value, query, mask=mask)
        prod = q @ k.transpose(1, 2)

        scaled = prod * (self.embed_dim**-0.5)  # qk^T / sqrt(d_key)

        # mask should be a square matrix with the lower triangle lit with the sequences to keep
        # masked_fill is the most straightforward implementation. Skips the padding with infinity (pulled from annotated Transformer [1])
        masked = scaled if mask is None else scaled.masked_fill(mask == 0, -1e6)
        masked = F.softmax(masked.contiguous(), dim=self.attention_axis_index)  # attended probabilities
        masked = self.dropout(masked)
        # print(f"{masked = }")

        # value projection, followed by selection
        v = self.value_projection(value)
        # print(f"{masked.shape = }")
        # print(f"projected{v.shape = }")
        attended = masked @ v

        out = self.linear_out(attended).contiguous()

        return out


class MultiheadedAttention3(Module):
    """Multiheaded Attention that overloads the functionality of the BasicAttention module to ease implementation.

    Caveat: there are extra mapping layers needed to support the inner BasicAttention, one before and one after

    Args:
        n_heads (int): number of attention heads, must be positive integer. Computational time scales sublinear in
            number of heads up to 8 heads.
        embed_dim (int): dimension of the inner Linear layers internal to BasicAttention
        dropout (float, optional): percentage in [0.0, 1.0] to apply dropout. Best values depend on other parameters.
            Defaults to 0.0 (no dropout).
        attention_axis (_AttentionalAxis, optional): axis to attend to either "feature" or "temporal".
            Defaults to "temporal".
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        dropout: float = 0.0,
        *,
        attention_axis: _AttentionalAxis = "temporal",
    ):
        super().__init__()
        validate(
            isinstance(n_heads, int),
            "Number of heads must be an integer. You can't have half a head paying attention",
        )
        validate(isinstance(embed_dim, int), "Embedding dimension should be an integer")
        validate(embed_dim > 0, "Embedding dimension should be positive")
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.attention_concat_dim = n_heads * embed_dim
        self.dropout_p = dropout
        self.attention_axis = attention_axis

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        self.inner_attention = BasicAttention(
            self.embed_dim,
            self.dropout_p,
            key_dim=self.attention_concat_dim,
            value_dim=self.attention_concat_dim,
            query_dim=self.attention_concat_dim,
            attention_axis=self.attention_axis,
        )
        self.linear_out = Linear(self.attention_concat_dim, self.embed_dim) if self.n_heads != 1 else Identity()

        if self.attention_axis == "temporal":
            att_dim = 0
        elif self.attention_axis == "feature":
            att_dim = -1
        else:
            raise ValueError(f"Supplied invalid attention_axis = {self.attention_axis}")
        self.attention_axis_index = att_dim

    def forward(self, key: Tensor, value: Tensor, query: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Apply (optionally masked) causal, multi-head attention over ``key``, ``value``, and ``query``

        Args:
            key (Tensor): some transformation of the input, shape expected to match query and value
            value (Tensor): some transformation of the input, shape expected to match key and query
            query (Tensor): some transformation of the input, shape expected to match key and value
            mask (Tensor, optional): either our sequence-length mask or a square matrix matching the attention_axis dimension.
                Defaults to None.

        Returns:
            Tensor: attended values mapped back to the input space. Shape should exactly match ``key``
        """
        attended = self.inner_attention(key, value, query, mask=mask)
        out = self.linear_out(attended).contiguous()
        return out


MhaInterface = Union[Type[MultiheadedAttention], Type[MultiheadedAttention2], Type[MultiheadedAttention3]]


class EncoderBlock(Module):
    """The Encoder block (sans coupling to the input embedding block and dropout) from Attention is All You Need.

    Omitting the coupling was to free us to make more drastic changes to the attention algorithm internal to this
    module.

    Args:
        n_attention_heads (int, optional): number of attention heads for the attention implementation. Defaults to 4.
        model_dimension (int, optional): feature dimension of the input tensors. Defaults to 128.
        feed_forward_internal_dimension (int, optional): dimension of the inner feed forward layers. Defaults to 512.
        feed_forward_activation (Type[Module], optional): a Torch Module or objectified-function that can be
        initialized in ``reset_parameters()``. Defaults to ReLU.
        dropout (float, optional): percentage in [0.0, 1.0] to apply dropout. Best values depend on other parameters.
            Defaults to 0.0 (no dropout).
        layer_norm_epsilon (float, optional): epsilon fed to the LayerNorm internal to each (Encoder|Decoder)Block.
            Defaults to 1e-5.
        attention_axis (str, optional): defines whether attention is applied on features (like most transformers) or
            across the time axis (unique to this work and some attention-augmented RNNs). Defaults to "temporal".
        t_attention (MhaInterface, optional): the Multihead attention algorithm / implementation to use.
            Defaults to MultiheadedAttention3.
    """

    def __init__(
        self,
        n_attention_heads: int = 4,
        model_dimension: int = 128,
        feed_forward_internal_dimension: int = 512,
        feed_forward_activation: Type[Module] = ReLU,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        *,
        attention_axis: _AttentionalAxis = "temporal",
        t_attention: MhaInterface = MultiheadedAttention,
    ):
        super().__init__()
        self.n_heads = n_attention_heads
        self.d_model = model_dimension
        self.d_ff = feed_forward_internal_dimension
        self.ff_activation = feed_forward_activation
        self.dropout_p = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attention_axis = attention_axis
        self._attention_type = t_attention

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        attention = Residual(
            self._attention_type(
                n_heads=self.n_heads,
                embed_dim=self.d_model,
                attention_axis=self.attention_axis,
            )
        )
        attending_sublayer = MultiArgSequence(attention, Dropout(self.dropout_p), LayerNorm(self.d_model))

        position_wise_ff_sublayer = Sequential(
            SkipConnected(
                Linear(self.d_model, self.d_ff),
                self.ff_activation(),
                Dropout(self.dropout_p),
                Linear(self.d_ff, self.d_model),
            ),
            LayerNorm(self.d_model, eps=self.layer_norm_epsilon),
        )
        self.assembled = MultiArgSequence(attending_sublayer, position_wise_ff_sublayer)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Activate the encoder, optionally applying a mask

        Args:
            x (Tensor): tensor of shape [time, entity, feature]
            mask (Tensor): either our sequence-length mask or a square matrix matching the attention_axis dimension.

        Returns:
            Tensor: encoding of ``x`` shape should match input ``x`` exactly.
        """
        return self.assembled(x, x, x, mask)


class DecoderBlock(Module):
    """
    The Decoder block, based on the architecture from Attention is All You Need

    Like the encoder, we omit the coupling to the output embedding block along with its additional dropout.

    Args:
        n_attention_heads (int, optional): number of attention heads for the attention implementation. Defaults to 4.
        model_dimension (int, optional): feature dimension of the input tensors. Defaults to 128.
        feed_forward_internal_dimension (int, optional): dimension of the inner feed forward layers. Defaults to 512.
        feed_forward_activation (Type[Module], optional): a Torch Module or objectified-function that can be
        initialized in ``reset_parameters()``. Defaults to ReLU.
        dropout (float, optional): percentage in [0.0, 1.0] to apply dropout. Best values depend on other parameters.
            Defaults to 0.0 (no dropout).
        layer_norm_epsilon (float, optional): epsilon fed to the LayerNorm internal to each (Encoder|Decoder)Block.
            Defaults to 1e-5.
        attention_axis (str, optional): defines whether attention is applied on features (like most transformers) or
            across the time axis (unique to this work and some attention-augmented RNNs). Defaults to "temporal".
        t_attention (MhaInterface, optional): the Multihead attention algorithm / implementation to use.
            Defaults to MultiheadedAttention3.

    Raises:
        ValueError: some implementations of ``MhaInferface`` may raise if ``n_attention_heads == 1``
    """

    def __init__(
        self,
        n_attention_heads: int = 4,
        model_dimension: int = 128,
        feed_forward_internal_dimension: int = 512,
        feed_forward_activation: Type[Module] = ReLU,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        *,
        attention_axis: _AttentionalAxis = "temporal",
        t_attention: MhaInterface = MultiheadedAttention,
    ):
        super().__init__()
        self.n_heads = n_attention_heads
        self.d_model = model_dimension
        self.d_ff = feed_forward_internal_dimension
        self.ff_activation = feed_forward_activation
        self.dropout_p = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attention_axis = attention_axis
        self._attention_type = t_attention

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        attention = Residual(
            self._attention_type(
                n_heads=self.n_heads,
                embed_dim=self.d_model,
                attention_axis=self.attention_axis,
            )
        )
        self.masked_self_attention = MultiArgSequence(attention, Dropout(self.dropout_p), LayerNorm(self.d_model))

        # The thing that's most important for the training architecture is having the encoder's memory output
        # be fed into this decoder's internal encoder as the key and value, while the query is the actual input
        # this attention receives key and value from the encoder, query from the decoder's (masked) self-attention
        merging_attention = self._attention_type(
            n_heads=self.n_heads,
            embed_dim=self.d_model,
            attention_axis=self.attention_axis,
        )
        merging_attention_sublayer = MultiArgSequence(
            Residual(merging_attention),
            LayerNorm(self.d_model, eps=self.layer_norm_epsilon),
        )

        position_wise_ff_sublayer = Sequential(
            SkipConnected(
                Linear(self.d_model, self.d_ff),
                self.ff_activation(),
                Dropout(self.dropout_p),
                Linear(self.d_ff, self.d_model),
            ),
            LayerNorm(self.d_model, eps=self.layer_norm_epsilon),
        )

        self.internal_encoder = MultiArgSequence(merging_attention_sublayer, position_wise_ff_sublayer)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_attention_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode the given input sequence ``x`` and ``memory`` (output from an encoder), optionally applying masking.

        Masking applied to self-attention is to prevent peering into the future of time series or signal data.
        Masking applied to the memory on the attention given to the memory, minimizing dependency on the memory

        Args:
            x (Tensor): tensor of shape [time, entity, feature]
            memory (Tensor): tensor of shape [time, entity, feature]
            self_attention_mask (Tensor, optional): either our "rectangular" sequence-length mask or a square matrix
                matching the attention_axis dimension.
            memory_mask (Tensor, optional): either our "rectangular" sequence-length mask or a square matrix matching
                the attention_axis dimension.

        Returns:
            Tensor: decoding of ``x`` and ``memory`` shape should match input ``x`` exactly.
        """
        # these function calls are noticeably slower with kwargs in my tests, but it's more readable.
        attended = self.masked_self_attention(x, x, x, mask=self_attention_mask)
        merger = self.internal_encoder(memory, memory, attended, mask=memory_mask)
        # would love an object that allowed a branching pattern - a more objectified Either<A, B>, probabl - but alas
        # this is the best we can do.

        return merger


class Transformer(Module):
    """A transformer nearly ready for our predict-parameters-from-an-embedding paradigm.

    Args:
        model_dimension (int, optional): number of input features necessary to infer on an example from the dataset.
            Defaults to 128.
        feed_forward_internal_dimension (int, optional): size of all feedforward layers used throughout.
            Defaults to 512.
        n_layers (int, optional): number of layers in each (Encoder|Decoder)Block. Defaults to 2.
        num_heads (int, optional): number of attention heads used in the multihead attention algorithm; thereby
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

    Raises:
        ValueError: some implementations of ``MhaInferface`` may raise if ``n_attention_heads == 1``
    """

    def __init__(
        self,
        model_dimension: int = 128,
        feed_forward_internal_dimension: int = 512,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        positional_encoding_dim: int = 2000,
        *,
        attention_axis: _AttentionalAxis = "temporal",
        t_attention: MhaInterface = MultiheadedAttention,
    ):
        super().__init__()
        self.d_model = model_dimension
        self.d_ff = feed_forward_internal_dimension
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.dropout_p = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.positional_encoding_p = positional_encoding_dim
        self.attention_axis = attention_axis
        self.t_attention = t_attention

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        self.embedding = Linear(self.d_model, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout_p, max_len=self.positional_encoding_p)
        self.encoders = _get_clones(
            EncoderBlock(
                model_dimension=self.d_model,
                feed_forward_internal_dimension=self.d_ff,
                n_attention_heads=self.n_heads,
                dropout=self.dropout_p,
                layer_norm_epsilon=self.layer_norm_epsilon,
                attention_axis=self.attention_axis,
                t_attention=self.t_attention,
            ),
            self.n_layers,
        )
        self.decoders = _get_clones(
            DecoderBlock(
                model_dimension=self.d_model,
                feed_forward_internal_dimension=self.d_ff,
                n_attention_heads=self.n_heads,
                dropout=self.dropout_p,
                layer_norm_epsilon=self.layer_norm_epsilon,
                attention_axis=self.attention_axis,
                t_attention=self.t_attention,
            ),
            self.n_layers,
        )

    def forward(
        self,
        x: Tensor,
        encoder_mask: Optional[Tensor] = None,
        decoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Perform encoder-decoder pass on ``x``, per the Transformer algorithm in Attention is All You Need

                Transformer inference is where things tend to vary between architectures. I'm thinking about making the decoder mask
        for some `ProvidenceTransformer` class to be implicit in the model, so the model __can't__ cheat.

        Perhaps for the final "Providence Transformer?


        Args:
            x (Tensor): tensor of shape [time, entity, feature]
            encoder_mask (Optional[Tensor], optional): mask passed to the encoder, with shape matching the
                attention_axis dimension. Defaults to None.
            decoder_mask (Optional[Tensor], optional): mask passed to the decoder, with shape matching the
                attention_axis dimension. Defaults to None.

        Returns:
            Tensor: tensor of attention paid along the attention_axis, exactly matching shape of ``x``

            This implementation does not directly output attented values, so one should not expect output tensor
            to sum to (approx.) 1 along the attended axis (as you might when inspecting the correctness of attention)
        """

        x = self.embedding(x)
        x.add_(self.positional_encoding(x))

        enc_out = x
        for encoder in self.encoders:
            enc_out = encoder(enc_out, encoder_mask)

        # NOTE: there isn't some "target" parameter like in PyTorch or many of the language-model-based Transformer
        # This Decoder architecture uses the encoder's "memory" of the input in tandem with the input itself and
        # its own self-attention. This self attention is called "merged" in this implementation. The Decoder activates
        # an encoder-style internal memory about the memory of the encoder's memory and its self-attended memory.
        dec_mem = x
        for decoder in self.decoders:
            dec_mem = decoder(dec_mem, enc_out, decoder_mask)

        return dec_mem


# courtesy of Pytorch: torch/nn/modue/transformer.py::_get_clones
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def set_attention_axis(transformer: Transformer, new_axis: _AttentionalAxis) -> None:
    """Propogate the (re)setting of the attention axis. Used for experiments.

    Args:
        transformer (Transformer): model on which the attention axis will change
        new_axis (str): the new attention axis

    Raises:
        ValueError: if ``new_axis`` is not one of "temporal" or "feature"
    """
    if new_axis == "temporal":
        new_dim = 0
    elif new_axis == "feature":
        new_dim = -1
    else:
        raise ValueError(f"Supplied {new_axis =} is invalid")

    for module in transformer.modules():
        if isinstance(module, BasicAttention):
            module.attention_axis = new_axis
            module.attention_dim = new_dim
        elif hasattr(module, "attention_axis"):  # i.e. for container types that have this property
            module.attention_axis = new_axis


class PositionalEncoding(Module):
    """Sinusoidal positional encoding, computed ahead-of-time, to decorate tensors with additional ordinality.

    Implement the PE function in a temporal context: all the sequence be decorated with ordering information,
    before we look across (i.e. backwards in) time

    Args:
        d_model (int): dimension of the model input
        dropout (float, optional): percentage in [0.0, 1.0] to apply dropout. Defaults to 0.0 (no dropout).
        max_len (int, optional): maximum length of a sequence that will be passed to this. Defaults to 5000.
    """

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        import math

        self.dropout = Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = pt.zeros(max_len, d_model)
        position = pt.arange(0, max_len).unsqueeze(1)
        div_term = pt.exp(pt.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = pt.sin(position * div_term)

        # Weird solution to bug: with div_term having size = (d_model // 2) + int(d_model is odd), you can end up with
        # a positional encoding of odd size trying to receive (d_model // 2) + 1 into (d_model // 2) slots.
        # so this makes sure we don't try to get the (extra) last slot.
        # This doesn't need to be more "elegant" because it's computed once then held in memory
        if (enc_sub_size := pe[:, 1::2].size(-1)) != div_term.size(0):
            pe[:, 1::2] = pt.cos(position * div_term[:enc_sub_size])
        else:
            pe[:, 1::2] = pt.cos(position * div_term)
        # pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        """Positonally encode ``x``

        Args:
            x (Tensor): input matching the dimensionality of the ``d_model``

        Returns:
            Tensor: dropout-applied, positionally encoded ``x``
        """
        pe = self.pe[: x.size(0)]

        batches = x.size(1)  # dependent on data being passed in with [sequence, batch or device, feature]
        full_pe = pt.stack([pe] * batches, dim=1)

        x = x + full_pe  # FIXME(stephen): is this a double addition?

        return self.dropout(x)
