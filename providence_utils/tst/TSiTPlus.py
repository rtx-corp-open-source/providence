# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/124_models.TSiTPlus.ipynb (unless otherwise specified).
"""
Time series transformer model based on ViT (Vision Transformer)

    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020).
    An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

courtesy of TSAI, pulled September 21st, 2022.

This code was pulled under an Apache 2.0 license. See the LICENSE in this directory.
(Originally, the Apache 2.0 license here - https://github.com/timeseriesAI/tsai/blob/b66bf0eeb32be9cc4e42b30456f68104d341f377/LICENSE#L1)

TODO: use this implementation to get a feel for how vision models handle time series data?
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
__all__ = ["TSiTPlus", "TSiT"]

# Cell
from ..imports import *
from .utils import *
from .layers import *
from typing import Callable


# Cell
class _TSiTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        q_len: int = None,
        attn_dropout: float = 0.0,
        dropout: float = 0,
        drop_path_rate: float = 0.0,
        mlp_ratio: int = 1,
        lsa: bool = False,
        qkv_bias: bool = True,
        act: str = "gelu",
        pre_norm: bool = False,
    ):
        super().__init__()
        self.mha = MultiheadAttention(
            d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout, lsa=lsa, qkv_bias=qkv_bias
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.pwff = PositionwiseFeedForward(d_model, dropout=dropout, act=act, mlp_ratio=mlp_ratio)
        self.ff_norm = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate != 0 else nn.Identity()
        self.pre_norm = pre_norm

        if lsa and q_len is not None:
            self.register_buffer("attn_mask", torch.eye(q_len).reshape(1, 1, q_len, q_len).bool())
        else:
            self.attn_mask = None

    def forward(self, x):
        if self.pre_norm:
            if self.attn_mask is not None:
                x = self.drop_path(self.mha(self.attn_norm(x), attn_mask=self.attn_mask)[0]) + x
            else:
                x = self.drop_path(self.mha(self.attn_norm(x))[0]) + x
            x = self.drop_path(self.pwff(self.ff_norm(x))) + x
        else:
            if self.attn_mask is not None:
                x = self.attn_norm(self.drop_path(self.mha(x, attn_mask=self.attn_mask)[0]) + x)
            else:
                x = self.attn_norm(self.drop_path(self.mha(x)[0]) + x)
            x = self.ff_norm(self.drop_path(self.pwff(x)) + x)
        return x


# Cell
class _TSiTEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        depth: int = 6,
        q_len: int = None,
        attn_dropout: float = 0.0,
        dropout: float = 0,
        drop_path_rate: float = 0.0,
        mlp_ratio: int = 1,
        lsa: bool = False,
        qkv_bias: bool = True,
        act: str = "gelu",
        pre_norm: bool = False,
    ):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        layers = []
        for i in range(depth):
            layer = _TSiTEncoderLayer(
                d_model,
                n_heads,
                q_len=q_len,
                attn_dropout=attn_dropout,
                dropout=dropout,
                drop_path_rate=dpr[i],
                mlp_ratio=mlp_ratio,
                lsa=lsa,
                qkv_bias=qkv_bias,
                act=act,
                pre_norm=pre_norm,
            )
            layers.append(layer)
        self.encoder = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        return x


# Cell
class _TSiTBackbone(Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        depth: int = 6,
        d_model: int = 128,
        n_heads: int = 16,
        act: str = "gelu",
        lsa: bool = False,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        mlp_ratio: int = 1,
        pre_norm: bool = False,
        use_token: bool = True,
        use_pe: bool = True,
        n_cat_embeds: Optional[list] = None,
        cat_embed_dims: Optional[list] = None,
        cat_padding_idxs: Optional[list] = None,
        cat_pos: Optional[list] = None,
        feature_extractor: Optional[Callable] = None,
        token_size: int = None,
        tokenizer: Optional[Callable] = None,
    ):
        # Categorical embeddings
        if n_cat_embeds is not None:
            n_cat_embeds = listify(n_cat_embeds)
            if cat_embed_dims is None:
                cat_embed_dims = [emb_sz_rule(s) for s in n_cat_embeds]
            self.to_cat_embed = MultiEmbedding(
                c_in, n_cat_embeds, cat_embed_dims=cat_embed_dims, cat_padding_idxs=cat_padding_idxs, cat_pos=cat_pos
            )
            c_in, seq_len = output_size_calculator(self.to_cat_embed, c_in, seq_len)
        else:
            self.to_cat_embed = nn.Identity()

        # Sequence embedding
        if token_size is not None:
            self.tokenizer = SeqTokenizer(c_in, d_model, token_size)
            c_in, seq_len = output_size_calculator(self.tokenizer, c_in, seq_len)
        elif tokenizer is not None:
            if isinstance(tokenizer, nn.Module):
                self.tokenizer = tokenizer
            else:
                self.tokenizer = tokenizer(c_in, d_model)
            c_in, seq_len = output_size_calculator(self.tokenizer, c_in, seq_len)
        else:
            self.tokenizer = nn.Identity()

        # Feature extractor
        if feature_extractor is not None:
            if isinstance(feature_extractor, nn.Module):
                self.feature_extractor = feature_extractor
            else:
                self.feature_extractor = feature_extractor(c_in, d_model)
            c_in, seq_len = output_size_calculator(self.feature_extractor, c_in, seq_len)
        else:
            self.feature_extractor = nn.Identity()

        # Linear projection
        if token_size is None and tokenizer is None and feature_extractor is None:
            self.linear_proj = nn.Conv1d(c_in, d_model, 1)
        else:
            self.linear_proj = nn.Identity()

        self.transpose = Transpose(1, 2)

        # Position embedding & token
        if use_pe:
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.use_pe = use_pe
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.use_token = use_token
        self.emb_dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = _TSiTEncoder(
            d_model,
            n_heads,
            depth=depth,
            q_len=seq_len + use_token,
            qkv_bias=qkv_bias,
            lsa=lsa,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            act=act,
            pre_norm=pre_norm,
        )

    def forward(self, x):
        # Categorical embeddings
        x = self.to_cat_embed(x)

        # Sequence embedding
        x = self.tokenizer(x)

        # Feature extractor
        x = self.feature_extractor(x)

        # Linear projection
        x = self.linear_proj(x)

        # Position embedding & token
        x = self.transpose(x)
        if self.use_pe:
            x = x + self.pos_embed
        if (
            self.use_token
        ):  # token is concatenated after position embedding so that embedding can be learned using self.supervised learning
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.emb_dropout(x)

        # Encoder
        x = self.encoder(x)

        # Output
        x = x.transpose(1, 2)
        return x


# Cell
class TSiTPlus(nn.Sequential):
    r"""Time series transformer model based on ViT (Vision Transformer):

    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020).
    An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

    This implementation is a modified version of Vision Transformer that is part of the grat timm library
    (https://github.com/rwightman/pytorch-image-models/blob/72b227dcf57c0c62291673b96bdc06576bb90457/timm/models/vision_transformer.py)

    Args:
        c_in:               the number of features (aka variables, dimensions, channels) in the time series dataset.
        c_out:              the number of target classes.
        seq_len:            number of time steps in the time series.
        d_model:            total dimension of the model (number of features created by the model).
        depth:              number of blocks in the encoder.
        n_heads:            parallel attention heads. Default:16 (range(8-16)).
        act:                the activation function of positionwise feedforward layer.
        lsa:                locality self attention used (see Lee, S. H., Lee, S., & Song, B. C. (2021). Vision Transformer for Small-Size Datasets.
                            arXiv preprint arXiv:2112.13492.)
        attn_dropout:       dropout rate applied to the attention sublayer.
        dropout:            dropout applied to to the embedded sequence steps after position embeddings have been added and
                            to the mlp sublayer in the encoder.
        drop_path_rate:     stochastic depth rate.
        mlp_ratio:          ratio of mlp hidden dim to embedding dim.
        qkv_bias:           determines whether bias is applied to the Linear projections of queries, keys and values in the MultiheadAttention
        pre_norm:           if True normalization will be applied as the first step in the sublayers. Defaults to False.
        use_token:          if True, the output will come from the transformed token. This is meant to be use in classification tasks.
        use_pe:             flag to indicate if positional embedding is used.
        n_cat_embeds:       list with the sizes of the dictionaries of embeddings (int).
        cat_embed_dims:     list with the sizes of each embedding vector (int).
        cat_padding_idxs:       If specified, the entries at cat_padding_idxs do not contribute to the gradient; therefore, the embedding vector at cat_padding_idxs
                            are not updated during training. Use 0 for those categorical embeddings that may have #na# values. Otherwise, leave them as None.
                            You can enter a combination for different embeddings (for example, [0, None, None]).
        cat_pos:            list with the position of the categorical variables in the input.
        token_size:         Size of the embedding function used to reduce the sequence length (similar to ViT's patch size)
        tokenizer:          nn.Module or callable that will be used to reduce the sequence length
        feature_extractor:  nn.Module or callable that will be used to preprocess the time series before
                            the embedding step. It is useful to extract features or resample the time series.
        flatten:            flag to indicate if the 3d logits will be flattened to 2d in the model's head if use_token is set to False.
                            If use_token is False and flatten is False, the model will apply a pooling layer.
        concat_pool:        if True the head begins with fastai's AdaptiveConcatPool2d if concat_pool=True; otherwise, it uses traditional average pooling.
        fc_dropout:         dropout applied to the final fully connected layer.
        use_bn:             flag that indicates if batchnorm will be applied to the head.
        bias_init:          values used to initialized the output layer.
        y_range:            range of possible y values (used in regression tasks).
        custom_head:        custom head that will be applied to the network. It must contain all kwargs (pass a partial function)
        verbose:            flag to control verbosity of the model.

    Input:
        x: bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        d_model: int = 128,
        depth: int = 6,
        n_heads: int = 16,
        act: str = "gelu",
        lsa: bool = False,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        mlp_ratio: int = 1,
        qkv_bias: bool = True,
        pre_norm: bool = False,
        use_token: bool = True,
        use_pe: bool = True,
        cat_pos: Optional[list] = None,
        n_cat_embeds: Optional[list] = None,
        cat_embed_dims: Optional[list] = None,
        cat_padding_idxs: Optional[list] = None,
        token_size: int = None,
        tokenizer: Optional[Callable] = None,
        feature_extractor: Optional[Callable] = None,
        flatten: bool = False,
        concat_pool: bool = True,
        fc_dropout: float = 0.0,
        use_bn: bool = False,
        bias_init: Optional[Union[float, list]] = None,
        y_range: Optional[tuple] = None,
        custom_head: Optional[Callable] = None,
        verbose: bool = True,
    ):
        if use_token and c_out == 1:
            use_token = False
            pv("use_token set to False as c_out == 1", verbose)
        backbone = _TSiTBackbone(
            c_in,
            seq_len,
            depth=depth,
            d_model=d_model,
            n_heads=n_heads,
            act=act,
            lsa=lsa,
            attn_dropout=attn_dropout,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            pre_norm=pre_norm,
            mlp_ratio=mlp_ratio,
            use_pe=use_pe,
            use_token=use_token,
            n_cat_embeds=n_cat_embeds,
            cat_embed_dims=cat_embed_dims,
            cat_padding_idxs=cat_padding_idxs,
            cat_pos=cat_pos,
            feature_extractor=feature_extractor,
            token_size=token_size,
            tokenizer=tokenizer,
        )

        self.head_nf = d_model
        self.c_out = c_out
        self.seq_len = seq_len

        # Head
        if custom_head:
            if isinstance(custom_head, nn.Module):
                head = custom_head
            else:
                head = custom_head(self.head_nf, c_out, seq_len)
        else:
            nf = d_model
            layers = []
            if use_token:
                layers += [TokenLayer()]
            elif flatten:
                layers += [Reshape(-1)]
                nf = nf * seq_len
            else:
                if concat_pool:
                    nf *= 2
                layers = [GACP1d(1) if concat_pool else GAP1d(1)]
            if use_bn:
                layers += [nn.BatchNorm1d(nf)]
            if fc_dropout:
                layers += [nn.Dropout(fc_dropout)]

            # Last layer
            linear = nn.Linear(nf, c_out)
            if bias_init is not None:
                if isinstance(bias_init, float):
                    nn.init.constant_(linear.bias, bias_init)
                else:
                    linear.bias = nn.Parameter(torch.as_tensor(bias_init, dtype=torch.float32))
            layers += [linear]

            if y_range:
                layers += [SigmoidRange(*y_range)]
            head = nn.Sequential(*layers)
        super().__init__(OrderedDict([("backbone", backbone), ("head", head)]))


TSiT = TSiTPlus
