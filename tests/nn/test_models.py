# -*- coding: utf-8 -*-
"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import OrderedDict

import numpy as np
import pytest
import torch as pt

from providence.dataloaders import providence_collate_fn
from providence.datasets.core import ProvidenceDataset
from providence.distributions import Weibull
from providence.distributions import Weibull3
from providence.loss import DiscreteWeibull3Loss
from providence.loss import DiscreteWeibullLoss
from providence.nn import ProvidenceGRU
from providence.nn import ProvidenceLSTM
from providence.nn import transformer
from providence.nn.rnn import ProvidenceRNN
from providence.nn.rnn import ProvidenceVanillaRNN
from providence.nn.transformer import deepmind as dm
from providence.nn.transformer.transformer import MultiheadedAttention2
from providence.nn.transformer.transformer import MultiheadedAttention3
from providence.training import unpack_label_and_censor

pt.autograd.set_detect_anomaly(True)

N_FEATURES = 20
TIME_STEPS = 100
BATCH_SIZE = 30


@pytest.fixture
def input_tensor():
    return pt.randn(TIME_STEPS, BATCH_SIZE, N_FEATURES)


@pytest.mark.parametrize(
    "providence_model",
    [
        (
            ProvidenceGRU,
            {"input_size": 20, "dropout": 0.3, "num_layers": 2, "hidden_size": 100},
            2,
        ),
        (
            ProvidenceLSTM,
            {"input_size": 20, "dropout": 0.3, "num_layers": 2, "hidden_size": 100},
            2,
        ),
        (
            ProvidenceVanillaRNN,
            {"input_size": 20, "dropout": 0.3, "num_layers": 2, "hidden_size": 100},
            2,
        ),
        (
            ProvidenceGRU,
            {
                "input_size": 20,
                "dropout": 0.3,
                "num_layers": 2,
                "hidden_size": 100,
                "activation": "weibull3",
            },
            3,
        ),
        (
            ProvidenceLSTM,
            {
                "input_size": 20,
                "dropout": 0.3,
                "num_layers": 2,
                "hidden_size": 100,
                "activation": "weibull3",
            },
            3,
        ),
        (
            ProvidenceVanillaRNN,
            {
                "input_size": 20,
                "dropout": 0.3,
                "num_layers": 2,
                "hidden_size": 100,
                "activation": "weibull3",
            },
            3,
        ),
    ],
)
def test_model_train(providence_model, input_tensor):
    model, kwargs, expected_output_size = providence_model
    model = model(**kwargs)
    params = model(input_tensor, np.arange(30, 0, -1).tolist())
    assert len(params) == expected_output_size
    for p in params:
        assert p.shape == (30, BATCH_SIZE, 1)


@pytest.mark.parametrize(
    "model_and_kwargs",
    [
        (
            ProvidenceGRU,
            {
                "dropout": 0.3,
                "num_layers": 2,
                "hidden_size": 100,
                "activation": "weibull3",
            },
        ),
        (
            ProvidenceLSTM,
            {
                "dropout": 0.3,
                "num_layers": 2,
                "hidden_size": 100,
                "activation": "weibull3",
            },
        ),
        (
            ProvidenceVanillaRNN,
            {
                "dropout": 0.3,
                "num_layers": 2,
                "hidden_size": 100,
                "activation": "weibull3",
            },
        ),
    ],
)
def test_rnn_successful_end2end(simple_providence_ds, model_and_kwargs):
    "Make sure that we can invoke the model with the full dressings."
    model_type, kwargs = model_and_kwargs
    kwargs["input_size"] = simple_providence_ds.n_features
    model = model_type(**kwargs)
    examples, lengths, targets = providence_collate_fn([simple_providence_ds[i] for i in range(3)])

    output_params = model(examples, lengths)

    y, censor = unpack_label_and_censor(targets)

    loss_fn = DiscreteWeibull3Loss()
    loss = loss_fn(Weibull3.Params(*output_params), y, censor, lengths)

    loss.backward()


@pytest.mark.parametrize(
    "providence_model",
    [
        (
            transformer.ProvidenceTransformer,
            {
                "model_dimension": N_FEATURES,
                "dropout": 0.3,
                "n_layers": 2,
                "hidden_size": 100,
                "n_attention_heads": 2,
            },
        ),
    ],
)
def test_model_train_transformer(providence_model, input_tensor):
    model_init, kwargs = providence_model
    model = model_init(**kwargs)
    alpha, beta = model(input_tensor, np.arange(30, 0, -1).tolist())
    assert tuple(alpha.shape) == (TIME_STEPS, BATCH_SIZE, 1)
    assert tuple(beta.shape) == (TIME_STEPS, BATCH_SIZE, 1)


@pytest.mark.parametrize(
    "providence_model",
    [
        (ProvidenceGRU, {"input_size": 24}),
        (ProvidenceLSTM, {"input_size": 24}),
        (ProvidenceVanillaRNN, {"input_size": 24}),
    ],
)
def test_model_init(providence_model):
    model, kwargs = providence_model
    model = model(**kwargs)

    want_input_size = 24
    want_dropout = 0.0
    want_num_layers = 2
    want_hidden_size = 24

    got_input_size = model.input_size
    got_dropout = model.dropout
    got_num_layers = model.num_layers
    got_hidden_size = model.hidden_size

    assert got_input_size == want_input_size
    assert got_dropout == want_dropout
    assert got_num_layers == want_num_layers
    assert got_hidden_size == want_hidden_size


@pytest.mark.parametrize(
    "providence_model",
    [
        (ProvidenceGRU, {"input_size": 24}),
        (ProvidenceLSTM, {"input_size": 24}),
        (ProvidenceVanillaRNN, {"input_size": 24}),
    ],
)
def test_reset_parameters(providence_model):
    model, kwargs = providence_model
    model_1: ProvidenceRNN = model(**kwargs)

    def copy_rnn_parameters(state_dict: OrderedDict[str, pt.nn.parameter.Parameter]) -> OrderedDict[str, pt.Tensor]:
        new = OrderedDict()
        for k, v in state_dict.items():
            if isinstance(v, pt.nn.parameter.UninitializedParameter):  # skip over lazy tensors and stuff
                continue
            new[k] = v.detach().clone()
        return new

    initial_state = copy_rnn_parameters(model_1.state_dict())  # copy the state for comparison, later

    model_1.reset_parameters()
    model_1.reset_parameters()  # second time should get different results, right?

    new_state = copy_rnn_parameters(model(**kwargs).state_dict())  # reinitialized model should have a different state

    for k in initial_state.keys():
        assert not pt.equal(initial_state[k], new_state[k]), "Should be impossible to get the same exact state again"


################################################################################
#
# Transformer Tests In-Depth
#
################################################################################


@pytest.mark.transformers
class TestProvidenceTransformer:
    @pytest.fixture
    def model_temporal(self, simple_providence_ds) -> transformer.ProvidenceTransformer:
        return transformer.ProvidenceTransformer(
            model_dimension=simple_providence_ds.n_features,
            hidden_size=12,
            n_attention_heads=2,
        )

    # TODO: test against the dataset
    """
    1. masking is applied appropriately
    2. parameterization of the masking changes to be offsets rather than traditional Transformer parameters
    """

    def test_successful_end2end(self, simple_providence_ds, model_temporal: transformer.ProvidenceTransformer):
        "Make sure that we can invoke the model with the full dressings."

        examples, lengths, targets = providence_collate_fn([simple_providence_ds[i] for i in range(3)])
        print(f"{examples.size() = }")
        print(f"{lengths = }")
        print(f"{targets.size() = }")

        # NOTE(stephen): will need to be updated when the ProvidenceTransformer gets another distribution
        alpha_tens, beta_tens = model_temporal(examples, lengths)

        y, censor = unpack_label_and_censor(targets)

        loss_fn = DiscreteWeibullLoss()
        loss = loss_fn(Weibull.Params(alpha_tens, beta_tens), y, censor, lengths)

        loss.backward()

    def test_successful_end2end__mha2(self, simple_providence_ds, model_temporal: transformer.ProvidenceTransformer):
        "Make sure that we can invoke the model with the full dressings."
        model_temporal = transformer.ProvidenceTransformer(
            model_temporal.d_model,
            model_temporal.ff_dimension,
            n_layers=model_temporal.n_layers,
            n_attention_heads=model_temporal.n_attention_heads,
            dropout=model_temporal.dropout,
            layer_norm_epsilon=model_temporal.layer_norm_epsilon,
            positional_encoding_dimension=model_temporal.positional_encoding_dimension,
            attention_axis=model_temporal.attention_axis,
            device=model_temporal.device,
            t_attention=MultiheadedAttention2,
        )

        examples, lengths, targets = providence_collate_fn([simple_providence_ds[i] for i in range(3)])
        alpha_tens, beta_tens = model_temporal(examples, lengths)

        y, censor = unpack_label_and_censor(targets)

        loss_fn = DiscreteWeibullLoss()
        loss = loss_fn(Weibull.Params(alpha_tens, beta_tens), y, censor, lengths)
        loss.backward()

    def test_successful_end2end__mha3(self, simple_providence_ds, model_temporal: transformer.ProvidenceTransformer):
        "Make sure that we can invoke the model with the full dressings."
        model_temporal = transformer.ProvidenceTransformer(
            model_temporal.d_model,
            model_temporal.ff_dimension,
            n_layers=model_temporal.n_layers,
            n_attention_heads=model_temporal.n_attention_heads,
            dropout=model_temporal.dropout,
            layer_norm_epsilon=model_temporal.layer_norm_epsilon,
            positional_encoding_dimension=model_temporal.positional_encoding_dimension,
            attention_axis=model_temporal.attention_axis,
            device=model_temporal.device,
            t_attention=MultiheadedAttention3,
        )

        examples, lengths, targets = providence_collate_fn([simple_providence_ds[i] for i in range(3)])
        alpha_tens, beta_tens = model_temporal(examples, lengths)

        y, censor = unpack_label_and_censor(targets)

        loss_fn = DiscreteWeibullLoss()
        loss = loss_fn(Weibull.Params(alpha_tens, beta_tens), y, censor, lengths)
        loss.backward()

    def test_successful_end2end_BERTish_model(
        self,
        simple_providence_ds: ProvidenceDataset,
        model_temporal: transformer.ProvidenceTransformer,
    ):
        "Make sure that we can invoke the model with the full dressings."
        model = dm.ProvidenceBertTransformer(
            n_heads=model_temporal.n_attention_heads,
            n_layers=model_temporal.n_layers,
            n_features_in=simple_providence_ds.n_features,
            n_embedding=8,  # NOTE: embedding is relevant
            mask_mode=dm.MaskMode.backward_only,
            att_inner_dim=128,  # NOTE: you can swap this out. Defaults to 256
            ff_dim=model_temporal.ff_dimension,
            # NOTE(stephen): this is how the positional_encoding limitation corresponds to the new arch
            max_seq_len=model_temporal.positional_encoding_dimension,
        )
        print(f"{len(simple_providence_ds) = }")

        examples, lengths, targets = providence_collate_fn(
            [simple_providence_ds[i] for i in range(len(simple_providence_ds))]
        )
        alpha_tens, beta_tens = model(examples, lengths)

        y, censor = unpack_label_and_censor(targets)

        loss_fn = DiscreteWeibullLoss()
        loss = loss_fn(Weibull.Params(alpha_tens, beta_tens), y, censor, lengths)
        loss.backward()
