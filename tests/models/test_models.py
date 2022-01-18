# -*- coding: utf-8 -*-
import copy

import numpy as np
import pytest
import torch
from providence.model import gru, lstm, rnn, transformer

torch.autograd.set_detect_anomaly(True)

N_FEATURES = 20
TIME_STEPS = 100
BATCH_SIZE = 30


@pytest.fixture
def input_tensor():
    return torch.randn(TIME_STEPS, BATCH_SIZE, N_FEATURES)


@pytest.mark.parametrize(
    "providence_model",
    [
        (gru.ProvidenceGRU, {"input_size": 20, "dropout": 0.3, "n_layers": 2, "hidden_size": 100}),
        (lstm.ProvidenceLSTM, {"input_size": 20, "dropout": 0.3, "n_layers": 2, "hidden_size": 100}),
        (rnn.ProvidenceVanillaRNN, {"input_size": 20, "dropout": 0.3, "n_layers": 2, "hidden_size": 100}),
    ],
)
def test_model_train(providence_model, input_tensor):
    model, kwargs = providence_model
    model = model(**kwargs)
    alpha, beta = model(input_tensor, np.arange(30, 0, -1).tolist())
    assert tuple(alpha.shape) == (30, BATCH_SIZE, 1)
    assert tuple(beta.shape) == (30, BATCH_SIZE, 1)


@pytest.mark.parametrize(
    "providence_model",
    [
        (
            transformer.ProvidenceTransformer,
            {"model_dimension": N_FEATURES, "dropout": 0.3, "n_layers": 2, "hidden_size": 100, "n_attention_heads": 2},
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
    [(gru.ProvidenceGRU, {"input_size": 24}), (lstm.ProvidenceLSTM, {"input_size": 24}), (rnn.ProvidenceVanillaRNN, {"input_size": 24})],
)
def test_model_init(providence_model):
    model, kwargs = providence_model
    model = model(**kwargs)

    want_input_size = 24
    want_dropout = 0.0
    want_n_layers = 1
    want_hidden_size = 20

    got_input_size = model.input_size
    got_dropout = model.dropout
    got_n_layers = model.n_layers
    got_hidden_size = model.hidden_size

    assert got_input_size == want_input_size
    assert got_dropout == want_dropout
    assert got_n_layers == want_n_layers
    assert got_hidden_size == want_hidden_size


@pytest.mark.parametrize(
    "providence_model",
    [(gru.ProvidenceGRU, {"input_size": 24}), (lstm.ProvidenceLSTM, {"input_size": 24}), (rnn.ProvidenceVanillaRNN, {"input_size": 24})],
)
def test_reset_parameters(providence_model):
    model, kwargs = providence_model
    model_1 = model(**kwargs)

    # clone the model parameters into a new model object so we can compare
    model_clone = model(**kwargs)
    model_clone.load_state_dict(copy.deepcopy(model_1.state_dict()))

    # reset the model params
    model_1.reset_parameters()

    got = torch.equal(model_1.activation.alpha.weight.data, model_clone.activation.alpha.weight.data)
    want = False

    assert got == want
