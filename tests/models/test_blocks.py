# -*- coding: utf-8 -*-
import pytest
import torch
import numpy as np

from providence.model.blocks import activation


N_FEATURES = 20
TIME_STEPS = 100
BATCH_SIZE = 30


@pytest.fixture
def input_tensor():
    return torch.randn(TIME_STEPS, BATCH_SIZE, N_FEATURES)


class TestActivation:
    def test_init(self):
        layer = activation.WeibullActivation(input_size=20)

        assert layer.input_size == 20


@pytest.mark.parametrize("providence_layer", [(activation.WeibullActivation, {"input_size": 20})])
def test_train(providence_layer, input_tensor):
    layer, kwargs = providence_layer
    layer = layer(**kwargs)
    alpha, beta = layer(input_tensor)

    assert alpha.shape == (TIME_STEPS, BATCH_SIZE, 1)
    assert beta.shape == (TIME_STEPS, BATCH_SIZE, 1)
