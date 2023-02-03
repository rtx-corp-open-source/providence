# -*- coding: utf-8 -*-
"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import pytest
import torch as pt

from providence.nn import weibull


N_FEATURES = 20
TIME_STEPS = 100
BATCH_SIZE = 30


@pytest.fixture
def input_tensor():
    return pt.randn(TIME_STEPS, BATCH_SIZE, N_FEATURES)


class TestActivation:
    def test_init(self):
        layer = weibull.WeibullActivation(input_size=20)

        assert layer.input_size == 20


@pytest.mark.parametrize("providence_layer", [(weibull.WeibullActivation, {"input_size": 20}),
    (weibull.WeibullHead, {}),
    (weibull.WeibullHead2, {})
])
def test_weibull_activations(providence_layer, input_tensor):
    layer, kwargs = providence_layer
    layer = layer(**kwargs)
    alpha, beta = layer(input_tensor)

    assert alpha.shape == (TIME_STEPS, BATCH_SIZE, 1)
    assert beta.shape == (TIME_STEPS, BATCH_SIZE, 1)

def test_weibull3_activations(input_tensor):
    layer = weibull.Weibull3Head()

    alpha, beta, k = layer(input_tensor)

    for p in (alpha, beta, k):
        assert p.shape == (TIME_STEPS, BATCH_SIZE, 1)