"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.paper_reproductions import (
    BackblazeExtendedRNN, BackblazeExtendedTransformer, BackblazeRNN, BackblazeTransformer, NasaFD001Transformer, NasaFD002Transformer,
    NasaFD003Transformer, NasaFD004Transformer, NasaRNN, NasaTransformer
)


def test_models_can_be_instantiated():

    model_constructors = [
        NasaTransformer, NasaRNN, NasaFD001Transformer, NasaFD002Transformer, NasaFD003Transformer, NasaFD004Transformer,
        BackblazeTransformer, BackblazeRNN, BackblazeExtendedTransformer, BackblazeExtendedRNN
    ]

    for ctor in model_constructors:
        ctor()

    assert True, "Should be able to instantiate all the models"
