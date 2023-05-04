"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.paper_reproductions import BackblazeExtendedRNN
from providence.paper_reproductions import BackblazeExtendedTransformer
from providence.paper_reproductions import BackblazeRNN
from providence.paper_reproductions import BackblazeTransformer
from providence.paper_reproductions import NasaFD001Transformer
from providence.paper_reproductions import NasaFD002Transformer
from providence.paper_reproductions import NasaFD003Transformer
from providence.paper_reproductions import NasaFD004Transformer
from providence.paper_reproductions import NasaRNN
from providence.paper_reproductions import NasaTransformer


def test_models_can_be_instantiated():
    model_constructors = [
        NasaTransformer,
        NasaRNN,
        NasaFD001Transformer,
        NasaFD002Transformer,
        NasaFD003Transformer,
        NasaFD004Transformer,
        BackblazeTransformer,
        BackblazeRNN,
        BackblazeExtendedTransformer,
        BackblazeExtendedRNN,
    ]

    for ctor in model_constructors:
        ctor()

    assert True, "Should be able to instantiate all the models"
