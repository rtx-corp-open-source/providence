"""
Named for the facilities it provides being extensions that Python doesn't make easy to pull off otherwise.
This is most notable in `with_weibull(...)`

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from ..type_utils import patch

from torch.nn import Module
from torchtyping import TensorType

from .module import ProvidenceModule
from .weibull import WeibullHead


@patch(cls_method=True)
def with_weibull(cls: ProvidenceModule, model: ProvidenceModule):
    # TODO: unit test me
    return WeibullWrapper(model)


class WeibullWrapper(Module):
    """
    A wrapper on any dense-output Module that complies with the ProvidenceModule protocol.
    This should allow a simplifying reimplementation of (say) ProvidenceTransformer a la
    >>> class Transformer(Module):
        ...
        def forward(self, input, input_lengths) -> Tensor:
            mask = mask_from_lengths(input_lengths)
            output = self.model(input, mask)
            return output
    >>> def ProvidenceTransformer(**kwargs):
            transformer = Transformer(**kwargs)
            return ProvidenceModule.with_weibull(transformer)
    
    Again, this will expedite the construction of Providence models, reduce code base size, flatten the API, while
    improving the granularity coverage. That is, we'll have mid-level (e.g. ProvidenceModule.with_xxx(model)),
    high-level (e.g. ProvidenceTransformer(), ProvidenceLSTM()), and lower-level faculties (e.g. this file)
    """
    def __init__(self, model: ProvidenceModule):
        super().__init__()
        self.model = model
        self.weibull = WeibullHead()

    def reset_parameters(self):
        self.model.reset_parameters()
        self.weibull.reset_parameters()

    def forward(self, x: TensorType["time", "device", "features"], x_lengths: TensorType["device"]):
        output = self.model(x, x_lengths)
        return self.weibull(output)

