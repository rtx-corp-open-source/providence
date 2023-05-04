"""
Named for the facilities it provides being extensions that Python doesn't make easy to pull off otherwise.
This is most notable in `with_weibull(...)`

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Tuple
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module

from providence.types import LengthsTensor

from ..type_utils import patch
from .module import ProvidenceModule
from .weibull import WeibullHead


@patch(cls_method=True)
def with_weibull(cls: ProvidenceModule, model: ProvidenceModule):
    """Give a module that follows the Providence Protocol a WeibullHead.

    HACK: this is hacking around the Python type system, method resolution order, and dependency resolver
    Created to facilitate the following usage - chiefly, for Transfer learning - to minimize lines of code
    to write a neural network, train for the Providence, and then leverage the core model without the Weibull head.

    Example:
    >>> myNetworkWithProvidenceHead = ProvidenceModule.with_weibull(myNeuralNetwork)
    >>> train(myNetworkWithProvidenceHead)
    >>> myNeuralNetwork # is now trained, and usable for other things.

    Args:
        cls (ProvidenceModule): ignored
        model (ProvidenceModule): the instance to wrap with the new WeibullHead

    Returns:
        WeibullWrapper: containing ``model`` which outputs will be fed to WeibullHead,
            producing Providence Distributions
    """
    # TODO: unit test me
    return WeibullWrapper(model)


class WeibullWrapper(Module):
    """Convert any dense-output Module that complies with the ProvidenceModule protocol *to* the providence protocol.

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
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        self.model.reset_parameters()
        self.weibull.reset_parameters()

    def forward(
        self, x: Float[Tensor, "... time entity features"], x_lengths: LengthsTensor
    ) -> Tuple[Float[Tensor, "... time entity 1"], ...]:
        """Perform the standard Providence forward pass.

        Args:
            x (Tensor): input, shape ``[time, device, features]``
            x_lengths (Tensor): lengths of entities, shape ``[device, ]``

        Returns:
            Tuple[Tensor, Tensor]: a tuple of two tensors, where the first and second are a Weibull's alpha and beta,
                respectively.
        """
        output = self.model(x, x_lengths)  # type: ignore[name-defined]
        return self.weibull(output)
