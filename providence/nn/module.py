"""
Specification of a `torch.nn.Module` that would work with Providence, in theory. In our testing, this is the minimal skeleton that
you need to properly perform inference and a backwards pass, while also behaving nicely with our automatic metric evaluation.

If you already have a model that should work with our code, but are having difficulties, try
>>> my_model = ... # your model initialization code here
>>> my_model.device = device('cuda')

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from typing import Protocol, Tuple

from torch import Tensor, device
from torchtyping import TensorType
class ProvidenceModule(Protocol):
    """Expectations of a model that's going to be supported by the framework"""

    def reset_parameters() -> None:
        """Reset the parameters or reinitialize the model"""
        ...

    def forward(self, input: TensorType["time", "entity", "features"], input_lengths: TensorType["entity"]) -> Tuple[Tensor, ...]:
        """Inference should return one or more tensors"""
        ...

    @property
    def device(self) -> device:
        """A device on which the Module will be trained or deployed i.e. a GPU, CPU, or TPU"""
        ...

