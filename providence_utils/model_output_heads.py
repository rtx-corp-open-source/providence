"""
Experimntal module "heads" that would support additional functionality or easy transfer learning.
EXPERIMENTAL. DO NOT USE FOR PRODUCTION ANYTHING

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from jaxtyping import Float
from torch import einsum
from torch import Tensor
from torch.nn import functional as F
from torch.nn import LazyLinear
from torch.nn import Module


time_entity_features = "time entity features"
entity_probabilities = "entity probabilities"


class ClassificationHead(Module):
    """EXPERIMENTAL: Unused classification head for a module.

    Created in pursuit of transfer learning from Providence models that perform well on the regression-based
    distribution-sequence learning problem

    Example:

        >>> nn.Sequential(my_architecture, ClassificationHead(n_classes))(examples).shape[-1] == n_classes
    """

    def __init__(self, *, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.inner = LazyLinear(self.n_classes)

    def forward(self, examples: Float[Tensor, time_entity_features]) -> Float[Tensor, entity_probabilities]:
        mapped = self.inner(einsum("...ef -> ef", examples))
        return F.softmax(mapped, dim=-1)
