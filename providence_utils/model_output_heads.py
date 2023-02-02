"""
Experimntal module "heads" that would support additional functionality or easy transfer learning.
EXPERIMENTAL. DO NOT USE FOR PRODUCTION ANYTHING

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from torch import einsum
from torch.nn import LazyLinear, Module, functional as F
from torchtyping import TensorType

class ClassificationHead(Module):
    def __init__(self, *, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.inner = LazyLinear(self.n_classes)

    def forward(self, examples: TensorType["time", "entity", "features"]) -> TensorType["entity", "probabilities"]:
        mapped = self.inner(einsum("...ef -> ef", examples))
        return F.softmax(mapped, dim=-1)
