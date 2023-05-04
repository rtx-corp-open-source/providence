"""
Specification of a `torch.nn.Module` that would work with Providence, in theory. In our testing, this is the minimal skeleton that
you need to properly perform inference and a backwards pass, while also behaving nicely with our automatic metric evaluation.

If you already have a model that should work with our code, but are having difficulties, try
>>> my_model = ... # your model initialization code here
>>> my_model.device = device('cuda')

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Any, Protocol, TypeVar, runtime_checkable
from typing import Iterator
from typing import Tuple

from torch import device
from torch import Tensor
from torch.nn.parameter import Parameter

from providence.types import LengthsTensor
from providence.types import ProvidenceTensor

T = TypeVar("T")


@runtime_checkable  # this is newly necessary. No clear reason why.
class ProvidenceModule(Protocol):
    """Contract of a model that's supported by the framework."""

    def reset_parameters(self) -> None:
        """Reset the parameters or reinitialize the model."""
        ...

    def forward(self, X: ProvidenceTensor, lengths: LengthsTensor) -> Tuple[Tensor, ...]:
        """Inference should return one or more tensors."""
        # NOTE: the type on ``lengths`` is a vector, and we need to use the weird reference for mypy to chill.
        # Yes, it should work with just ``Int[Tensor, "batch"]``, but it doesn't.
        # It was supposedly fixed in v 0.2.5 (https://github.com/google/jaxtyping/issues/29#issuecomment-1255511781) but wasn't.
        # We will see about a PR.
        ...

    def __call__(self, X: ProvidenceTensor, lengths: LengthsTensor, *args: Any, **kwds: Any) -> Any:
        """Support the fast-inference of forward with Module.__call__. Should NOT be overridden

        See torch.nn.Module docs for more.
        """
        raise NotImplementedError()

    @property
    def device(self) -> device:
        """A device on which the Module will be trained or deployed i.e. a GPU, CPU, or TPU."""
        ...

    # NOTE(stephen): the following correspond to nn.Module essentials until we find a better way to implement this.
    # docstrings are adapted from PyTorch documentation

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Returns an iterator over this and all child Modules if ``recurse``, otherwise only this Module."""
        ...

    def train(self: T, mode: bool = True) -> T:
        """Sets the module in training mode.

        This has any effect only on certain modules, but should recursively traverse child modules. See documentations
        of particular modules for details of their behaviors in training/evaluation mode, and if they are affected,
        e.g. ``Dropout``, ``BatchNorm``, etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        ...

    def eval(self: "ProvidenceModule") -> "ProvidenceModule":
        """Sets the module in evaluation mode.

        This has any effect only on certain modules, but should recursively traverse child modules. See documentations
        of particular modules for details of their behaviors in training/evaluation mode, and if they are affected,
        e.g. ``nn.Dropout``, ``nn.BatchNorm``, etc.

        This is equivalent with ``self.train(False)``

        Returns:
            Module: self
        """
        return self.train(False)

    def to(self, *args, **kwargs):
        """Moves and/or casts the parameters and buffers to devices or types, respectively.

        Returns:
            Module: self
        """
