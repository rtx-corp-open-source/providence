"""
A handful of seperate activation "heads" that will produce outputs that should be fully compatible with our results.
In the initial research, `WeibullActivation` was used, but in the redesign of the library I discovered the `Lazy_` API
and felt we should adhere to better design principles when possible (e.g. avoid unnamed integer constructor arguments).

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Tuple

from torch import exp
from torch import split
from torch import Tensor
from torch.nn import functional as F
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module


class WeibullActivation(Module):
    """Activation layer for a two parameter Weibull.

    Args:
        input_size (int): The number of input features to this module i.e. the size of a network's last hidden layer

    Returns:
        Alpha and Beta tensors for the Weibull distribution
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.input_size: int = input_size
        self.output_size: int = 1

        self.reset_parameters()

    def reset_parameters(self):
        self.alpha = Linear(self.input_size, self.output_size)
        self.beta = Linear(self.input_size, self.output_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Map the input ``x`` (usually an embedding) to alpha, and beta tensors

        Args:
            x (Tensor): input tensor of rank 1 or more

        Returns:
            Tuple[Tensor, Tensor, Tensor]: alpha, beta Tensors of shape ``x.shape[:-1] + [1]``
        """
        # taken from https://github.com/ragulpr/wtte-rnn/blob/162f5c17f21db79a316d563b60835d178142fd69/python/wtte/wtte.py#L31

        alpha = self.alpha(x)
        alpha = exp(alpha)

        beta = self.beta(x)
        beta = F.softplus(beta)

        return alpha, beta


class WeibullHead(Module):
    """Output head for a two parameter Weibull

    Follows after WeibullActivation, but uses LazyLinear to not depend on the dimensionality i.e. improve decoupling
    w/o losing functionality

    Returns:
        Alpha and Beta tensors for the Weibull distribution
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset_parameters()

    def reset_parameters(self):
        self.alpha_proj = LazyLinear(1, bias=True)
        self.beta_proj = LazyLinear(1, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Map the input ``x`` (usually an embedding) to alpha, and beta tensors

        Args:
            x (Tensor): input tensor of rank 1 or more

        Returns:
            Tuple[Tensor, Tensor, Tensor]: alpha, beta Tensors of shape ``x.shape[:-1] + [1]``
        """
        alpha_in = self.alpha_proj(x)
        beta_in = self.beta_proj(x)
        alpha = exp(alpha_in)
        beta = F.softplus(beta_in)

        return alpha, beta


class WeibullHead2(Module):
    """EXPERIMENTAL: Output head for a two parameter Weibull

    Rather than `WeibullActivation`, output two regression targets from the same layer,
    then compute the Weibull values as desired.

    NOTE: idk if this is a useful approach, but I'm curious to test it out - especially as we're benchmarking

    Returns:
        Alpha and Beta tensors for the Weibull distribution
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset_parameters()

    def reset_parameters(self):
        self.dense = LazyLinear(2, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Map the input ``x`` (usually an embedding) to alpha, and beta tensors

        Args:
            x (Tensor): input tensor of rank 1 or more

        Returns:
            Tuple[Tensor, Tensor, Tensor]: alpha, beta Tensors of shape ``x.shape[:-1] + [1]``
        """
        _2d_output = self.dense(x)
        # split(...)'s API is weird. We want segments of size=1, on the last dimension
        alpha_in, beta_in = split(_2d_output, 1, dim=-1)
        alpha = exp(alpha_in)
        beta = F.softplus(beta_in)

        return alpha, beta


class Weibull3Head(Module):
    """EXPERIMENTAL: Output head for the three-parameter Weibull

    Follows after WeibullHead (see this module), but for the three-parameter Weibull model

    returns:
        Alpha, Beta, and K (translation / location) parameters-as-tensors for the Weibull distribution
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset_parameters()

    def reset_parameters(self):
        self.alpha_proj = LazyLinear(1, bias=True)
        self.beta_proj = LazyLinear(1, bias=True)
        self.k_proj = LazyLinear(1, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Map the input ``x`` (usually an embedding) to alpha, beta, and k tensors

        Args:
            x (Tensor): input tensor of rank 1 or more

        Returns:
            Tuple[Tensor, Tensor, Tensor]: alpha, beta, k Tensors of shape ``x.shape[:-1] + [1]``
        """
        alpha_in = self.alpha_proj(x)
        beta_in = self.beta_proj(x)
        k_in = self.k_proj(x)

        alpha = exp(alpha_in)
        beta = F.softplus(beta_in)
        k = k_in  # NOTE: we must experiment with necessary controls for the translation parameter
        return alpha, beta, k
