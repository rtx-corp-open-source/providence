"""
All of the Recurrent Neural Networks (RNNs) that are shipped with Providence.
We provide simple constructor functions to use LSTMs, GRUs, or basic 'vanilla' RNNs, none of which have docstrings
for their two line implementations

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Final
from typing import List
from typing import Tuple
from typing import Union

from torch import device
from torch.nn import GRU
from torch.nn import LSTM
from torch.nn import Module
from torch.nn import RNN
from torch.nn import RNNBase
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from providence.types import LengthsTensor
from providence.types import ProvidenceTensor


# NOTE: This is a long, deferred type reference to the activation layer because of Python's poor module structure.
_SUPPORTED_ACTIVIATIONS: Final = frozenset(["weibull", "weibull3"])


def get_activation(activation_name: str) -> Module:
    """Load the activation head that matches ``activation_name``.

    Args:
        activation_name (str): name of the activation. Use a supported activation to avoid an error

    Raises:
        ValueError: if the activation name is not one of the support activations
            ``"weibull"``, or ``"weibull3"``

    Returns:
        Module: activation "Head" that can be used in any nn.Module for producing Providence Distributions.
    """
    from .weibull import (
        WeibullHead,
        Weibull3Head,
    )  # done late because Python imports are headache inducing.

    # NOTE: should be a case-match
    if activation_name == "weibull":
        return WeibullHead()

    if activation_name == "weibull3":
        return Weibull3Head()

    raise ValueError(f"Activation '{activation_name}' not supported by Providence models")


_ProvidenceLengths = Union[List[int], LengthsTensor]


class ProvidenceRNN(Module):
    """Base Module type for arbitrary RNNs, given the simplicity of the RNN class hierarchy in PyTorch.

    Organized around two principles
    1. Alternative activations are easy to swap into
    2. The forward() implementation is RNN-implementation-agnostic.

    As an added benefit, it minimizes the code surface for implementing a new Providence recurrent model.

    Args:
        rnn (RNNBase): some instance of an RNN e.g. nn.LSTM
        activation (str, optional): Providence distribution activatian. Defaults to "weibull".
        device (pt.device, optional): Torch device for training this instance. Defaults to device("cpu").

    Raises:
        ValueError: if ``activation`` is not supported. See ``get_activation()``
    """

    def __init__(self, rnn: RNNBase, *, activation: str = "weibull", device=device("cpu")):
        super().__init__()
        self.rnn = rnn
        if activation not in _SUPPORTED_ACTIVIATIONS:
            raise ValueError(f"Activation '{activation}' not supported by Providence models")

        self.activation_p = activation
        self.device = device
        self.reset_parameters()

    @property
    def input_size(self) -> int:
        """Input size of - or number of features received by - the inner ``rnn``.

        Returns:
            int: input size
        """
        return self.rnn.input_size

    @property
    def hidden_size(self) -> int:
        """hidden size of - or number of features in the hidden dimension received by - the inner ``rnn``.

        Returns:
            int: hidden size
        """
        return self.rnn.hidden_size

    @property
    def num_layers(self) -> int:
        """Number of recurrent cells attached - or the number of layers used - in the inner ``rnn``.

        Returns:
            int: number of layers
        """
        return self.rnn.num_layers

    @property
    def dropout(self) -> float:
        """Dropout rate used in the model.

        Returns:
            float: dropout rate in range [0.0, 1.0]
        """
        return self.rnn.dropout

    def reset_parameters(self):
        """Initialize model parameters based on fields.

        Used to programmatically (re-)initialize this instance
        """
        if getattr(self, "activation", None) is None:
            self.activation = get_activation(self.activation_p)
            # WeibullActivation(self.rnn.hidden_size)

        self.rnn.reset_parameters()  # for RNNBase, Modules uses standard normalization
        self.activation.reset_parameters()

    def forward(self, x: ProvidenceTensor, x_lengths: _ProvidenceLengths) -> Tuple[ProvidenceTensor, ...]:  # type: ignore[valid-type]
        """Perform RNN forward pass.

        Args:
            x (ProvidenceTensor): pt.Tensor of inputs
            x_lengths (_ProvidenceLengths): the lengths corresponding to those inputs

        Returns:
            Tuple[ProvidenceTensor, ...]: n-tensors with a final dimension of 1
        """

        # NOTE: leave this inline so Python doesn't waste time with dereference in calling `providence_rnn_infer(...)`
        packed = pack_padded_sequence(x, x_lengths, batch_first=False)

        rnn_outputs, _ = self.rnn(packed)

        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=False)

        # In the feature, you would add invocations here
        output = self.activation(rnn_outputs)

        return output


def providence_rnn_infer(
    rnn: RNNBase,
    examples: ProvidenceTensor,
    x_lengths: _ProvidenceLengths,
):
    """Inference for an Providence RNN, without the Providence Distribution activation.

    Meant to many support ideas like transfer learning and new versions of Providence-compatible RNN
    with lower implementation overhead.

    Args:
        rnn (RNNBase): an instance of an RNN
        examples (Tensor): examples to perform inference on
        x_lengths (List[int]): lengths of the entities in ``examples``

    Returns:
        Tensor: the pre-activation inference of a Providence RNN
    """
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    packed = pack_padded_sequence(examples, x_lengths, batch_first=False, enforce_sorted=False)
    rnn_outputs, _ = rnn(packed)
    # Packing padded sequences is a trick to speed up computation
    # for more information check out https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
    rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=False)
    return rnn_outputs


def ProvidenceGRU(
    input_size: int,
    hidden_size: int = 24,
    num_layers: int = 2,
    dropout: float = 0.0,
    *,
    activation: str = "weibull",
    device=device("cpu"),
) -> ProvidenceRNN:
    """Construct a GRU for the Providence problem.

    Args:
        input_size (int): number of input features
        hidden_size (int, optional): hidden dimension of the model. Defaults to 24.
        num_layers (int, optional): number of inner recurrent cells. Defaults to 2.
        dropout (float, optional): dropout percentage rate in range [0.0, 1.0]. Defaults to 0.0.
        activation (str, optional): name of the Providence activation. Defaults to "weibull".
        device (pt.device, optional): Torch device for inference and training. Defaults to device("cpu").

    Returns:
        ProvidenceRNN: a Providence-ready GRU
    """
    gru = GRU(input_size, hidden_size, num_layers, dropout=dropout)
    return ProvidenceRNN(gru, activation=activation, device=device)


def ProvidenceLSTM(
    input_size: int,
    hidden_size: int = 24,
    num_layers: int = 2,
    dropout: float = 0.0,
    *,
    activation: str = "weibull",
    device=device("cpu"),
) -> ProvidenceRNN:
    """Construct a LSTM for the Providence problem.

    Args:
        input_size (int): number of input features
        hidden_size (int, optional): hidden dimension of the model. Defaults to 24.
        num_layers (int, optional): number of inner recurrent cells. Defaults to 2.
        dropout (float, optional): dropout percentage rate in range [0.0, 1.0]. Defaults to 0.0.
        activation (str, optional): name of the Providence activation. Defaults to "weibull".
        device (pt.device, optional): Torch device for inference and training. Defaults to device("cpu").

    Returns:
        ProvidenceRNN: a Providence-ready LSTM
    """
    lstm = LSTM(input_size, hidden_size, num_layers, dropout=dropout)
    return ProvidenceRNN(lstm, activation=activation, device=device)


def ProvidenceVanillaRNN(
    input_size: int,
    hidden_size: int = 24,
    num_layers: int = 2,
    dropout: float = 0.0,
    *,
    activation: str = "weibull",
    rnn_nonlinearity: str = "tanh",
    device=device("cpu"),
) -> ProvidenceRNN:
    """Construct a Vanilla RNN (the most simple recurrent neural network module) for the Providence problem.

    Args:
        input_size (int): number of input features
        hidden_size (int, optional): hidden dimension of the model. Defaults to 24.
        num_layers (int, optional): number of inner recurrent cells. Defaults to 2.
        dropout (float, optional): dropout percentage rate in range [0.0, 1.0]. Defaults to 0.0.
        activation (str, optional): name of the Providence activation. Defaults to "weibull".
        device (pt.device, optional): Torch device for inference and training. Defaults to device("cpu").

    Returns:
        ProvidenceRNN: a Providence-ready Vanilla RNN
    """
    vanilla_rnn = RNN(
        input_size,
        hidden_size,
        num_layers,
        dropout=dropout,
        nonlinearity=rnn_nonlinearity,
    )
    return ProvidenceRNN(vanilla_rnn, activation=activation, device=device)
