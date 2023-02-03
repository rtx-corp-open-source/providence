"""
All of the Recurrent Neural Networks (RNNs) that are shipped with Providence.
We provide simple constructor functions to use LSTMs, GRUs, or basic 'vanilla' RNNs, none of which have docstrings
for their two line implementations

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from typing import Final, List, Tuple, Union

from torch import device
from torch.nn import LSTM, GRU, RNN, Module, RNNBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtyping import TensorType


# NOTE: This is a long, deferred type reference to the activation layer because of Python's poor module structure.
_SUPPORTED_ACTIVIATIONS: Final = frozenset(['weibull', 'weibull3'])

def get_activation(activation_name: str) -> Module:
    from .weibull import WeibullHead, Weibull3Head # done late because Python imports are headache inducing.
    # NOTE: should be a case-match
    if activation_name == 'weibull':
        return WeibullHead()
    
    if activation_name == 'weibull3':
        return Weibull3Head()

    raise ValueError(f"Activation '{activation_name}' not supported by Providence models")

_ProvidenceLengths = Union[List[int], TensorType["length_per_sequence"]]
ProvidenceTensor = TensorType["time", "entity", "feature"]

class ProvidenceRNN(Module):
    """
    A container Module type for arbitrary RNNs, given the simplicity of the RNN class hierarchy in PyTorch.
    1. Alternative activations (to be implemented in the future)
    2. The forward() implementation is RNN-implementation-agnostic.

    As an added benefit, it minimizes the code surface for implementing a new Providence recurrent model.
    """
    def __init__(self, rnn: RNNBase, *, activation: str = 'weibull', device = device('cpu')):
        super().__init__()
        self.rnn = rnn
        if activation not in _SUPPORTED_ACTIVIATIONS:
            raise ValueError(f"Activation '{activation}' not supported by Providence models")

        self.activation_p = activation
        self.device = device
        self.reset_parameters()

    @property
    def input_size(self) -> int:
        return self.rnn.input_size

    @property
    def hidden_size(self) -> int:
        return self.rnn.hidden_size

    @property
    def num_layers(self) -> int:
        return self.rnn.num_layers

    @property
    def dropout(self) -> float:
        return self.rnn.dropout

    def reset_parameters(self):
        if getattr(self, 'activation', None) is None:
            self.activation = get_activation(self.activation_p)
            # WeibullActivation(self.rnn.hidden_size)

        self.rnn.reset_parameters()  # for RNNBase, Modules uses standard normalization
        self.activation.reset_parameters()

    def forward(self, x: ProvidenceTensor, x_lengths: _ProvidenceLengths) -> Tuple[ProvidenceTensor, ...]:
        # NOTE: leave this inline so Python doesn't waste time with dereference in calling `providence_rnn_infer(...)`
        packed = pack_padded_sequence(x, x_lengths, batch_first=False)

        rnn_outputs, _ = self.rnn(packed)

        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=False)

        # In the feature, you would add invocations here
        output = self.activation(rnn_outputs)

        return output


def providence_rnn_infer(rnn: RNNBase, examples: TensorType["timestep", "batch", "feature"], x_lengths: List[int]):
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
    dropout: float = 0.,
    *,
    activation: str = 'weibull',
    device = device('cpu')
) -> ProvidenceRNN:
    gru = GRU(input_size, hidden_size, num_layers, dropout=dropout)
    return ProvidenceRNN(gru, activation=activation, device=device)


def ProvidenceLSTM(
    input_size: int,
    hidden_size: int = 24,
    num_layers: int = 2,
    dropout: float = 0.,
    *,
    activation: str = 'weibull',
    device = device('cpu')
) -> ProvidenceRNN:
    lstm = LSTM(input_size, hidden_size, num_layers, dropout=dropout)
    return ProvidenceRNN(lstm, activation=activation, device=device)


def ProvidenceVanillaRNN(
    input_size: int,
    hidden_size: int = 24,
    num_layers: int = 2,
    dropout: float = 0.,
    *,
    activation: str = 'weibull',
    rnn_nonlinearity: str = 'tanh',
    device = device('cpu')
) -> ProvidenceRNN:
    vanilla_rnn = RNN(input_size, hidden_size, num_layers, dropout=dropout, nonlinearity=rnn_nonlinearity)
    return ProvidenceRNN(vanilla_rnn, activation=activation, device=device)
