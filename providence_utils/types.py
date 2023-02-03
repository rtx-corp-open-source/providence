"""
Bottom of the type stack for the utility classes
TODO: delete and use the types in core providence

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Tuple, Union

import torch

LossValueType = Union[float, torch.Tensor]


class Losses:
    def __init__(self, train: LossValueType, validation: LossValueType):
        self.training = train
        self.validation = validation

    def to_tuple(self) -> Tuple[LossValueType, LossValueType]:
        return self.training, self.validation

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.to_tuple()}"
