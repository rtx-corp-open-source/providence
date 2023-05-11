from typing import Any, Iterator, Tuple
from typing_extensions import Self

from torch import Tensor, device, nn

from providence.nn.module import ProvidenceModule
from providence.types import LengthsTensor
from providence.types import ProvidenceTensor


class TestTypeSpec:
    # NOTE: instances of testC have `with_weibull()` to tolerate the late binding of that *class* method.
    def test_class_most_faithful_implementation(self):
        class testC:
            def reset_parameters(self) -> None:
                raise NotImplementedError()

            def forward(self, X: ProvidenceTensor, lengths: LengthsTensor) -> Tuple[Tensor, ...]:
                raise NotImplementedError()

            def __call__(self, X: ProvidenceTensor, lengths: LengthsTensor, *args: Any, **kwds: Any) -> Any:
                return self.forward(X, lengths)

            @property
            def device(self) -> device:
                ...

            def parameters(self, recurse=True) -> Iterator[nn.parameter.Parameter]:
                raise NotImplementedError()

            def train(self: 'testC', b=True) -> 'testC':
                return self

            def eval(self: 'testC') -> 'testC':
                return self.train(False)

            def to(self, *args, **kwargs) -> Any:
                return self

            def with_weibull(self, model: ProvidenceModule):
                raise NotImplementedError()

        # pdb.set_trace()
        assert isinstance(testC, ProvidenceModule)

    def test_class_field_rather_than_property(self):
        class testC:
            device = device("cpu")  # this is the difference

            def reset_parameters(self) -> None:
                raise NotImplementedError()

            def forward(self, X: ProvidenceTensor, lengths: LengthsTensor) -> Tuple[Tensor, ...]:
                raise NotImplementedError()

            def __call__(self, X: ProvidenceTensor, lengths: LengthsTensor, *args: Any, **kwds: Any) -> Any:
                return self.forward(X, lengths)

            def parameters(self, recurse=True) -> Iterator[nn.parameter.Parameter]:
                raise NotImplementedError()

            @property
            def device(self) -> device:
                ...

            def train(self, b=True) -> Self:
                return self

            def eval(self) -> Self:
                return self.train(False)

            def to(self, *args, **kwargs) -> Any:
                return self

            def with_weibull(self, model: ProvidenceModule):
                raise NotImplementedError()

        assert isinstance(testC, ProvidenceModule)

    def test_minimal_instance_of_TorchModule(self):
        """Test minimal API implementation to comply with ``ProvidenceModule``

        NOTE: you don't have to comply with the signature of the forward function...
        This is unfortunate, but getting stricter (e.g. forcing inheritance) raises the adoption cieling and design burden
        without actual functional benefit.
        """

        class testC(nn.Module):
            device = device("cpu")

            def __init__(self) -> None:
                super().__init__()

            def reset_parameters(self):
                ...

            def with_weibull(self, model: ProvidenceModule):
                raise NotImplementedError()

        assert isinstance(testC, ProvidenceModule)
