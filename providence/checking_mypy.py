from typing import Protocol


class FType(Protocol):
    def __call__(self, a: int) -> float:
        ...


def f(a: int) -> float:
    return float(a)


mycheck: FType = f


from providence.loss import ProvidenceLossInterface, discrete_weibull_loss_fn


chekc2: ProvidenceLossInterface = discrete_weibull_loss_fn
