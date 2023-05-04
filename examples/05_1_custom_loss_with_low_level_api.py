"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import List
from typing import Union

from providence.dataloaders import NasaFD00XDataLoaders
from providence.datasets.adapters import NASA_FEATURE_NAMES
from providence.datasets.adapters import NasaTurbofanTest
from providence.distributions import Weibull
from providence.loss import discrete_weibull_loss_fn
from providence.loss import discrete_weibull_mse
from providence.loss import T_Loss
from providence.nn import ProvidenceLSTM
from providence.paper_reproductions import NasaRnnOptimizer
from providence.training import generic_training
from providence.training import use_gpu_if_available
from providence.types import LengthsTensor, TimeTensor

model = ProvidenceLSTM(input_size=len(NASA_FEATURE_NAMES), device=use_gpu_if_available())
optimizer = NasaRnnOptimizer(model)._replace(num_epochs=5)
nasa_dls = NasaFD00XDataLoaders(NasaTurbofanTest.FD001, batch_size=optimizer.batch_size)


def custom_loss_fn(
    params: Weibull.Params,
    y: TimeTensor,
    censor: TimeTensor,
    x_lengths: Union[List[int], LengthsTensor],
) -> T_Loss:
    loss1 = discrete_weibull_loss_fn(params, y, censor, x_lengths, epsilon=1e-5)  # 100x epsilon
    loss2 = discrete_weibull_mse(params, y)

    return loss1 + loss2  # not NaN-safe. Just for example


losses = generic_training(model, optimizer, nasa_dls, loss_fn=custom_loss_fn)
