"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.dataloaders import NasaFD00XDataLoaders
from providence.datasets.adapters import NASA_FEATURE_NAMES, NasaTurbofanTest
from providence.distributions import Weibull
from providence.loss import discrete_weibull_loss_fn, discrete_weibull_mse
from providence.nn.module import ProvidenceLSTM
from providence.paper_reproductions import NasaFD001RnnOptimizer
from providence.training import generic_training, use_gpu_if_available
from torch.nn import Module as TorchModule
from torchtyping import TensorType

model = ProvidenceLSTM(input_size=len(NASA_FEATURE_NAMES), device=use_gpu_if_available())
optimizer = NasaFD001RnnOptimizer(model)._replace(num_epochs=5)
nasa_dls = NasaFD00XDataLoaders(NasaTurbofanTest.FD001, batch_size=optimizer.batch_size)


class ProvidenceRmsePlusWeibullLoss(TorchModule):
    def forward(
        self, params: Weibull.Params, y: TensorType["time"], censor: TensorType["time"], x_lengths: TensorType["time"]
    ) -> 'Union[float, Tuple[float, ...]]':
        loss1 = discrete_weibull_loss_fn(params, y, censor, x_lengths, epsilon=1e-5)  # 100x epsilon
        loss2 = discrete_weibull_mse(params, y)

        return loss1 + loss2  # not NaN-safe. Just for example


losses = generic_training(model, optimizer, nasa_dls, loss_fn=ProvidenceRmsePlusWeibullLoss())
