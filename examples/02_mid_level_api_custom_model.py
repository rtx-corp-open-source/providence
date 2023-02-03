"""
This example shows how to use a custom model that's provided by the Providence library

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from providence.datasets.adapters import NASA_FEATURE_NAMES, NasaTurbofanTest
from providence.dataloaders import NasaFD00XDataLoaders
from providence.nn.module import ProvidenceRNN
from providence.training import use_gpu_if_available
from providence.paper_reproductions import NasaFD001RnnOptimizer, NasaTraining
from torch.nn import LSTM


def ProvidenceBidirectionalLstm(input_size: int, hidden_size: int = 24, num_layers: int = 2, dropout: float = 0.):
    bidiLstm = LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True)
    return ProvidenceRNN(bidiLstm, device=use_gpu_if_available())


model = ProvidenceBidirectionalLstm(input_size=len(NASA_FEATURE_NAMES))
optimizer = NasaFD001RnnOptimizer(model)._replace(num_epochs=5)
nasa_dls = NasaFD00XDataLoaders(NasaTurbofanTest.FD001, batch_size=optimizer.batch_size)

losses = NasaTraining(model, optimizer, nasa_dls)

# then do whatever you want with the losses, model and optimizer
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(16, 12))

ax.plot(losses.training_losses, label='training')
ax.plot(losses.validation_losses, label='validation')
ax.set_title("Training Losses")

plt.show()
