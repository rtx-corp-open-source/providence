"""
We want to show the extensibility of the library, so that one may implement some novel training approaches.
In this example, multi-pass training epochs are such a 'novel' offering.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import sys

# sys.path.append('./')  # let us readily access providence.*

from random import randint
from typing import List, NamedTuple

from pathlib import Path

from providence.dataloaders import BackblazeDataLoaders
from providence.datasets.adapters import BackblazeQuarter
from providence.paper_reproductions import BackblazeTransformer, BackblazeTransformerOptimizer
from providence.training import OptimizerWrapper, training_pass, validation_pass, use_gpu_if_available
from providence.types import DataLoaders

model = BackblazeTransformer()
optimizer = BackblazeTransformerOptimizer(model)
backblaze_dls = BackblazeDataLoaders(quarter=BackblazeQuarter._2019_Q4, batch_size=optimizer.batch_size)


def custom_training(model, optim: OptimizerWrapper, dataloaders: DataLoaders):
    "perform multi-pass training before validating each epoch"

    maybe_gpu = use_gpu_if_available()
    model.to(maybe_gpu)
    dataloaders.to_device(maybe_gpu)

    loss_agg = NamedTuple(
        'MultipassAggregates', [('training_losses', List[List[float]]), ('validation_losses', List[float])]
    )([], [])
    n_passes = randint(1, 5)

    for _ in range(optimizer.num_epochs):
        training_losses = [training_pass(dataloaders.train, model, optim.opt) for _ in range(n_passes)]
        validation_loss = validation_pass(dataloaders.validation, model)
        loss_agg.training_losses.append(training_losses)
        loss_agg.validation_losses.append(validation_loss)

    # send back to the CPU for metrics calculation
    model.to('cpu')
    dataloaders.to_device('cpu')

    return loss_agg


losses = custom_training(model, optimizer, backblaze_dls)
print(losses)

# then do whatever you want with the losses, model and optimizer
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(16, 12))

training_loss_means = []
# scatterplot the multiple losses achieved of a given epoch, plotting their mean
for x_tick, loss_set in enumerate(losses.training_losses):
    training_loss_means.append(loss_set.mean())
    ax.scatter([x_tick] * len(loss_set), loss_set, color='b')
ax.plot(training_loss_means, label='training')
ax.plot(losses['validation'], label='validation')

ax.legend()

fig.suptitle(f"Learning Curves on Backblaze: (epoch={optimizer.num_epochs}, batch_size={optimizer.batch_size})")
Path("outputs").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/learning-curves.png")
