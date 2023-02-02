"""
This shows a naive implementation of lite callback support in Providence.
It mirrors the previous work done in the legacy train.py, but with a clearer abstraction separation that makes for
easier experimentation and future encapsulation.

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from logging import getLogger
from pathlib import Path
from typing import List

from providence.dataloaders import BackblazeDataLoaders
from providence.datasets.adapters import BackblazeQuarter
from providence.paper_reproductions import BackblazeTransformer
from providence.training import LossAggregates, OptimizerWrapper, training_epoch
from providence.types import DataLoaders
from providence_utils.callbacks import Callback, NthEpochLoggerCallback, check_before_epoch
from torch.optim import Adam


def callback_training(model, optimizer: OptimizerWrapper, dataloaders: DataLoaders, cbs: List[Callback]):
    model.to(model.device)
    dataloaders.to_device(model.device)

    loss_agg = LossAggregates([], [])
    for current_epoch in range(1, optimizer.num_epochs + 1):
        terminate_training, termination_message = check_before_epoch(callbacks=cbs)
        if terminate_training:
            print(termination_message)
            break
        losses = training_epoch(dataloaders, model, optimizer.opt)
        loss_agg.append_losses(losses)

        for cb in cbs:
            cb.after_epoch(current_epoch, model, optimizer.opt, losses, dataloaders)

    for cb in cbs:
        cb.after_training(current_epoch, model, optimizer.opt, losses, dataloaders)

    dataloaders.to_device('cpu')
    model.to('cpu')
    return loss_agg


logger = getLogger(__file__)

output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':

    model = BackblazeTransformer()
    optimizer = OptimizerWrapper(Adam(model.parameters(), lr=1e-3), batch_size=128, num_epochs=3)
    backblaze_dls = BackblazeDataLoaders(quarter=BackblazeQuarter._2019_Q4, batch_size=optimizer.batch_size)

    losses = callback_training(model, optimizer, backblaze_dls, [
        NthEpochLoggerCallback(5, logger=logger),
    ])

    # then do whatever you want with the losses, model and optimizer
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 12))

    ax.plot(losses.training_losses, label='training')
    ax.plot(losses.validation_losses, label='validation')
    ax.set_title("Training Losses")
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig("outputs/training-losses.png")
