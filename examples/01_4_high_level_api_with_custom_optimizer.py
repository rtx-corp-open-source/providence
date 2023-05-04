"""
This example shows that only three lines of code need to change to facilitate a new optimizer

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from pathlib import Path

from torch.optim import Adam

from providence.dataloaders import BackblazeDataLoaders
from providence.datasets.adapters import BackblazeQuarter
from providence.paper_reproductions import BackblazeTraining
from providence.paper_reproductions import BackblazeTransformer
from providence.training import OptimizerWrapper

model = BackblazeTransformer()
optimizer = OptimizerWrapper(  # use the wrapper so it can store these useful things
    Adam(model.parameters(), lr=1e-3), batch_size=128, num_epochs=3
)
# alternatively, you can do something like the following:
# optimizer = BackblazeTransformerOptimizer(model)._replace(
#     opt=Adam(model.parameters(), lr=1e-3)
# )
backblaze_dls = BackblazeDataLoaders(quarter=BackblazeQuarter._2019_Q4, batch_size=optimizer.batch_size)

losses = BackblazeTraining(model, optimizer, backblaze_dls)
print(losses)
# then do whatever you want with the losses, model and optimizer
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(16, 12))

ax.plot(losses.training_losses, label="training")
ax.plot(losses.validation_losses, label="validation")
ax.set_title("Training Losses")
ax.legend()

Path("outputs").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/training-losses.png")
