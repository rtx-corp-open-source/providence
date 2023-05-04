"""
With a nod to functional / immutable data, you can override things easily while still getting the
benefits of the high level API. Thankfully, Python makes short work of this with namedtuple._replace()
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.dataloaders import BackblazeDataLoaders
from providence.datasets.adapters import BackblazeQuarter
from providence.paper_reproductions import BackblazeTraining
from providence.paper_reproductions import BackblazeTransformer
from providence.paper_reproductions import BackblazeTransformerOptimizer

model = BackblazeTransformer()
# NOTE: _replace() is actually meant to be used. It is prefixed with `_` so there isn't a naming conflict (https://stackoverflow.com/a/2166161/7024476)
optimizer = BackblazeTransformerOptimizer(model)._replace(batch_size=32, num_epochs=2)
backblaze_dls = BackblazeDataLoaders(quarter=BackblazeQuarter._2019_Q4, batch_size=optimizer.batch_size)

losses = BackblazeTraining(model, optimizer, backblaze_dls)

# then do whatever you want with the losses, model and optimizer
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(16, 12))

ax.plot(losses.training_losses, label="training")
ax.plot(losses.validation_losses, label="validation")
ax.set_title("Training Losses")

plt.show()
