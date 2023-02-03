"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.dataloaders import BackblazeDataLoaders
from providence.datasets.adapters import BackblazeQuarter
from providence.paper_reproductions import BackblazeTransformer, BackblazeTransformerOptimizer, BackblazeTraining, GeneralMetrics

model = BackblazeTransformer()
optimizer = BackblazeTransformerOptimizer(model)
backblaze_dls = BackblazeDataLoaders(quarter=BackblazeQuarter._2019_Q4, batch_size=optimizer.batch_size)

losses = BackblazeTraining(model, optimizer, backblaze_dls)

# then do whatever you want with the losses, model, optimizers
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(16, 12))

ax.plot(losses.training_losses, label='training')
ax.plot(losses.validation_losses, label='validation')
ax.set_title("Training Losses")

plt.show()

# ... and dataset
metrics_dataframe = GeneralMetrics(model, backblaze_dls.validation_ds, losses)

metrics_dataframe["loss_train_total"]
metrics_dataframe["loss_train_min"]
