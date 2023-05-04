"""
This example shows how to use a custom model that's provided by the Providence library

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from providence.dataloaders import BackblazeDataLoaders
from providence.datasets.adapters import BACKBLAZE_FEATURE_NAMES
from providence.datasets.adapters import BackblazeQuarter
from providence.nn.transformer import ProvidenceTransformer
from providence.paper_reproductions import BackblazeTraining
from providence.paper_reproductions import BackblazeTransformerOptimizer

model = ProvidenceTransformer(
    model_dimension=len(BACKBLAZE_FEATURE_NAMES),
    n_attention_heads=3,
    dropout=0.9,
    layer_norm_epsilon=1e-3,
    positional_encoding_dimension=1000,
)
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
