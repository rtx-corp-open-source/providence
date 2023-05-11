"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import importlib
from logging import getLogger
from pathlib import Path

import torch  # noqa: F401
from pandas import DataFrame

from providence.dataloaders import BackblazeDataLoaders
from providence.dataloaders import ProvidenceDataLoader
from providence.datasets.adapters import BackblazeQuarter
from providence.datasets.core import ProvidenceDataset
from providence.distributions import Weibull
from providence.paper_reproductions import BackblazeTransformer
from providence.paper_reproductions import BackblazeTransformerOptimizer
from providence.paper_reproductions import GeneralMetrics
from providence.training import LossAggregates
from providence.training import use_gpu_if_available
from providence.utils import now_dt_string
from providence_utils.callbacks import CachedIntervalMetricsVisualizer
from providence_utils.callbacks import EarlyStopping
from providence_utils.callbacks import EmergencyBrake
from providence_utils.callbacks import ModelCheckpointer
from providence_utils.callbacks import NthEpochLoggerCallback
from providence_utils.callbacks import WriteModelOutputs
from providence_utils.visualization import plot_loss_curves

# this is a weird import, just so we don't duplicate a bunch of code
cb_training_example = importlib.import_module("04_callbacks_via_mid_level_api", "../examples")
print(cb_training_example.callback_training)

model = BackblazeTransformer()
model.device = use_gpu_if_available()
optimizer = BackblazeTransformerOptimizer(model)
backblaze_dls = BackblazeDataLoaders(quarter=BackblazeQuarter._2019_Q4, batch_size=optimizer.batch_size)

logger = getLogger(__file__)

output_dir = Path(f"./outputs-{now_dt_string()}")
output_dir.mkdir(parents=True, exist_ok=True)

callbacks = [
    NthEpochLoggerCallback(10, logger=logger),
    CachedIntervalMetricsVisualizer(optimizer.num_epochs // 4, output_dir, logger=logger),
    WriteModelOutputs(optimizer.num_epochs // 2, output_dir, logger=logger),
    ModelCheckpointer(output_dir=(output_dir / "model-checkpoints"), track="val_loss", logger=logger),
    EarlyStopping(patience=20, track="val_loss"),
    EmergencyBrake(20, 1.0),  # our weibull loss takes a bit longer to bit below 1.0, but still begets strong results
]

losses: LossAggregates = cb_training_example.callback_training(model, optimizer, backblaze_dls, callbacks)

# To get the metrics that we use in the paper
metrics = GeneralMetrics(model, backblaze_dls.test_ds, losses)

plot_loss_curves(
    losses.training_losses,
    losses.validation_losses,
    str(output_dir / "learning_curves.png"),
    y_lim=50,
)
"""
At this point, the model is trained and we have many outputs to review the progression of training.
The metrics that are visualized are under directories of their name, roughly "outputs-{dtstring}/plot-{metric}/".
Models are saved under "outputs-{dtstring}/model-checkpoints/".
- we save every better model. Because the learning rates are so high, the models don't train for very long
  because they go off the rails and the EarlyStopping cuts the training short.
  Obviously, the best model configurations train for longer.
Assessing the visualizations and the metrics leads us to ascertain the qualitatively and quantitatively best models
"""

# COMMENT: demonstrative of how to design an inference engine with Providence primitives. It does NOT actually run
# from providence.visualization import plot_weibull

new_devices = DataFrame()  # initialized with new devices

ds = ProvidenceDataset(new_devices, ...)  # type: ignore[arg-type, call-arg, misc]

dl = ProvidenceDataLoader(ds, batch_size=1)

# optional: get the targets to compare
# targets = [targets for (_, _, targets) in dl]

# recall "model" is in scope, initialized and trained up above.
inferred_weibull_params = [model(inputs, lengths) for (inputs, lengths, _) in dl]

# torch.save() would be called here

predicted_rul = [Weibull.mode(params) for params in inferred_weibull_params]

# plot_weibull() would be called here
