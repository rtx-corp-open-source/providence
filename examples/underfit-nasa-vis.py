"""
Demonstration of the myriad visualizations that we have supporting Providence

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import matplotlib.pyplot as plt
import torch

from providence import visualization as vis
from providence.datasets.core import DataSubsetId
from providence.datasets.nasa import NasaDataset
from providence.distributions import Weibull
from providence.metrics import MetricsCalculator
from providence.nn import ProvidenceVanillaRNN

# load model
model = ProvidenceVanillaRNN(input_size=24, hidden_size=120, num_layers=2)
model.load_state_dict(torch.load("../outputs/model.pt"))
model.eval()

# load test data
test_data = NasaDataset(DataSubsetId.Test)

metrics_calculator = MetricsCalculator(model, Weibull(), test_data)
# plot MSE
ax = vis.plot_mse_by_timestep(metrics_calculator, max_timestep=500, min_timestep=0)
ax.get_figure().savefig("figs/mse-vs-timestep_underfit-nasa.png")

ax = vis.plot_mfe_by_timestep(metrics_calculator, max_timestep=500, min_timestep=0)
ax.get_figure().savefig("figs/mfe-vs-timestep_underfit-nasa.png")

ax = vis.scatterplot_overshoot(metrics_calculator, max_timestep=500, min_timestep=0, stat="mean")
ax.get_figure().savefig("figs/overshoot-mean_underfit-nasa.png")

ax = vis.scatterplot_overshoot(metrics_calculator, max_timestep=500, min_timestep=0, stat="median")
ax.get_figure().savefig("figs/overshoot-median_underfit-nasa.png")

ax = vis.scatterplot_overshoot(metrics_calculator, max_timestep=500, min_timestep=0, stat="mode")
ax.get_figure().savefig("figs/overshoot-mode_underfit-nasa.png")

ax = vis.plot_percent_overshot_by_tte(metrics_calculator, max_timestep=500, min_timestep=0)
ax.get_figure().savefig("figs/percent-overshot-by-tte_underfit-nasa.png")

plt.show()
