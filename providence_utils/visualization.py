"""
Visualizations that are somewhat dependent on what is emitted from providence_utils.

TODO: move to providence core for their basic interspection abilities

**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from typing import List, Literal, Optional, Tuple, Union

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import torch
from torch.utils.data import Dataset
from providence import metrics
from providence.visualization import make_error_plot
from providence.metrics import SDist


VisualizableDataset = Union[Dataset, List[Tuple[torch.Tensor, torch.Tensor]]]
_Error_Kind = Literal["reg", "kde"]


def plot_error_plots(
    model: torch.nn.Module,
    distribution: SDist,
    fleet_object: VisualizableDataset,
    kind: _Error_Kind,
) -> Figure:
    print(f"Plotting fleet object where {len(fleet_object) =}")
    device_outputs = metrics.output_per_device(model, distribution, fleet_object)

    grid = make_error_plot(device_outputs["mean"], device_outputs["tte"], kind=kind)
    return grid.fig


def plot_loss_curves(train_loss: List[int], val_loss: List[int], fname: Optional[str] = None, *, y_lim: int = 20) -> None:
    assert len(train_loss) == len(val_loss)
    x = [x for x in range(1, len(train_loss) + 1)]
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    ax.plot(x, train_loss, label="train")
    ax.plot(x, val_loss, label="validation")

    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    # Default behavior: If you got max(train_loss) < 40, you can stick with the auto-scaling.
    if max(train_loss) > 2 * y_lim:
        ax.set_ylim(0, y_lim)

    if fname is not None:
        fig.savefig(fname)
    # plt.close("all")
