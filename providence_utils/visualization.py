"""
Visualizations that are somewhat dependent on what is emitted from providence_utils.

TODO: move to providence core for their basic interspection abilities

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from typing import Any, Sequence
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from providence import metrics
from providence.datasets import ProvidenceDataset
from providence.metrics import SDist
from providence.utils import validate
from providence.visualization import make_error_plot


VisualizableDataset = Union[ProvidenceDataset, Sequence[Tuple[Any, torch.Tensor]]]
"""What we can visualize. Rather than leverage metrics.FleetType, narrow to Sequence because we current log the length."""

_Error_Kind = Literal["reg", "kde"]


def plot_error_plots(
    model: torch.nn.Module,
    distribution: SDist,
    fleet_object: VisualizableDataset,
    kind: _Error_Kind,
) -> Figure:
    """Plot the error visualization from fresh model outputs.

    Args:
        model (torch.nn.Module): model to predict distributions
        distribution (SDist): Distribution type
        fleet_object (VisualizableDataset): dataset or iterable of id and features, wrt which we visualize the metrics
        kind (_Error_Kind): _description_

    Returns:
        Figure: PyPlot output figure
    """
    print(f"Plotting fleet object where {len(fleet_object) =}")
    device_outputs = metrics.output_per_device(model, distribution, fleet_object)

    grid = make_error_plot(device_outputs["mean"], device_outputs["tte"], kind=kind)
    return grid.fig


def plot_loss_curves(
    train_loss: List[float],
    val_loss: List[float],
    fname: Optional[str] = None,
    *,
    y_lim: int = 20,
) -> None:
    """Plot the learning rate curves, assuming a pair per epoch, with a better zoom factor.

    Args:
        train_loss (List[int]): losses from the training phase, drawn with pyplot's default first color
        val_loss (List[int]): loss from the validation phase, drawn with pyplot's default second color
        fname (Optional[str], optional): file path to save the figure, if provided. Defaults to None.
        y_lim (int, optional): vertical limit of the figure, for easing inspection. Defaults to 20.
    """
    validate(
        len(train_loss) == len(val_loss), f"Loss sequences must be equal. Got: {len(train_loss)=} and {len(val_loss)=}"
    )
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
