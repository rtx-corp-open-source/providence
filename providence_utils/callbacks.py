"""
Callbacks and supporting utilities for them, all meant to hook into simplified training to automate typical tasks (while adhering to a simple interface):
- Checkpointing models
- Tracking loss curves
- Plotting our custom metrics at fixed intervals

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from abc import ABC
from collections import deque
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal

import pandas as pd
import progressbar
import torch
from torch import nn
from torch import optim

from providence.distributions import SurvivalAnalysisDistribution, Weibull
from providence.metrics import MetricsCalculator
from providence.metrics import SDist
from providence.nn.module import ProvidenceModule
from providence.training import EpochLosses as Losses
from providence.type_utils import once
from providence.type_utils import type_name
from providence.types import DataLoaders
from providence.visualization import plot_mfe_by_timestep
from providence.visualization import plot_mse_by_timestep
from providence.visualization import plot_percent_overshot_by_tte
from providence.visualization import scatterplot_overshoot
from providence.visualization import Visualizer
from providence_utils.visualization import plot_loss_curves


class EpochTermination:
    def __init__(self, terminate: bool, reason: str = "") -> None:
        self.terminate = terminate
        self.message = (f"Exiting training early. Reason: {reason}") if terminate else ""


EpochTermination_CONTINUE = EpochTermination(False)


class Callback(ABC):
    """Base claass to provide customizable callback behavior."""

    def before_epoch(self) -> EpochTermination:
        """Return whether `to proceed` and, if you should not, a termination message."""
        return EpochTermination_CONTINUE

    def after_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """The core functionality of Callbacks is handled after the epoch. These arguments are fixed and consistent.

        While some of the behavior can get rather intricate, everything should be documented at the class-level, much
        like a __call__ isn't documented more heavily than the class / constructor itself

        Args:
            epoch (int): the epoch that just elapsed
            model (nn.Module): the model trained this epoch
            optimizer (optim.Optimizer): the optimizer that was used during this epoch of training
            losses (Losses): losses generated from this epoch
            dls (DataLoaders): DataLoaders used during this epoch
        """
        pass

    def after_training(
        self,
        last_epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """
        Args:
            last_epoch: the last epoch that elapsed before training terminated (prematurely or otherwise)
            model: the model that was being trained
            optimizer: the torch Optimizer used last epoch
            losses: losses generated from the last epoch
            dls: the dataloaders used to training during this epoch
        """
        pass


def check_before_epoch(callbacks: List[Callback]):
    for cb in callbacks:
        if (epoch_check := cb.before_epoch()).terminate:
            return True, epoch_check.message
    return False, ""


class Every:
    """Invoke some Callable "every" n function invocations.

    Used to callbacks

    Example:

        >>> my_bad_printer = Every(3, print)
        >>> my_bad_printer("Hello")
        >>> my_bad_printer("Hello")
        >>> my_bad_printer("Hello")
        "Hello"

    Args:
        every (int): the number of invocations required to produce one invocation
        func (Callable): the underlying function or Callable to interact with, once per ``every`` invocations.
    """

    def __init__(self, every: int, func: Callable) -> None:
        super().__init__()
        self.every = int(every)
        self.ticks = 0
        self.invocation_count = 0
        self.func = func

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.ticks += 1
        if self.ticks % self.every == 0:
            res = self.func(*args, **kwds)
            self.invocation_count += 1
            return res


class EveryNthCallback(Callback):
    """Callback that only operates on every nth invocation.

    You might use this Callback for early-stopping, (expensive) metric tracking, or visualization behavior.
    Leverages the ``Every`` type to facilitate this behavior.

    Args:
        every (int): the number of invocations required to produce one invocation
        cb (Callable[[int, nn.Module, optim.Optimizer, Losses, DataLoaders], Any]): the underlying callback to engage
            periodically, per ``every`` invocations.
    """

    def __init__(
        self,
        every: int,
        cb: Callable[[int, nn.Module, optim.Optimizer, Losses, DataLoaders], Any],
    ):
        self.callable = Every(every, cb)

    def after_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """Invokes underlying ``Every`` instance, delegating functionality.

        Args:
            epoch (int): the epoch that just elapsed
            model (nn.Module): the model trained this epoch
            optimizer (optim.Optimizer): the optimizer that was used during this epoch of training
            losses (Losses): losses generated from this epoch
            dls (DataLoaders): DataLoaders used during this epoch

        Returns:
            Any: Up to the implementation of ``cb`` at ``__init__`` time.
        """
        return self.callable(epoch, model, optimizer, losses, dls)

    @property
    def invocation_count(self) -> int:
        return self.callable.invocation_count


class EpochLoggerCallback(Callback):
    def __init__(self, logger) -> None:
        super().__init__()
        self.logger = logger

    def after_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """Logs losses and epoch every epoch.

        Args:
            epoch (int): the epoch that just elapsed
            model (nn.Module): *ignored* the model trained this epoch
            optimizer (optim.Optimizer): *ignored* the optimizer that was used during this epoch of training
            losses (Losses): losses generated from this epoch
            dls (DataLoaders): *ignored* DataLoaders used during this epoch
        """
        self.logger.info(
            f"[epoch: {epoch:03d}] training loss: {losses.training:>15.8f} ||| validation loss: {losses.validation:>15.8f}"
        )


class LearningRateScheduler(Callback):
    """EXPERIMENTAL: Implements a CosineAnnealingWarmRestarts Learning Rate Scheduler with our Callback hierarchy.

    All arguments to ``after_epoch`` are ignored, as this only cares about how many times it is invoked - irrespective of arguments

    Args:
        optimizer (optim.Optimizer): the optimizer that will have its learning rate adjusted.
    """

    def __init__(self, optimizer: optim.Optimizer, **cosine_annealing_schedule_kwargs) -> None:
        super().__init__()
        t_0 = cosine_annealing_schedule_kwargs.pop("T_0", 10)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0)
        self.history: List[float] = []

    def after_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        self.scheduler.step()
        self.history.append(self.scheduler.get_lr())


class NthEpochLoggerCallback(EveryNthCallback):
    """Adjust the ``EpochLoggerCallback`` to print ``every`` epochs, rather than every epoch.

    Args:
        every (int): the number of epochs to wait before logging.
    """

    def __init__(self, every: int, logger) -> None:
        self.logger = logger

        def log(
            epoch: int,
            model: nn.Module,
            optimizer: optim.Optimizer,
            losses: Losses,
            dls: DataLoaders,
        ):
            self.logger.info(
                f"[epoch: {epoch:03d}] training loss: {losses.training:>12.5f} ||| validation loss: {losses.validation:>12.5f}"
            )

        super().__init__(every, log)


class EmergencyBrake(EveryNthCallback):
    """Terminate training if a model hasn't achieved sufficiently low loss by some epoch.

    Function:
    The e-brake check occurs at the start of an epoch, aligning with the collaquial usage of emergency brakes i.e. right before
    something bad happens.
    In other words, setting `check_at: int = 3` means this callback will check-in at the end of epoch 3 - after its *third* invocation of `after_epoch()`.
    - Continuing with that example, `check_before_epoch()` called at the start of epoch 4 will produce a "should_terminate" signal, from this instance

    Mentality:
    Sometimes sweeps have the right vector, but are too far from the goal to matter.
    This callback is meant to alleviate that problem, allowing for wider seaches of viable parameters, while not incurring the cost
    of waiting / wading through a full run where a model was asymptotic to a failure.

    Historical note:
    This was implemented with the ``EveryNthCallback`` just to build it with minimal new code.
    Ideally, this would be removed (or remove itself) from a list of callbacks checked once it was invoked.
    Using `@once` to turn it into a no-op is a short-term solution for the redesign of the callback system.

    Args:
        check_at (int): the epoch to check, to see if the loss has met the minimality criteria
        requisite_loss (float): the loss which must be greater than or equal to the epoch loss
            i.e. (train|val) loss must be less than or equal to this number else training will terminate.

    """

    @classmethod
    def format_message(cls, losses: Losses, requisite_level: float) -> str:
        return f"{losses} failed to descend below requisite_loss = {requisite_level}"

    def __init__(self, check_at: int, requisite_loss: float) -> None:
        self.requisite = requisite_loss
        self.terminate = False

        @once
        def should_we_terminate(
            epoch: int,
            model: nn.Module,
            optimizer: optim.Optimizer,
            losses: Losses,
            dls: DataLoaders,
        ):
            # terminate if neither train nor validation loss has descended below the requisite loss threshold
            self.terminate = not ((losses.training < self.requisite) or (losses.validation < self.requisite))
            if self.terminate:
                self._termination_message = self.format_message(losses, self.requisite)

        super().__init__(check_at, should_we_terminate)

    def before_epoch(self) -> EpochTermination:
        """Produce a termination message if appropriate: that the loss didn't fall low enough.

        Returns:
            EpochTermination: (True, termination message) if neither train nor validation loss dropped below ``self.requisite`` else (False, "")
        """
        if self.terminate:
            return EpochTermination(True, self._termination_message)
        return super().before_epoch()


class LearningCurveTracker(EveryNthCallback):
    """Plot the learning (train and validation loss) curves every n epochs.

    Args:
        every (int):
        output_dir (Path): output directory, underwhich the learning_curves.png will be updated
        y_limit (int, optional): Vertical limit for the plots. Defaults to 50.

    """

    def __init__(self, every: int, output_dir: Path, *, y_limit: int = 50):
        self.loss_tracking: Dict[str, List] = {"train": [], "val": []}
        self.y_limit = y_limit

        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        def loss_plotting(*args):
            return plot_loss_curves(
                self.loss_tracking["train"],
                self.loss_tracking["val"],
                str(self.output_dir / "learning_curves.png"),
                y_lim=self.y_limit,
            )

        super().__init__(every, loss_plotting)

    def after_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """Track the losses and visualize on the ``every`` period.

        Args:
            epoch (int): the epoch that just elapsed
            model (nn.Module): the model trained this epoch
            optimizer (optim.Optimizer): the optimizer that was used during this epoch of training
            losses (Losses): losses generated from this epoch
            dls (DataLoaders): DataLoaders used during this epoch

        Returns:
            None: nothing to return from plotting.
        """
        # accumulate every time
        self.loss_tracking["train"].append(losses.training)
        self.loss_tracking["val"].append(losses.validation)
        # plot periodically
        return super().after_epoch(epoch, model, optimizer, losses, dls)

    def after_training(
        self,
        last_epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """Draw one final plot, and export tracked losses to ``losses.csv``

        As this happens after ``after_epoch``, all arguments are ignored
        """
        # plot the losses
        self.callable.func()
        # save them off for future inspection
        pd.DataFrame(self.loss_tracking).to_csv(self.output_dir / "losses.csv", index=False)


from operator import attrgetter

_ES_Trackable = Literal["val_loss", "train_loss"]


class EarlyStopping(Callback):
    """EarlyStopping in case a loss isn't monotonically decreasing. Meant to prevent wasted training runs.

    Background: read
    - "Early Stopping - but when?" (Prechelt, 1997), or
    - "A fast learning algorithm for deep belief nets" (Hinton et al, 2006)

    Function:
    If patience is set to 0, there should be 'no patience' for a non-decreasing behavior.
    If patience is set to 1, there should be an allowance of a slight up-tick; exactly one epoch's worth
    etc.

    Args:
        patience (int): see extended summary
        track (str): "train_loss" or "val_loss". Defaults to "train_loss".
        verbose (bool): whether to print debug messages to stdout. Defaults to False.
    """

    def __init__(
        self,
        patience: int = 5,
        track: _ES_Trackable = "train_loss",
        verbose: bool = False,
    ):
        super().__init__()
        self.patience = patience
        self._tracked_prop = track
        self.selector = attrgetter("train" if track == "train_loss" else "validation")
        self.most_recent: List[Any] = []
        self.best = None
        self.verbose = verbose

    def _construct_message(self) -> str:
        return f"Waited {self.patience} epochs to find a better {self._tracked_prop}. Best = {self.best}. Rest = {self.most_recent}"

    def before_epoch(self):
        if len(self.most_recent) > self.patience:
            return EpochTermination(True, self._construct_message())
        return EpochTermination_CONTINUE

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def after_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """Perform the early stopping algorithm: tracking the best loss thus far, and a queue of intermediary losses.

        Args:
            epoch (int): *ignored* the epoch that just elapsed
            model (nn.Module): *ignored* the model trained this epoch
            optimizer (optim.Optimizer): *ignored* the optimizer that was used during this epoch of training
            losses (Losses): losses generated from this epoch
            dls (DataLoaders): *ignored* DataLoaders used during this epoch
        """
        to_track = self.selector(losses)
        if self.best is None or (to_track < self.best):
            msg = f"{self.best is not None = }"
            if self.best is not None:
                msg += f" {to_track < self.best = }"
            self._log(msg)
            self.best = to_track
            self.most_recent.clear()
        else:
            self.most_recent.append(to_track)


class DeferrableEarlyStopping(Callback):
    """EarlyStopping that doesn't start tracking until after a fixed number of epochs.

    Occasionally, some architectures need to have erratic loss behavior before

    For other considerations
    Args:
        patience (int): this many epochs without a "lower than best" loss will be tolerated.
        defer_until (int): instructs the callback to lie dormant until *after* this epoch i.e. allow ``defer_until``-many epochs to elapse without
            consideration for the ``patience`` and standard operation
    """

    def __init__(
        self,
        track: _ES_Trackable = "train_loss",
        *,
        patience: int,
        defer_until: int,
        verbose: bool = False,
    ):
        super().__init__()
        self.early_stopper = EarlyStopping(patience, track, verbose)
        self.defer_until = defer_until
        self._invocation_count = 0

    def _construct_message(self) -> str:
        return f"Waited {self.patience} epochs to find a better {self._tracked_prop}. Best = {self.best}. Rest = {self.most_recent}"

    # delegated properties
    @property
    def _tracked_prop(self):
        return self.early_stopper._tracked_prop

    @property
    def patience(self):
        return self.early_stopper.patience

    @property
    def selector(self):
        return self.early_stopper.selector

    @property
    def most_recent(self):
        return self.early_stopper.most_recent

    @property
    def best(self):
        return self.early_stopper.best

    @property
    def verbose(self):
        return self.early_stopper.verbose

    @verbose.setter
    def verbose(self, val: bool):
        self.early_stopper.verbose = val

    def before_epoch(self):
        if len(self.most_recent) > self.patience:
            return EpochTermination(True, self._construct_message())
        return EpochTermination_CONTINUE

    def after_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """Waits ``defer_until``-many epochs before invoking the EarlyStopping behavior.

        All arguments are ignored until that time, after which this class behaves exactly as the simpler version

        Args:
            epoch (int): the epoch that just elapsed
            model (nn.Module): the model trained this epoch
            optimizer (optim.Optimizer): the optimizer that was used during this epoch of training
            losses (Losses): losses generated from this epoch
            dls (DataLoaders): DataLoaders used during this epoch
        """
        if self._invocation_count >= self.defer_until:
            self.early_stopper.after_epoch(epoch, model, optimizer, losses, dls)

        self._invocation_count += 1


class ModelCheckpointer(Callback):
    """Checkpoint a model whenever it performs better on a target metric.

    Args:
        output_dir (Path): directory under which model checkpoints will be created and deleted.
        track (str): a string metric to monitor, measuring the "best" model, either "train_loss" or "val_loss".
        logger (logging.Logger):
        verbose (bool): *ignored* whether to log verbosely about this callback's operations. Defaults to False
        keep_old (int): keep ``keep_old``-many checkpoints if ``True`` or greater than 0. Defaults to 0.

    TODO(stephen): address dependencies on this checkpointer. Not the priority right now.
    """

    def __init__(
        self,
        output_dir: Path,
        track: _ES_Trackable,
        logger,
        *,
        verbose: bool = False,
        keep_old: int = 0,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        # metric stuff
        self.selector = attrgetter("train" if track == "train_loss" else "validation")
        self._tracked_prop = track
        self.best = None

        # model book-keeping
        if isinstance(keep_old, bool):
            # keep 10_000_000 models
            self.n_keep = 10_000_000 if keep_old else 1
        elif isinstance(keep_old, int):
            "If keep_old = -1, disable. If it = 0, keep only the best model. If it = 1, keep the top 2, etc..."
            self.n_keep = keep_old + 1
        else:
            raise ValueError("`keep_old` parameter must be either an integer or bool.")

        if self.n_keep:
            self.previous_models: deque = deque([], maxlen=self.n_keep)

    def after_epoch(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """Save a full copy of ``model``, marked with ``epoch`` if it performed better on the tracked loss this epoch.

        Args:
            epoch (int): the epoch that just elapsed
            model (nn.Module): the model trained this epoch
            optimizer (optim.Optimizer): *ignored* the optimizer that was used during this epoch of training
            losses (Losses): losses generated from this epoch
            dls (DataLoaders): *ignored* DataLoaders used during this epoch
        """
        to_track = self.selector(losses)
        if not self.best or (to_track < self.best):
            self.best = to_track

            if self.n_keep:
                model_name = f"{type_name(model)}-epoch{epoch:03d}-checkpoint-full.pt"
                self.logger.info(f"Saving {model_name}")
                model_name_on_disk: Path = self.output_dir / model_name
                torch.save(model, str(model_name_on_disk))

                # invariant: if we're going to exceed our allowance, delete in advance
                if len(self.previous_models) == self.n_keep:  # this will hit **every** time once the queue is filled.
                    self.previous_models[0].unlink(
                        missing_ok=True
                    )  # shouldn't be missing, but if the drive moves or something...

                self.previous_models.append(model_name_on_disk)

    @property
    def best_model(self) -> Path:
        """The model that performed the best on the metric to track."""
        return self.previous_models[-1]


def scatterplot_overshoot_mode(calc: MetricsCalculator, max_timestep: int, min_timestep: int):
    """The overshoot scatterplot for the "mode" average.

    Historical note: this exists just to keep its usage site clean

    Args:
        calc (MetricsCalculator): calculator that may already have computed the outputs necessary
        max_timestep (int): oldest timestep of concern
        min_timestep (int): most recent timestep of concern

    Returns:
        pyplot.Ax: axis upon which the plot was drawn
    """
    return scatterplot_overshoot(calc, max_timestep=max_timestep, min_timestep=min_timestep, stat="mode")


class IntervalMetricsVisualizer(EveryNthCallback):
    """For accumulating and plotting metrics visualizations every n epochs.

    Args:
        every (int): the periodic number of epochs to elapse before this callback acts
            i.e. for ``every`` number of invocations, you get 1 functional invocation
        output_dir (Path): path for the metrics subfolders and csvs to be written to
        logger (logging.Logger): optional reference to the owning module's logger
        dist_t (SDist): type of a ``SurvivalAnalysisDistribution`` e.g. ``Weibull``
    """

    def __init__(self, every: int, output_dir: Path, logger, dist_t: SDist):
        self.logger = logger
        self.dist_t = dist_t

        self.logger.info(f"Creating {type_name(self)} that would send plots to {output_dir.as_posix()}")
        # all visualizations return plt.Axes and we use this later.
        metric_visualizations = [
            plot_mfe_by_timestep,
            plot_mse_by_timestep,
            scatterplot_overshoot_mode,
            plot_percent_overshot_by_tte,
        ]

        for viz in metric_visualizations:
            (output_dir / viz.__name__).mkdir(parents=True, exist_ok=True)

        self.metric_visualizations = [((output_dir / viz.__name__), viz) for viz in metric_visualizations]

        def plot_metrics_visualizations(
            epoch: int,
            model: ProvidenceModule,
            optimizer: optim.Optimizer,
            losses: Losses,
            dls: DataLoaders,
        ):
            """Plots the visualizations of metrics, ``every``-th epoch."""
            self.logger.info(f"{epoch=} generating metrics visualizations")

            model.to("cpu", non_blocking=True)
            dls.to_device("cpu")

            train_calc = MetricsCalculator(model, self.dist_t, dls.train_ds, rul_stat="mode")
            validation_calc = MetricsCalculator(model, self.dist_t, dls.validation_ds, rul_stat="mode")

            for visualization_directory, plot_visualization in progressbar.progressbar(self.metric_visualizations):
                # plot trainting metric
                plot_ret = plot_visualization(train_calc, min_timestep=0, max_timestep=45)
                plot_ret.figure.savefig(visualization_directory / f"train-epoch_{epoch:03d}.png")

                # plot validation metric
                plot_ret = plot_visualization(validation_calc, min_timestep=0, max_timestep=45)
                plot_ret.figure.savefig(visualization_directory / f"validation-epoch_{epoch:03d}.png")

            # send it back where it came from
            model.to(model.device, non_blocking=True)
            dls.to_device(model.device)

        super().__init__(every, plot_metrics_visualizations)

    def after_training(
        self,
        last_epoch: int,
        model: ProvidenceModule,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """Plot the visualizations one last time.

        Args:
            last_epoch (int): the last epoch of training that happened
            model (nn.Module): the model which was trained
            optimizer (optim.Optimizer): *ignored* the optimizenr for training
            losses (Losses): *ignored* the losses from the last epoch
            dls (DataLoaders): training and validation data used for the last training epoch
        """
        self.callable.func(last_epoch, model, optimizer, losses, dls)


class CachedIntervalMetricsVisualizer(EveryNthCallback):
    """Accumulate and plot metrics visualizations every n epochs.

    Follows a slightly different code path to minimize computation, and focus the CPU time on plotting pictures.

    Args:
        every (int): the periodic number of epochs to elapse before this callback acts
            i.e. for ``every`` number of invocations, you get 1 functional invocation
        output_dir (Path): directory for the metrics subfolders and csvs to be written to.
        logger (logging.Logger): optional reference to the owning module's logger
        verbose (bool, optional): verbose logging of visualization progress
    """

    def __init__(self, every: int, output_dir: Path, logger, *, verbose=False):
        self.logger = logger
        self.verbose = verbose
        track_progress = progressbar.progressbar if self.verbose else (lambda x: x)

        self.logger.info(f"Creating {type_name(self)} that would send plots to {output_dir.as_posix()}")
        # all visualizations return plt.Axes and we use this later.
        metric_visualizations = [
            Visualizer.plot_mfe_by_timestep,
            Visualizer.plot_mse_by_timestep,
            Visualizer.scatterplot_overshoot_mode,
            Visualizer.plot_percent_overshot_by_tte,
        ]

        for viz in metric_visualizations:
            (output_dir / viz.__name__).mkdir(parents=True, exist_ok=True)

        self.metric_visualizations = [((output_dir / viz.__name__), viz) for viz in metric_visualizations]

        def plot_metrics_visualizations(
            epoch: int,
            model: ProvidenceModule,
            optimizer: optim.Optimizer,
            losses: Losses,
            dls: DataLoaders,
        ):
            """Plots the visualizations of metrics, ``every``-th epoch.

            Args:
                epoch (int): the epoch for which these performance metrics are generated.
                model (ProvidenceModule): PyTorch nn.Module that will be invoked (calling ``model.forward(...)``) to
                    generate metrics.
                optimizer (optim.Optimizer): *ignored*
                losses (Losses): *ignored*
                dls (DataLoaders): training and validaiton data that is used for visualizations
            """
            self.logger.info(f"{epoch=} generating metrics visualizations")

            model.to("cpu", non_blocking=True)
            dls.to_device("cpu")
            if getattr(self, "validation_visualizable", None) is None:
                # send data to cpu so it can be visualized.
                if self.verbose:
                    self.logger.info("Loading training data onto CPU")
                self.train_visualizable = dls.train_ds
                self.validation_visualizable = dls.validation_ds

            # train visualizations
            visualizer = Visualizer(model, Weibull(), self.train_visualizable)
            visualizer.warm_up_cache(min_timestep=0, max_timestep=45)
            for vis_output_dir, vis_func in track_progress(self.metric_visualizations):
                vis_func(visualizer, min_timestep=0, max_timestep=45).figure.savefig(
                    vis_output_dir / f"train-epoch_{epoch:03d}.png"
                )

            # validation visualizations, freeing memory
            visualizer = Visualizer(model, Weibull(), self.validation_visualizable)
            visualizer.warm_up_cache(min_timestep=0, max_timestep=45)
            for vis_output_dir, vis_func in track_progress(self.metric_visualizations):
                vis_func(visualizer, min_timestep=0, max_timestep=45).figure.savefig(
                    vis_output_dir / f"validation-epoch_{epoch:03d}.png"
                )

            # send it back where it came from
            model.to(model.device, non_blocking=True)
            dls.to_device(model.device)

        super().__init__(every, plot_metrics_visualizations)

    def after_training(
        self,
        last_epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """Plot the visualizations one last time.

        Args:
            last_epoch (int): the last epoch of training that happened
            model (nn.Module): the model which was trained
            optimizer (optim.Optimizer): *ignored* the optimizenr for training
            losses (Losses): *ignored* the losses from the last epoch
            dls (DataLoaders): training and validation data used for the last training epoch
        """
        self.callable.func(last_epoch, model, optimizer, losses, dls)


class WriteModelOutputs(EveryNthCallback):
    """Write the model outputs (the foundation for metrics) and the fleet-level metrics (see ``MetricsCalculator``) periodically.

    Writes not only the model outputs, but also the default fleet metrics and the paper comparison / "forecasting" metrics.
    TODO(stephen): parameterize the fleet metrics

    Usage:
    You can use this callback independent on the Visualizer, and then produce visualizations after training.

    Historical note:
    Was initially intended to easily "record final model performance", but became more of a performance checkpointer.
    TODO(stephen): rename first parameter.

    Args:
        last_epoch_number (int): the ``every`` epoch number, i.e. how many epoch should elapse between recording model
            outputs
        output_dir (Path): parent directory under which we can create a directory for model outputs (wherein the outputs
            of this callback will be written.)
        logger (logging.Logger): logger to use for
        verbose (bool, optional): Whether to write debug information to INFO and stdout. Defaults to False.
        dist_t (SDist, optional): type of the SurvivalAnalysisDistribution e.g. Weibull. Defaults to Weibull.

    Callback Args:
        epoch (int): the last epoch of training that happened before a given set of outputs are recorded.
        model (nn.Module): the model whose outputs are to be recorderd.
        optimizer (optim.Optimizer): *ignored* the optimizenr for training
        losses (Losses): *ignored* the losses from the last epoch
        dls (DataLoaders): training and validation data used for the individual recordings
    """

    def __init__(
        self,
        last_epoch_number: int,
        output_dir: Path,
        logger,
        *,
        verbose: bool = False,
        dist_t: SurvivalAnalysisDistribution = Weibull(),
    ):
        self.logger = logger
        self.verbose = verbose
        self.dist_t = dist_t
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        def write_metrics(
            epoch: int,
            model: ProvidenceModule,
            optimizer: optim.Optimizer,
            losses: Losses,
            dls: DataLoaders,
        ):
            """Write model outputs, fleet metrics, and single-point forecasting metrics to ``output_dir / "metrics/"``.

            Args:
                last_epoch (int): the last epoch of training that happened
                model (nn.Module): the model which was trained
                optimizer (optim.Optimizer): *ignored* the optimizenr for training
                losses (Losses): *ignored* the losses from the last epoch
                dls (DataLoaders): training and validation data used for the last training epoch
            """
            self.logger.info(f"Writing model outputs on epoch {epoch}")

            # TODO(stephen): encapsulate the mandatory marshalling to- and from- CPU/GPU because it's error prone and time wasteful
            model.to("cpu", non_blocking=True)
            dls.to_device("cpu")

            train_calc = MetricsCalculator(model, self.dist_t, dls.train_ds)
            val_calc = MetricsCalculator(model, self.dist_t, dls.validation_ds)

            with torch.no_grad():
                if self.verbose:
                    print("computing training outputs")
                    self.logger.info("computing training outputs")
                train_outputs_df = train_calc.outputs_per_device()

                if self.verbose:
                    print("computing validation outputs")
                    self.logger.info("computing validation outputs")
                validation_outputs_df = val_calc.outputs_per_device()

                if self.verbose:
                    print("writing training outputs")
                    self.logger.info("writing training outputs")
                train_outputs_df.to_csv(output_dir / f"training_{epoch:03}.csv", index=False)

                if self.verbose:
                    print("writing validation outputs")
                validation_outputs_df.to_csv(output_dir / f"validation_{epoch:03}.csv", index=False)

                if self.verbose:
                    print("computing fleet metrics: training")
                    self.logger.info("computing fleet metrics: training")
                train_fm = train_calc.fleet_metrics()

                if self.verbose:
                    print("writing fleet metrics: training")
                    self.logger.info("writing fleet metrics: training")
                train_fm.to_csv(metrics_dir / f"training_{epoch:03}.csv", index=False)

                if self.verbose:
                    print("computing fleet metrics: validation")
                    self.logger.info("computing fleet metrics: validation")
                validation_fm = val_calc.fleet_metrics()

                if self.verbose:
                    print("writing fleet metrics: validation")
                    self.logger.info("writing fleet metrics: validation")
                validation_fm.to_csv(metrics_dir / f"validation_{epoch:03}.csv", index=False)

                if self.verbose:
                    print("computing paper metrics")
                    self.logger.info("computing paper metrics")
                paper_metrics = val_calc.fleet_metrics(tte_cutoff=1, rul_stat="mode")

                if self.verbose:
                    print("writing paper metrics")
                    self.logger.info("writing paper metrics")
                paper_metrics.to_csv(metrics_dir / f"paper_metrics_{epoch:03}.csv", index=False)

            # final output writer is assumed to be last, so if take care to order our writers accordingly
            if self.verbose:
                print("Writing global outputs df")
                self.logger.info("Writing global outputs df")
            train_outputs_df.to_csv(output_dir / "training.csv", index=False)
            validation_outputs_df.to_csv(output_dir / "validation.csv", index=False)

            # send it back where it came from
            model.to(model.device, non_blocking=True)
            dls.to_device(model.device)

        super().__init__(last_epoch_number, write_metrics)

    def after_training(
        self,
        last_epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        losses: Losses,
        dls: DataLoaders,
    ):
        """Write the model outputs again, in all situations (this is particularly useful for early-stopping scenarios).

        Args:
            last_epoch (int): the last epoch of training that happened
            model (nn.Module): the model which was trained
            optimizer (optim.Optimizer): *ignored* the optimizenr for training
            losses (Losses): *ignored* the losses from the last epoch
            dls (DataLoaders): training and validation data used for the last training epoch
        """
        self.callable.func(last_epoch, model, optimizer, losses, dls)
