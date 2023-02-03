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
from typing import Any, Callable, List, Literal

import pandas as pd
import progressbar
import torch
from providence.distributions import Weibull
from providence.metrics import MetricsCalculator, SDist
from providence.nn.module import ProvidenceModule
from providence.training import EpochLosses as Losses
from providence.type_utils import once, type_name
from providence.types import DataLoaders
from providence.visualization import (Visualizer, plot_mfe_by_timestep,
                                      plot_mse_by_timestep,
                                      plot_percent_overshot_by_tte,
                                      scatterplot_overshoot)
from torch import nn, optim

from providence_utils.visualization import plot_loss_curves


class EpochTermination:
    def __init__(self, terminate: bool, reason: str = "") -> None:
        self.terminate = terminate
        self.message = (f"Exiting training early. Reason: {reason}") if terminate else ""


EpochTermination_CONTINUE = EpochTermination(False)


class Callback(ABC):
    """
    A class used to provide customizable callback behavior
    """
    def before_epoch(self) -> EpochTermination:
        """Return whether `to proceed` and, if you should not, a termination message"""
        return EpochTermination_CONTINUE

    def after_epoch(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders):
        pass

    def after_training(
        self, last_epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders
    ):
        """
        Arguments:
        - last_epoch: the last epoch that elapsed before training terminated (prematurely or otherwise)
        - model: the model that was being trained
        - optimizer
        """
        pass

def check_before_epoch(callbacks: List[Callback]):
    for cb in callbacks:
        if (epoch_check := cb.before_epoch()).terminate:
            return True, epoch_check.message
    return False, ""


class Every(Callable):
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
    "call back that only operates on every nth invocation"

    def __init__(
        self,
        every: int,
        cb: Callable[[int, nn.Module, optim.Optimizer, Losses, DataLoaders], Any],
    ):
        self.callable = Every(every, cb)

    def after_epoch(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders):
        return self.callable(epoch, model, optimizer, losses, dls)

    @property
    def invocation_count(self) -> int:
        return self.callable.invocation_count


class EpochLoggerCallback(Callback):
    def __init__(self, logger) -> None:
        super().__init__()
        self.logger = logger

    def after_epoch(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders):
        self.logger.info(
            f"[epoch: {epoch:03d}] training loss: {losses.training:>15.8f} ||| validation loss: {losses.validation:>15.8f}"
        )


class LearningRateScheduler(Callback):
    def __init__(self, optimizer: optim.Optimizer, **cosine_annealing_schedule_kwargs) -> None:
        super().__init__()
        t_0 = cosine_annealing_schedule_kwargs.pop('T_0', 10)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0)
        self.history = []

    def after_epoch(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders):
        self.scheduler.step()
        self.history.append(self.scheduler.get_lr())


class NthEpochLoggerCallback(EveryNthCallback):
    """Like the `EpochLoggerCallback` to printing `every` epochs, rather than every epoch"""
    def __init__(self, every: int, logger) -> None:
        self.logger = logger

        def log(epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders):
            self.logger.info(
                f"[epoch: {epoch:03d}] training loss: {losses.training:>12.5f} ||| validation loss: {losses.validation:>12.5f}"
            )

        super().__init__(every, log)


class EmergencyBrake(EveryNthCallback):
    """
    If the model hasn't achieved loss less than some requisite, terminate training.

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
    This was implemented with the `EveryNthCallback` just to build it with minimal new code.
    Ideally, this would be removed (or remove itself) from a list of callbacks checked once it was invoked.
    Using `@once` to turn it into a no-op is a short-term solution for the redesign of the callback system.
    """
    @classmethod
    def format_message(cls, losses: Losses, requisite_level: float) -> str:
        return f"{losses} failed to descend below requisite_loss = {requisite_level}"

    def __init__(self, check_at: int, requisite_loss: float) -> None:
        self.requisite = requisite_loss
        self.terminate = False

        @once
        def should_we_terminate(
            epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders
        ) -> bool:
            # terminate if neither train nor validation loss has descended below the requisite loss threshold
            self.terminate = not ((losses.training < self.requisite) or (losses.validation < self.requisite))
            if self.terminate:
                self._termination_message = self.format_message(losses, self.requisite)

        super().__init__(check_at, should_we_terminate)

    def before_epoch(self) -> EpochTermination:
        if self.terminate:
            return EpochTermination(True, self._termination_message)
        return super().before_epoch()


class LearningCurveTracker(EveryNthCallback):
    """For plotting the learning curves every n epochs"""
    def __init__(self, every: int, output_dir: Path, *, y_limit: int = 50):
        self.loss_tracking = {"train": [], "val": []}
        self.y_limit = y_limit

        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        def loss_plotting(*args):
            return plot_loss_curves(
                self.loss_tracking["train"],
                self.loss_tracking["val"],
                str(self.output_dir / "learning_curves.png"),
                y_lim=self.y_limit
            )

        super().__init__(every, loss_plotting)

    def after_epoch(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders):
        # accumulate every time
        self.loss_tracking["train"].append(losses.training)
        self.loss_tracking["val"].append(losses.validation)
        # plot periodically
        return super().after_epoch(epoch, model, optimizer, losses, dls)

    def after_training(
        self, last_epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders
    ):
        # plot the losses
        self.callable.func()
        # save them off for future inspection
        pd.DataFrame(self.loss_tracking).to_csv(self.output_dir / "losses.csv", index=False)


from operator import attrgetter

_ES_Trackable = Literal["val_loss", "train_loss"]


class EarlyStopping(Callback):
    """A callback for EarlyStopping in case a loss isn't monotonically decreasing. Meant to prevent wasted training runs.

    If patience is set to 0, there should be 'no patience' for a non-decreasing behavior.
    If patience is set to 1, there should be an allowance of a slight up-tick; exactly one epoch's worth
    etc.
    """
    def __init__(self, patience: int = 5, track: _ES_Trackable = "train_loss", verbose: bool = False):
        super().__init__()
        self.patience = patience
        self._tracked_prop = track
        self.selector = attrgetter("train" if track == "train_loss" else "validation")
        self.most_recent = []
        self.best = None
        self.verbose = verbose

    def _construct_message(self) -> str:
        return f"Waited {self.patience} epochs to find a better {self._tracked_prop}. Best = {self.best}. Rest = {self.most_recent}"

    def before_epoch(self):
        if len(self.most_recent) > self.patience:
            return EpochTermination(True, self._construct_message())
        return EpochTermination_CONTINUE

    def _log(self, *args, **kwargs):
        if self.verbose: print(*args, **kwargs)

    def after_epoch(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders):
        to_track = self.selector(losses)
        if self.best is None or (to_track < self.best):
            msg = f"{self.best is not None = }"
            if self.best is not None: msg += f" {to_track < self.best = }"
            self._log(msg)
            self.best = to_track
            self.most_recent.clear()
        else:
            self.most_recent.append(to_track)


class DeferrableEarlyStopping(Callback):
    """A callback for EarlyStopping in case a loss isn't monotonically decreasing. Meant to prevent wasted training runs.

    If patience is set to 0, there should be 'no patience' for a non-decreasing behavior.
    If patience is set to 1, there should be an allowance of a slight up-tick; exactly one epoch's worth
    etc.

    `defer_until` instructs the callback to lie dormant until *after* that epoch i.e. allow `defer_until`-many epochs to elapse without
    consideration for the `patience` and standard operation
    """
    def __init__(self, track: _ES_Trackable = "train_loss", *, patience: int, defer_until: int, verbose: bool = False):
        super().__init__()
        self.early_stopper = EarlyStopping(patience, track, verbose)
        self.defer_until = defer_until
        self._invocation_count = 0

    def _construct_message(self) -> str:
        return f"Waited {self.patience} epochs to find a better {self._tracked_prop}. Best = {self.best}. Rest = {self.most_recent}"

    # delegated properties
    @property
    def _tracked_prop(self): return self.early_stopper._tracked_prop

    @property
    def patience(self): return self.early_stopper.patience

    @property
    def selector(self): return self.early_stopper.selector

    @property
    def most_recent(self): return self.early_stopper.most_recent

    @property
    def best(self): return self.early_stopper.best

    @property
    def verbose(self): return self.early_stopper.verbose

    @verbose.setter
    def verbose(self, val: bool):
        self.early_stopper.verbose = val


    def before_epoch(self):
        if len(self.most_recent) > self.patience:
            return EpochTermination(True, self._construct_message())
        return EpochTermination_CONTINUE

    def after_epoch(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders):
        if self._invocation_count >= self.defer_until:
            self.early_stopper.after_epoch(epoch, model, optimizer, losses, dls)
        
        self._invocation_count += 1


class ModelCheckpointer(Callback):
    """A callback for checkpointing model whenever it performs better on a target metric

    Arguments:
        track: a string metric to monitor, measuring the "best" model
            Acceptable metrics: train_loss, val_loss
        logger: Python logging.Logger
        verbose: whether to log verbosely about this callback's operations
        keep_old: (default 0) keep `keep_old`-many checkpoints if `True` or greater than 0

    TODO(stephen): address dependencies on this checkpointer. Not the priority right now.
    """
    def __init__(self, output_dir: Path, track: str, logger, *, verbose: bool = False, keep_old: int = 0) -> None:
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
            raise ValueError("`keep_old` parameter must be eiter an integer or bool.")

        if self.n_keep:
            self.previous_models = deque([], maxlen=self.n_keep)

    def after_epoch(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders):
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
                    self.previous_models[0].unlink(missing_ok=True) # shouldn't be missing, but if the drive moves or something...

                self.previous_models.append(model_name_on_disk)

    @property
    def best_model(self) -> Path:
        """Give back the model that performed the best on the metric to track"""
        return self.previous_models[-1]



def scatterplot_overshoot_mode(calc: MetricsCalculator, max_timestep: int, min_timestep: int):
    return scatterplot_overshoot(calc, max_timestep=max_timestep, min_timestep=min_timestep, stat="mode")


class IntervalMetricsVisualizer(EveryNthCallback):
    """For accumulating and plotting metrics visualizations every n epochs

    :param every: the periodic number of epochs to elapse before this callback acts
                    i.e. for `every` number of invocations, you get 1 functional invocation
    :param output_dir: path for the metrics subfolders and csvs to be written to
    :param logger: optional reference to the owning module's logger
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
            epoch: int, model: ProvidenceModule, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders
        ):
            """Plots the visualizations of metrics, `every`-th epoch"""
            self.logger.info(f"{epoch=} generating metrics visualizations")

            model.to("cpu", non_blocking=True)
            dls.to_device('cpu')

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
        self, last_epoch: int, model: ProvidenceModule, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders
    ):
        self.callable.func(last_epoch, model, optimizer, losses, dls)


class CachedIntervalMetricsVisualizer(EveryNthCallback):
    """For accumulating and plotting metrics visualizations every n epochs

    :param every: the periodic number of epochs to elapse before this callback acts
                    i.e. for `every` number of invocations, you get 1 functional invocation
    :param output_dir: path for the metrics subfolders and csvs to be written to
    :param logger: optional reference to the owning module's logger
    :param verbose: verbose logging of visualization progress
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
            epoch: int, model: ProvidenceModule, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders
        ):
            """Plots the visualizations of metrics, `every`-th epoch"""
            self.logger.info(f"{epoch=} generating metrics visualizations")

            model.to("cpu", non_blocking=True)
            dls.to_device('cpu')
            if getattr(self, "validation_visualizable", None) is None:
                # send data to cpu so it can be visualized.
                if self.verbose: self.logger.info("Loading training data onto CPU")
                self.train_visualizable = dls.train_ds
                self.validation_visualizable = dls.validation_ds

            # train visualizations
            visualizer = Visualizer(model, Weibull, self.train_visualizable)
            visualizer.warm_up_cache(min_timestep=0, max_timestep=45)
            for vis_output_dir, vis_func in track_progress(self.metric_visualizations):
                vis_func(visualizer, min_timestep=0,
                         max_timestep=45).figure.savefig(vis_output_dir / f"train-epoch_{epoch:03d}.png")

            # validation visualizations, freeing memory
            visualizer = Visualizer(model, Weibull, self.validation_visualizable)
            visualizer.warm_up_cache(min_timestep=0, max_timestep=45)
            for vis_output_dir, vis_func in track_progress(self.metric_visualizations):
                vis_func(visualizer, min_timestep=0,
                         max_timestep=45).figure.savefig(vis_output_dir / f"validation-epoch_{epoch:03d}.png")

            # send it back where it came from
            model.to(model.device, non_blocking=True)
            dls.to_device(model.device)

        super().__init__(every, plot_metrics_visualizations)

    def after_training(
        self, last_epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders
    ):
        self.callable.func(last_epoch, model, optimizer, losses, dls)


class WriteModelOutputs(EveryNthCallback):
    """
    Write the model outputs (the foundation for metrics) and the fleet-level metrics (see `MetricsCalculator`)
    periodically.
    """
    def __init__(self, last_epoch_number: int, output_dir: Path, logger, *, verbose: bool = False, dist_t: SDist = Weibull):
        self.logger = logger
        self.verbose = verbose
        self.dist_t = dist_t
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        def write_metrics(
            epoch: int, model: ProvidenceModule, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders
        ):
            self.logger.info(f"Writing model outputs on epoch {epoch}")

            # TODO(stephen): encapsulate the mandatory marshalling to- and from- CPU/GPU because it's error prone and time wasteful
            model.to("cpu", non_blocking=True)
            dls.to_device('cpu')

            train_calc = MetricsCalculator(model, self.dist_t, dls.train_ds)
            val_calc = MetricsCalculator(model, self.dist_t, dls.validation_ds)

            with torch.no_grad():
                if self.verbose:
                    print("computing training outputs")
                train_outputs_df = train_calc.outputs_per_device()

                if self.verbose:
                    print("computing validation outputs")
                validation_outputs_df = val_calc.outputs_per_device()

                if self.verbose:
                    print("writing training outputs")
                train_outputs_df.to_csv(output_dir / f"training_{epoch:03}.csv", index=False)

                if self.verbose:
                    print("writing validation outputs")
                validation_outputs_df.to_csv(output_dir / f"validation_{epoch:03}.csv", index=False)

                if self.verbose:
                    print("computing fleet metrics: training")
                train_fm = train_calc.fleet_metrics()

                if self.verbose:
                    print("writing fleet metrics: training")
                train_fm.to_csv(metrics_dir / f"training_{epoch:03}.csv", index=False)

                if self.verbose:
                    print("computing fleet metrics: validation")
                validation_fm = val_calc.fleet_metrics()

                if self.verbose:
                    print("writing fleet metrics: validation")
                validation_fm.to_csv(metrics_dir / f"validation_{epoch:03}.csv", index=False)

                if self.verbose:
                    print("computing paper metrics")
                paper_metrics = val_calc.fleet_metrics(tte_cutoff=1, rul_stat="mode")

                if self.verbose:
                    print("writing paper metrics")
                paper_metrics.to_csv(metrics_dir / f"paper_metrics_{epoch:03}.csv", index=False)

            # final output writer is assumed to be last, so if take care to order our writers accordingly
            if self.verbose:
                print("Writing global outputs df")
            train_outputs_df.to_csv(output_dir / "training.csv", index=False)
            validation_outputs_df.to_csv(output_dir / "validation.csv", index=False)

            # send it back where it came from
            model.to(model.device, non_blocking=True)
            dls.to_device(model.device)

        super().__init__(last_epoch_number, write_metrics)

    def after_training(
        self, last_epoch: int, model: nn.Module, optimizer: optim.Optimizer, losses: Losses, dls: DataLoaders
    ):
        # in all situations, write the outputs again
        # this is great for early-stopping scenarios
        self.callable.func(last_epoch, model, optimizer, losses, dls)
