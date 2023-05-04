"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from time import perf_counter

from providence import paper_reproductions
from providence.dataloaders import BackblazeDataLoaders
from providence.datasets.adapters import BackblazeQuarter
from providence.training import use_gpu_if_available
from providence.utils import now_dt_string
from providence.utils import set_seed


def time_training_full_example(n_laps: int = 5, random_seed: int = 1234):
    experiment_start_timestamp = now_dt_string()
    print(f"Experiment started at {experiment_start_timestamp}")

    model = paper_reproductions.BackblazeTransformer()
    model.device = use_gpu_if_available()
    model_optim_init = paper_reproductions.BackblazeTransformerOptimizer
    dataloaders = BackblazeDataLoaders(quarter=BackblazeQuarter._2019_Q4, batch_size=64, random_seed=random_seed)

    # run training.
    training_run_metrics = []
    for training_lap in range(n_laps):
        print(f"{training_lap = }")
        set_seed(random_seed)  # global set seeds just before model initialization.
        model.reset_parameters()
        model_optim = model_optim_init(model)

        start_time = perf_counter()
        paper_reproductions.BackblazeTraining(model, model_optim, dataloaders)
        training_time = perf_counter() - start_time

        completion_time = now_dt_string()

        metrics = {
            "completion_time": completion_time,
            "training_seconds": training_time,
        }
        training_run_metrics.append(metrics)
