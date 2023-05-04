"""
For more on Horovod with Pytorch, see
- Horovod's documentation: https://horovod.readthedocs.io/en/latest/pytorch.html
- Databricks' tutorial: https://docs.databricks.com/machine-learning/train-model/distributed-training/mnist-pytorch.html
- Pyro's documentation: https://pyro.ai/examples/svi_horovod.html

This script is meant to be a full hyperparameter sweep of the Providence GRU with the Weibull3 distribution on a given ProvidenceDataset.
We are using Horovod to accelerate a given run (i.e. one model, one dataset, all epochs, and evaluation) to see how it turns around experimentation.

Protocol:
- For the Backblaze dataset, models should be trained on Q4 2019 (per the paper), validated with that quarter's split, and tested for generalization
  on Q1 2020
- For the NASA dataset, a validation set can be extracted from the training set to perform more ceremonial model selection. However, due to the nature
  of the PHM 2008 challenge and (moderate) promise of generalizability, a test set can be omitted. After all, solving the combined dataset problem is
  hard enough.
- For the BackblazeExtended dataset, the training set would be the whole of 2019 (four quarters, per the paper), the validation set is extracted from the
  same, and the test set is also Q1 2020.

NOTE: below is an outline of the work to do, in no particular order

- [ ] Introduce model evaluation code
- [ ] Assess Horovod's weight sharing *correctness* to make sure that our models are doing well after the speed-up in training

These two are crucial, without which it is hard to design intelligent action items to the completion of the remaining tasks.
- If the models aren't performing as well on our evaluations metrics as previously, clearly something is wrong and needs correcting.
- If they are doing fine, I just need to deal with the "god function" and renaming things for encapsulation.

- [ ] Reduce checkpointing and/or refactor to the callback, which checkpoints on an improved metric (if allowed)

Speaking of, what happens to the metrics from the models trained on the non-root-rank machines? Unclear.

- [ ] Refactor to separate model instantiation from training code (if allowed)
- [ ] Refactor to separate data instantiation from training code (if allowed)

**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from time import time
from typing import Callable

import horovod.torch as hvd
import mlflow
import torch as pt
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam

import providence.nn.transformer.deepmind as dm
from providence.dataloaders import ProvidenceDataLoader
from providence.datasets import BackblazeDataset
from providence.datasets import BackblazeDatasets
from providence.datasets.adapters import BackblazeQuarter
from providence.datasets.core import DataSubsetId
from providence.datasets.core import ProvidenceDataset
from providence.datasets.nasa import NasaDataset
from providence.distributions import Weibull
from providence.loss import discrete_weibull_loss_fn
from providence.loss import ProvidenceLossInterface
from providence.nn import ProvidenceGRU
from providence.nn.module import ProvidenceModule
from providence.paper_reproductions import GranularMetrics
from providence.training import LossAggregates
from providence.training import unpack_label_and_censor
from providence.type_utils import type_name
from providence_utils.mlflow import create_or_set_experiment

PYTORCH_DIR = "/dbfs/FileStore/AIML/scratch/Pytorch-Distributed/horovod_providence_pytorch"

dm._DEBUG = False

from pathlib import Path


def save_checkpoint(log_dir: Path, model: pt.nn.Module, optimizer: pt.optim.Optimizer, epoch: int):
    filepath = log_dir / "checkpoint-{epoch}.pth.tar".format(epoch=epoch)
    save_dict = {
        "model": model,
        "optimizer": optimizer.state_dict(),  # state dict because optimizers are more recoverable
    }
    pt.save(save_dict, filepath)


def load_checkpoint(log_dir: Path, epoch: int):
    filepath = log_dir / "checkpoint-{epoch}.pth.tar".format(epoch=epoch)
    return pt.load(filepath)


def create_log_dir() -> Path:
    log_dir = Path(PYTORCH_DIR, str(time()), "MNISTDemo")
    log_dir.mkdir(parents=True)
    #   os.makedirs(log_dir)
    return log_dir


def testing_basics():
    pt.cuda.set_device(hvd.local_rank())
    z = pt.zeros(1)
    print(z)
    print(z.device)
    o = pt.ones(1)
    print(o)
    print(o.device)

    print(pt.randn(1, device="cuda"))


def get_train_set() -> ProvidenceDataset:
    return BackblazeDataset(
        subset_choice=DataSubsetId.Train,
        quarter=BackblazeQuarter._2019_Q4,
        train_percentage=0.7,
        consider_validation=False,
        data_dir="/dbfs/FileStore/datasets/providence",
    )


def get_test_set() -> ProvidenceDataset:
    return BackblazeDataset(
        subset_choice=DataSubsetId.Test,
        quarter=BackblazeQuarter._2019_Q4,
        train_percentage=0.7,
        consider_validation=True,
        data_dir="/dbfs/FileStore/datasets/providence",
    )


def epoch_training_pass(
    train_dl: pt.utils.data.DataLoader,
    model: ProvidenceModule,
    optimizer: pt.optim.Optimizer,
    *,
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_output_type=Weibull.Params,
    should_clip_gradients=True,
):
    train_loss = pt.zeros(1, device=model.device)
    model.train()
    for feats, lengths, targets in train_dl:
        optimizer.zero_grad()

        outputs = model(feats.to(model.device), lengths)
        distribution_params = model_output_type(*outputs)

        y_true, censor_ = unpack_label_and_censor(targets.to(model.device))
        loss = loss_criterion(distribution_params, y_true, censor_, lengths)
        loss.backward()

        if should_clip_gradients:
            clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

        optimizer.step()
        train_loss += loss

    return train_loss.item()


def epoch_validation_pass(
    validation_dl: pt.utils.data.DataLoader,
    model: ProvidenceModule,
    *,
    loss_criterion: ProvidenceLossInterface = discrete_weibull_loss_fn,
    model_output_type=Weibull.Params,
    should_clip_gradients=True,
):
    val_loss = pt.zeros(1, device=model.device)
    model.eval()
    for feats, lengths, targets in validation_dl:
        outputs = model(feats.to(model.device), lengths)
        distribution_params = model_output_type(*outputs)

        y_true, censor_ = unpack_label_and_censor(targets.to(model.device))
        loss = loss_criterion(distribution_params, y_true, censor_, lengths)

        if should_clip_gradients:
            clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

        val_loss += loss

    return val_loss.item()


# Actual run
# epoch_training_pass(train_loader, model, optimizer)
_HVD_LOG_DIR = create_log_dir()


def train_hvd(
    model_func: Callable[[int], ProvidenceModule],
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    momentum: float = 0.5,
):
    from torch.utils.data.distributed import DistributedSampler

    dm._DEBUG = False

    # Initialize Horovod
    hvd.init()
    # device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

    # if device.type == 'cuda':
    #     # Pin GPU to local rank
    #     pt.cuda.set_device(hvd.local_rank())

    # Define dataset...
    train_ds = train_dataset = get_train_set()

    # Partition dataset among workers using DistributedSampler
    # Configure the sampler so that each worker gets a distinct sample of the input dataset
    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    # Use train_sampler to load a different sample of data on each worker
    train_loader = ProvidenceDataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    # tutorial stuff...
    n_model_features = train_ds.n_features
    model = model_func(n_model_features)

    # Pin GPU to be used to process local rank (one GPU per process)
    model.device = pt.device(f"cuda:{hvd.local_rank()}")  # get the above device?
    print(f"{model.device = }")
    model.to(model.device)

    # The effective batch size in synchronous distributed training is scaled by the number of workers
    # Increase learning_rate to compensate for the increased batch size
    optimizer = Adam(model.parameters(), lr=learning_rate * hvd.size())
    # wrap for horovod
    # Wrap the local optimizer with hvd.DistributedOptimizer so that Horovod handles the distributed optimization
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Broadcast the model and optimizer state to all ranks, so all workers start with the same parameters
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    validation_ds = get_test_set()

    for epoch in range(1, num_epochs + 1):
        training_loss = epoch_training_pass(train_loader, model, optimizer)
        # Save checkpoints only on worker 0 to prevent conflicts between workers
        if hvd.rank() == 0:
            save_checkpoint(_HVD_LOG_DIR, model, optimizer, epoch)
            mlflow.log_metric("train_loss", training_loss, step=epoch)

            if epoch % 5 == 0:
                # log intermediary metrics
                val_loss = epoch_validation_pass(ProvidenceDataLoader(validation_ds, batch_size=batch_size), model)
                mlflow.log_metric("val_loss", val_loss, step=epoch)


def create_model(n_model_features: int) -> ProvidenceModule:
    return dm.ProvidenceDeepMindTransformer(
        n_heads=2,
        n_layers=2,
        n_features_in=n_model_features,
        n_embedding=512,
        max_seq_len=1000,
    )

    return dm.ProvidenceBertTransformer(
        n_heads=2,
        n_layers=2,
        n_features_in=n_model_features,
        n_embedding=64,
        max_seq_len=1000,
    )


def main():
    print(f"{pt.__version__ = }")
    # hvd.init()

    from sparkdl import HorovodRunner

    from datetime import datetime
    from time import perf_counter

    run_index = 0
    start_time = datetime.now().isoformat()

    print(f"Run {run_index} started at:", start_time)

    with mlflow.start_run(
        experiment_id=create_or_set_experiment("/Users/40000889@azg.utccgl.com/TEST Providence Weibull-3")
    ) as ml_run:
        start_perf_time = perf_counter()
        run_params = {  # yapf: skip
            "model_seed": pt.initial_seed(),
            "batch_size": 128,
            "dataset_name": "Backblaze",
            "model_arch": type_name(create_model(1)),
            "optim.num_epochs": 500,
            "optim.lr": 1e-3,
            "hvd.np": 2,
        }
        # run_params["optim.lr"] *= run_params["hvd.np"]

        mlflow.log_params(run_params)

        hr = HorovodRunner(np=run_params["hvd.np"])
        hr.run(
            train_hvd,
            model_func=create_model,
            learning_rate=run_params["optim.lr"],
            num_epochs=run_params["optim.num_epochs"],
            batch_size=run_params["batch_size"],
        )
        end_perf_time = perf_counter()

        # log evaluation metrics
        # model = load_checkpoint(_HVD_LOG_DIR, run_params["optim.num_epochs"])["model"]
        # test_ds = get_test_set()
        # test_ds.device = model.device
        # test_ds.use_device_for_iteration(True)
        # # TODO: get the model and data on the same device. It's way faster (which is what we need) to do the metrics calculations on the GPU
        # metrics_series = GranularMetrics(model, test_ds, LossAggregates([0.], [0.]), all_on_cpu=True).iloc[0] # DF -> Series
        # mlflow.log_metrics(metrics_series.to_dict())

        end_time = datetime.now().isoformat()

    print(f"Run {run_index} ended at:", end_time)
    print(f"Performance clock says run took {end_perf_time - start_perf_time} seconds")


if __name__ == "__main__":
    main()
