"""
Using the documentation outlined
- here: https://horovod.readthedocs.io/en/latest/pytorch.html
- here: https://docs.databricks.com/machine-learning/train-model/distributed-training/mnist-pytorch.html

create some of our code that can leverage Horovod to run faster.
It doesn't seem to actually be model-paralell (split the model across multiple GPUs), so it won't actually win us any memory
battles, but it will speed up certain flows if we're able.


**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from time import time

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
from providence.nn.module import ProvidenceModule
from providence.training import unpack_label_and_censor
from providence_utils.mlflow import create_or_set_experiment

PYTORCH_DIR = "/dbfs/FileStore/AIML/scratch/Pytorch-Distributed/horovod_providence_pytorch"

dm._DEBUG = False

from pathlib import Path


def save_checkpoint(log_dir: Path, model: pt.nn.Module, optimizer: pt.optim.Optimizer, epoch: int):
    filepath = log_dir / "/checkpoint-{epoch}.pth.tar".format(epoch=epoch)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    pt.save(state, filepath)


def load_checkpoint(log_dir, epoch):
    filepath = log_dir + "/checkpoint-{epoch}.pth.tar".format(epoch=epoch)
    return pt.load(filepath)


def create_log_dir() -> Path:
    log_dir = Path(PYTORCH_DIR, str(time()), "MNISTDemo")
    log_dir.mkdir(parents=True)
    #   os.makedirs(log_dir)
    return log_dir


print(f"{pt.__version__ = }")
hvd.init()

dm._DEBUG = False


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


# Actual run
# epoch_training_pass(train_loader, model, optimizer)
_HVD_LOG_DIR = create_log_dir()


def train_hvd(learning_rate: float, num_epochs: int, batch_size: int, momentum: float = 0.5):
    from torch.utils.data.distributed import DistributedSampler

    # Initialize Horovod
    hvd.init()
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    if device.type == "cuda":
        # Pin GPU to local rank
        pt.cuda.set_device(hvd.local_rank())

    # Define dataset...
    train_ds = train_dataset = get_train_set()

    # Partition dataset among workers using DistributedSampler
    # Configure the sampler so that each worker gets a distinct sample of the input dataset
    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    # Use train_sampler to load a different sample of data on each worker
    train_loader = ProvidenceDataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    # tutorial stuff...

    model = dm.ProvidenceBertTransformer(
        n_heads=2,
        n_layers=2,
        n_features_in=train_ds.n_features,
        n_embedding=64,
        max_seq_len=1000,
    )

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

    for epoch in range(1, num_epochs + 1):
        epoch_training_pass(train_loader, model, optimizer)
        # Save checkpoints only on worker 0 to prevent conflicts between workers
        if hvd.rank() == 0:
            save_checkpoint(_HVD_LOG_DIR, model, optimizer, epoch)


from sparkdl import HorovodRunner

from datetime import datetime
from time import perf_counter

for run_index in range(1, 3):
    start_time = datetime.now().isoformat()

    print(f"Run {run_index} started at:", start_time)

    start_perf_time = perf_counter()

    with mlflow.start_run(
        experiment_id=create_or_set_experiment("/Users/40000889@azg.utccgl.com/Providence on Horovod")
    ) as ml_run:
        run_params = {  # yapf: skip
            "model_seed": pt.initial_seed(),
            "batch_size": 64,
            "dataset_name": "Backblaze",
            "model_arch": (dm.ProvidenceBertTransformer).__name__,
            "optim.num_epochs": 200,
            "optim.lr": 0.001,
            "hvd.np": 3,
        }

        mlflow.log_params(run_params)

        hr = HorovodRunner(np=run_params["hvd.np"])
        hr.run(
            train_hvd,
            learning_rate=run_params["optim.lr"],
            num_epochs=run_params["optim.num_epochs"],
            batch_size=run_params["batch_size"],
        )
        # TODO: log evaluation metrics

    end_perf_time = perf_counter()

    end_time = datetime.now().isoformat()
    print(f"Run {run_index} ended at:", end_time)
    print(f"Performance clock says run took {end_perf_time - start_perf_time} seconds")
