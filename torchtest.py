"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
import sys
from contextlib import nullcontext

import torch

print("Python version", sys.version, sys.version_info)

print("Torch CUDA device count:", torch.cuda.device_count())

from providence.training import use_gpu_if_available

get_model = lambda: torch.nn.Linear(10, 2)

print("Moving shallow model to GPU")

with nullcontext() as fake_scope:
    model = get_model()

    model.to(use_gpu_if_available())

    print(model)
    params = list(model.parameters())
    print(params)

print("Doing smart assignment")

with nullcontext():
    model = get_model().to(use_gpu_if_available())

    print(model)
    params = list(model.parameters())
    print(params)


print("Doing forced assignment")
with nullcontext():
    model = get_model().to("cuda")

    print(model)
    params = list(model.parameters())
    print(params)
