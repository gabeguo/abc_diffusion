import json
import os
import random
from argparse import Namespace
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def save_json(data: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def namespace_to_dict(args: Namespace) -> Dict:
    return {k: v for k, v in vars(args).items()}


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def init_distributed() -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_cuda = torch.cuda.is_available()
    if world_size <= 1:
        return torch.device("cuda" if use_cuda else "cpu")

    requested_backend = os.environ.get("DIST_BACKEND", "").strip().lower()
    if requested_backend:
        backend = requested_backend
    else:
        backend = "nccl" if use_cuda else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if use_cuda:
        rank = dist.get_rank()
        device_index = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        return torch.device("cuda", device_index)
    return torch.device("cpu")


def warmup_distributed_collectives(device: torch.device):
    if not is_dist_avail_and_initialized():
        return
    backend = dist.get_backend()
    tensor_device = device if (backend == "nccl" and device.type == "cuda") else torch.device("cpu")
    tensor = torch.ones(1, device=tensor_device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def shard_dataset_for_rank(dataset):
    world_size = get_world_size()
    if world_size == 1:
        return dataset
    indices = list(range(get_rank(), len(dataset), world_size))
    return Subset(dataset, indices)
