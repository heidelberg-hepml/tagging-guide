import datetime
import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.tagging.atlastopexperiment import ATLASTopExperiment
from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.finetuneexperiment import TopTaggingFineTuneExperiment
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
from experiments.tagging.toptagxlexperiment import TopTagXLExperiment

EXPERIMENTS = {
    "toptagging": TopTaggingExperiment,
    "toptaggingft": TopTaggingFineTuneExperiment,
    "toptagxl": TopTagXLExperiment,
    "jetclass": JetClassTaggingExperiment,
    "atlastop": ATLASTopExperiment,
}


@hydra.main(config_path="config_quick", config_name="toptagging", version_base=None)
def main(cfg):
    # under torchrun, world size / ranks come from the environment; otherwise single process
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, local_rank, world_size = 0, 0, 1

    use_cuda = cfg.gpu and torch.cuda.is_available()
    distributed = world_size > 1

    if distributed:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        dist.init_process_group(
            backend="nccl" if use_cuda else "gloo",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=30),
        )

    if use_cuda:
        torch.cuda.set_device(local_rank)

    if cfg.exp_type not in EXPERIMENTS:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    try:
        exp = EXPERIMENTS[cfg.exp_type](cfg, rank, world_size, local_rank)
        exp()
    finally:
        if distributed:
            dist.barrier(device_ids=[local_rank] if use_cuda else None)
            dist.destroy_process_group()


if __name__ == "__main__":
    # CUDA-safe DataLoader workers (workers fork from a clean server, not main)
    mp.set_start_method("forkserver", force=True)
    main()
