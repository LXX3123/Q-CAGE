from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedState:
    import torch
    import torch.distributed as dist

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistributedState(enabled=False, rank=0, local_rank=0, world_size=1)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistributedState(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def barrier(state: DistributedState) -> None:
    if not state.enabled:
        return
    import torch.distributed as dist

    dist.barrier()


def cleanup_distributed(state: DistributedState) -> None:
    if not state.enabled:
        return
    import torch.distributed as dist

    dist.destroy_process_group()

