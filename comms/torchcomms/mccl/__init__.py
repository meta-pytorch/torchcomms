#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""MCCL-owned Python registration hooks for the TorchComms plugin."""

from typing import Any, Optional


def _register_c10d_backend() -> None:
    """Register the TorchComms MCCL implementation as a CUDA c10d backend."""
    import torch.distributed as dist
    import torchcomms._comms_mccl  # noqa: F401
    from torch.distributed.distributed_c10d import _create_torchcomms_backend

    def create_backend(opts: Any, backend_options: Optional[object]) -> Any:
        process_group = opts.process_group
        return _create_torchcomms_backend(
            "mccl",
            "cuda",
            group_rank=opts.group_rank,
            group_size=opts.group_size,
            group_name=opts.group_id,
            store=opts.store,
            device_id=(
                process_group.bound_device_id if process_group is not None else None
            ),
            backend_options=backend_options,
            timeout=opts.timeout,
            enable_reconfigure=opts.enable_reconfigure,
        )

    dist.Backend.register_backend(
        "mccl",
        create_backend,
        extended_api=True,
        devices=["cuda"],
    )
