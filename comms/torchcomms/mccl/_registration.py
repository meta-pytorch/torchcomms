# Copyright (c) Meta Platforms, Inc. and affiliates.

"""c10d registration for the TorchComms MCCL backend."""

from typing import Any, Callable, Optional

from ._identity import require_c10d_torchcomms_factory


def register_c10d_backend(load_backend: Callable[[], None]) -> None:
    """Register MCCL after validating and loading its native backend."""
    import torch.distributed as dist

    create_torchcomms_backend = require_c10d_torchcomms_factory()
    load_backend()

    def create_backend(opts: Any, backend_options: Optional[object]) -> Any:
        process_group = opts.process_group
        return create_torchcomms_backend(
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
