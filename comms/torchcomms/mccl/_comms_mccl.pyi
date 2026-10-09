#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

from datetime import timedelta
from typing import Any, Dict, List, Tuple

class BroadcastOptions:
    """Options for broadcast operations."""
    def __init__(self) -> None: ...
    hints: Dict[str, str]
    timeout: timedelta

class CollectiveStat:
    """Per-(collective_op.msg_size) host-drive timing (microseconds)."""

    count: int
    total_us: int
    min_us: int
    max_us: int

    # Launch geometry; 0 when unknown.
    num_blocks: int
    block_size: int
    blocks_per_sm: int

    # SM residency: sum of ceil(num_blocks / blocks_per_sm) * duration.
    total_sm_us: int

    # Quantiles of the duration, set on the "<op>.all" and "all" roll-ups; 0
    # when unknown. queue_p99_us is enqueue to start.
    p50_us: int
    p90_us: int
    p99_us: int
    queue_p99_us: int

class TorchCommMCCL:
    """MCCL backend for TorchComm.

    This class is obtained via TorchCommBackend when the backend is "mccl".
    Only the MCCL-specific methods without a public TorchComm equivalent
    (setTimeout, getAndClearCollectiveStats) are available on this type.
    """

    def setTimeout(self, duration: timedelta) -> None: ...
    def getAndClearCollectiveStats(self) -> Dict[str, CollectiveStat]: ...
    def colltrace_get_comm_id(self) -> int | None: ...
    def colltrace_get_latest_coll_id(self) -> int | None: ...
    def colltrace_get_unread_events(
        self,
    ) -> List[Tuple[int | None, int, int, int, str, float]]: ...

class TorchWorkMCCL:
    """Async work handle for MCCL operations.

    Returned by collective and point-to-point operations. Use wait() for
    GPU-stream synchronization or wait_blocking() for CPU-blocking wait.
    """

    def is_completed(self) -> bool:
        """Check if the operation has completed (non-blocking)."""
        ...
    def wait(self) -> None:
        """Wait for completion. Non-blocking for GPU ops; blocks CPU for CPU ops."""
        ...
    def wait_blocking(self) -> None:
        """Block the CPU thread until completion."""
        ...
