# Copyright (c) Meta Platforms, Inc. and affiliates.
# Confidential and proprietary.
# pyre-unsafe
"""
Pure-Python auto-tuning for Triton LL kernels.

No torch or triton dependencies — importable by unit tests without GPU.
"""

from __future__ import annotations


def ll_auto_tune(nbytes: int) -> dict[str, int]:
    """Return {num_blocks, block_size} for unidirectional P2P LL.

    Uses BLOCK_SIZE=32 (1 warp) for small messages to avoid cross-warp
    tl.min() coupling in the polling loop. Scales to BLOCK_SIZE=512
    (16 warps) for larger messages where throughput matters more.

    The tuning table mirrors C++ LlAutoTune.cuh with Triton-specific
    block_size adjustments for the cross-warp polling tradeoff.
    """
    if nbytes <= 256:
        return {"num_blocks": 1, "block_size": 32}
    if nbytes <= 2048:
        return {"num_blocks": 1, "block_size": 512}
    if nbytes <= 4096:
        return {"num_blocks": 2, "block_size": 512}
    if nbytes <= 8192:
        return {"num_blocks": 4, "block_size": 512}
    if nbytes <= 16384:
        return {"num_blocks": 8, "block_size": 512}
    if nbytes <= 32768:
        return {"num_blocks": 16, "block_size": 512}
    if nbytes <= 65536:
        return {"num_blocks": 32, "block_size": 512}
    if nbytes <= 131072:
        return {"num_blocks": 64, "block_size": 512}
    if nbytes <= 262144:
        return {"num_blocks": 128, "block_size": 512}
    return {"num_blocks": 256, "block_size": 512}


def ll_auto_tune_bidirectional(nbytes: int) -> dict[str, int]:
    """Return {num_blocks, block_size} for bidirectional P2P LL.

    Doubles the block count vs unidirectional (capped at 1024) since
    partition_interleaved(2) halves the warps per direction.
    """
    config = ll_auto_tune(nbytes)
    return {
        "num_blocks": min(config["num_blocks"] * 2, 1024),
        "block_size": config["block_size"],
    }
