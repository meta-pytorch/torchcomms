# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-unsafe

"""Pure-Python auto-tuning configuration for the Triton AlltoAllv kernel.

This module contains topology-aware parameter selection logic with zero
torch/triton dependencies. Separated from device_alltoallv_dynamic.py
so that unit tests can import without pulling in the GPU runtime.
"""

from __future__ import annotations


def _tune_for_nvl(max_msg_size_bytes: int) -> dict:
    """NVLink-optimized tuning parameters.

    NVLink scales with block parallelism (each block independently
    saturates a portion of the NVLink bandwidth via cooperative memcpy).
    """
    if max_msg_size_bytes <= 1 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 4, "chunk_size": 64 * 1024}
    if max_msg_size_bytes <= 2 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 8, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 4 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 8, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 8 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 16, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 16 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 16, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 32 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 16, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 64 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 32, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 128 * 1024:
        return {"blocks_per_peer": 8, "num_warps": 16, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 256 * 1024:
        return {"blocks_per_peer": 8, "num_warps": 16, "chunk_size": 64 * 1024}
    else:
        return {"blocks_per_peer": 16, "num_warps": 16, "chunk_size": 64 * 1024}


def _tune_for_ib(max_msg_size_bytes: int) -> dict:
    """IB/RDMA-optimized tuning parameters.

    Empirical data from exhaustive sweep on 2x8 H100 (64 param combos
    per message size, CUDA graph mode). Key findings:
    - 1 bpp optimal up to 1MB (RDMA NIC pipelines internally)
    - 4 bpp for 2-4MB (some block parallelism helps)
    - 8 bpp for >=8MB (more blocks to saturate bandwidth)
    - 512KB chunks dominate for medium/large messages
    """
    if max_msg_size_bytes <= 1 * 1024:  # 1KB: 2.35x vs NCCL
        return {"blocks_per_peer": 1, "num_warps": 4, "chunk_size": 256 * 1024}
    elif max_msg_size_bytes <= 4 * 1024:  # 4KB: 2.05x
        return {"blocks_per_peer": 1, "num_warps": 16, "chunk_size": 128 * 1024}
    elif max_msg_size_bytes <= 16 * 1024:  # 16KB: 2.35x
        return {"blocks_per_peer": 1, "num_warps": 8, "chunk_size": 256 * 1024}
    elif max_msg_size_bytes <= 64 * 1024:  # 64KB: 2.76x
        return {"blocks_per_peer": 1, "num_warps": 4, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 256 * 1024:  # 256KB: 1.91x
        return {"blocks_per_peer": 1, "num_warps": 8, "chunk_size": 512 * 1024}
    elif max_msg_size_bytes <= 512 * 1024:  # 512KB: 1.92x
        return {"blocks_per_peer": 1, "num_warps": 4, "chunk_size": 512 * 1024}
    elif max_msg_size_bytes <= 1 * 1024 * 1024:  # 1MB: 1.76x
        return {"blocks_per_peer": 1, "num_warps": 16, "chunk_size": 256 * 1024}
    elif max_msg_size_bytes <= 4 * 1024 * 1024:  # 2-4MB: 1.51-1.60x
        return {"blocks_per_peer": 4, "num_warps": 16, "chunk_size": 512 * 1024}
    else:  # >=8MB: 1.42-1.47x
        return {"blocks_per_peer": 8, "num_warps": 16, "chunk_size": 512 * 1024}


def auto_tune_alltoallv_params(
    max_msg_size_bytes: int,
    peer_is_nvl: list[bool] | None = None,
) -> dict:
    """
    Select optimal kernel parameters based on maximum per-peer message size.

    When peer_is_nvl is provided, returns topology-aware config with
    per-peer block counts. NVL peers get NVLink-optimized parameters,
    IB peers get RDMA-optimized parameters. The kernel launches with
    the max blocks_per_peer as a constexpr upper bound and masks excess
    blocks at runtime.

    Args:
        max_msg_size_bytes: Maximum per-peer message size in bytes.
        peer_is_nvl: Optional list of booleans per peer. True = NVLink,
            False = IB. None = all NVLink (backward compatible).

    Returns:
        dict with keys: blocks_per_peer, num_warps, chunk_size,
            per_peer_blocks (list[int] or None)
    """
    nvl_config = _tune_for_nvl(max_msg_size_bytes)

    if peer_is_nvl is None or all(peer_is_nvl):
        return {
            "blocks_per_peer": nvl_config["blocks_per_peer"],
            "num_warps": nvl_config["num_warps"],
            "chunk_size": nvl_config["chunk_size"],
            "per_peer_blocks": None,
        }

    ib_config = _tune_for_ib(max_msg_size_bytes)

    max_bpp = max(nvl_config["blocks_per_peer"], ib_config["blocks_per_peer"])

    per_peer_blocks = [
        nvl_config["blocks_per_peer"] if is_nvl else ib_config["blocks_per_peer"]
        for is_nvl in peer_is_nvl
    ]

    return {
        "blocks_per_peer": max_bpp,
        "num_warps": max(nvl_config["num_warps"], ib_config["num_warps"]),
        "chunk_size": nvl_config["chunk_size"],
        "per_peer_blocks": per_peer_blocks,
    }
