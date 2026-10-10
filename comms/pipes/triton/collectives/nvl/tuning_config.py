# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""JSON-based tuning config for NVLink sendrecv kernels.

Each kernel launch looks up optimal parameters from a JSON config file
keyed by ``(hardware, num_peers, msg_size_range)``. This replaces
environment-variable-based tuning with a declarative, per-kernel config
that can be populated by a sweep script.

Two kernel variants have separate config sections:

- **Stable kernel** (``sendrecv.py``): supports ``signal_bytes`` for
  sub-slot signaling (latency vs throughput trade-off).
- **WS kernel** (``sendrecv_ws.py``): no sub-slot signaling (TLX barrier
  limitation), but has ``sender_warps`` / ``receiver_warps`` split.

Failure semantics:
  * Malformed or missing JSON → ``json.JSONDecodeError`` /
    ``FileNotFoundError`` propagates from ``_load_json``. This is
    deliberately fail-loud: silent fallback would hide bad sweep
    output. If you need a recoverable load path, wrap callers, do not
    add a try/except here.
  * Schema mismatch (missing keys, wrong types) → the per-section
    ``_parse_*`` functions fall back to hardcoded defaults via
    ``dict.get(..., default)``. Sweep tooling should validate JSON
    upstream rather than rely on this fallback.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch

logger: logging.Logger = logging.getLogger(__name__)

_TUNING_JSON_PATH: Path = Path(__file__).parent / "tuning_sendrecv.json"


@dataclass(frozen=True)
class StableTuningConfig:
    tile_rows: int
    tile_cols: int
    signal_bytes: int
    num_blocks: int
    num_warps: int


@dataclass(frozen=True)
class WsTuningConfig:
    tile_rows: int
    tile_cols: int
    num_blocks: int
    sender_warps: int
    receiver_warps: int


@dataclass(frozen=True)
class SendRecvTuningConfig:
    stable: StableTuningConfig
    ws: WsTuningConfig | None


def _detect_hardware() -> str:
    """Bucket the local GPU into a coarse hardware family.

    Note: this inspects ``cuda:0`` only. On heterogeneous hosts (rare
    in current production) the per-device family is not honored and
    callers may receive a tuning config that does not match the device
    actually used by the process.
    """
    if not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name(0)
    if "H100" in name or "H200" in name:
        return "H100"
    # GB200 NVL72 (Grace+B200 superchip, NVLink fabric across superchips)
    # has substantially different NVLink topology vs standalone B200 PCIe
    # (8x B200 + discrete NVLink switch in one host). Tuning sweet spots
    # diverge — keep them in separate buckets so the sweep can populate
    # them independently.
    if "GB200" in name:
        return "GB200"
    if "B200" in name:
        return "B200"
    if "A100" in name:
        return "A100"
    return name


_CONFIG_CACHE: dict[tuple[str, int, int], SendRecvTuningConfig] = {}

_JSON_CACHE: dict[str, object] | None = None


def _load_json() -> dict[str, object]:
    global _JSON_CACHE
    if _JSON_CACHE is not None:
        return _JSON_CACHE

    override = os.environ.get("TRITON_NVL_TUNING_JSON")
    path = Path(override) if override else _TUNING_JSON_PATH

    with open(path) as f:
        data = json.load(f)

    _JSON_CACHE = data
    return data


def _parse_stable(d: dict[str, object], element_size: int) -> StableTuningConfig:
    tile_rows = int(d.get("tile_rows", 32))
    row_bytes = int(d.get("tile_row_bytes", 2048))
    return StableTuningConfig(
        tile_rows=tile_rows,
        tile_cols=row_bytes // element_size,
        signal_bytes=int(d.get("signal_bytes", d.get("block_stride_bytes", 262144))),
        num_blocks=int(d.get("num_blocks", 16)),
        num_warps=int(d.get("num_warps", 8)),
    )


def _parse_ws(d: dict[str, object] | None, element_size: int) -> WsTuningConfig | None:
    if d is None:
        return None
    tile_rows = int(d.get("tile_rows", 32))
    row_bytes = int(d.get("tile_row_bytes", 2048))
    return WsTuningConfig(
        tile_rows=tile_rows,
        tile_cols=row_bytes // element_size,
        num_blocks=int(d.get("num_blocks", 16)),
        sender_warps=int(d.get("sender_warps", 4)),
        receiver_warps=int(d.get("receiver_warps", 4)),
    )


def get_sendrecv_config(
    msg_bytes: int,
    element_size: int = 4,
    num_peers: int = 1,
    hardware: str | None = None,
) -> SendRecvTuningConfig:
    """Lookup optimal tuning config for given parameters.

    Falls back to defaults if no matching config entry is found.
    """
    if hardware is None:
        hardware = _detect_hardware()

    cache_key = (hardware, num_peers, msg_bytes)
    cached = _CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    data = _load_json()
    configs = data.get("configs", [])

    best: dict[str, object] | None = None
    for entry in configs:
        if entry.get("hardware", hardware) != hardware:
            continue
        if entry.get("num_peers", num_peers) != num_peers:
            continue
        lo = int(entry.get("msg_bytes_min", 0))
        hi = int(entry.get("msg_bytes_max", 2**63))
        if lo <= msg_bytes <= hi:
            best = entry
            break

    if best is not None:
        stable_d = best.get("stable", data.get("default_stable", {}))
        ws_d = best.get("ws", data.get("default_ws"))
    else:
        stable_d = data.get("default_stable", {})
        ws_d = data.get("default_ws")

    result = SendRecvTuningConfig(
        stable=_parse_stable(stable_d, element_size),
        ws=_parse_ws(ws_d, element_size),
    )

    _CONFIG_CACHE[cache_key] = result
    return result
