# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Public host entry for the fused all_to_all in the CuTe DSL (symmetric-memory NVLink).

The host half of the CuTe all_to_all: :func:`all_to_all` (staging copy schedule, writes
the caller's output). It resolves a ``CuteA2AConfig`` (explicit or the analytic default),
picks the analytic launch shape (``_pick_tile`` / ``_pick_slots``), and
``cute.compile`` / launches the device kernel in :mod:`comms.dsl.cute.a2a.schedules`.

Persistent counters plus HEAD/TAIL credits make fixed-geometry calls and CUDA Graph
replays safe back-to-back. :func:`comms.dsl.tuning_base.check_geometry` rejects staging
ownership changes on a reused transport. The fused signal-spinning grid must be fully
co-resident, so :func:`all_to_all` rejects a grid above the conservative capacity rather
than risking deadlock.
"""

import logging
import os
from importlib import import_module
from typing import Any

import torch
from comms.dsl.transport import check_transfer, NvlTransport
from comms.dsl.tuning_base import check_geometry

# Importing send_recv runs the one-time CuTe setup (CUTE_DSL_ARCH detection +
# cuda-bindings shim) AND imports cutlass; the os.environ statement below is a barrier so
# the formatter cannot hoist the cutlass imports above this setup.
from ..send_recv import (
    _ensure_cuda_rt_compat,
    _pick_slots,
    _pick_tile,
    _resolve_cute_dsl_arch,
)

os.environ.setdefault("CUTE_DSL_ARCH", _resolve_cute_dsl_arch())

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .schedules import _A2ACfg, _launch_a2a
from .tuning import (
    _max_coresident_ctas,
    CUTE_A2A_GEOMETRY_FIELDS,
    CuteA2AConfig,
    DEFAULT_A2A_CONFIG,
)

logger: logging.Logger = logging.getLogger(__name__)

_cuda_driver: Any = import_module("cuda.bindings.driver")

_MAX_PORTABLE_CLUSTER: int = 8  # Hopper/Blackwell portable cluster cap


# Minimal dtype support: (cutlass dtype, bits-per-element).
_CUTLASS_DTYPE: dict[torch.dtype, tuple[Any, int]] = {
    torch.float32: (cutlass.Float32, 32),
    torch.bfloat16: (cutlass.BFloat16, 16),
}

_COMPILED: dict[tuple[Any, ...], object] = {}


def _env_int(name: str, default: int) -> int:
    """int() of env ``name`` with a clear error naming the offending knob (dev/tuning knobs)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"env {name}={raw!r} must be an integer") from None


def all_to_all(  # noqa: C901
    transport: NvlTransport,
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    config: CuteA2AConfig | None = None,
) -> None:
    """Equal-split all_to_all_single via the fused CuTe copy kernel.

    Launch tunables come from ``config`` (a ``CuteA2AConfig``); ``config is None`` uses the
    analytic adaptive defaults (``DEFAULT_A2A_CONFIG``). ``numel`` must be divisible by the
    world size (equal split). Input and output must not overlap. Fixed-geometry calls may
    run back-to-back, but one transport must not execute concurrently on multiple streams.
    """
    if not (input.is_cuda and output.is_cuda):
        raise ValueError("cute a2a requires CUDA input/output tensors")
    if not (input.is_contiguous() and output.is_contiguous()):
        raise ValueError("cute a2a requires contiguous input/output tensors")
    if input.dtype != output.dtype:
        raise ValueError("cute a2a requires matching input/output dtype")
    if input.device != output.device:
        raise ValueError("cute a2a requires input/output on the same CUDA device")
    if input.device.index != torch.cuda.current_device():
        raise ValueError(
            "cute a2a requires the input device to be the current CUDA device"
        )
    if input.dtype not in _CUTLASS_DTYPE:
        raise ValueError(f"cute a2a supports {list(_CUTLASS_DTYPE)}, got {input.dtype}")
    if input.data_ptr() % 16 or output.data_ptr() % 16:
        raise ValueError("cute a2a requires 16-byte-aligned input/output tensors")
    if transport.per_peer_bytes % 16:
        raise ValueError("cute a2a requires a 16-byte-aligned per-peer staging stride")
    ws = transport.world_size
    numel = input.numel()
    if numel != output.numel():
        raise ValueError("cute a2a requires input.numel() == output.numel()")
    span_bytes = numel * input.element_size()
    if max(input.data_ptr(), output.data_ptr()) < min(
        input.data_ptr() + span_bytes, output.data_ptr() + span_bytes
    ):
        raise ValueError("cute a2a requires non-overlapping input/output tensors")
    if numel == 0:
        raise ValueError("cute a2a requires non-empty input/output tensors")
    if numel % ws != 0:
        raise ValueError("equal-split requires numel % world_size == 0")
    chunk = numel // ws
    cdtype, dbits = _CUTLASS_DTYPE[input.dtype]
    if config is None:
        config = DEFAULT_A2A_CONFIG
    if config.primitive != "copy":
        raise ValueError(
            f"primitive {config.primitive!r} is unsupported; only 'copy' is available"
        )
    nb = config.num_blocks
    num_threads, vec = _pick_tile(chunk, dbits, nb * ws)
    # num_threads: analytic _pick_tile, overridden by a tuned/explicit config (0 = analytic),
    # an env knob winning over the config; the override cascades into num_tiles / num_slots.
    if "A2A_CUTE_NT" not in os.environ and config.num_threads:
        units = chunk // vec
        nt = min(config.num_threads, units, 1024)  # CUDA threads/block hardware cap
        while nt > 1 and units % nt:
            nt -= 1
        num_threads = nt
    # Co-residency guard: the fused kernel spins on cross-CTA signals, so all nb*ws CTAs
    # must be simultaneously resident or a resident CTA can wait forever on a signal from a
    # never-scheduled one (a systematic deadlock at high world_size, e.g. NVL72). Reject a
    # grid that provably cannot co-reside rather than hang.
    props = torch.cuda.get_device_properties(input.device)
    grid_ctas = nb * ws
    cap = min(
        props.multi_processor_count,
        _max_coresident_ctas(
            props.multi_processor_count,
            props.max_threads_per_multi_processor,
            num_threads,
        ),
    )
    if grid_ctas > cap:
        raise ValueError(
            f"cute a2a grid {grid_ctas} CTAs (num_blocks={nb} x world_size={ws}) exceeds "
            f"the conservative signal-spin CTA capacity {cap} (num_threads={num_threads}) on "
            f"{props.name}; the signal-spin kernel needs every CTA co-resident or it "
            "deadlocks -- reduce num_blocks."
        )
    num_tiles = chunk // (num_threads * vec)
    num_slots, tiles_per_slot = _pick_slots(num_tiles, chunk * input.element_size())
    if "A2A_CUTE_SLOTS" not in os.environ and config.num_slots:
        want = max(1, min(config.num_slots, num_tiles))
        tiles_per_slot = (num_tiles + want - 1) // want
        num_slots = (num_tiles + tiles_per_slot - 1) // tiles_per_slot
    check_transfer(transport, chunk, input.dtype, nb)
    cap_elems = transport.per_peer_bytes // input.element_size()
    mbp = transport.max_blocks_per_peer

    table = transport.endpoints_device()
    if table.buffer_ptrs.device != input.device:
        raise ValueError("cute a2a transport and input must be on the same CUDA device")
    send_ctr, recv_ctr = transport.step_state()

    in2d = from_dlpack(input.view(ws, chunk), assumed_align=16)
    out2d = from_dlpack(output.view(ws, chunk), assumed_align=16)
    buf_c = from_dlpack(table.buffer_ptrs, assumed_align=8)
    sig_c = from_dlpack(table.signal_pad_ptrs, assumed_align=8)
    send_c = from_dlpack(send_ctr, assumed_align=8)
    recv_c = from_dlpack(recv_ctr, assumed_align=8)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    # Register-blocking unroll for the NVLink store loop (in-flight stores/thread): the
    # per-SM injection lever. Default to 8 once the per-peer chunk is large enough for the
    # unrolled body to engage (tiny sizes keep 1); a tuned/explicit config (0 = analytic)
    # overrides, an env knob winning over it.
    _u_default = 8 if chunk * input.element_size() >= 64 * 1024 else 1
    unroll = max(1, _env_int("A2A_CUTE_UNROLL", _u_default))
    if "A2A_CUTE_UNROLL" not in os.environ and config.unroll:
        unroll = max(1, config.unroll)
    # CGA cluster size along the block axis (co-locate the blocks targeting one peer on one
    # GPC): raises the per-SM NVLink send rate at the >=256MB band but regresses the mid
    # band, so 0 = analytic default (cluster the large band, else off), -1 = max, >0 =
    # explicit. Capped at the portable cluster size and collapsed to 1 unless it divides
    # num_blocks (so a large SM-budget nb runs cluster-off instead of tripping
    # CUDA_ERROR_INVALID_CLUSTER_SIZE). An env knob wins over the config.
    _cl_default = 0 if chunk * input.element_size() >= 64 * 1024 * 1024 else 1
    _cl = _env_int("A2A_CUTE_CLUSTER", config.cluster or _cl_default)
    cluster = min(nb if _cl <= 0 else _cl, _MAX_PORTABLE_CLUSTER)
    # cluster_y is capped by the portable cluster size too (like cluster): an uncapped large
    # value would combine with cluster on the block axis to exceed the hardware cluster cap
    # and trip CUDA_ERROR_INVALID_CLUSTER_SIZE instead of collapsing gracefully.
    cluster_y = min(
        max(1, _env_int("A2A_CUTE_CLUSTER_Y", config.cluster_y)), _MAX_PORTABLE_CLUSTER
    )
    if nb % cluster:
        cluster = 1
    if ws % cluster_y:
        cluster_y = 1

    # Verify/commit the geometry baseline now that every OTHER pre-dispatch guard (primitive /
    # co-residency / check_transfer) has passed, but BEFORE the expensive cute.compile: a call
    # that will be rejected for a geometry switch should not pay compile cost or populate
    # _COMPILED for a key that never dispatches. check_geometry does not seed the baseline on
    # rejection, so this still cannot pollute a reused transport.
    check_geometry(
        transport,
        config,
        CUTE_A2A_GEOMETRY_FIELDS,
        resolved=(
            ("chunk", chunk),
            ("dtype", input.dtype),
            ("num_threads", num_threads),
            ("vec", vec),
            ("num_slots", num_slots),
            ("tiles_per_slot", tiles_per_slot),
        ),
    )
    transport._record_a2a_launch_metadata(
        {
            "primitive": config.primitive,
            "grid_ctas": grid_ctas,
            "num_blocks": nb,
            "num_threads": num_threads,
            "vec": vec,
            "num_slots": num_slots,
            "tiles_per_slot": tiles_per_slot,
            "unroll": unroll,
            "cluster": cluster,
            "cluster_y": cluster_y,
            "per_peer_bytes": transport.per_peer_bytes,
        }
    )

    key = (
        "a2a",
        # Arch is baked into the compiled kernel: a runtime CUTE_DSL_ARCH change must not reuse
        # a kernel built for a different arch (matches send_recv's cache key).
        os.environ.get("CUTE_DSL_ARCH"),
        nb,
        num_threads,
        vec,
        input.dtype,
        ws,
        chunk,
        cap_elems,
        mbp,
        num_slots,
        tiles_per_slot,
        unroll,
        cluster,
        cluster_y,
    )
    compiled: Any = _COMPILED.get(key)
    if compiled is None:
        _ensure_cuda_rt_compat()  # lazy: shim cuda-bindings before this cute.compile path
        logger.info(
            "compiling cute a2a: ws=%s chunk=%s nb=%s nt=%s vec=%s",
            ws,
            chunk,
            nb,
            num_threads,
            vec,
        )
        # The cfg is UNPACKED into individual constexpr args at the cute.compile boundary --
        # never passed as a single Constexpr object (cute's tree-walk would crash on dtype).
        cfg = _A2ACfg(
            num_blocks=nb,
            num_threads=num_threads,
            vec=vec,
            dtype=cdtype,
            dbits=dbits,
            world_size=ws,
            local_rank=transport.local_rank,
            chunk=chunk,
            cap_elems=cap_elems,
            mbp=mbp,
            num_slots=num_slots,
            tiles_per_slot=tiles_per_slot,
            unroll=unroll,
            cluster=cluster,
            cluster_y=cluster_y,
        )
        compiled = cute.compile(
            _launch_a2a,
            in2d,
            out2d,
            buf_c,
            sig_c,
            send_c,
            recv_c,
            cfg.num_blocks,
            cfg.num_threads,
            cfg.vec,
            cfg.dtype,
            cfg.dbits,
            cfg.world_size,
            cfg.local_rank,
            cfg.chunk,
            cfg.cap_elems,
            cfg.mbp,
            cfg.num_slots,
            cfg.tiles_per_slot,
            cfg.unroll,
            cfg.cluster,
            cfg.cluster_y,
            stream,
        )
        _COMPILED[key] = compiled
    compiled(in2d, out2d, buf_c, sig_c, send_c, recv_c, stream)
