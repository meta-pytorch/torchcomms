# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Public host entry for the fused all_to_all in the CuTe DSL (symmetric-memory NVLink).

The host half of the CuTe all_to_all: :func:`all_to_all` (staging copy schedule, writes
the caller's output). It resolves a ``CuteA2AConfig`` (explicit or the analytic default),
picks the analytic launch shape (``_pick_tile`` / ``_pick_slots``), and
``cute.compile`` / launches the device kernel in :mod:`comms.dsl.cute.a2a.schedules`.

Reuse: persistent step counters plus HEAD/TAIL credits make a transport safe across
back-to-back calls and CUDA-graph replays without a host barrier. Switching the resolved
staging geometry on a reused transport is caught by
:func:`comms.dsl.tuning_base.check_geometry`. The fused grid is ``num_blocks *
world_size`` signal-spinning CTAs that must all be co-resident, so :func:`all_to_all`
conservatively limits the grid to one CTA per SM (reduce ``num_blocks`` at high world size)
rather than risk deadlock.
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

from .schedules import (
    _A2ACfg,
    _launch_a2a,
    _launch_a2a_channel_full,
    _launch_a2a_channel_ring,
)
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

_RANGE_COPY_PRIMITIVES: frozenset[str] = frozenset(
    {
        "copy_channel_full",
        "copy_channel_ring",
    }
)


def _env_int(name: str, default: int) -> int:
    """int() of env ``name`` with a clear error naming the offending knob (dev/tuning knobs)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"env {name}={raw!r} must be an integer") from None


def _resolve_launch_shape(
    chunk: int, dbits: int, world_size: int, config: CuteA2AConfig
) -> tuple[int, int]:
    """Return the effective threads/CTA and vector width after all overrides."""
    num_threads, vec = _pick_tile(chunk, dbits, config.num_blocks * world_size)
    if "A2A_CUTE_NT" not in os.environ:
        if config.primitive == "copy_channel_full" and config.peer_fanout in (2, 4):
            return config.num_threads or 1024, vec
        if config.primitive in _RANGE_COPY_PRIMITIVES and config.num_threads:
            return min(config.num_threads, 1024), vec
        if config.primitive in _RANGE_COPY_PRIMITIVES:
            return min(1024, max(128, ((num_threads + 31) // 32) * 32)), vec
        if config.num_threads:
            units = chunk // vec
            num_threads = min(config.num_threads, units, 1024)
            while num_threads > 1 and units % num_threads:
                num_threads -= 1
    return num_threads, vec


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
    world size (equal split). Input and output must not overlap. Calls using one transport
    may be back-to-back but must not execute concurrently on different streams.
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
    if (
        config.num_blocks <= 0
        or config.num_threads < 0
        or config.num_slots < 0
        or config.unroll < 0
        or config.send_threads < 0
        or config.peer_fanout < 0
    ):
        raise ValueError(
            "cute a2a launch counts must be non-negative and num_blocks > 0"
        )
    if config.primitive not in (
        "copy",
        "copy_channel_full",
        "copy_channel_ring",
    ):
        raise ValueError(
            f"primitive {config.primitive!r} is not shipped yet; only 'copy' / "
            "'copy_channel_full' / 'copy_channel_ring' are available"
        )
    if config.primitive == "copy_channel_full":
        if config.peer_fanout not in (0, 1, 2, 4):
            raise ValueError(
                "copy_channel_full requires peer_fanout in {1, 2, 4} "
                "(0 selects the default of 1)"
            )
    elif config.peer_fanout:
        raise ValueError(f"{config.primitive} does not use peer_fanout")
    nb = config.num_blocks
    # Explicit configs override the analytic pick unless the environment override is set.
    # Range-copy schedules handle tails; tiled classic schedules decrement to an exact
    # divisor so zipped_divide never leaves an uncovered partial tile.
    num_threads, vec = _resolve_launch_shape(chunk, dbits, ws, config)
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
    if config.primitive in (
        "copy_channel_full",
        "copy_channel_ring",
    ):
        _dispatch_a2a_channel(
            transport,
            input,
            output,
            cdtype,
            dbits,
            ws,
            chunk,
            nb,
            num_threads,
            vec,
            config,
        )
        return
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
    if cluster * cluster_y > _MAX_PORTABLE_CLUSTER:
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
            "send_threads": 0,
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
        transport.local_rank,
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


def _resolve_channel_roles(config: CuteA2AConfig, num_threads: int) -> int:
    """Validate and resolve the warp-specialized send/receive split."""
    send_threads = config.send_threads or num_threads // 2
    recv_threads = num_threads - send_threads
    if (
        num_threads < 128
        or num_threads % 32
        or send_threads < 32
        or recv_threads < 32
        or send_threads % 32
        or recv_threads % 32
    ):
        raise ValueError(
            f"{config.primitive} requires warp-aligned non-empty send/recv groups"
        )
    return send_threads


def _dispatch_a2a_channel(  # noqa: C901
    transport: NvlTransport,
    input: torch.Tensor,
    output: torch.Tensor,
    cdtype: Any,
    dbits: int,
    ws: int,
    chunk: int,
    nb: int,
    num_threads: int,
    vec: int,
    config: CuteA2AConfig,
) -> None:
    """Compile and launch a peer-packed warp-specialized channel schedule."""
    grid_ctas = nb * ws
    ring = config.primitive == "copy_channel_ring"
    peer_fanout = 0 if ring else config.peer_fanout or 1
    if not ring and config.num_slots:
        raise ValueError(f"{config.primitive} does not use num_slots")
    ring_slots = max(1, config.num_slots or 8) if ring else 0
    if peer_fanout > 1:
        if ws != 8:
            raise ValueError(
                f"{config.primitive} with peer_fanout={peer_fanout} requires world_size == 8"
            )
        if grid_ctas != 32:
            raise ValueError(
                f"{config.primitive} with peer_fanout={peer_fanout} requires exactly "
                f"32 CTAs, got {grid_ctas}"
            )
        if num_threads != 1024:
            raise ValueError(
                f"{config.primitive} with peer_fanout={peer_fanout} requires exactly "
                "1024 threads per CTA"
            )
    if ring:
        elem = input.element_size()
        if transport.per_peer_bytes % elem:
            raise ValueError(
                f"per_peer_bytes={transport.per_peer_bytes} is not a multiple of {elem}"
            )
    else:
        check_transfer(transport, chunk, input.dtype, grid_ctas)
    if config.cluster not in (0, 1) or config.cluster_y != 1:
        raise ValueError(
            f"{config.primitive} requires cluster in {{0, 1}} and cluster_y == 1"
        )
    cap_elems = transport.per_peer_bytes // input.element_size()
    mbp = transport.max_blocks_per_peer
    signal_channels = grid_ctas
    if not 1 <= signal_channels <= mbp:
        raise ValueError(
            f"{config.primitive} signal channels {signal_channels} must be in [1, {mbp}] for the "
            "signal-pad channel capacity"
        )
    if ring and cap_elems // vec < grid_ctas * ring_slots:
        raise ValueError(
            "copy_channel_ring staging cannot hold one vector per channel/slot"
        )
    send_threads = _resolve_channel_roles(config, num_threads)
    if peer_fanout > 1 and send_threads != 512:
        raise ValueError(
            f"{config.primitive} with peer_fanout={peer_fanout} requires exactly "
            "512 send_threads"
        )
    unroll = max(1, _env_int("A2A_CUTE_UNROLL", 8))
    if "A2A_CUTE_UNROLL" not in os.environ and config.unroll:
        unroll = max(1, config.unroll)

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

    resolved_geometry: tuple[tuple[str, Any], ...] = (
        ("chunk", chunk),
        ("dtype", input.dtype),
        ("vec", vec),
        ("peer_fanout", peer_fanout),
    )
    if ring:
        resolved_geometry += (
            ("cap_elems", cap_elems),
            ("ring_slots", ring_slots),
        )
    check_geometry(
        transport,
        config,
        CUTE_A2A_GEOMETRY_FIELDS,
        resolved=resolved_geometry,
    )
    launch_metadata: dict[str, Any] = {
        "primitive": config.primitive,
        "grid_ctas": grid_ctas,
        "num_blocks": nb,
        "num_threads": num_threads,
        "vec": vec,
        "num_slots": ring_slots if ring else 0,
        "tiles_per_slot": 0,
        "unroll": unroll,
        "loop_unroll": (
            2
            if config.primitive == "copy_channel_full"
            and peer_fanout == 1
            and num_threads == 1024
            else 1
        ),
        "cluster": 1,
        "cluster_y": 1,
        "send_threads": send_threads,
        "peer_fanout": peer_fanout,
        "role_ctas": 0,
        "per_peer_bytes": transport.per_peer_bytes,
    }
    transport._record_a2a_launch_metadata(launch_metadata)

    key = (
        config.primitive,
        os.environ.get("CUTE_DSL_ARCH"),
        grid_ctas,
        num_threads,
        send_threads,
        vec,
        input.dtype,
        ws,
        transport.local_rank,
        chunk,
        cap_elems,
        mbp,
        unroll,
        peer_fanout,
        ring_slots if ring else 0,
    )
    compiled: Any = _COMPILED.get(key)
    if compiled is None:
        _ensure_cuda_rt_compat()
        if ring:
            launch_fn = _launch_a2a_channel_ring
        else:
            launch_fn = _launch_a2a_channel_full
        logger.info(
            "compiling cute %s: ws=%s chunk=%s grid=%s nt=%s send_nt=%s vec=%s fanout=%s",
            config.primitive,
            ws,
            chunk,
            grid_ctas,
            num_threads,
            send_threads,
            vec,
            peer_fanout,
        )
        dynamic_args = (in2d, out2d, buf_c, sig_c, send_c, recv_c)
        common_constants = (
            vec,
            cdtype,
            dbits,
            ws,
            transport.local_rank,
            chunk,
            cap_elems,
            mbp,
            unroll,
        )
        if ring:
            compiled = cute.compile(
                launch_fn,
                *dynamic_args,
                grid_ctas,
                num_threads,
                send_threads,
                *common_constants,
                ring_slots,
                stream,
            )
        else:
            compiled = cute.compile(
                launch_fn,
                *dynamic_args,
                grid_ctas,
                num_threads,
                send_threads,
                *common_constants,
                peer_fanout,
                stream,
            )
        _COMPILED[key] = compiled
    compiled(in2d, out2d, buf_c, sig_c, send_c, recv_c, stream)
