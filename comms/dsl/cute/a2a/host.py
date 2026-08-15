# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Public host entries for the fused all_to_all in the CuTe DSL (symmetric-memory NVLink).

The host half of the CuTe all_to_all: :func:`all_to_all` (staging, writes the
caller's output) and :func:`all_to_all_zc` (zero-copy DirectWrite / copy-engine,
returns the transport's symmetric-memory output view, dispatching to
``_all_to_all_direct`` / ``_all_to_all_ce``). Each resolves a tuned
``CuteA2AConfig``, picks the analytic launch shape, and ``cute.compile``/launches
the device kernels in :mod:`comms.dsl.cute.a2a.schedules`.

Mirrors :mod:`comms.dsl.triton.a2a.host`: launch tunables come from ``config``;
when ``config is None`` it is looked up from the tuned table by the runtime key,
falling back to the analytic adaptive defaults. The per-element staging schedules
(copy) apply a non-default ``produce``/``consume`` value hook; ``rows>0`` (the
block-tile layout transpose) delegates to :func:`all_to_all_transpose` (zero-copy).
The TMA bulk-copy and zero-copy direct/ce paths move raw bytes and reject a hook.
"""

import logging
import os
from importlib import import_module
from typing import Any

import torch
from comms.dsl.transport import check_transfer, NvlTransport
from comms.dsl.tuning_base import check_geometry

# Importing schedules runs the one-time CuTe setup (CUTE_DSL_ARCH detection +
# cuda-bindings shim) AND imports cutlass; the os.environ statement below is a
# barrier so the formatter cannot hoist the cutlass imports above this setup.
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

from ..hooks import copy_consume, copy_produce, transpose_tile
from .schedules import (
    _A2ACfg,
    _CeSignalKernel,
    _launch_a2a,
    _launch_a2a_tma,
    _launch_a2a_transpose,
    _MAX_PORTABLE_CLUSTER,
    _pick_cluster,
)
from .tuning import (
    CUTE_A2A_GEOMETRY_FIELDS,
    CuteA2AConfig,
    get_a2a_config,
    make_a2a_key,
)

logger: logging.Logger = logging.getLogger(__name__)

_cuda_driver: Any = import_module("cuda.bindings.driver")


# Minimal dtype support: (cutlass dtype, bits-per-element).
_CUTLASS_DTYPE: dict[torch.dtype, tuple[Any, int]] = {
    torch.float32: (cutlass.Float32, 32),
    torch.bfloat16: (cutlass.BFloat16, 16),
}

_COMPILED: dict[tuple[Any, ...], Any] = {}


def all_to_all(  # noqa: C901
    transport: NvlTransport,
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    produce=copy_produce,
    consume=copy_consume,
    rows: int = 0,
    config: CuteA2AConfig | None = None,
    variant: str = "",
) -> None:
    """Equal-split all_to_all_single via the fused CuTe kernel.

    Mirrors ``triton.all_to_all``: launch tunables come from ``config`` (a
    ``CuteA2AConfig``); when ``config is None`` it is looked up from the tuned table by
    the runtime key, falling back to the analytic adaptive defaults. The per-element staging
    schedules (copy) apply a non-default ``produce``/``consume`` value hook; ``rows > 0`` (the
    per-chunk ``[rows, cols] -> [cols, rows]`` block-tile transpose) delegates to
    :func:`all_to_all_transpose` (zero-copy) and writes the result into ``output``. A value hook
    combined with ``rows > 0`` is rejected (a value hook cannot compose with the layout hook); the
    TMA path moves raw bytes and rejects a hook."""
    if not (input.is_cuda and output.is_cuda):
        raise ValueError("cute a2a requires CUDA input/output tensors")
    if not (input.is_contiguous() and output.is_contiguous()):
        raise ValueError("cute a2a requires contiguous input/output tensors")
    if input.dtype != output.dtype:
        raise ValueError("cute a2a requires matching input/output dtype")
    # A non-identity produce/consume runs on the per-element staging schedule (copy); the TMA
    # bulk-copy and the zero-copy direct/ce paths move raw bytes and cannot apply a per-tile
    # hook, so a hook on those raises (TMA below; direct/ce rejected after the config is
    # resolved -- those must route to all_to_all_zc).
    hooked = produce is not copy_produce or consume is not copy_consume
    # rows>0 = the per-chunk [rows, cols] -> [cols, rows] transpose: the fast BLOCK-TILE HOOK
    # path (all_to_all_transpose), a fused block-cooperative SMEM transpose written zero-copy
    # into peers' buffers and returned like all_to_all_zc. It is a LAYOUT hook, not a per-element
    # value hook, so a produce/consume value transform cannot compose with it here -- reject that
    # combination rather than silently dropping the hook. (The old per-element gather hook is
    # superseded -- it could not coalesce the strided access, ~0.26x; the block-tile tier fixes it.)
    if rows > 0:
        if hooked:
            raise ValueError(
                "cute a2a: rows>0 selects the block-tile transpose (a LAYOUT hook) and cannot be "
                "combined with a per-element produce/consume value hook; drop rows= or the hook"
            )
        # all_to_all_transpose is zero-copy (returns the transport's symm-mem buffer); the
        # all_to_all facade contract is to WRITE `output`, so copy the result in. Perf callers
        # (benchmark) call all_to_all_transpose directly to skip this copy.
        output.copy_(all_to_all_transpose(transport, input, rows, config=config))
        return
    if input.dtype not in _CUTLASS_DTYPE:
        raise ValueError(f"cute a2a supports {list(_CUTLASS_DTYPE)}, got {input.dtype}")
    ws = transport.world_size
    numel = input.numel()
    if numel != output.numel():
        raise ValueError("cute a2a requires input.numel() == output.numel()")
    if numel % ws != 0:
        raise ValueError("equal-split requires numel % world_size == 0")
    chunk = numel // ws
    cdtype, dbits = _CUTLASS_DTYPE[input.dtype]
    if config is None:
        config = get_a2a_config(
            make_a2a_key(
                input,
                transport,
                rows=rows,
                produce=produce,
                consume=consume,
                variant=variant,
            )
        )
    check_geometry(transport, config, CUTE_A2A_GEOMETRY_FIELDS)
    if config.primitive in ("direct", "ce"):
        raise ValueError(
            f"primitive {config.primitive!r} is zero-copy (symm-mem output); call "
            f"all_to_all_zc(primitive={config.primitive!r}) instead of all_to_all"
        )
    nb = config.num_blocks
    num_threads, vec = _pick_tile(chunk, dbits, nb * ws)
    # Variant selection: config.primitive picks the staging schedule; an env flag still
    # overrides for A-B (so a default config + env sweep keeps working).
    tma_variant = os.environ.get("A2A_CUTE_TMA") == "1" or config.primitive == "tma"
    send_only = os.environ.get("A2A_CUTE_SENDONLY") == "1"
    if hooked and tma_variant:
        raise NotImplementedError(
            "cute fused hooks are wired into the per-element schedules (copy); the TMA "
            "bulk-copy path moves raw bytes and cannot apply a per-tile transform"
        )
    # TMA-drain adds D drain warps on top of the send threads, so the TMA send width clamps
    # to 512 (the largest power-of-2 with +32 <= the 1024-thread block cap). D=1 is the
    # validated single-warp drain (~305 GB/s on GB300); D>1 parallelises it but is WIP.
    tma_dwarps = max(
        1, int(os.environ.get("A2A_CUTE_TMA_DWARPS", str(config.tma_drain_warps)))
    )
    if tma_variant and num_threads > 1024 - tma_dwarps * 32:
        num_threads = 512
    # num_threads: analytic _pick_tile, overridden by a tuned/explicit config (0 = analytic),
    # an env knob winning over the config; the override cascades into num_tiles / num_slots.
    if "A2A_CUTE_NT" not in os.environ and config.num_threads:
        units = chunk // vec
        nt = min(config.num_threads, units)
        while nt > 1 and units % nt:
            nt -= 1
        num_threads = nt
    # Re-apply the TMA block-thread budget AFTER the config.num_threads override: that override
    # is NOT gated on `not tma_variant` (num_threads IS a tuned knob for the tma primitive), so a
    # tuned value (e.g. 1024) could make the block dim num_threads + tma_dwarps*32 exceed the
    # 1024-thread cap and fail the launch. Re-clamp keeps a fitting tuned value and only clamps
    # when it would overflow.
    if tma_variant and num_threads > 1024 - tma_dwarps * 32:
        num_threads = 512
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
    send_ctr, recv_ctr = transport.step_state()

    in2d = from_dlpack(input.view(ws, chunk), assumed_align=16)
    out2d = from_dlpack(output.view(ws, chunk), assumed_align=16)
    # TMA-drain reads MY local staging buffer (where peers wrote) as [ws, chunk] (row
    # stride cap_elems) -- the descriptor source for the drain bounce. Built only for
    # the TMA variant; other variants pass out2d as an unused placeholder so the
    # compiled signature stays uniform.
    if tma_variant:
        stg_full = transport.handle.get_buffer(
            transport.local_rank, sizes=[ws, cap_elems], dtype=input.dtype
        )
        stg_arg = from_dlpack(stg_full[:, :chunk], assumed_align=16)
    else:
        stg_arg = out2d
    buf_c = from_dlpack(table.buffer_ptrs, assumed_align=8)
    sig_c = from_dlpack(table.signal_pad_ptrs, assumed_align=8)
    send_c = from_dlpack(send_ctr, assumed_align=8)
    recv_c = from_dlpack(recv_ctr, assumed_align=8)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    # Register-blocking unroll for the NVLink store loop (in-flight stores/thread): the
    # per-SM injection lever (GB300 ncu: unroll 1->8 cut the 64MB send leg ~1.55x and
    # lifted the large band ~0.62->~0.90x NCCL; 16 spills the register file). Default to 8
    # once the per-peer chunk is large enough for the unrolled body to engage (tiny sizes
    # keep 1); a tuned/explicit config (0 = analytic) overrides, an env knob winning over it.
    _u_default = 8 if chunk * input.element_size() >= 64 * 1024 else 1
    unroll = max(1, int(os.environ.get("A2A_CUTE_UNROLL", str(_u_default))))
    if "A2A_CUTE_UNROLL" not in os.environ and config.unroll:
        unroll = max(1, config.unroll)
    # CGA cluster size along the block axis (co-locate the blocks targeting one peer on one
    # GPC): raises the per-SM NVLink send rate at the >=256MB band but regresses the mid
    # band, so 0 = analytic default (cluster the large band, else off), -1 = max, >0 =
    # explicit. Capped at the portable cluster size and collapsed to 1 unless it divides
    # num_blocks (so a large SM-budget nb, e.g. 19, runs cluster-off instead of tripping
    # CUDA_ERROR_INVALID_CLUSTER_SIZE). An env knob wins over the config.
    _cl_default = 0 if chunk * input.element_size() >= 64 * 1024 * 1024 else 1
    _cl_env = os.environ.get("A2A_CUTE_CLUSTER")
    _cl = int(_cl_env) if _cl_env is not None else (config.cluster or _cl_default)
    cluster = min(nb if _cl <= 0 else _cl, _MAX_PORTABLE_CLUSTER)
    cluster_y = max(1, int(os.environ.get("A2A_CUTE_CLUSTER_Y", str(config.cluster_y))))
    if nb % cluster:
        cluster = 1
    if ws % cluster_y:
        cluster_y = 1

    key = (
        "a2a",
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
        send_only,
        cluster,
        cluster_y,
        tma_variant,
        tma_dwarps,
        produce,
        consume,
        rows,
    )
    compiled = _COMPILED.get(key)
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
            send_only=send_only,
            cluster=cluster,
            cluster_y=cluster_y,
            tma=tma_variant,
            tma_drain_warps=tma_dwarps,
        )
        # Pick the @cute.jit entry at the host level (real Python branch): the TMA
        # variant routes through _launch_a2a_tma (extra stg_arg + smem), everything
        # else through _launch_a2a. This keeps the dead variant's body out of the
        # trace. The cfg is UNPACKED into individual constexpr args at the
        # cute.compile boundary -- never passed as a single Constexpr object.
        if tma_variant:
            compiled = cute.compile(
                _launch_a2a_tma,
                in2d,
                out2d,
                stg_arg,
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
                cfg.tma_stages,
                cfg.tma_drain_warps,
                cfg.cluster,
                cfg.cluster_y,
                stream,
            )
        else:
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
                cfg.send_only,
                cfg.direct,
                cfg.cluster,
                cfg.cluster_y,
                produce,
                consume,
                rows,
                stream,
            )
        _COMPILED[key] = compiled
    if tma_variant:
        compiled(in2d, out2d, stg_arg, buf_c, sig_c, send_c, recv_c, stream)
    else:
        compiled(in2d, out2d, buf_c, sig_c, send_c, recv_c, stream)


def _check_zc_input(input: torch.Tensor, transport: NvlTransport) -> None:
    """Shared zero-copy (direct/ce) input preconditions -- raise (not ``assert``, so they still
    guard under ``python -O``)."""
    if not (input.is_cuda and input.is_contiguous()):
        raise ValueError("all_to_all_zc requires a contiguous CUDA input")
    if input.dtype not in _CUTLASS_DTYPE:
        raise ValueError(f"cute a2a supports {list(_CUTLASS_DTYPE)}, got {input.dtype}")
    if input.numel() % transport.world_size != 0:
        raise ValueError("equal-split requires numel % world_size == 0")


def _all_to_all_direct(
    transport: NvlTransport,
    input: torch.Tensor,
    *,
    config: CuteA2AConfig,
) -> torch.Tensor:
    """Zero-copy DirectWrite all_to_all: each rank writes its chunks STRAIGHT into
    peers' symmetric-memory output buffers (no staging, no drain), then reads the
    result from its own symm-mem buffer. Returns that buffer view as the output, so
    the transport must be sized ``per_peer_bytes == chunk * elem``. This is the
    theoretical-minimum-work kernel (one NVLink store/elem + one fence); its busbw
    is the per-SM send-rate ceiling that bounds every staging variant from above.

    **Lifetime contract:** returned view is transport-backed and only valid until
    the next collective on the same transport can overwrite it (see ``all_to_all_zc``
    docstring). Caller must ``.clone()`` if it needs to hold the result across calls.

    Honors the same tuned ``CuteA2AConfig`` knobs the staging path does -- a field
    left at its sentinel (``0``) takes the analytic adaptive pick (``_pick_tile`` /
    ``_pick_cluster``), an explicit value (from a tuned direct entry) overrides, and
    an env knob wins over both -- so a tuned direct config drives ``num_threads`` /
    ``unroll`` / ``cluster`` / ``cluster_y`` at launch, not just ``num_blocks``."""
    _check_zc_input(input, transport)
    ws = transport.world_size
    numel = input.numel()
    chunk = numel // ws
    num_blocks = config.num_blocks
    cdtype, dbits = _CUTLASS_DTYPE[input.dtype]
    num_threads, vec = _pick_tile(chunk, dbits, num_blocks * ws)
    # num_threads: analytic _pick_tile, overridden by a tuned config (0 = analytic),
    # an env knob still winning over the config -- mirrors the staging path.
    if "A2A_CUTE_NT" not in os.environ and config.num_threads:
        units = chunk // vec
        nt = min(config.num_threads, units)
        while nt > 1 and units % nt:
            nt -= 1
        num_threads = nt
    num_tiles = chunk // (num_threads * vec)
    check_transfer(transport, chunk, input.dtype, num_blocks)
    cap_elems = transport.per_peer_bytes // input.element_size()
    # The flat get_buffer([numel]) return assumes slot s starts at s*chunk, but slots live at
    # s*cap_elems; equality is the only sizing where the flat view IS the contiguous a2a result
    # while keeping the true zero-copy return (a strided [:,:chunk].reshape would force a copy).
    if cap_elems != chunk:
        raise ValueError("direct needs per_peer_bytes == chunk * elem")
    mbp = transport.max_blocks_per_peer
    # Register-block N independent 16B loads then N stores (NCCL COLL_UNROLL) so N
    # NVLink stores stay in flight per thread -- the dominant per-SM send-rate lever.
    # Mirror the staging path's size-aware default: a hardcoded unroll of 1 leaves
    # ~12-18% busbw on the table (GB300 measured 0.98->1.09x NCCL at 96 MiB/peer,
    # 1.04->1.16x at 48 MiB/peer going unroll 1->8). A tuned config (0 = analytic)
    # overrides; an env knob wins over it.
    _u_default = 8 if chunk * input.element_size() >= 64 * 1024 else 1
    unroll = max(1, int(os.environ.get("A2A_CUTE_UNROLL", str(_u_default))))
    if "A2A_CUTE_UNROLL" not in os.environ and config.unroll:
        unroll = max(1, config.unroll)
    # Direct is the pure-send path -> CGA cluster raises its send-rate ceiling; pick the adaptive
    # default (cluster every block per peer at the large band). Precedence: an A2A_CUTE_CLUSTER env
    # value wins (routed through _pick_cluster, which parses int(env): <=0 = max = num_blocks
    # sentinel, >0 = explicit count, both capped at the portable cap and dropped to 1 if they don't
    # divide num_blocks), else a tuned config.cluster (0 = analytic, -1 = max, >0 = explicit), else
    # the analytic _pick_cluster. The env value IS parsed here (same as staging) -- it just goes
    # through _pick_cluster rather than being read inline.
    _cl_env = os.environ.get("A2A_CUTE_CLUSTER")
    if _cl_env is not None:
        cluster = _pick_cluster(num_blocks, chunk * input.element_size())
    elif config.cluster:
        _cl = num_blocks if config.cluster < 0 else config.cluster
        cluster = min(_cl, _MAX_PORTABLE_CLUSTER)
        if num_blocks % cluster:
            cluster = 1
    else:
        cluster = _pick_cluster(num_blocks, chunk * input.element_size())
    cluster_y = max(1, int(os.environ.get("A2A_CUTE_CLUSTER_Y", str(config.cluster_y))))
    if ws % cluster_y:
        cluster_y = 1

    table = transport.endpoints_device()
    send_ctr, recv_ctr = transport.step_state()
    # The output IS this rank's symm-mem buffer: after the kernel, slot s holds the
    # chunk sender s wrote, i.e. the a2a output. A fresh per-call view (graph-safe).
    output = transport.handle.get_buffer(
        transport.local_rank, sizes=[numel], dtype=input.dtype, storage_offset=0
    )

    in2d = from_dlpack(input.view(ws, chunk), assumed_align=16)
    buf_c = from_dlpack(table.buffer_ptrs, assumed_align=8)
    sig_c = from_dlpack(table.signal_pad_ptrs, assumed_align=8)
    send_c = from_dlpack(send_ctr, assumed_align=8)
    recv_c = from_dlpack(recv_ctr, assumed_align=8)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    key = (
        "a2a_direct",
        num_blocks,
        num_threads,
        vec,
        input.dtype,
        ws,
        chunk,
        cap_elems,
        mbp,
        unroll,
        cluster,
        cluster_y,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        _ensure_cuda_rt_compat()  # lazy: shim cuda-bindings before this cute.compile path
        cfg = _A2ACfg(
            num_blocks=num_blocks,
            num_threads=num_threads,
            vec=vec,
            dtype=cdtype,
            dbits=dbits,
            world_size=ws,
            local_rank=transport.local_rank,
            chunk=chunk,
            cap_elems=cap_elems,
            mbp=mbp,
            # direct is drain-free (no slot pipeline): the direct kernel never reads
            # num_slots/tiles_per_slot, so these are unused placeholders that only satisfy
            # the (required) _A2ACfg fields. They are deliberately out of the compile key.
            num_slots=1,
            tiles_per_slot=num_tiles,
            unroll=unroll,
            direct=True,
            cluster=cluster,
            cluster_y=cluster_y,
        )
        # Unpack cfg into individual constexpr args at the cute.compile boundary
        # (in2d passed twice: out2d is unused on the direct path).
        compiled = cute.compile(
            _launch_a2a,
            in2d,
            in2d,
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
            cfg.send_only,
            cfg.direct,
            cfg.cluster,
            cfg.cluster_y,
            copy_produce,
            copy_consume,
            0,
            stream,
        )
        _COMPILED[key] = compiled
    compiled(in2d, in2d, buf_c, sig_c, send_c, recv_c, stream)  # out2d unused
    return output


# Block-cooperative SMEM transpose tile (bank-conflict-free classic transpose): 32x32 tile,
# 8 thread-rows -> 256 threads/CTA, each thread streams 4 rows. rows/cols must be tile-multiples.
_TR_TILE: int = 32
_TR_BLOCK_ROWS: int = 8

# Per-peer byte crossover for the transpose variant auto-select. GB300 8xGB300 A/B (bit-exact,
# vs the production reference): the STAGED orchestrated variant wins/on-par small+mid
# (2MB 1.47x, 8MB 0.97x, 32MB 1.06x) but its ``empty_like`` scratch crashes above (2GB CUDA
# illegal access); the scratch-free FUSED kernel wins the large band (128MB-2GB, 1.3-11.8x) and
# is OOM-safe. So route <=32MB/peer -> orchestrated, above -> fused.
_TR_ORCH_MAX_BYTES: int = 32 * 1024 * 1024


def all_to_all_transpose(
    transport: NvlTransport,
    input: torch.Tensor,
    rows: int,
    *,
    config: CuteA2AConfig | None = None,
) -> torch.Tensor:
    """Non-contiguous all_to_all: each received ``[rows, cols]`` chunk arrives TRANSPOSED to
    ``[cols, rows]``, via the block-tile ``transpose_tile`` HOOK. Auto-selects the best variant
    per size from the GB300 A/B: the STAGED :func:`all_to_all_transpose_orchestrated` (block-hook
    transpose -> tuned zero-copy a2a) for <=32MB/peer (small+mid, where it wins/on-par vs the
    production reference:
    2MB 1.47x, 8MB 0.97x, 32MB 1.06x), and the mem-efficient single-fused
    :func:`all_to_all_transpose_fused` above (the large band 128MB-2GB, where it wins 1.3-11.8x
    and is scratch-free/OOM-safe -- orchestrated's ``empty_like`` scratch crashes at 2GB).
    ``A2A_TR_ORCH`` forces one path (``1`` = orchestrated, ``0`` = fused). ``rows`` and
    ``chunk // rows`` (= cols) must be multiples of 32.

    **Lifetime contract (zero-copy):** this entry returns a transport-backed view
    (like ``all_to_all_zc``), not an owned copy. The buffer can be overwritten by
    the next collective on the same transport while a delayed local consumer is
    still reading it (faster peer starts next call). Either ``.clone()`` immediately
    or use ``all_to_all`` (staging, writes caller-owned output) if you need ownership.
    This is the same class as the direct/CE zero-copy lifetime issue.
    """
    _orch = os.environ.get("A2A_TR_ORCH")
    if _orch == "1":
        return all_to_all_transpose_orchestrated(transport, input, rows, config=config)
    if _orch != "0":
        per_peer_bytes = (input.numel() // transport.world_size) * input.element_size()
        if per_peer_bytes <= _TR_ORCH_MAX_BYTES:
            return all_to_all_transpose_orchestrated(
                transport, input, rows, config=config
            )
    return all_to_all_transpose_fused(transport, input, rows, config=config)


def all_to_all_transpose_orchestrated(
    transport: NvlTransport,
    input: torch.Tensor,
    rows: int,
    *,
    config: CuteA2AConfig | None = None,
) -> torch.Tensor:
    """Block-hook SMEM transpose into a scratch, then the tuned zero-copy a2a (each leg at its
    peak). Wins the mid band on GB300, but the scratch OOM/crashes at the top of the ladder
    (2GB) -- use :func:`all_to_all_transpose` (fused, mem-efficient) for the full range. Kept as
    an autotuner-selectable mid-band variant (pending the top-of-ladder memory hardening).

    **Lifetime contract:** returns transport-backed view (zero-copy) valid only until
    next collective on same transport (see ``all_to_all_zc``). Caller must clone for longer hold.
    """
    from ..transpose import transpose_chunks

    _check_zc_input(input, transport)
    ws = transport.world_size
    chunk = input.numel() // ws
    if rows <= 0 or chunk % rows:
        raise ValueError(f"rows={rows} must divide the per-peer chunk={chunk}")
    cols = chunk // rows
    if rows % _TR_TILE or cols % _TR_TILE:
        raise ValueError(
            f"transpose needs rows/cols % {_TR_TILE} == 0, got {rows}x{cols}"
        )
    scratch = torch.empty_like(input)
    transpose_chunks(scratch, input, ws, rows, cols)
    return all_to_all_zc(transport, scratch, primitive="direct", config=config)


def all_to_all_transpose_fused(
    transport: NvlTransport,
    input: torch.Tensor,
    rows: int,
    *,
    config: CuteA2AConfig | None = None,
) -> torch.Tensor:
    """Single-fused-kernel variant: the block-cooperative SMEM transpose (``transpose_tile``
    block-hook) is written zero-copy STRAIGHT into peers' symm-mem output buffers in ONE kernel
    (no scratch). Fastest at the small/large bands; the default single-buffer store leg is
    barrier-per-tile, so it loses the mid band on GB300 (8MB 0.75x vs the orchestrated 0.99x).
    ``A2A_TR_PIPELINE=2`` selects the depth-2 load/store software pipeline (overlaps each tile's
    NVLink stores with the next tile's HBM load, one barrier/tile) to close that gap. The fused-vs-
    orchestrated *variant* is size-selected by :func:`all_to_all_transpose`, but this kernel's
    ``num_blocks`` is analytic (occupancy ramp below) -- it does NOT consume a tuned ``config``.
    ``rows``/``cols`` must be multiples of 32.

    **Lifetime contract:** zero-copy - returns transport-backed view whose contents
    can be overwritten by next collective while delayed consumer still reading
    (faster peer starts next call). Caller must ``.clone()`` immediately for ownership.
    """
    _check_zc_input(input, transport)
    ws = transport.world_size
    numel = input.numel()
    chunk = numel // ws
    if rows <= 0 or chunk % rows:
        raise ValueError(f"rows={rows} must divide the per-peer chunk={chunk}")
    cols = chunk // rows
    if rows % _TR_TILE or cols % _TR_TILE:
        raise ValueError(
            f"transpose needs rows/cols % {_TR_TILE} == 0, got {rows}x{cols}"
        )
    cdtype, dbits = _CUTLASS_DTYPE[input.dtype]
    # No tuned-config lookup here: num_blocks is analytic (occupancy ramp below) and the geometry
    # guard is built from a synthetic config reflecting the real launch, so a lookup would be dead.
    # `config` stays in the signature only for symmetry with the orchestrated variant.
    mbp = transport.max_blocks_per_peer
    # The fused SMEM transpose is occupancy-bound: each tile does load->barrier->store, so it
    # needs many CTA waves to hide the barrier stalls (measured 8xH100 8MB: 4 blocks/peer =
    # 0.33x reference, 32 blocks/peer = 0.99x). UNLIKE the plain a2a's small SM-matched grid, so
    # default to a full blocks/peer grid -- but cap at the actual per-peer 2D-tile count so a
    # tiny chunk (e.g. 64x64 = 4 tiles) does not over-launch idle CTAs (which cost 0.39x at 8KB).
    # num_blocks*ws is a full-SM grid at scale, keeping a fair full-device budget.
    # Size ramp: signal/sync overhead scales with block count, so small chunks want few blocks
    # (8xH100 64KB: 4 blocks = 1.88x reference, 32 = 0.54x) while large chunks want a full grid for
    # occupancy (8MB: 32 = 1.00x). Ramp ~1 block per 256KB, floored at 4, capped by mbp and the
    # actual per-peer tile count. The autotuner refines this per size (transpose tuned entries).
    ntiles_pp = (rows // _TR_TILE) * (cols // _TR_TILE)
    chunk_bytes = chunk * input.element_size()
    num_blocks = min(ntiles_pp, max(4, min(mbp, chunk_bytes // (256 * 1024))))
    num_blocks = max(1, num_blocks)
    _tr_blk = os.environ.get("A2A_TR_BLOCKS")
    if _tr_blk:
        num_blocks = max(1, min(int(_tr_blk), mbp, ntiles_pp))
    # Opt-in depth-2 load/store software pipeline: overlaps each tile's NVLink stores with the
    # next tile's coalesced HBM load (one barrier/tile, not two) to close the mid-band gap.
    # Default 1 = the GB300-validated single-buffer path; flip the default once depth-2 is
    # A/B-validated on GB300 (bit-exact is verified on H100 for both depths).
    # Store-unroll factor U (A2A_TR_PIPELINE): batch U 2D-tiles per barrier group so U*(tile/
    # block_rows) NVLink stores are in flight and barriers drop to 2/U-per-tile (vs 2 for U=1),
    # closing the mid-band gap. Default 1 = the GB300-validated single-buffer path; flip the
    # default once a U is A/B-validated on GB300 (bit-exact is verified on H100 for U>1).
    pipeline_depth = max(1, min(8, int(os.environ.get("A2A_TR_PIPELINE", "1"))))
    check_transfer(transport, chunk, input.dtype, num_blocks)
    cap_elems = transport.per_peer_bytes // input.element_size()
    if cap_elems != chunk:
        raise ValueError("transpose (zero-copy) needs per_peer_bytes == chunk * elem")
    # Guard on the ACTUAL launch geometry, not the resolved `config`: this path computes
    # num_blocks analytically (the occupancy ramp above, independent of config.num_blocks) and
    # always transfers zero-copy DirectWrite. Validating with `config` (which may carry a copy
    # primitive / a different num_blocks) would let a real geometry switch slip past the guard on
    # a reused transport, since the stashed signature would not reflect the real per-(peer,block)
    # step-counter shape. Build the signature from the values actually used at launch (mirrors the
    # ce path's synthetic CuteA2AConfig(primitive="ce")).
    check_geometry(
        transport,
        CuteA2AConfig(num_blocks=num_blocks, primitive="direct"),
        CUTE_A2A_GEOMETRY_FIELDS,
    )

    table = transport.endpoints_device()
    send_ctr, recv_ctr = transport.step_state()
    output = transport.handle.get_buffer(
        transport.local_rank, sizes=[numel], dtype=input.dtype, storage_offset=0
    )
    in2d = from_dlpack(input.view(ws, chunk), assumed_align=16)
    buf_c = from_dlpack(table.buffer_ptrs, assumed_align=8)
    sig_c = from_dlpack(table.signal_pad_ptrs, assumed_align=8)
    send_c = from_dlpack(send_ctr, assumed_align=8)
    recv_c = from_dlpack(recv_ctr, assumed_align=8)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    key = (
        "a2a_transpose",
        num_blocks,
        input.dtype,
        ws,
        rows,
        cols,
        cap_elems,
        mbp,
        pipeline_depth,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        _ensure_cuda_rt_compat()
        compiled = cute.compile(
            _launch_a2a_transpose,
            in2d,
            in2d,  # out2d unused (zero-copy)
            buf_c,
            sig_c,
            send_c,
            recv_c,
            cdtype,
            dbits,
            num_blocks,
            transport.local_rank,
            chunk,
            cap_elems,
            mbp,
            ws,
            rows,
            cols,
            _TR_TILE,
            _TR_BLOCK_ROWS,
            pipeline_depth,
            transpose_tile,
            stream,
        )
        _COMPILED[key] = compiled
    compiled(in2d, in2d, buf_c, sig_c, send_c, recv_c, stream)
    return output


def _ce_buf_host(transport: NvlTransport, table) -> list[int]:
    """Host-side peer symm-mem buffer base addresses (cached on the transport).

    Read once (a device->host copy) and reused, so the timed/CUDA-graph-captured
    region issues only ``cuMemcpyAsync`` + the signal kernel (no host sync)."""
    cached = getattr(transport, "_a2a_ce_buf_host", None)
    if cached is None:
        cached = table.buffer_ptrs.cpu().tolist()
        transport._a2a_ce_buf_host = cached  # pyre-ignore[16]: runtime cache
    return cached


def _all_to_all_ce(transport: NvlTransport, input: torch.Tensor) -> torch.Tensor:
    """Copy-engine zero-copy all_to_all: move data with ``cuMemcpyAsync`` (the GB300
    copy engines, zero SM occupancy) straight into peers' symmetric-memory output
    buffers, then a 1-CTA kernel does the cross-rank completion handshake. Frees all
    SMs for compute overlap; wins the very-large band where the copy engines sustain
    more BW than per-SM NVLink stores (ported from Cen's ``ce_a2a_zc``, D108841080).
    Returns this rank's symm-mem buffer view (slot s = chunk from sender s).

    **Lifetime contract:** same as ``all_to_all_zc`` / direct - returned view is
    transport-backed and valid only until next collective on same transport.
    Caller must clone if holding across calls.
    """
    _check_zc_input(input, transport)
    ws = transport.world_size
    numel = input.numel()
    chunk = numel // ws
    elem = input.element_size()
    per_peer_bytes = chunk * elem
    cap_elems = transport.per_peer_bytes // elem
    # The final get_buffer([numel]) is a contiguous view that equals the a2a output only when
    # cap_elems == chunk (cuMemcpyAsync writes slot s at s*cap_bytes); == keeps the flat
    # zero-copy return correct without a strided reshape/copy.
    if cap_elems != chunk:
        raise ValueError("ce needs per_peer_bytes == chunk * elem")
    cap_bytes = cap_elems * elem
    rank = transport.local_rank
    mbp = transport.max_blocks_per_peer
    table = transport.endpoints_device()
    buf_host = _ce_buf_host(transport, table)
    src_base = input.data_ptr()
    raw_stream = torch.cuda.current_stream().cuda_stream

    # Round-robin peer order: every rank starts at a different peer so the N remote
    # writes do not all incast onto one destination at once.
    for step in range(ws):
        peer = (rank + step) % ws
        src = src_base + peer * per_peer_bytes
        dst = buf_host[peer] + rank * cap_bytes
        (err,) = _cuda_driver.cuMemcpyAsync(dst, src, per_peer_bytes, raw_stream)
        # External driver return code must survive -O (an assert can be stripped, turning a
        # failed copy into a silent read of undefined peer data).
        if err != _cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyAsync failed: {err}")

    send_ctr, recv_ctr = transport.step_state()
    sig_c = from_dlpack(table.signal_pad_ptrs, assumed_align=8)
    send_c = from_dlpack(send_ctr, assumed_align=8)
    recv_c = from_dlpack(recv_ctr, assumed_align=8)
    stream = _cuda_driver.CUstream(raw_stream)
    key = ("a2a_ce", ws, rank, mbp)
    compiled = _COMPILED.get(key)
    if compiled is None:
        _ensure_cuda_rt_compat()  # lazy: shim cuda-bindings before this cute.compile path
        k = _CeSignalKernel(world_size=ws, local_rank=rank, mbp=mbp)
        compiled = cute.compile(k, sig_c, send_c, recv_c, stream)
        _COMPILED[key] = compiled
    compiled(sig_c, send_c, recv_c, stream)
    return transport.handle.get_buffer(
        rank, sizes=[numel], dtype=input.dtype, storage_offset=0
    )


def all_to_all_zc(
    transport: NvlTransport,
    input: torch.Tensor,
    *,
    primitive: str = "direct",
    config: CuteA2AConfig | None = None,
) -> torch.Tensor:
    """Zero-copy all_to_all: write each chunk straight into peers' symmetric-memory buffers
    and return this rank's buffer view (slot ``s`` = the chunk from sender ``s``); the buffer
    is read back in place, so the transport must be sized ``per_peer_bytes == chunk * elem``.
    ``primitive='direct'`` is the DirectWrite NVLink-store path; ``'ce'`` is the copy-engine
    path (``cuMemcpyAsync`` on the GB300 copy engines, zero SM occupancy). For ``'direct'``,
    ``config`` supplies the tuned launch knobs (``num_blocks`` plus ``num_threads`` / ``unroll``
    / ``cluster`` / ``cluster_y``; a sentinel ``0`` takes the analytic pick); the copy-engine
    path has no grid and ignores ``config``.

    **Lifetime / ownership contract (enforced API):** the returned tensor is a *view* into the
    transport's symmetric-memory buffer (``transport.handle.get_buffer``), not an owned copy.
    Its contents are only valid until the next collective that uses the same transport
    (``all_to_all``, ``all_to_all_zc``, ``all_to_all_transpose``) on *any* rank can
    overwrite it. A delayed consumer that keeps the view across calls will observe the
    next call's payload (cross-rank skew). Callers that need the result longer must
    ``.clone()`` or copy out immediately after the call, before the next collective.

    This is intentional - zero-copy avoids the HBM copy at the cost of caller-managed
    lifetime, exactly like the staging HEAD free-credit for the pipelined path, but for
    the *output* buffer. If a handshake is desired, call ``transport`` with ``all_to_all``
    (staging, writes caller-owned ``output``) instead of ``all_to_all_zc``.
    """
    if primitive == "direct":
        if config is None:
            # Mirror the staging path + the Triton twin: a None config looks up the
            # tuned table by the runtime key so a tuned direct entry's knobs apply,
            # falling back to the analytic default.
            config = get_a2a_config(make_a2a_key(input, transport))
        # Same geometry guard as the staging all_to_all: the zero-copy paths drive the
        # transport's persistent step counters too, so switching num_blocks/primitive on a
        # reused transport without a drain is the identical hazard. Guard AFTER the config is
        # resolved, BEFORE dispatch.
        check_geometry(transport, config, CUTE_A2A_GEOMETRY_FIELDS)
        return _all_to_all_direct(transport, input, config=config)
    if primitive == "ce":
        # ce has no grid (1-CTA signal kernel) so no num_blocks knob; check the geometry
        # signature against a config that names this primitive so a copy/direct -> ce switch
        # on a reused transport is still flagged.
        check_geometry(
            transport, CuteA2AConfig(primitive="ce"), CUTE_A2A_GEOMETRY_FIELDS
        )
        return _all_to_all_ce(transport, input)
    raise ValueError(
        f"all_to_all_zc supports primitive 'direct' or 'ce'; got {primitive!r}"
    )
