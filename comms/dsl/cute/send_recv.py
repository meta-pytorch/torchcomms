# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# pyre-ignore-all-errors[6, 35, 58]: @cute.kernel / @cute.jit constexpr params are
# annotated cutlass.Constexpr; pyre models that as Constexpr[Any] and rejects the
# arithmetic / range() / dataclass-field uses the kernel bodies do on them -- the values
# are real compile-time ints at trace time.

"""Shared CuTe device substrate for collectives over ``NvlTransport``.

The module owns vectorized copy leaves, per-slot send/receive primitives, CUDA DSL setup,
and analytic tile/slot sizing. Collective schedules compose these helpers and supply their
own peer ownership, address mapping, barriers, and counter progression.
"""

# NOTE: do NOT add ``from __future__ import annotations`` here. The @cute.jit
# functions below (``_send_slot`` / ``_recv_slot``) classify their compile-time params via the REAL
# ``cutlass.Constexpr`` annotation object, which ``inspect.getfullargspec`` only exposes
# when annotations are NOT stringized -- stringizing makes cute marshal a constexpr dtype
# as a dynamic arg and crash (Float32.__c_pointers__ TypeError).
import logging
import os
from importlib import import_module
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)


def _resolve_cute_dsl_arch() -> str:
    """``sm_<major><minor>[a]`` for the local device; honors CUTE_DSL_ARCH."""
    explicit = os.environ.get("CUTE_DSL_ARCH")
    if explicit:
        return explicit
    try:
        import torch as _torch

        major, minor = _torch.cuda.get_device_capability(_torch.cuda.current_device())
    except (ImportError, RuntimeError, AssertionError) as e:
        logger.warning(
            "could not detect CUDA device capability (%s); "
            "defaulting CUTE_DSL_ARCH to sm_90a",
            e,
        )
        return "sm_90a"
    if (major, minor) == (9, 0):
        return "sm_90a"
    if (major, minor) in {(10, 0), (10, 1)}:
        return "sm_100a"
    if (major, minor) == (8, 0):
        return "sm_80"
    return f"sm_{major}{minor}"


# Must be set before importing cutlass.cute.
os.environ.setdefault("CUTE_DSL_ARCH", _resolve_cute_dsl_arch())

import cutlass
import cutlass.cute as cute

from . import nvl_ops

_cuda_rt: Any = import_module("cuda.bindings.runtime")


def _ensure_cuda_rt_compat() -> None:
    """Idempotently shim cuda-bindings symbols the cutlass DSL JIT executor expects.

    Some cuda-bindings versions lack ``cudaLibrary_t`` / ``cudaLibraryUnload``, so kernel
    compilation cannot load the cuda library. This patches them in on first use (NOT at
    import scope, per the don't-mutate-imported-module-attributes-at-import rule).
    The installed ``cudaLibraryUnload`` is a process-wide no-op, so it can leak library
    handles for OTHER callers of cuda.bindings.runtime -- we log a warning when installing
    it so that side effect is visible rather than silent. Invoked lazily from the JIT paths
    (just before cute.compile)."""
    if hasattr(_cuda_rt, "cudaLibrary_t"):
        return

    class _cudaLibrary_t:
        __slots__ = ("value",)

        def __init__(self, value: int = 0) -> None:
            self.value = value

    logger.warning(
        "installing process-wide cuda.bindings.runtime shim (cudaLibrary_t + "
        "no-op cudaLibraryUnload); cudaLibraryUnload becomes a no-op for ALL callers"
    )
    _cuda_rt.cudaLibrary_t = _cudaLibrary_t
    _cuda_rt.cudaLibraryUnload = lambda lib: (_cuda_rt.cudaError_t(0),)


# ---------------------------------------------------------------------------
# Shared CuTe send/recv substrate: the vectorized copy leaf + the per-slot
# credit-ring primitives (``_send_slot`` / ``_recv_slot``). This is the backend
# substrate home. A schedule composes these primitives by supplying its own per-peer
# region/offset math, so a perf/codegen change here lands once and every composer
# inherits it. The dependency is one-way: a composer imports these from here; nothing
# here imports a schedule.
# ---------------------------------------------------------------------------


def _copy_atom(dtype: Any, vec: int, dbits: int):
    """Build the vectorized gmem copy atom shared by every schedule."""
    return cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=vec * dbits
    )


def _copy_u(thr_copy, copy_atom, g_src, g_dst, t, n, num_blocks):
    """Copy ``n`` consecutive (stride ``num_blocks``) tiles starting at ``t``:
    issue all ``n`` loads, then all ``n`` stores, so ``n`` NVLink stores are in
    flight per thread (mirrors NCCL's ``COLL_UNROLL`` -- the per-SM-extraction
    lever Triton/TLX could not use because the wider per-thread footprint
    spilled). STRAIGHT-LINE only (``n`` is constexpr; no control flow) so it is
    safe to call from the kernel body -- CuTe only rewrites control flow in the
    ``@cute.kernel`` function itself, so the grid-stride ``while`` stays inline
    there. Correctness is unroll-independent: src and dst share one ``thr_copy``
    partition, so any order copies the same elements."""
    nb = num_blocks
    frags = [
        nvl_ops.get(copy_atom, thr_copy.partition_S(g_src[(None, t + i * nb)]))
        for i in range(n)
    ]
    for i in range(n):
        nvl_ops.put(
            copy_atom, frags[i], thr_copy.partition_D(g_dst[(None, t + i * nb)])
        )


def _copy_u2(thr_copy, copy_atom, g_src, s0, g_dst, d0, n, num_blocks):
    """Like :func:`_copy_u` but with INDEPENDENT src/dst tile bases: copy ``n``
    stride-``num_blocks`` tiles from ``g_src[s0 + i*nb]`` to ``g_dst[d0 + i*nb]``.
    Used by the bounded-ring schedule, where the source tile index (global chunk
    offset) and the destination tile index (ring-slot-local offset) differ. All
    ``n`` loads then all ``n`` stores, so ``n`` NVLink stores are in flight."""
    nb = num_blocks
    frags = [
        nvl_ops.get(copy_atom, thr_copy.partition_S(g_src[(None, s0 + i * nb)]))
        for i in range(n)
    ]
    for i in range(n):
        nvl_ops.put(
            copy_atom, frags[i], thr_copy.partition_D(g_dst[(None, d0 + i * nb)])
        )


@cute.jit
def _send_slot(
    thr_copy,
    copy_atom,
    g_in,
    g_send,
    tail_remote,
    start_send,
    b,
    tidx,
    s: cutlass.Constexpr,
    tiles_per_slot: cutlass.Constexpr,
    num_tiles: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
) -> None:
    """One credit-ring SEND slot: stage slot ``s`` into the peer's staging over NVLink,
    then publish its data-ready TAIL.

    Extracted as a ``@cute.jit`` device sub-function so every schedule composes the SAME
    slot primitive. It carries the runtime grid-stride ``while`` + barrier + leader
    signal, so it MUST be ``@cute.jit`` -- a plain helper would not get CuTe's
    control-flow rewrite (see ``_copy_u``)."""
    u = unroll
    nb = num_blocks
    s_lo = s * tiles_per_slot
    s_hi = min((s + 1) * tiles_per_slot, num_tiles)
    t = s_lo + b
    while t + (u - 1) * nb < s_hi:
        _copy_u(thr_copy, copy_atom, g_in, g_send, t, u, num_blocks)
        t += u * nb
    while t < s_hi:
        _copy_u(thr_copy, copy_atom, g_in, g_send, t, 1, num_blocks)
        t += nb
    cute.arch.barrier()
    if tidx == 0:
        # data-ready for slots 0..s (monotonic counter).
        nvl_ops.signal(tail_remote, start_send + s + 1)


@cute.jit
def _recv_slot(
    thr_copy,
    copy_atom,
    g_recv,
    g_out,
    tail_local,
    start_recv,
    b,
    tidx,
    s: cutlass.Constexpr,
    tiles_per_slot: cutlass.Constexpr,
    num_tiles: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
) -> None:
    """One credit-ring RECV slot: wait the peer's data-ready TAIL for slot ``s``, then
    drain its matching staging slot into the output. Symmetric twin of :func:`_send_slot`;
    same ``@cute.jit`` rationale."""
    u = unroll
    nb = num_blocks
    p_lo = s * tiles_per_slot
    p_hi = min((s + 1) * tiles_per_slot, num_tiles)
    if tidx == 0:
        nvl_ops.wait(tail_local, start_recv + s + 1)
    cute.arch.barrier()
    t = p_lo + b
    while t + (u - 1) * nb < p_hi:
        _copy_u(thr_copy, copy_atom, g_recv, g_out, t, u, num_blocks)
        t += u * nb
    while t < p_hi:
        _copy_u(thr_copy, copy_atom, g_recv, g_out, t, 1, num_blocks)
        t += nb


# ---------------------------------------------------------------------------
# Pipelined credit-ring send/recv collective (graph-safe), composed from the shared
# _send_slot / _recv_slot substrate above. Single-peer-pair whole-buffer transfer,
# keeping the slot send/drain overlap. uni (send-only / recv-only) and bidir are selected
# by the do_send / do_recv constexprs. Graph-safe via the transport's persistent monotonic
# counters + the symmetric-memory signal pad.
# ---------------------------------------------------------------------------


# Analytic tile / slot sizing for the pipelined send/recv geometry. Owned here so the
# send/recv collective is self-contained; a fused multi-peer schedule reuses these.
_SATURATION_THREADS: int = 32768
_MIN_TILES: int = 32  # keep at least this many tiles so the pipeline has slots
# Target number of pipeline slots per (peer, block) once pipelining is on. The
# send/drain overlap approaches the send-only NVLink ceiling as slots grow (the
# un-overlapped tail is one slot's drain), with diminishing returns vs per-slot
# signal overhead; ~8 is the measured knee on H100. Overridable for the autotuner.
_NUM_SLOTS: int = 8
# Only pipeline when the per-peer chunk is large enough that the bandwidth-bound
# send/drain overlap win beats the per-slot sync overhead. Below this the message
# is latency-bound and a single shot (no extra barriers/signals/waits) is faster.
# Measured on 8xH100: chunks <4MB regress under pipelining, >=8MB chunks gain
# ~15-25%. Gated on bytes, not tiles -- 16MB and 64MB land at the same tile count
# but opposite optima, so absolute size is the right signal.
_MIN_PIPELINE_CHUNK_BYTES: int = 4 * 1024 * 1024
# Per-peer chunk at/above which the deep (8-slot) run-ahead beats the shallow
# (4-slot) one -- below it the deeper pipeline's per-slot sub-chunk is too small and
# the sync overhead dominates (GB300, unroll=8). 64 MiB chunk.
_DEEP_PIPELINE_CHUNK_BYTES: int = 64 * 1024 * 1024


def _pick_tile(chunk: int, dbits: int, total_ctas: int) -> tuple[int, int]:
    """Pick (num_threads, vec) so the per-(peer,block) chunk tiles EXACTLY.

    Vec is the widest 128-bit-down-to-scalar copy the chunk allows (vec drops to 1
    for tiny/odd chunks, so the whole 32B-2GB ladder tiles with no tail). Threads
    are chosen CTA-aware: ``_SATURATION_THREADS / total_ctas`` (more warps per CTA
    when few CTAs are in the SM budget), bounded so a small chunk keeps
    ``_MIN_TILES`` tiles, floored so a chunk with real work is not under-threaded,
    and capped at the 1024 hardware limit. An explicit ``A2A_CUTE_NT`` env overrides
    the analytic pick (for sweeps / the autotuner).
    """
    env_nt = os.environ.get("A2A_CUTE_NT")
    for vbits in (128, 64, 32, 16):
        if vbits < dbits:
            continue
        vec = vbits // dbits
        if chunk % vec:
            continue
        units = chunk // vec  # number of vectors to copy across the whole chunk
        if env_nt is not None:
            # Floor at 1 and cap at 1024 (mirrors the analytic branch): A2A_CUTE_NT="0" would
            # otherwise return nt=0 and divide-by-zero in the caller's num_tiles math, and
            # A2A_CUTE_NT>1024 would exceed the CUDA per-block thread cap and fail the launch.
            nt = max(1, min(int(env_nt), units, 1024))
        else:
            nt = max(1, _SATURATION_THREADS // max(1, total_ctas))
            nt = min(nt, max(1, units // _MIN_TILES))  # leave enough tiles
            nt = min(nt, 1024)  # hardware cap
            nt = max(nt, min(256, units))  # don't under-thread a chunk with work
        nt = min(nt, units)
        while nt > 1 and units % nt:
            nt -= 1
        # Final floor: the `min(nt, units)` above re-introduces nt=0 when units==0
        # (chunk==0), which would divide-by-zero in the caller's num_tiles math.
        return max(1, nt), vec
    return 1, 1  # scalar fallback (chunk not a multiple of any vec width)


def _pick_slots(num_tiles: int, chunk_bytes: int) -> tuple[int, int]:
    """Split ``num_tiles`` into (num_slots, tiles_per_slot) for the send/drain
    pipeline. Returns ``(1, num_tiles)`` (single shot, no pipeline) for per-peer
    chunks below ``_MIN_PIPELINE_CHUNK_BYTES`` -- latency-bound sizes are faster
    without the per-slot sync. An explicit ``A2A_CUTE_SLOTS`` env forces the slot
    count (for sweeps / the autotuner)."""
    if num_tiles <= 1:
        return 1, 1
    env_slots = os.environ.get("A2A_CUTE_SLOTS")
    if env_slots is not None:
        want = max(1, min(int(env_slots), num_tiles))
    elif chunk_bytes < _MIN_PIPELINE_CHUNK_BYTES:
        return 1, num_tiles  # single shot: latency-bound band
    else:
        # The mid-large band over-pipelines at 8 slots -- each slot's sub-chunk gets
        # too small and the per-slot sync dominates, so 4 slots is the knee there; the
        # >=64MB chunk band keeps 8 (the deeper run-ahead still wins).
        want = min(
            4 if chunk_bytes < _DEEP_PIPELINE_CHUNK_BYTES else _NUM_SLOTS, num_tiles
        )
    tps = (num_tiles + want - 1) // want
    slots = (num_tiles + tps - 1) // tps  # re-derive exact slot count for this tps
    return slots, tps
