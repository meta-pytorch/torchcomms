# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# pyre-ignore-all-errors[6, 29, 35, 58, 61]: @cute.kernel / @cute.jit constexpr params are
# annotated cutlass.Constexpr; pyre models that as Constexpr[Any] and rejects the
# arithmetic / range() / dataclass-field uses the kernel bodies do on them, and the calls
# to the produce/consume hook constexprs ([29]) -- the values are real compile-time ints /
# callables at trace time. [61]: locals set under a ``do_send``/``do_recv`` constexpr guard
# and used under the same guard are always defined at runtime. Same idiom as the schedules twin.

"""Minimal CuTe DSL send/recv kernel for the composable framework.

The CuTe realization of the device send/recv primitive. It is the DSL twin of
``framework/triton/send_recv.py``: same contract (consumes a `PeerEndpoint` from
the shared `NvlTransport`), same minimal semantics (no pipeline, single-shot,
per-block chunk + one data-ready signal), just written in CuTe instead of Triton.
Its existence validates that the host layer (transport, ctx, signal protocol) is
genuinely DSL-agnostic.

Minimal scope (correctness only; performance is not a goal): no slots /
double-buffer / credit; ``seq = 1`` single-shot; requires ``numel`` divisible by
the CTA thread count (no tail handling); fixed 1-element-per-thread copy.
"""

# NOTE: do NOT add ``from __future__ import annotations`` here. The @cute.jit /
# @cute.kernel launchers below (``_send_slot`` / ``_recv_slot`` / ``_sendrecv_kernel`` /
# ``_launch_sendrecv``) classify their compile-time params via the REAL
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

        major, minor = _torch.cuda.get_device_capability(0)
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
import torch
from cutlass.cute.runtime import from_dlpack

from ..transport import check_transfer, NvlTransport
from . import nvl_ops
from .ctx import BlockCtx, Ctx
from .hooks import copy_consume, copy_produce

_cuda_driver: Any = import_module("cuda.bindings.driver")
_cuda_rt: Any = import_module("cuda.bindings.runtime")


def _ensure_cuda_rt_compat() -> None:
    """Idempotently shim cuda-bindings symbols the cutlass DSL JIT executor expects.

    Some cuda-bindings versions lack ``cudaLibrary_t`` / ``cudaLibraryUnload``, so kernel
    compilation cannot load the cuda library. This patches them in on first use (NOT at
    import scope, per python.md's don't-mutate-imported-module-attributes-at-import rule).
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


_NUM_THREADS: int = 128

# Minimal dtype support: (cutlass dtype, bits-per-element).
_CUTLASS_DTYPE: dict[torch.dtype, tuple[Any, int]] = {
    torch.float32: (cutlass.Float32, 32),
    torch.bfloat16: (cutlass.BFloat16, 16),
}

_COMPILED: dict[tuple[Any, ...], object] = {}


# ---------------------------------------------------------------------------
# Shared CuTe send/recv substrate: the vectorized copy leaves + the per-slot
# credit-ring primitives (``_send_slot`` / ``_recv_slot``). This is the backend
# substrate home -- symmetric with the Triton ``triton/send_recv.py`` holding
# ``send_step`` / ``recv_step``. A schedule composes these primitives by supplying its
# own per-peer region/offset math, so a perf/codegen change here lands once and every
# composer inherits it. The dependency is one-way: a composer imports these from here;
# nothing here imports a schedule.
# ---------------------------------------------------------------------------


def _copy_atom(dtype: Any, vec: int, dbits: int):
    """The vectorized gmem copy atom every schedule shares: VEC contiguous elems per
    thread per copy, so the NVLink store is a vectorized st.global (up to 128-bit)
    instead of a scalar store -- the single biggest copy-bandwidth lever (mirrors the
    Triton 128-bit-store fix). The TV-tiled copy is built per schedule on top of it."""
    return cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        dtype,
        num_bits_per_copy=vec * dbits,
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


# Hook-aware twins of _copy_u, one per leg. The default copy hooks make them identical to
# _copy_u (nvl_ops.get IS copy_produce, nvl_ops.put IS copy_consume -- both cute.copy over
# the atom), so the identity path is bit-for-bit unchanged; a non-default hook transforms
# the tile on its owning leg. Same unroll shape (issue all n loads, then all n stores) so n
# NVLink transfers stay in flight, and the same straight-line (n constexpr) contract so they
# are safe to call from the kernel body.
def _send_u(
    thr_copy,
    copy_atom,
    g_in,
    g_send,
    t,
    n,
    num_blocks,
    produce,
    src=None,
    tiler=None,
    rows=0,
    chunk=0,
    peer=None,
):
    """Send leg: produce n input tiles (HBM -> frag, transform) then NVLink-store to staging.

    ``src``/``tiler``/``rows``/``chunk`` are the gather-hook context (this peer's raw chunk +
    tile shape); a value-transform hook ignores them, a gather hook (transpose) uses them to
    re-index ``src`` itself. ``peer`` is the position/identity fact for per-peer transforms.
    Defaults keep the identity/value path unchanged."""
    nb = num_blocks
    frags = [
        produce(
            Ctx(
                part=thr_copy.partition_S(g_in[(None, t + i * nb)]),
                atom=copy_atom,
                coord=t + i * nb,
                rows=rows,
                chunk=chunk,
                peer=peer,
            )
        )
        for i in range(n)
    ]
    for i in range(n):
        nvl_ops.put(
            copy_atom, frags[i], thr_copy.partition_D(g_send[(None, t + i * nb)])
        )


def _recv_u(thr_copy, copy_atom, g_recv, g_out, t, n, num_blocks, consume, peer=None):
    """Recv leg: NVLink-load n staging tiles then consume them (frag -> HBM, transform).

    ``coord``/``peer`` are the position/identity facts a consume hook may use (masking,
    per-peer dequant scale); a value/identity consume ignores them."""
    nb = num_blocks
    frags = [
        nvl_ops.get(copy_atom, thr_copy.partition_S(g_recv[(None, t + i * nb)]))
        for i in range(n)
    ]
    for i in range(n):
        consume(
            Ctx(
                part=thr_copy.partition_D(g_out[(None, t + i * nb)]),
                atom=copy_atom,
                coord=t + i * nb,
                peer=peer,
            ),
            frags[i],
        )


def _local_u(
    thr_copy,
    copy_atom,
    g_in,
    g_out,
    t,
    n,
    num_blocks,
    produce,
    consume,
    src=None,
    tiler=None,
    rows=0,
    chunk=0,
    peer=None,
):
    """Diagonal (local) leg: produce from input then consume to output (no NVLink hop).

    ``src``/``tiler``/``rows``/``chunk`` carry the gather-hook context (see ``_send_u``);
    ``coord``/``peer`` are the position/identity facts for both legs."""
    nb = num_blocks
    frags = [
        produce(
            Ctx(
                part=thr_copy.partition_S(g_in[(None, t + i * nb)]),
                atom=copy_atom,
                coord=t + i * nb,
                rows=rows,
                chunk=chunk,
                peer=peer,
            )
        )
        for i in range(n)
    ]
    for i in range(n):
        consume(
            Ctx(
                part=thr_copy.partition_D(g_out[(None, t + i * nb)]),
                atom=copy_atom,
                coord=t + i * nb,
                peer=peer,
            ),
            frags[i],
        )


def _block_tile_u(
    sA, src, dst, br, bc, tile, block_rows, tx, ty, hook, war_barrier=True
):
    """Block-tile (layout) hook leaf: the CTA-cooperative twin of :func:`_send_u`/:func:`_copy_u`.

    Coalesced-loads the ``[tile, tile]`` tile at 2D coord ``(br, bc)`` from ``src`` (a per-peer
    ``[rows, cols]`` view) into the padded SMEM tile ``sA``, barriers (RAW), then the block-tile
    ``hook`` (a :class:`BlockCtx` consumer such as ``transpose_tile``) transforms it in SMEM and
    coalesced-stores it into ``dst`` at the transformed position. Both gmem legs stay coalesced
    (``tx`` = lane is the contiguous dim on load AND store) -- the whole point of the block-tile
    tier vs the per-element value hook.

    ``war_barrier=False`` drops the trailing WAR barrier for a ROTATING-buffer (store-unroll)
    caller: a buffer is reused ``depth`` tiles later, covered by the intervening RAW load-barriers,
    so one barrier per tile suffices and each store overlaps the next load. Straight-line (only
    barriers + a constexpr ``r``-loop, no control flow) so it composes into any schedule's
    grid-stride loop -- exactly the contract of the value leaf ``_send_u``. This is the SHARED
    substrate leaf for the block-tile hook tier, composed by the a2a transpose schedules in the
    next layer (symmetric with how the value leaves ``_send_u``/``_send_slot`` are composed)."""
    for r in range(0, tile, block_rows):
        sA[(ty + r, tx)] = src[(br * tile + ty + r, bc * tile + tx)]
    cute.arch.barrier()
    hook(BlockCtx(sA, dst, tile, block_rows, tx, ty, br, bc))
    if war_barrier:
        cute.arch.barrier()


@cute.jit
def _send_slot(
    thr_copy,
    copy_atom,
    g_in,
    g_send,
    tail_remote,
    head_local,
    start_send,
    b,
    tidx,
    s: cutlass.Constexpr,
    tiles_per_slot: cutlass.Constexpr,
    num_tiles: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    num_slots: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    produce: cutlass.Constexpr,
    src=None,
    tiler=None,
    rows=0,
    chunk=0,
    peer=None,
) -> None:
    """One credit-ring SEND slot (CuTe twin of the Triton ``send_step``): wait the peer's
    free-credit HEAD for this slot's prior occupant, stage slot ``s`` into the peer's
    staging over NVLink, then publish its data-ready TAIL.

    Extracted as a ``@cute.jit`` device sub-function so every schedule composes the SAME
    slot primitive (symmetric with how the Triton ``send_step`` is shared). It carries the
    runtime grid-stride ``while`` + barrier + leader signal, so it MUST be ``@cute.jit`` --
    a plain helper would not get CuTe's control-flow rewrite (see ``_copy_u``).
    ``src``/``tiler``/``rows``/``chunk`` are the gather-hook context forwarded to ``_send_u``.

    Free-credit (HEAD): the whole per-peer staging region is ``num_slots`` slots, rewritten
    every call, so this physical slot's memory is reused every ``num_slots`` absolute steps.
    Before overwriting it we wait until the peer has drained its PREVIOUS occupant --
    ``head_local >= (this slot's seq) - num_slots`` -- so a faster rank cannot clobber a slot
    the peer has not finished reading (monotonic TAIL counters alone do not prevent that).
    The target is non-positive on the first ring-fill (no prior occupant), which the SIGNED
    :func:`nvl_ops.wait_free` treats as already satisfied, so the first call never blocks."""
    u = unroll
    nb = num_blocks
    s_lo = s * tiles_per_slot
    s_hi = min((s + 1) * tiles_per_slot, num_tiles)
    if tidx == 0:
        nvl_ops.wait_free(head_local, start_send + s + 1 - num_slots)
    cute.arch.barrier()  # free-credit observed before any thread overwrites the slot
    t = s_lo + b
    while t + (u - 1) * nb < s_hi:
        _send_u(
            thr_copy,
            copy_atom,
            g_in,
            g_send,
            t,
            u,
            num_blocks,
            produce,
            src=src,
            tiler=tiler,
            rows=rows,
            chunk=chunk,
            peer=peer,
        )
        t += u * nb
    while t < s_hi:
        _send_u(
            thr_copy,
            copy_atom,
            g_in,
            g_send,
            t,
            1,
            num_blocks,
            produce,
            src=src,
            tiler=tiler,
            rows=rows,
            chunk=chunk,
            peer=peer,
        )
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
    head_remote,
    start_recv,
    b,
    tidx,
    s: cutlass.Constexpr,
    tiles_per_slot: cutlass.Constexpr,
    num_tiles: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    consume: cutlass.Constexpr,
    peer=None,
) -> None:
    """One credit-ring RECV slot (CuTe twin of the Triton ``recv_step``): wait the
    peer's data-ready TAIL for slot ``s``, drain its matching staging slot into the
    output, then publish the free-credit HEAD so the sender may reuse the slot.
    Symmetric twin of :func:`_send_slot`; same ``@cute.jit`` rationale.

    The trailing barrier + :func:`nvl_ops.signal_free` hand the drained slot back to the
    sender (the HEAD credit :func:`_send_slot` waits on): the barrier guarantees every
    thread finished reading staging before the slot is declared free."""
    u = unroll
    nb = num_blocks
    p_lo = s * tiles_per_slot
    p_hi = min((s + 1) * tiles_per_slot, num_tiles)
    if tidx == 0:
        nvl_ops.wait(tail_local, start_recv + s + 1)
    cute.arch.barrier()
    t = p_lo + b
    while t + (u - 1) * nb < p_hi:
        _recv_u(
            thr_copy, copy_atom, g_recv, g_out, t, u, num_blocks, consume, peer=peer
        )
        t += u * nb
    while t < p_hi:
        _recv_u(
            thr_copy, copy_atom, g_recv, g_out, t, 1, num_blocks, consume, peer=peer
        )
        t += nb
    cute.arch.barrier()  # all staging reads done before the slot is handed back
    if tidx == 0:
        # free-credit for slots 0..s (monotonic counter): tell the sender it may reuse.
        nvl_ops.signal_free(head_remote, start_recv + s + 1)


class _SendTilesKernel:
    """Send direction: produce hook (HBM -> frag), write frag to staging, signal."""

    def __init__(
        self, *, num_blocks, num_threads, dtype, dbits, hook, put, signal
    ) -> None:
        self.num_blocks = num_blocks
        self.num_threads = num_threads
        self.dtype = dtype
        self.dbits = dbits
        self.hook = hook  # produce hook; called per tile with a Ctx
        self.put = put  # transport op: write produced frag to staging
        self.signal = signal  # transport op: publish data-ready

    @cute.jit
    def __call__(self, data, staging, sig_addr, stream) -> None:
        tiler = cute.make_layout(self.num_threads)
        g_data = cute.zipped_divide(data, tiler)
        g_staging = cute.zipped_divide(staging, tiler)
        num_tiles = cute.size(g_data, mode=[1])
        self.kernel(g_data, g_staging, sig_addr, num_tiles).launch(
            grid=(self.num_blocks, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, g_data, g_staging, sig_addr, num_tiles: cutlass.Int32) -> None:
        tidx = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[0]
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=self.dbits
        )
        tiled_copy = cute.make_tiled_copy_tv(
            copy_atom, cute.make_layout(self.num_threads), cute.make_layout(1)
        )
        thr_copy = tiled_copy.get_slice(tidx)
        addr = sig_addr[0] + bid * 8  # per-block signal slot (slot == block id)

        tile_idx = bid
        while tile_idx < num_tiles:
            # produce hook owns the input leg (HBM -> frag); the primitive writes
            # the returned fragment to staging.
            in_part = thr_copy.partition_S(g_data[(None, tile_idx)])
            stg_part = thr_copy.partition_D(g_staging[(None, tile_idx)])
            frag = self.hook(Ctx(part=in_part, atom=copy_atom))
            self.put(copy_atom, frag, stg_part)
            tile_idx += self.num_blocks

        cute.arch.barrier()  # all staging writes visible before the data-ready signal
        if tidx == 0:
            self.signal(addr, 1)


class _RecvTilesKernel:
    """Recv direction: wait, load staging -> frag, consume hook (frag -> HBM)."""

    def __init__(
        self, *, num_blocks, num_threads, dtype, dbits, hook, get, wait
    ) -> None:
        self.num_blocks = num_blocks
        self.num_threads = num_threads
        self.dtype = dtype
        self.dbits = dbits
        self.hook = hook  # consume hook; called per tile with (Ctx, frag)
        self.get = get  # transport op: read staging into frag
        self.wait = wait  # transport op: wait for data-ready

    @cute.jit
    def __call__(self, data, staging, sig_addr, stream) -> None:
        tiler = cute.make_layout(self.num_threads)
        g_data = cute.zipped_divide(data, tiler)
        g_staging = cute.zipped_divide(staging, tiler)
        num_tiles = cute.size(g_data, mode=[1])
        self.kernel(g_data, g_staging, sig_addr, num_tiles).launch(
            grid=(self.num_blocks, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, g_data, g_staging, sig_addr, num_tiles: cutlass.Int32) -> None:
        tidx = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[0]
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=self.dbits
        )
        tiled_copy = cute.make_tiled_copy_tv(
            copy_atom, cute.make_layout(self.num_threads), cute.make_layout(1)
        )
        thr_copy = tiled_copy.get_slice(tidx)
        addr = sig_addr[0] + bid * 8  # per-block signal slot (slot == block id)

        if tidx == 0:
            self.wait(addr, 1)
        cute.arch.barrier()  # data-ready before any thread reads staging

        tile_idx = bid
        while tile_idx < num_tiles:
            # primitive loads staging -> frag; consume hook owns the output leg
            # (frag -> HBM).
            stg_part = thr_copy.partition_S(g_staging[(None, tile_idx)])
            out_part = thr_copy.partition_D(g_data[(None, tile_idx)])
            frag = self.get(copy_atom, stg_part)
            self.hook(Ctx(part=out_part, atom=copy_atom), frag)
            tile_idx += self.num_blocks


def _launch(data_buf, staging, sig_slice, *, kernel_cls, num_blocks, hook, ops) -> None:
    dtype = data_buf.dtype
    if dtype not in _CUTLASS_DTYPE:
        raise ValueError(
            f"cute minimal backend supports {list(_CUTLASS_DTYPE)}, got {dtype}"
        )
    cdtype, dbits = _CUTLASS_DTYPE[dtype]

    # Signal-slot base address passed as data (read on device); sig_slice is the
    # (remote for send, local for recv) address of this peer's slot region.
    #
    # NOTE (CUDA graph): this per-call allocation is NOT graph-capture-safe — the
    # tensor's lifetime ends with this call, so a captured graph would reference
    # freed memory on replay. A persistent, transport-owned sig-addr buffer is the
    # follow-up fix (lands with the pipelined/graph work).
    sig_addr = torch.tensor(
        [sig_slice.data_ptr()], dtype=torch.int64, device=data_buf.device
    )
    _ensure_cuda_rt_compat()  # lazy: shim cuda-bindings before the cute.compile JIT path
    data_c = from_dlpack(data_buf, assumed_align=16)
    staging_c = from_dlpack(staging, assumed_align=16)
    sig_c = from_dlpack(sig_addr, assumed_align=8)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    key = (
        kernel_cls.__name__,
        num_blocks,
        _NUM_THREADS,
        dtype,
        data_buf.numel(),
        hook,
        *(ops[k] for k in sorted(ops)),
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        logger.info(
            "compiling cute %s: blocks=%s numel=%s hook=%s",
            kernel_cls.__name__,
            num_blocks,
            data_buf.numel(),
            getattr(hook, "__name__", None),
        )
        kernel = kernel_cls(
            num_blocks=num_blocks,
            num_threads=_NUM_THREADS,
            dtype=cdtype,
            dbits=dbits,
            hook=hook,
            **ops,
        )
        compiled = cute.compile(kernel, data_c, staging_c, sig_c, stream)
        _COMPILED[key] = compiled
    compiled(data_c, staging_c, sig_c, stream)


def send_tiles(data_buf, staging, sig_slice, *, num_blocks, hook, put, signal) -> None:
    """Send-direction device transfer (impl behind the ``send`` launcher)."""
    _launch(
        data_buf,
        staging,
        sig_slice,
        kernel_cls=_SendTilesKernel,
        num_blocks=num_blocks,
        hook=hook,
        ops={"put": put, "signal": signal},
    )


def recv_tiles(data_buf, staging, sig_slice, *, num_blocks, hook, get, wait) -> None:
    """Recv-direction device transfer (impl behind the ``recv`` launcher)."""
    _launch(
        data_buf,
        staging,
        sig_slice,
        kernel_cls=_RecvTilesKernel,
        num_blocks=num_blocks,
        hook=hook,
        ops={"get": get, "wait": wait},
    )


# ---------------------------------------------------------------------------
# Pipelined credit-ring send/recv collective (graph-safe), composed from the shared
# _send_slot / _recv_slot substrate above -- the standalone CuTe twin of the Triton
# pipelined send/recv. Single-peer-pair whole-buffer transfer, keeping the slot
# send/drain overlap. uni (send-only / recv-only) and bidir are selected by the
# do_send / do_recv constexprs. Graph-safe via the transport's persistent monotonic
# counters + the symmetric-memory signal pad (NOT the per-call sig tensor the minimal
# send_tiles uses).
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


@cute.kernel
def _sendrecv_kernel(  # noqa: C901
    in_t,
    out_t,
    buf_ptrs,
    sig_ptrs,
    send_ctr,
    recv_ctr,
    dtype: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    dbits: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    send_peer: cutlass.Constexpr,
    recv_peer: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    numel: cutlass.Constexpr,
    cap_elems: cutlass.Constexpr,
    mbp: cutlass.Constexpr,
    num_slots: cutlass.Constexpr,
    tiles_per_slot: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    do_send: cutlass.Constexpr,
    do_recv: cutlass.Constexpr,
    produce: cutlass.Constexpr,
    consume: cutlass.Constexpr,
) -> None:
    tidx = cute.arch.thread_idx()[0]
    b = cute.arch.block_idx()[0]
    # Signal pad: tail region [0, ws*mbp), then head (free-credit) region [ws*mbp, 2*ws*mbp).
    # head_off is the byte offset to the head region; head slots are indexed by RECEIVER rank
    # (symmetric with tail's sender-rank index), so both legs of a channel address one slot.
    head_off = world_size * mbp * 8
    copy_atom = _copy_atom(dtype, vec, dbits)
    tiled_copy = cute.make_tiled_copy_tv(
        copy_atom, cute.make_layout(num_threads), cute.make_layout(vec)
    )
    thr_copy = tiled_copy.get_slice(tidx)
    tiler = cute.make_layout(num_threads * vec)
    g_in = cute.zipped_divide(in_t, tiler)
    g_out = cute.zipped_divide(out_t, tiler)
    num_tiles = cute.size(g_in, mode=[1])
    elem_bytes = dbits // 8
    cap_bytes = cap_elems * elem_bytes

    # SEND: stream the whole buffer into send_peer's staging at MY slot, signal TAIL.
    # const_expr so the branch folds at trace time -- a bare `if` on a constexpr makes
    # CuTe emit a dynamic scf.if whose nested scope hides g_send from the slot loop below.
    if cutlass.const_expr(do_send):
        send_ptr = cute.make_ptr(
            dtype,
            buf_ptrs[send_peer] + local_rank * cap_bytes,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        g_send = cute.zipped_divide(
            cute.make_tensor(send_ptr, cute.make_layout(numel)), tiler
        )
        tail_remote = sig_ptrs[send_peer] + (local_rank * mbp + b) * 8
        # Free-credit HEAD I poll in MY OWN pad (the receiver stores it here): head slot is
        # indexed by the RECEIVER rank == send_peer.
        head_local = sig_ptrs[local_rank] + head_off + (send_peer * mbp + b) * 8
        start_send = send_ctr[send_peer * mbp + b]
    # RECV: wait recv_peer's TAIL, drain recv_peer's slot in MY staging into the output.
    if cutlass.const_expr(do_recv):
        recv_ptr = cute.make_ptr(
            dtype,
            buf_ptrs[local_rank] + recv_peer * cap_bytes,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        g_recv = cute.zipped_divide(
            cute.make_tensor(recv_ptr, cute.make_layout(numel)), tiler
        )
        tail_local = sig_ptrs[local_rank] + (recv_peer * mbp + b) * 8
        # Free-credit HEAD I store into the SENDER's pad (recv_peer) after draining: head slot
        # is indexed by the RECEIVER rank == local_rank.
        head_remote = sig_ptrs[recv_peer] + head_off + (local_rank * mbp + b) * 8
        start_recv = recv_ctr[recv_peer * mbp + b]

    if cutlass.const_expr(do_send and do_recv):
        # bidir: send slot s then drain slot s-1 so the in-flight stores overlap the
        # drain (disjoint regions: my-staging drain vs peer-staging store).
        for s in range(num_slots):
            _send_slot(
                thr_copy,
                copy_atom,
                g_in,
                g_send,
                tail_remote,
                head_local,
                start_send,
                b,
                tidx,
                s,
                tiles_per_slot,
                num_tiles,
                num_blocks,
                num_slots,
                unroll,
                produce,
            )
            if s >= 1:
                _recv_slot(
                    thr_copy,
                    copy_atom,
                    g_recv,
                    g_out,
                    tail_local,
                    head_remote,
                    start_recv,
                    b,
                    tidx,
                    s - 1,
                    tiles_per_slot,
                    num_tiles,
                    num_blocks,
                    unroll,
                    consume,
                )
        _recv_slot(
            thr_copy,
            copy_atom,
            g_recv,
            g_out,
            tail_local,
            head_remote,
            start_recv,
            b,
            tidx,
            num_slots - 1,
            tiles_per_slot,
            num_tiles,
            num_blocks,
            unroll,
            consume,
        )
    elif cutlass.const_expr(do_send):
        for s in range(num_slots):
            _send_slot(
                thr_copy,
                copy_atom,
                g_in,
                g_send,
                tail_remote,
                head_local,
                start_send,
                b,
                tidx,
                s,
                tiles_per_slot,
                num_tiles,
                num_blocks,
                num_slots,
                unroll,
                produce,
            )
    elif cutlass.const_expr(do_recv):
        for s in range(num_slots):
            _recv_slot(
                thr_copy,
                copy_atom,
                g_recv,
                g_out,
                tail_local,
                head_remote,
                start_recv,
                b,
                tidx,
                s,
                tiles_per_slot,
                num_tiles,
                num_blocks,
                unroll,
                consume,
            )

    if cutlass.const_expr(do_send):
        if tidx == 0:
            send_ctr[send_peer * mbp + b] = start_send + num_slots
    if cutlass.const_expr(do_recv):
        if tidx == 0:
            recv_ctr[recv_peer * mbp + b] = start_recv + num_slots


@cute.jit
def _launch_sendrecv(
    in_t,
    out_t,
    buf_ptrs,
    sig_ptrs,
    send_ctr,
    recv_ctr,
    num_blocks: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    dbits: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    send_peer: cutlass.Constexpr,
    recv_peer: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    numel: cutlass.Constexpr,
    cap_elems: cutlass.Constexpr,
    mbp: cutlass.Constexpr,
    num_slots: cutlass.Constexpr,
    tiles_per_slot: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    do_send: cutlass.Constexpr,
    do_recv: cutlass.Constexpr,
    produce: cutlass.Constexpr,
    consume: cutlass.Constexpr,
    stream,
) -> None:
    _sendrecv_kernel(
        in_t,
        out_t,
        buf_ptrs,
        sig_ptrs,
        send_ctr,
        recv_ctr,
        dtype,
        vec,
        dbits,
        num_blocks,
        num_threads,
        local_rank,
        send_peer,
        recv_peer,
        world_size,
        numel,
        cap_elems,
        mbp,
        num_slots,
        tiles_per_slot,
        unroll,
        do_send,
        do_recv,
        produce,
        consume,
    ).launch(grid=(num_blocks, 1, 1), block=(num_threads, 1, 1), stream=stream)


def pipelined_sendrecv(
    transport: NvlTransport,
    send_buf,
    recv_buf,
    send_peer: int,
    recv_peer: int,
    *,
    num_blocks: int,
    mode: str = "bidir",
    produce=copy_produce,
    consume=copy_consume,
) -> None:
    """Graph-safe pipelined CuTe send/recv over the shared NvlTransport.

    ``mode``: ``"bidir"`` (send to send_peer + recv from recv_peer), ``"send"`` (send
    only), or ``"recv"`` (recv only). Whole-buffer transfer (single peer pair), so the
    transport must be sized ``per_peer_bytes >= numel * elem``. Composes the shared
    ``_send_slot``/``_recv_slot`` substrate over the analytic tile/slot sizing above.
    """
    if mode not in ("bidir", "send", "recv"):
        raise ValueError(f"mode must be one of bidir/send/recv, got {mode!r}")
    do_send = mode != "recv"
    do_recv = mode != "send"
    # Validate each leg that actually reaches from_dlpack (send_buf on the send leg, recv_buf
    # on the recv leg) so an invalid buffer fails at this guard, not deep inside from_dlpack.
    for nm, t, used in (
        ("send_buf", send_buf, do_send),
        ("recv_buf", recv_buf, do_recv),
    ):
        if used:
            assert t.is_cuda and t.is_contiguous(), f"{nm} must be CUDA+contiguous"
    # bidir bakes numel/dtype from the active leg into the compiled kernel for BOTH in_t and
    # out_t, so a recv_buf that disagrees with send_buf would OOB-write g_out / mismatch the
    # constexpr dtype. Require the two legs to agree before any device work.
    if do_send and do_recv:
        assert (
            recv_buf.numel() == send_buf.numel() and recv_buf.dtype == send_buf.dtype
        ), "bidir requires recv_buf.numel()/dtype == send_buf.numel()/dtype"
    buf = send_buf if do_send else recv_buf  # dtype/numel source for the active leg
    dtype = buf.dtype
    if dtype not in _CUTLASS_DTYPE:
        raise ValueError(f"cute sendrecv supports {list(_CUTLASS_DTYPE)}, got {dtype}")
    cdtype, dbits = _CUTLASS_DTYPE[dtype]
    numel = buf.numel()
    elem = buf.element_size()
    num_threads, vec = _pick_tile(numel, dbits, num_blocks)
    num_tiles = numel // (num_threads * vec)
    num_slots, tiles_per_slot = _pick_slots(num_tiles, numel * elem)
    check_transfer(transport, numel, dtype, num_blocks)
    cap_elems = transport.per_peer_bytes // elem
    mbp = transport.max_blocks_per_peer
    local_rank = transport.local_rank
    world_size = (
        transport.world_size
    )  # sizes the signal-pad head (free-credit) region base
    # Register-blocking unroll for the NVLink store loop (the per-SM injection lever):
    # default 8 once the buffer is large enough for the unrolled body to engage.
    unroll = 8 if numel * elem >= 64 * 1024 else 1

    table = transport.endpoints_device()
    send_ctr, recv_ctr = transport.step_state()
    _ensure_cuda_rt_compat()  # lazy: shim cuda-bindings before the cute.compile JIT path
    in_t = from_dlpack(send_buf if do_send else recv_buf, assumed_align=16)
    # out_t is only read on device under do_recv (g_out is consumed inside do_recv-guarded
    # slot loops); on send-only recv_buf is unvalidated and may be None, so source it from
    # send_buf to keep from_dlpack and the unconditional g_out=zipped_divide(out_t) valid.
    out_t = from_dlpack(recv_buf if do_recv else send_buf, assumed_align=16)
    buf_c = from_dlpack(table.buffer_ptrs, assumed_align=8)
    sig_c = from_dlpack(table.signal_pad_ptrs, assumed_align=8)
    send_c = from_dlpack(send_ctr, assumed_align=8)
    recv_c = from_dlpack(recv_ctr, assumed_align=8)
    stream = _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    key = (
        "sendrecv",
        num_blocks,
        num_threads,
        vec,
        dtype,
        numel,
        cap_elems,
        mbp,
        num_slots,
        tiles_per_slot,
        unroll,
        local_rank,
        send_peer,
        recv_peer,
        world_size,
        do_send,
        do_recv,
        produce,
        consume,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            _launch_sendrecv,
            in_t,
            out_t,
            buf_c,
            sig_c,
            send_c,
            recv_c,
            num_blocks,
            num_threads,
            vec,
            cdtype,
            dbits,
            local_rank,
            send_peer,
            recv_peer,
            world_size,
            numel,
            cap_elems,
            mbp,
            num_slots,
            tiles_per_slot,
            unroll,
            do_send,
            do_recv,
            produce,
            consume,
            stream,
        )
        _COMPILED[key] = compiled
    compiled(in_t, out_t, buf_c, sig_c, send_c, recv_c, stream)
