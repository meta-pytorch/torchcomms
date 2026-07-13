# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

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

from __future__ import annotations

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

from .ctx import Ctx

_cuda_driver: Any = import_module("cuda.bindings.driver")
_cuda_rt: Any = import_module("cuda.bindings.runtime")

# Some cuda-bindings versions lack symbols the cutlass DSL JIT executor expects
# (cudaLibrary_t / cudaLibraryUnload). Patch them in, mirroring the existing CuTe
# sendrecv runtime, so kernel compilation can load the cuda library.
if not hasattr(_cuda_rt, "cudaLibrary_t"):

    class _cudaLibrary_t:
        __slots__ = ("value",)

        def __init__(self, value: int = 0) -> None:
            self.value = value

    _cuda_rt.cudaLibrary_t = _cudaLibrary_t
    _cuda_rt.cudaLibraryUnload = lambda lib: (_cuda_rt.cudaError_t(0),)

_NUM_THREADS: int = 128

# Minimal dtype support: (cutlass dtype, bits-per-element).
_CUTLASS_DTYPE: dict[torch.dtype, tuple[Any, int]] = {
    torch.float32: (cutlass.Float32, 32),
    torch.bfloat16: (cutlass.BFloat16, 16),
}

_COMPILED: dict[tuple[Any, ...], object] = {}


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
