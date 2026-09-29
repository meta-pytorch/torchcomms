// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// PRIVATE header. Declares the raw `__global__` template behind the compressed
// sendrecv-tile collective. It is deliberately NOT exported by the
// `:sendrecv_tile_compressed` facade: callers get
// `pick_sendrecv_tile_compressed_kernel()` from the public
// `SendRecvTileCompressed.cuh` instead.
//
// Why the split. Exposing the template publicly put CUDA syntax
// (`__global__`, `__launch_bounds__`, `__grid_constant__`) in a header that
// host-only consumers include, which forced a duplicated `#ifdef __CUDACC__`
// host mirror of the same declaration -- two declarations that had to be kept
// character-identical by hand. It also made the supported launch
// configurations unstateable: the template accepts ANY `(NumWarps,
// MinBlocksPerSM)`, but only seven pairs are explicitly instantiated, so
// `<AnsCompressor, 8, 4>` compiled at the call site and failed at DEVICE LINK
// with an undefined symbol -- a diagnostic that points at the linker rather
// than at the caller's mistake. The picker makes the supported set
// authoritative in one place and rejects everything else host-side, before any
// launch.

#pragma once

#include "comms/prims/collectives/SendRecvTile.cuh"
#include "comms/prims/collectives/SendRecvTileCompressed.cuh"

namespace comms::prims {

/**
 * Compressed point-to-point tile send/recv. Same shape as
 * `sendrecv_tile_kernel`, but IBGDA sub-chunks may be ANS-compressed.
 *
 * Templated on a `Compressor` tag (e.g. `AnsCompressor`) and
 * `NumWarps = blockDim.x / 32`. The device TU maps `(Compressor, NumWarps)` to
 * a concrete CopyOp via `compressor_copyop`, so the per-block static
 * `__shared__` footprint is just that one CopyOp's scratch.
 *
 * Explicit instantiations live in `SendRecvTileCompressed.cu`, and the set of
 * them is what `pick_sendrecv_tile_compressed_kernel()` maps. Adding a pair
 * means adding it in BOTH places -- the picker is the contract, the
 * instantiation is the implementation, and a pair present in only one of them
 * is exactly the failure this split exists to prevent.
 */
template <typename Compressor, int NumWarps, int MinBlocksPerSM = 2>
__global__
    __launch_bounds__(NumWarps * 32, MinBlocksPerSM) void sendrecv_tile_compressed_kernel(
        const __grid_constant__ SendRecvTileArgs args,
        AbortDevice abortDevice = AbortDevice());

} // namespace comms::prims
