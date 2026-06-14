// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Plain (no-compression) flavour of a tile-based pipelined point-to-point
// send/recv with NVLink and IB support. Provided by the `:sendrecv_tile`
// BUCK target.
//
// SendRecvTile is the point-to-point specialisation of AllToAllvTile. A
// single launch can do any of:
//   - send only:   send `send_count` bytes from `send_data` to `send_peer`
//   - recv only:   recv `recv_count` bytes from `recv_peer` into `recv_data`
//   - send + recv: do BOTH simultaneously, to/from (possibly different)
//                  peers — half the grid's blocks drive the send, the
//                  other half drive the recv.
// selected by the `is_send` / `is_recv` flags. The per-tile send/recv
// mechanism (NVL-vs-IB dispatch, sub-chunk pipelining, signalling) is
// self-contained (see `SendRecvTileCommon.cuh`).
//
// For ANS-compressed IBGDA chunks, see the sibling
// `SendRecvTileCompressed.cuh` (provided by `:sendrecv_tile_compressed`).
//
// Uses MultiPeerDeviceHandle for NVLink/IB transport dispatch. For each
// direction, the two endpoints MUST agree on the per-direction active
// block count so the per-block tiling lines up: a one-directional launch
// uses all `gridDim.x` blocks; a bidirectional (send+recv) launch gives
// each direction `gridDim.x / 2` blocks, so the matching peer's launch
// must be bidirectional too (or a one-way launch of `gridDim.x / 2`).

#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"

namespace comms::prims {

/**
 * SendRecvTileArgs — Kernel arguments for the tile-based point-to-point
 * send/recv collective.
 *
 * The `is_send` / `is_recv` flags select the mode:
 *   - is_send && !is_recv : send `send_count` bytes from `send_data` to
 *                           `send_peer` (all blocks).
 *   - !is_send && is_recv : recv `recv_count` bytes from `recv_peer` into
 *                           `recv_data` (all blocks).
 *   - is_send && is_recv  : do both at once — the first half of the grid's
 *                           blocks send to `send_peer`, the second half
 *                           recv from `recv_peer`. `send_peer` and
 *                           `recv_peer` may differ.
 * A direction with a zero count is skipped. The matching peer(s) must
 * issue mirrored launches with the same per-direction active block count
 * (see header comment).
 *
 * Shared by both the plain and ANS-compressed kernel symbols; the
 * compress-vs-plain choice is encoded in the kernel symbol selected by
 * the caller, NOT a runtime flag on this struct.
 */
struct SendRecvTileArgs {
  MultiPeerDeviceHandle handle;

  // Direction enables. When both are true the grid is split in half.
  bool is_send;
  bool is_recv;

  // Peer ranks for each direction (may be equal for bidirectional pairs).
  int send_peer;
  int recv_peer;

  // Send-side source buffer + byte count (consulted only when is_send).
  char* send_data;
  std::size_t send_count;

  // Recv-side destination buffer + byte count (consulted only when
  // is_recv).
  char* recv_data;
  std::size_t recv_count;

  // Sub-chunk signaling hint (bytes). 0 = one signal per slot fill.
  // Applies to both directions.
  std::size_t max_signal_bytes;

  // Per-block 16-byte-aligned auxiliary buffer used by ANS-compressed
  // sends when the per-chunk source pointer is not 16-byte aligned.
  // Defaults to nullptr; ignored by the plain (Memcpy) kernel and only
  // consulted by the ANS-compressed send dispatcher when the per-chunk
  // `src` pointer is misaligned. Sized `gridDim.x * kAnsMaxUncompBytes`
  // (every block gets a slice keyed on its global `blockIdx.x`).
  char* aligned_aux_buf;
};

#ifdef __CUDACC__
/**
 * Plain (Memcpy) point-to-point tile send/recv. NVL and IB peers are
 * both supported (dispatched per-peer by the transport handle). For a
 * one-directional launch all `gridDim.x` blocks cooperate; for a
 * bidirectional (send+recv) launch the grid is split in half.
 */
__global__ __launch_bounds__(512, 2) void sendrecv_tile_kernel(
    const __grid_constant__ SendRecvTileArgs args,
    Timeout timeout = Timeout());
#else
void sendrecv_tile_kernel(SendRecvTileArgs args, Timeout timeout);
#endif

} // namespace comms::prims
