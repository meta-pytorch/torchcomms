// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Private (in-tree only) shared device helpers for the sendrecv-tile
// kernels. Included by both SendRecvTile.cu (plain Memcpy path) and
// SendRecvTileCompressed.cu (ANS-compressed path); each TU instantiates
// the templates below with the CopyOp policy that matches its build
// flavour.
//
// SendRecvTile is self-contained: it depends only on the core CopyOp /
// transport primitives (the `:copy_op` / `:copy_op_compress` layer), NOT
// on the AllToAllvTile collective. The compressed-specific dispatchers
// (`ibgda_send_compressed`, `ibgda_recv_compressed`, `kAnsMaxUncompBytes`,
// `AnsCopyOp`) live in the sibling SendRecvTileCompressed.cuh, which is
// conditionally included below ONLY when PIPES_ENABLE_ANS_COMPRESSION is
// defined — so the plain TU never sees any nvcompdx symbols.
//
// Do NOT include this header from public consumers — the only stable
// contracts are `SendRecvTile.cuh` (plain kernel + `SendRecvTileArgs`)
// and `SendRecvTileCompressed.cuh` (compressed kernels + stats).

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/collectives/SendRecvTile.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/core/TiledBuffer.cuh"

#ifdef PIPES_ENABLE_ANS_COMPRESSION
// Pulls in `kAnsMaxUncompBytes`, `ibgda_send_compressed`,
// `ibgda_recv_compressed`, `AnsCopyOp`. Must be included BEFORE the
// `send_to_peer` / `recv_from_peer` template definitions below so that
// non-dependent name lookup of the dispatchers (inside the
// `if constexpr (CopyOp::kVariableSize)` branch) succeeds at template
// definition time.
#include "comms/prims/collectives/SendRecvTileCompressed.cuh"
#endif

namespace comms::prims {

namespace {

// Send one tile to a single peer. CopyOp tags the build flavour:
//   - Memcpy (kVariableSize == false)        -> plain transport-level send
//   - AnsCopyOp (kVariableSize == true)      -> compressed dispatcher
// NVL transport is bandwidth-bound on intra-node links, so it always
// uses the plain `send()` regardless of CopyOp; only the IB path honours
// the compression policy. The `if constexpr` branch on
// `CopyOp::kVariableSize` resolves the discriminator at compile time and
// drops dead code per TU; the branch is preprocessor-removed when
// PIPES_ENABLE_ANS_COMPRESSION is undefined, so the plain TU never
// references `ibgda_send_compressed`.
template <typename CopyOp = Memcpy>
__device__ __forceinline__ void send_to_peer(
    MultiPeerDeviceHandle& handle,
    int peer,
    ThreadGroup& group,
    void* src,
    std::size_t nbytes,
    std::size_t peer_total_bytes,
    int active_blocks,
    std::size_t max_signal_bytes,
    char* aligned_aux_buf,
    const Timeout& timeout) {
  if (handle.get_type(peer) == TransportType::P2P_NVL) {
    handle.get_nvl(peer).send(group, src, nbytes, max_signal_bytes, timeout);
    return;
  }
#ifdef PIPES_ENABLE_ANS_COMPRESSION
  if constexpr (CopyOp::kVariableSize) {
    // Per-peer activation threshold: small tiles bypass the compressed
    // CopyOp and fall back to the plain Memcpy IB path. The discriminator
    // is the PER-PEER total byte count (`peer_total_bytes`), NOT this
    // block's per-tile `nbytes`, so sender + receiver + every block agree
    // on the same compress-vs-bypass decision per peer.
    if (peer_total_bytes >= CopyOp::kActivationThreshold) {
      // Per-block slice of the caller-provided 16-byte-aligned aux
      // buffer (sized `CopyOp::kMaxUncompBytes` per block, derived from the
      // compiled compressor rather than a hard-coded ANS constant). Only
      // consumed inside the compressor's `send` when `src` is misaligned;
      // null is fine for callers that always pass aligned `src`.
      char* perBlockAux = aligned_aux_buf == nullptr
          ? nullptr
          : aligned_aux_buf + blockIdx.x * CopyOp::kMaxUncompBytes;
      ibgda_send_compressed<CopyOp>(
          handle.get_ibgda(peer),
          group,
          src,
          nbytes,
          active_blocks,
          max_signal_bytes,
          timeout,
          perBlockAux);
      return;
    }
    // Fall through to the plain IB Memcpy path below for sub-threshold
    // peers.
  }
#endif
  handle.get_ibgda(peer).send(
      group, src, nbytes, active_blocks, max_signal_bytes, timeout);
}

template <typename CopyOp = Memcpy>
__device__ __forceinline__ void recv_from_peer(
    MultiPeerDeviceHandle& handle,
    int peer,
    ThreadGroup& group,
    void* dst,
    std::size_t nbytes,
    std::size_t peer_total_bytes,
    int active_blocks,
    std::size_t max_signal_bytes,
    const Timeout& timeout) {
  if (handle.get_type(peer) == TransportType::P2P_NVL) {
    handle.get_nvl(peer).recv(group, dst, nbytes, max_signal_bytes, timeout);
    return;
  }
#ifdef PIPES_ENABLE_ANS_COMPRESSION
  if constexpr (CopyOp::kVariableSize) {
    // Mirror the sender-side `kActivationThreshold` gate. Sender and
    // receiver MUST agree on this discriminator (both see the same
    // `peer_total_bytes`), or the receiver would try to ANS-decompress
    // raw payload.
    if (peer_total_bytes >= CopyOp::kActivationThreshold) {
      ibgda_recv_compressed<CopyOp>(
          handle.get_ibgda(peer),
          group,
          dst,
          nbytes,
          active_blocks,
          max_signal_bytes,
          timeout);
      return;
    }
    // Fall through to the plain IB Memcpy recv path below.
  }
#endif
  handle.get_ibgda(peer).recv(
      group, dst, nbytes, active_blocks, max_signal_bytes, timeout);
}

// Per-block plain/compressed selector for the mixed-mode dispatch. When
// `fraction` is in (0, 1), exactly `k = lround(fraction * activeBlocks)`
// of the `activeBlocks` blocks are chosen to use the plain `Memcpy`
// CopyOp, evenly interleaved across the block-id range via a
// Bresenham-style test. The decision depends only on (`fraction`,
// `blockId`, `activeBlocks`) — all identical on both peers for a given
// per-direction subgroup id — so a plain-sending block is always matched
// by a plain-receiving block on the peer. `fraction <= 0` -> never plain
// (all compiled CopyOp); `fraction >= 1` -> always plain.
__device__ __forceinline__ bool
sendrecv_use_plain_block(float fraction, int blockId, int activeBlocks) {
  if (fraction <= 0.0f || activeBlocks <= 0) {
    return false;
  }
  if (fraction >= 1.0f) {
    return true;
  }
  long k = lroundf(fraction * static_cast<float>(activeBlocks));
  if (k <= 0) {
    return false;
  }
  if (k >= activeBlocks) {
    return true;
  }
  // floor((blockId+1)*k / activeBlocks) > floor(blockId*k / activeBlocks)
  // selects exactly k evenly spread block ids.
  const long lo = (static_cast<long>(blockId) * k) / activeBlocks;
  const long hi = (static_cast<long>(blockId + 1) * k) / activeBlocks;
  return hi > lo;
}

// Drive the send direction with `group`'s blocks: tile `send_data` across
// the group and have each block ship its slice to `send_peer`.
template <typename CopyOp = Memcpy>
__device__ __forceinline__ void sendrecv_do_send(
    MultiPeerDeviceHandle& handle,
    const SendRecvTileArgs& args,
    ThreadGroup& group,
    const Timeout& timeout) {
  const int activeBlocks = group.total_groups;
  const int blockId = group.group_id;
  TiledBuffer<char> tiles(args.send_data, args.send_count, activeBlocks);
  // Mixed-mode: a fraction of blocks fall back to plain Memcpy so their
  // light-weight sends fill NIC bandwidth the compute-heavy CopyOp
  // blocks leave idle. Byte-identical to the all-CopyOp path when
  // plain_block_fraction == 0 (the common case).
  if (sendrecv_use_plain_block(
          args.plain_block_fraction, blockId, activeBlocks)) {
    send_to_peer<Memcpy>(
        handle,
        args.send_peer,
        group,
        tiles.tile_data(blockId),
        tiles.tile_bytes(blockId),
        args.send_count,
        activeBlocks,
        args.max_signal_bytes,
        args.aligned_aux_buf,
        timeout);
    return;
  }
  send_to_peer<CopyOp>(
      handle,
      args.send_peer,
      group,
      tiles.tile_data(blockId),
      tiles.tile_bytes(blockId),
      args.send_count,
      activeBlocks,
      args.max_signal_bytes,
      args.aligned_aux_buf,
      timeout);
}

// Drive the recv direction with `group`'s blocks: tile `recv_data` across
// the group and have each block pull its slice from `recv_peer`.
template <typename CopyOp = Memcpy>
__device__ __forceinline__ void sendrecv_do_recv(
    MultiPeerDeviceHandle& handle,
    const SendRecvTileArgs& args,
    ThreadGroup& group,
    const Timeout& timeout) {
  const int activeBlocks = group.total_groups;
  const int blockId = group.group_id;
  TiledBuffer<char> tiles(args.recv_data, args.recv_count, activeBlocks);
  // Mirror the send-side mixed-mode decision so a plain-sent tile is
  // plain-received (same fraction/blockId/activeBlocks on both peers).
  if (sendrecv_use_plain_block(
          args.plain_block_fraction, blockId, activeBlocks)) {
    recv_from_peer<Memcpy>(
        handle,
        args.recv_peer,
        group,
        tiles.tile_data(blockId),
        tiles.tile_bytes(blockId),
        args.recv_count,
        activeBlocks,
        args.max_signal_bytes,
        timeout);
    return;
  }
  recv_from_peer<CopyOp>(
      handle,
      args.recv_peer,
      group,
      tiles.tile_data(blockId),
      tiles.tile_bytes(blockId),
      args.recv_count,
      activeBlocks,
      args.max_signal_bytes,
      timeout);
}

// Point-to-point driver. Modes (selected by is_send / is_recv):
//   - send only:   all blocks send to send_peer.
//   - recv only:   all blocks recv from recv_peer.
//   - send + recv: split the grid — role 0 (first half) sends to
//                  send_peer, role 1 (second half) recvs from recv_peer
//                  (each direction gets gridDim.x / 2 blocks). The two
//                  directions use independent transport staging/signal
//                  state, so a bidirectional pair to the same peer is
//                  safe.
// `send_to_peer` / `recv_from_peer` dispatch to NVL or IB (and, for
// variable-size CopyOps, to the ANS-compressed path).
template <typename CopyOp = Memcpy>
__device__ __forceinline__ void sendrecv_tile_impl(
    const SendRecvTileArgs& args,
    Timeout& timeout) {
  auto handle = args.handle;
  const bool doSend = args.is_send && args.send_count > 0;
  const bool doRecv = args.is_recv && args.recv_count > 0;
  if (!doSend && !doRecv) {
    return;
  }

  auto group = make_block_group();

  if (doSend && doRecv) {
    // Bidirectional mode splits the grid in half via partition(2). An odd
    // gridDim.x would hand the two directions unequal block counts, breaking
    // the matching per-direction active-block invariant the peers rely on
    // (peer A's role-0 sender count must equal peer B's role-1 receiver
    // count). Require an even grid and fail loudly rather than hang.
    if ((gridDim.x % 2u) != 0u) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf(
            "[PIPES] FATAL: SendRecvTile bidirectional mode requires an even "
            "gridDim.x (got %u)\n",
            gridDim.x);
      }
      PIPES_DEVICE_TRAP();
    }
    auto [role, sub] = group.partition(2);
    if (role == 0) {
      sendrecv_do_send<CopyOp>(handle, args, sub, timeout);
    } else {
      sendrecv_do_recv<CopyOp>(handle, args, sub, timeout);
    }
    return;
  }

  if (doSend) {
    sendrecv_do_send<CopyOp>(handle, args, group, timeout);
  } else {
    sendrecv_do_recv<CopyOp>(handle, args, group, timeout);
  }
}

} // namespace

} // namespace comms::prims
