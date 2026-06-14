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
    handle.get_nvl(peer).send(
        group, src, nbytes, active_blocks, max_signal_bytes, timeout);
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
      // buffer (sized `kAnsMaxUncompBytes` per block). Only consumed
      // inside `AnsCompress::send` when `src` is misaligned; null is
      // fine for callers that always pass aligned `src`.
      char* perBlockAux = aligned_aux_buf == nullptr
          ? nullptr
          : aligned_aux_buf + blockIdx.x * kAnsMaxUncompBytes;
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
    handle.get_nvl(peer).recv(
        group, dst, nbytes, active_blocks, max_signal_bytes, timeout);
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
