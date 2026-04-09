// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/IbgdaBuffer.h"

namespace comms::pipes {

// Fixed: 2 signal slots per peer per channel (completion + back-pressure).
// Shared across host setup (MultiPeerIbgdaTransportSetup.cc) and device
// protocol (P2pIbgdaTransportDevice.cuh) so signal offset computations agree.
inline constexpr int kP2pSignalCount = 2;

/**
 * P2pIbgdaTransportState - Per-peer device state for IBGDA collectives
 *
 * Contains pre-baked buffer descriptors and configuration for one peer.
 * The kernel indexes into a DeviceSpan<P2pIbgdaTransportState> by peer rank
 * to get all the information needed for pipelined RDMA send/recv.
 *
 * Symmetric with NVLink's P2pNvlTransportDevice::LocalState/RemoteState,
 * but uses IBGDA buffer descriptors (lkey/rkey) instead of raw pointers.
 *
 * Offsets are pre-baked at host setup time (following the NVLink pattern
 * in MultiPeerNvlTransport::getP2pTransportDevice()), so the kernel only
 * applies pipeline-slot and chunk offsets within the staging buffer.
 */
struct P2pIbgdaTransportState {
  // Staging data buffer descriptors (allocated once, registered with NIC)
  // Send and recv use SEPARATE staging regions to avoid data races:
  //   - Sender writes own data to localStagingBuf, then RDMA puts to remote
  //   - RDMA from remote peer writes incoming data to recvStagingBuf
  //   - Receiver reads from recvStagingBuf
  IbgdaLocalBuffer localStagingBuf; // send staging (sender writes here)
  IbgdaRemoteBuffer
      remoteStagingBuf; // remote peer's recv staging (RDMA target)
  IbgdaLocalBuffer recvStagingBuf; // recv staging (receiver reads here)

  // Signal buffer descriptors (allocated once, registered with NIC)
  IbgdaLocalBuffer localSignalBuf;
  IbgdaRemoteBuffer remoteSignalBuf;

  // Signal slot IDs for the sender and receiver directions.
  // Each rank's signal buffer has nRanks * 2 slots (completion +
  // back-pressure). localSignalId:  slot in MY signal buffer where this peer
  // writes
  //                 (used by receiver to wait_signal)
  // remoteSignalId: slot in PEER's signal buffer where I write
  //                 (used by sender to signal_remote)
  // Layout per peer: [+0] = completion signal, [+1] = back-pressure signal
  int localSignalId{};
  int remoteSignalId{};

  // Per-pipeline-slot staging buffer size in bytes
  size_t dataBufferSize{};

  // Chunk size for sub-pipeline RDMA puts (bytes).
  //
  // CURRENTLY UNUSED by the IBGDA kernel. Each pipeline step
  // issues a single RDMA put_signal for the full dataBufferSize. Unlike the
  // NVLink path — where chunkSize subdivides each step into independent
  // chunks with per-chunk ChunkState signaling for warp-level parallelism —
  // the IBGDA path uses a single QP per peer, so issuing multiple smaller
  // WQEs per step would just serialize on the same QP and add WQE-posting
  // overhead without any parallelism benefit.
  //
  // This field is intentionally kept for forward compatibility: with
  // multi-QP support, chunks can be striped across QPs to achieve
  // sub-step parallelism analogous to the NVLink chunk distribution.
  size_t chunkSize{};

  // Number of pipeline slots
  int pipelineDepth{};

  // Multi-channel fields — subdivide per-peer staging across channels.
  // When maxChannelsPerPeer == 1 (default), single-channel mode:
  //   channelDataBufferSize == dataBufferSize
  //   channelStride == pipelineDepth * dataBufferSize
  int maxChannelsPerPeer{1};
  size_t channelDataBufferSize{0}; // dataBufferSize / maxChannelsPerPeer
  size_t channelStride{0}; // pipelineDepth * channelDataBufferSize
};

} // namespace comms::pipes
