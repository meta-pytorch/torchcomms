// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"

namespace comms::pipes {

/**
 * MultipeerIbgdaDeviceTransport - Multi-peer RDMA transport handle for GPU
 *
 * Device-side wrapper that provides access to per-peer RDMA transport handles.
 * This struct is passed to CUDA kernels and contains handles for communicating
 * with all peers in the communicator.
 *
 * USAGE:
 * ======
 *
 *   __global__ void allToAllKernel(
 *       MultipeerIbgdaDeviceTransport transport,
 *       IbgdaLocalBuffer localBufs[],
 *       IbgdaRemoteBuffer remoteBufs[],
 *       size_t nbytes) {
 *
 *     int myRank = transport.myRank;
 *     int nRanks = transport.nRanks;
 *
 *     // Send to all peers
 *     for (int peer = 0; peer < nRanks; peer++) {
 *       if (peer == myRank) continue;
 *
 *       auto& p2p = transport.getPeer(peer);
 *       auto ticket = p2p.put_signal(
 *           localBufs[peer], remoteBufs[myRank], nbytes, 1);
 *       p2p.wait(ticket);
 *     }
 *
 *     // Wait for all peers to send to us
 *     for (int peer = 0; peer < nRanks; peer++) {
 *       if (peer == myRank) continue;
 *       transport.getPeer(peer).wait_signal(1);
 *     }
 *   }
 *
 * MEMORY LAYOUT:
 * ==============
 *
 * The peerTransports span contains (nRanks - 1) elements, indexed by
 * logical peer index. The mapping from global rank to peer index
 * excludes self:
 *
 *   For rank 2 with nRanks=4:
 *     peerTransports[0] -> rank 0
 *     peerTransports[1] -> rank 1
 *     peerTransports[2] -> rank 3  (skips self)
 *
 * Use getPeer(globalRank) to handle this mapping automatically.
 */
struct MultipeerIbgdaDeviceTransport {
  int myRank{-1};
  int nRanks{0};
  DeviceSpan<P2pIbgdaTransportDevice> peerTransports;

  __host__ __device__ MultipeerIbgdaDeviceTransport() = default;

  __host__ __device__ MultipeerIbgdaDeviceTransport(
      int rank,
      int numRanks,
      DeviceSpan<P2pIbgdaTransportDevice> transports)
      : myRank(rank), nRanks(numRanks), peerTransports(transports) {}

  /**
   * getPeer - Get transport handle for a specific peer rank
   *
   * Returns a reference to the P2pIbgdaTransportDevice for the given
   * global peer rank. Handles the rank-to-peer-index mapping internally.
   *
   * @param peerRank Global rank of the peer (must be != myRank and < nRanks)
   * @return Reference to the peer's transport handle
   */
  __device__ __forceinline__ P2pIbgdaTransportDevice& getPeer(int peerRank) {
    // Convert global rank to peer index (skip self)
    int peerIndex = (peerRank < myRank) ? peerRank : (peerRank - 1);
    return peerTransports[peerIndex];
  }

  /**
   * getPeer - Const version for read-only access
   */
  __device__ __forceinline__ const P2pIbgdaTransportDevice& getPeer(
      int peerRank) const {
    int peerIndex = (peerRank < myRank) ? peerRank : (peerRank - 1);
    return peerTransports[peerIndex];
  }

  /**
   * numPeers - Get number of peer connections
   *
   * @return Number of peers (nRanks - 1)
   */
  __host__ __device__ __forceinline__ int numPeers() const {
    return nRanks - 1;
  }

  /**
   * getPeerByIndex - Get transport by peer index (not rank)
   *
   * Direct access to the transport array by index. Use when iterating
   * over all peers without needing rank translation.
   *
   * @param index Index into peerTransports (0 to numPeers()-1)
   * @return Reference to the peer's transport handle
   */
  __device__ __forceinline__ P2pIbgdaTransportDevice& getPeerByIndex(
      int index) {
    return peerTransports[index];
  }

  /**
   * getPeerByIndex - Const version for read-only access
   */
  __device__ __forceinline__ const P2pIbgdaTransportDevice& getPeerByIndex(
      int index) const {
    return peerTransports[index];
  }

  /**
   * peerIndexToRank - Convert peer index back to global rank
   *
   * @param index Index into peerTransports (0 to numPeers()-1)
   * @return Global rank of the peer at this index
   */
  __host__ __device__ __forceinline__ int peerIndexToRank(int index) const {
    // Reverse the mapping: if index < myRank, rank = index; else rank = index+1
    return (index < myRank) ? index : (index + 1);
  }
};

} // namespace comms::pipes
