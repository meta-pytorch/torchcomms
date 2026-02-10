// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/DeviceSpan.cuh"

// When compiling with CUDA, include full definition for device methods
// When compiling with host-only compiler, forward declaration is sufficient
#ifdef __CUDACC__
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#else
namespace comms::pipes {
// Forward declaration - full definition in P2pIbgdaTransportDevice.cuh
class P2pIbgdaTransportDevice;
} // namespace comms::pipes
#endif

namespace comms::pipes {

/**
 * MultipeerIbgdaDeviceTransport - Multi-peer RDMA transport handle for GPU
 *
 * Device-side wrapper that provides access to per-rank RDMA transport handles.
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
 *     constexpr int kSignalId = 0;
 *     constexpr uint64_t kSignalVal = 1;
 *
 *     // Send to all peers
 *     for (int rank = 0; rank < nRanks; rank++) {
 *       if (rank == myRank) continue;
 *
 *       auto& p2p = transport.get(rank);
 *       auto work = p2p.put_signal(
 *           localBufs[myRank], remoteBufs[rank], nbytes,
 *           kSignalId, kSignalVal);
 *       p2p.wait_local(work);
 *     }
 *
 *     // Wait for all peers to send to us
 *     for (int rank = 0; rank < nRanks; rank++) {
 *       if (rank == myRank) continue;
 *       transport.get(rank).wait_signal(kSignalId, IbgdaCmpOp::GE, kSignalVal);
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
 * Use get(globalRank) to handle this mapping automatically.
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
   * numPeers - Get number of peer connections
   *
   * @return Number of peers (nRanks - 1)
   */
  __host__ __device__ __forceinline__ int numPeers() const {
    return nRanks - 1;
  }

  /**
   * indexToRank - Convert index back to global rank
   *
   * @param index Index into peerTransports (0 to numPeers()-1)
   * @return Global rank at this index
   */
  __host__ __device__ __forceinline__ int indexToRank(int index) const {
    // Reverse the mapping: if index < myRank, rank = index; else rank = index+1
    return (index < myRank) ? index : (index + 1);
  }

// Device-only methods require full P2pIbgdaTransportDevice definition
// and are only available when compiling with nvcc
#ifdef __CUDACC__
  /**
   * get - Get transport handle for a specific rank
   *
   * Returns a reference to the P2pIbgdaTransportDevice for the given
   * global rank. Handles the rank-to-index mapping internally.
   *
   * @param rank Global rank (must be != myRank and < nRanks)
   * @return Reference to the transport handle for that rank
   */
  __device__ __forceinline__ P2pIbgdaTransportDevice& get(int rank) {
    // Convert global rank to index (skip self)
    int index = (rank < myRank) ? rank : (rank - 1);
    return peerTransports[index];
  }

  /**
   * get - Const version for read-only access
   */
  __device__ __forceinline__ const P2pIbgdaTransportDevice& get(
      int rank) const {
    int index = (rank < myRank) ? rank : (rank - 1);
    return peerTransports[index];
  }

  /**
   * getByIndex - Get transport by index (not rank)
   *
   * Direct access to the transport array by index. Use when iterating
   * over all peers without needing rank translation.
   *
   * @param index Index into peerTransports (0 to numPeers()-1)
   * @return Reference to the transport handle
   */
  __device__ __forceinline__ P2pIbgdaTransportDevice& getByIndex(int index) {
    return peerTransports[index];
  }

  /**
   * getByIndex - Const version for read-only access
   */
  __device__ __forceinline__ const P2pIbgdaTransportDevice& getByIndex(
      int index) const {
    return peerTransports[index];
  }
#endif // __CUDACC__
};

} // namespace comms::pipes
