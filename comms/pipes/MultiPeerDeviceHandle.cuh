// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#include "comms/pipes/DeviceSpan.cuh"

// In CUDA compilation, include full Transport definition for device accessors.
// In host-only compilation, a forward declaration suffices because DeviceSpan
// only stores a pointer (T*) — it doesn't need sizeof(T).
#ifdef __CUDACC__
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"
#else
namespace comms::pipes {
struct Transport;
enum class TransportType : uint8_t;
class P2pNvlTransportDevice;
class P2pIbgdaTransportDevice;
} // namespace comms::pipes
#endif

namespace comms::pipes {

/**
 * MultiPeerDeviceHandle - Unified device-side handle for mixed-transport
 * communication.
 *
 * Lightweight struct passed to CUDA kernels. Contains a single DeviceSpan
 * of Transport objects (one per rank) plus peer counts. The Transport union
 * already carries the type discriminant, so no separate type array is needed.
 *
 * Layout: transports[0..nRanks-1] where transports[myRank].type == SELF,
 *         NVL peers sorted first, followed by IBGDA-only peers.
 *
 * USAGE:
 *   __global__ void kernel(MultiPeerDeviceHandle handle, ...) {
 *     for (int rank = 0; rank < handle.nRanks; ++rank) {
 *       switch (handle.get_type(rank)) {
 *         case TransportType::SELF: ... break;
 *         case TransportType::P2P_NVL: handle.get_nvl(rank).send(...); break;
 *         case TransportType::P2P_IBGDA: handle.get_ibgda(rank).put(...);
 * break;
 *       }
 *     }
 *   }
 */
struct MultiPeerDeviceHandle {
  int myRank{-1};
  int nRanks{0};

  // Unified transport array indexed by global rank.
  // transports[rank].type gives the transport type for that rank.
  DeviceSpan<Transport> transports;

  // Number of NVL peers (excluding self)
  int numNvlPeers{0};

  // Number of IBGDA peers (= nRanks - 1, all non-self)
  int numIbPeers{0};

#ifdef __CUDACC__
  /** @return Transport type for the given global rank. */
  __device__ __forceinline__ TransportType get_type(int rank) const {
    return transports[rank].type;
  }

  /** @return Mutable reference to the NVL transport for the given rank. */
  __device__ __forceinline__ P2pNvlTransportDevice& get_nvl(int rank) {
    return transports[rank].p2p_nvl;
  }

  /** @return Const reference to the NVL transport for the given rank. */
  __device__ __forceinline__ const P2pNvlTransportDevice& get_nvl(
      int rank) const {
    return transports[rank].p2p_nvl;
  }

  /** @return Mutable reference to the IBGDA transport for the given rank. */
  __device__ __forceinline__ P2pIbgdaTransportDevice& get_ibgda(int rank) {
    return *transports[rank].p2p_ibgda;
  }

  /** @return Const reference to the IBGDA transport for the given rank. */
  __device__ __forceinline__ const P2pIbgdaTransportDevice& get_ibgda(
      int rank) const {
    return *transports[rank].p2p_ibgda;
  }
#endif
};

} // namespace comms::pipes
