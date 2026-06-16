// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/memory/DeviceSpan.cuh"
#include "comms/prims/transport/Transport.cuh"

#ifdef __CUDACC__
#include "comms/prims/transport/Transport.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"
#else
namespace comms::prims {
struct Transport;
enum class TransportType : uint8_t;
class P2pNvlTransportDevice;
class P2pIbgdaTransportDevice;
} // namespace comms::prims
#endif

namespace comms::prims {

/**
 * MultiPeerDeviceHandle - Unified device-side handle for mixed-transport
 * communication.
 *
 * Lightweight struct passed to CUDA kernels. Contains a single DeviceSpan
 * of Transport objects (one per rank) plus peer counts. The Transport union
 * already carries the type discriminant, so no separate type array is needed.
 *
 * Layout: transports[0..nRanks-1] where transports[myRank].type == SELF,
 *         NVL peers sorted first, followed by IB-only peers.
 *
 * USAGE:
 *   __global__ void kernel(MultiPeerDeviceHandle handle, ...) {
 *     for (int rank = 0; rank < handle.nRanks; ++rank) {
 *       switch (handle.get_type(rank)) {
 *         case TransportType::SELF: ...; break;
 *         case TransportType::P2P_NVL:
 *           handle.get_nvl(rank).send_group(...);
 *           break;
 *         case TransportType::P2P_IBGDA:
 *         case TransportType::P2P_IBRC:
 *           handle.get_ib(rank).put(...);
 *           break;
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

  // Number of IB peers (= nRanks - 1, all non-self)
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
    return *transports[rank].p2p_ib.ibgda;
  }

  /** @return Const reference to the IBGDA transport for the given rank. */
  __device__ __forceinline__ const P2pIbgdaTransportDevice& get_ibgda(
      int rank) const {
    return *transports[rank].p2p_ib.ibgda;
  }

  /** @return Mutable reference to the IBRC transport for the given rank. */
  __device__ __forceinline__ P2pIbrcTransportDevice& get_ibrc(int rank) {
    return *transports[rank].p2p_ib.ibrc;
  }

  /** @return Const reference to the IBRC transport for the given rank. */
  __device__ __forceinline__ const P2pIbrcTransportDevice& get_ibrc(
      int rank) const {
    return *transports[rank].p2p_ib.ibrc;
  }

  /** @return Backend-dispatching IB transport for the given rank. */
  __device__ __forceinline__ P2pIbTransportDevice get_ib(int rank) {
    return transports[rank].p2p_ib;
  }

  /** @return Backend-dispatching IB transport for the given rank. */
  __device__ __forceinline__ P2pIbTransportDevice get_ib(int rank) const {
    return transports[rank].p2p_ib;
  }

#endif
};

} // namespace comms::prims
