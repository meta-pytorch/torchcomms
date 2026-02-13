// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"

// When compiling with CUDA, include full IBGDA definition for device methods.
// When compiling with host-only compiler, forward declaration is sufficient.
#ifdef __CUDACC__
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#else
namespace comms::pipes {
class P2pIbgdaTransportDevice;
} // namespace comms::pipes
#endif

namespace comms::pipes {

enum class TransportType : uint8_t;

/**
 * MultiPeerDeviceHandle - Unified device-side handle for mixed-transport
 * communication.
 *
 * Lightweight struct (~80 bytes) passed to CUDA kernels. Contains DeviceSpans
 * pointing to GPU-allocated transport arrays and rank mapping arrays.
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

  // Per-rank transport type (indexed by global rank, SELF for myRank)
  DeviceSpan<TransportType> typePerRank;

  // GPU-allocated NVL transports (indexed by NVL peer index)
  DeviceSpan<P2pNvlTransportDevice> nvlTransports;

  // GPU-allocated IBGDA transports (indexed by IBGDA peer index)
  DeviceSpan<P2pIbgdaTransportDevice> ibgdaTransports;

  // Global rank → index into nvlTransports (-1 if not NVL peer)
  DeviceSpan<int> globalToNvlIndex;

  // Global rank → index into ibgdaTransports (-1 if not IBGDA peer)
  DeviceSpan<int> globalToIbgdaIndex;

  __host__ __device__ MultiPeerDeviceHandle() = default;

  /** @return Transport type for the given global rank. */
  __device__ __forceinline__ TransportType get_type(int rank) const {
    return typePerRank[rank];
  }

  /** @return Mutable reference to the NVL transport for the given rank. */
  __device__ __forceinline__ P2pNvlTransportDevice& get_nvl(int rank) {
    return nvlTransports[globalToNvlIndex[rank]];
  }

  /** @return Const reference to the NVL transport for the given rank. */
  __device__ __forceinline__ const P2pNvlTransportDevice& get_nvl(
      int rank) const {
    return nvlTransports[globalToNvlIndex[rank]];
  }

  /** @return Mutable reference to the IBGDA transport for the given rank. */
  __device__ __forceinline__ P2pIbgdaTransportDevice& get_ibgda(int rank) {
    return ibgdaTransports[globalToIbgdaIndex[rank]];
  }

  /** @return Const reference to the IBGDA transport for the given rank. */
  __device__ __forceinline__ const P2pIbgdaTransportDevice& get_ibgda(
      int rank) const {
    return ibgdaTransports[globalToIbgdaIndex[rank]];
  }
#endif
};

} // namespace comms::pipes
