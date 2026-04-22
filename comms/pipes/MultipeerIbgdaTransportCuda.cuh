// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <vector>

#include "comms/pipes/IbgdaBuffer.h"

// Forward declarations
struct doca_gpu_dev_verbs_qp;

namespace comms::pipes {

// Forward declaration for return type
class P2pIbgdaTransportDevice;

/**
 * Parameters for building a single P2pIbgdaTransportDevice.
 *
 * Contains host-side vectors of QP pointers (one per QP set).
 * The build function handles copying these to GPU memory.
 */
struct P2pIbgdaTransportBuildParams {
  std::vector<doca_gpu_dev_verbs_qp*> mainQps;
  std::vector<doca_gpu_dev_verbs_qp*> companionQps;
  NetworkLKey sinkLkey{};
  IbgdaRemoteBuffer remoteSignalBuf{};
  IbgdaLocalBuffer localSignalBuf{};
  IbgdaLocalBuffer counterBuf{};
  // Throwaway remote uint64_t slot used as the signal target for counter-only
  // puts. Required when counterBuf is set; ignored otherwise. See
  // P2pIbgdaTransportDevice::put_impl for rationale.
  IbgdaRemoteBuffer discardSignalSlot{};
  int numSignalSlots{0};
  int numCounterSlots{0};
};

/**
 * Build P2pIbgdaTransportDevice array on GPU.
 *
 * For each peer, allocates GPU arrays for QP pointers, copies them,
 * then constructs P2pIbgdaTransportDevice objects in GPU memory.
 * All GPU allocations are pushed into outGpuAllocations for cleanup.
 *
 * @param params Build parameters (one per peer)
 * @param numPeers Number of peers
 * @param outGpuAllocations Output: all GPU allocations (caller frees)
 * @return Pointer to GPU array of transport objects (also in outGpuAllocations)
 */
P2pIbgdaTransportDevice* buildDeviceTransportsOnGpu(
    const std::vector<P2pIbgdaTransportBuildParams>& params,
    int numPeers,
    std::vector<void*>& outGpuAllocations);

/**
 * Get size of P2pIbgdaTransportDevice struct.
 */
std::size_t getP2pIbgdaTransportDeviceSize();

} // namespace comms::pipes
