// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/pipes/IbgdaBuffer.h"

// Forward declarations
struct doca_gpu_dev_verbs_qp;

namespace comms::pipes {

// Forward declaration for return type
class P2pIbgdaTransportDevice;

/**
 * Parameters for building a single P2pIbgdaTransportDevice
 */
struct P2pIbgdaTransportBuildParams {
  doca_gpu_dev_verbs_qp* gpuQp{nullptr};
  doca_gpu_dev_verbs_qp* companionGpuQp{nullptr};
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
 * Build P2pIbgdaTransportDevice array on GPU
 *
 * Constructs an array of P2pIbgdaTransportDevice objects in GPU memory.
 *
 * @param params Array of build parameters (one per peer)
 * @param numPeers Number of peers
 * @return Pointer to GPU memory containing the array
 */
P2pIbgdaTransportDevice* buildDeviceTransportsOnGpu(
    const P2pIbgdaTransportBuildParams* params,
    int numPeers);

/**
 * Free P2pIbgdaTransportDevice array from GPU
 *
 * @param ptr Pointer returned by buildDeviceTransportsOnGpu
 */
void freeDeviceTransportsOnGpu(P2pIbgdaTransportDevice* ptr);

/**
 * Get size of P2pIbgdaTransportDevice struct
 *
 * Used for memory allocation calculations.
 */
std::size_t getP2pIbgdaTransportDeviceSize();

} // namespace comms::pipes
