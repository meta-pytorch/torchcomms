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
  uint64_t sinkAddr{0};
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

// Forward declaration
struct P2pIbgdaTransportState;

/**
 * Build a fully-formed P2pIbgdaTransportDevice on the host with QP handles
 * AND staging state. Used by
 * MultipeerIbgdaTransport::buildP2pTransportDevice().
 *
 * This free function exists because P2pIbgdaTransportDevice.cuh includes DOCA
 * device headers with CUDA-only intrinsics that can't compile in .cc files.
 * The .cu file can include the full definition.
 *
 * @param params QP build parameters
 * @param stagingState Pre-built staging state for this peer
 * @param sendCounter Pointer to per-peer send iteration counter
 * @param recvCounter Pointer to per-peer recv iteration counter
 * @return Fully-formed P2pIbgdaTransportDevice, allocated on GPU (cudaMalloc)
 */
P2pIbgdaTransportDevice* buildFullP2pIbgdaTransportDeviceOnGpu(
    const P2pIbgdaTransportBuildParams& params,
    const P2pIbgdaTransportState& stagingState,
    uint64_t* sendCounter,
    uint64_t* recvCounter);

} // namespace comms::pipes
