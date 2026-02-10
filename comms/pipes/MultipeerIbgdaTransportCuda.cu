// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultipeerIbgdaTransportCuda.cuh"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "comms/pipes/P2pIbgdaTransportDevice.cuh"

namespace comms::pipes {

P2pIbgdaTransportDevice* buildDeviceTransportsOnGpu(
    const P2pIbgdaTransportBuildParams* params,
    int numPeers) {
  // Build array on host first
  std::vector<P2pIbgdaTransportDevice> hostTransports;
  hostTransports.reserve(numPeers);

  for (int i = 0; i < numPeers; ++i) {
    // Buffers already have keys in network byte order
    hostTransports.emplace_back(
        params[i].gpuQp,
        params[i].localSignalBuf,
        params[i].remoteSignalBuf,
        params[i].numSignals);
  }

  // Allocate GPU memory
  P2pIbgdaTransportDevice* gpuPtr = nullptr;
  std::size_t totalSize = numPeers * sizeof(P2pIbgdaTransportDevice);
  cudaError_t err = cudaMalloc(&gpuPtr, totalSize);
  CHECK(err == cudaSuccess)
      << "Failed to allocate GPU memory for device transports: "
      << cudaGetErrorString(err);

  // Copy to GPU
  err = cudaMemcpy(
      gpuPtr, hostTransports.data(), totalSize, cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess)
      << "Failed to copy device transports to GPU: " << cudaGetErrorString(err);

  return gpuPtr;
}

void freeDeviceTransportsOnGpu(P2pIbgdaTransportDevice* ptr) {
  if (ptr != nullptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
      LOG(WARNING) << "Failed to free GPU memory: " << cudaGetErrorString(err);
    }
  }
}

std::size_t getP2pIbgdaTransportDeviceSize() {
  return sizeof(P2pIbgdaTransportDevice);
}

} // namespace comms::pipes
