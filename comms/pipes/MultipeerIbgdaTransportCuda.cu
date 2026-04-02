// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultipeerIbgdaTransportCuda.cuh"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/P2pIbgdaTransportState.h"

namespace comms::pipes {

P2pIbgdaTransportDevice* buildDeviceTransportsOnGpu(
    const P2pIbgdaTransportBuildParams* params,
    int numPeers) {
  // Build array on host first
  std::vector<P2pIbgdaTransportDevice> hostTransports;
  hostTransports.reserve(numPeers);

  for (int i = 0; i < numPeers; ++i) {
    hostTransports.emplace_back(
        params[i].gpuQp,
        params[i].companionGpuQp,
        params[i].sinkLkey,
        params[i].sinkAddr);
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

P2pIbgdaTransportDevice* buildFullP2pIbgdaTransportDeviceOnGpu(
    const P2pIbgdaTransportBuildParams& params,
    const P2pIbgdaTransportState& stagingState,
    uint64_t* sendCounter,
    uint64_t* recvCounter) {
  P2pIbgdaTransportDevice hostDev(
      params.gpuQp,
      params.companionGpuQp,
      params.sinkLkey,
      params.sinkAddr,
      stagingState.localStagingBuf,
      stagingState.remoteStagingBuf,
      stagingState.recvStagingBuf,
      stagingState.localSignalBuf,
      stagingState.remoteSignalBuf,
      stagingState.localSignalId,
      stagingState.remoteSignalId,
      stagingState.dataBufferSize,
      stagingState.pipelineDepth,
      sendCounter,
      recvCounter);

  P2pIbgdaTransportDevice* gpuPtr = nullptr;
  cudaError_t err = cudaMalloc(&gpuPtr, sizeof(P2pIbgdaTransportDevice));
  CHECK(err == cudaSuccess)
      << "Failed to allocate GPU memory for fully-formed IBGDA device: "
      << cudaGetErrorString(err);

  err = cudaMemcpy(
      gpuPtr,
      &hostDev,
      sizeof(P2pIbgdaTransportDevice),
      cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess)
      << "Failed to copy fully-formed IBGDA device to GPU: "
      << cudaGetErrorString(err);

  return gpuPtr;
}

} // namespace comms::pipes
