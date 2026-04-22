// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultipeerIbgdaTransportCuda.cuh"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "comms/pipes/P2pIbgdaTransportDevice.cuh"

namespace comms::pipes {

P2pIbgdaTransportDevice* buildDeviceTransportsOnGpu(
    const std::vector<P2pIbgdaTransportBuildParams>& params,
    int numPeers,
    std::vector<void*>& outGpuAllocations) {
  // All peers must have the same numQps
  int numQps = static_cast<int>(params[0].mainQps.size());
  std::size_t arraySize = numQps * sizeof(doca_gpu_dev_verbs_qp*);

  // 1. Allocate one contiguous GPU buffer for all QP pointer arrays:
  //    [peer0_main][peer0_comp][peer1_main][peer1_comp]...
  std::size_t totalArraySize = numPeers * 2 * arraySize;
  char* d_allArrays = nullptr;
  cudaError_t err = cudaMalloc(&d_allArrays, totalArraySize);
  CHECK(err == cudaSuccess)
      << "Failed to allocate GPU QP arrays: " << cudaGetErrorString(err);
  outGpuAllocations.push_back(d_allArrays);

  // Build contiguous host buffer with all QP pointer arrays
  std::vector<doca_gpu_dev_verbs_qp*> hostArrays;
  hostArrays.reserve(numPeers * 2 * numQps);
  for (int i = 0; i < numPeers; ++i) {
    hostArrays.insert(
        hostArrays.end(), params[i].mainQps.begin(), params[i].mainQps.end());
    hostArrays.insert(
        hostArrays.end(),
        params[i].companionQps.begin(),
        params[i].companionQps.end());
  }

  // One memcpy for all QP pointer arrays
  err = cudaMemcpy(
      d_allArrays, hostArrays.data(), totalArraySize, cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess)
      << "Failed to copy QP arrays to GPU: " << cudaGetErrorString(err);

  // 2. Build transport objects pointing into the contiguous GPU buffer
  //    Each peer gets 2 * numQps entries: [main QPs][companion QPs]
  auto* basePtr = reinterpret_cast<doca_gpu_dev_verbs_qp**>(d_allArrays);
  std::vector<P2pIbgdaTransportDevice> hostTransports;
  hostTransports.reserve(numPeers);

  for (int i = 0; i < numPeers; ++i) {
    auto* d_mainQps = basePtr + (i * 2 * numQps);
    auto* d_companionQps = basePtr + (i * 2 * numQps + numQps);
    hostTransports.emplace_back(
        DeviceSpan<doca_gpu_dev_verbs_qp*>(d_mainQps, numQps),
        DeviceSpan<doca_gpu_dev_verbs_qp*>(d_companionQps, numQps),
        params[i].sinkLkey,
        params[i].remoteSignalBuf,
        params[i].localSignalBuf,
        params[i].counterBuf,
        params[i].numSignalSlots,
        params[i].numCounterSlots,
        params[i].discardSignalSlot,
        params[i].sendRecvState);
  }

  // 3. Allocate and copy transport objects to GPU
  P2pIbgdaTransportDevice* gpuPtr = nullptr;
  std::size_t transportSize = numPeers * sizeof(P2pIbgdaTransportDevice);
  err = cudaMalloc(&gpuPtr, transportSize);
  CHECK(err == cudaSuccess) << "Failed to allocate GPU device transports: "
                            << cudaGetErrorString(err);
  outGpuAllocations.push_back(gpuPtr); // track before memcpy for leak safety
  err = cudaMemcpy(
      gpuPtr, hostTransports.data(), transportSize, cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess)
      << "Failed to copy device transports to GPU: " << cudaGetErrorString(err);

  return gpuPtr;
}

std::size_t getP2pIbgdaTransportDeviceSize() {
  return sizeof(P2pIbgdaTransportDevice);
}

} // namespace comms::pipes
