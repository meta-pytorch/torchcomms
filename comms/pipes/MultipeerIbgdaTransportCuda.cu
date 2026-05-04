// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultipeerIbgdaTransportCuda.cuh"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <cstring>

#include "comms/pipes/P2pIbgdaTransportDevice.cuh"

namespace comms::pipes {

P2pIbgdaTransportDevice* buildDeviceTransportsOnGpu(
    const std::vector<P2pIbgdaTransportBuildParams>& params,
    int numPeers,
    std::vector<void*>& outGpuAllocations) {
  // All peers must have the same shape: same numNics, and same per-NIC QP
  // counts. Take peer 0's layout as canonical and validate the rest.
  CHECK(!params.empty() && !params[0].h_nicDeviceIbgdaResources.empty())
      << "buildDeviceTransportsOnGpu: empty params or zero NICs";
  int numNics = static_cast<int>(params[0].h_nicDeviceIbgdaResources.size());
  int qpsPerNic =
      static_cast<int>(params[0].h_nicDeviceIbgdaResources[0].qps.size());
  for (int i = 0; i < numPeers; ++i) {
    CHECK_EQ(
        static_cast<int>(params[i].h_nicDeviceIbgdaResources.size()), numNics)
        << "All peers must have the same numNics";
    for (int n = 0; n < numNics; ++n) {
      CHECK_EQ(
          static_cast<int>(params[i].h_nicDeviceIbgdaResources[n].qps.size()),
          qpsPerNic)
          << "All peers' NICs must have the same QP count";
      CHECK_EQ(
          static_cast<int>(
              params[i].h_nicDeviceIbgdaResources[n].companionQps.size()),
          qpsPerNic)
          << "qps and companionQps must match per NIC";
    }
  }

  // Each NIC has two QP kinds packed back-to-back: primary + companion.
  constexpr int kQpKindsPerNic = 2;

  // 1. Allocate one contiguous GPU buffer for all QP pointer arrays.
  //    Layout per peer: [nic0_main][nic0_comp][nic1_main][nic1_comp]...
  //    Total: numPeers * numNics * kQpKindsPerNic * qpsPerNic pointers.
  std::size_t qpsPerPeer =
      static_cast<std::size_t>(kQpKindsPerNic) * numNics * qpsPerNic;
  std::size_t totalQpBytes =
      numPeers * qpsPerPeer * sizeof(doca_gpu_dev_verbs_qp*);
  doca_gpu_dev_verbs_qp** d_allQps = nullptr;
  cudaError_t err = cudaMalloc(&d_allQps, totalQpBytes);
  CHECK(err == cudaSuccess)
      << "Failed to allocate GPU QP arrays: " << cudaGetErrorString(err);
  outGpuAllocations.push_back(d_allQps);

  std::vector<doca_gpu_dev_verbs_qp*> h_qps;
  h_qps.reserve(numPeers * qpsPerPeer);
  for (int i = 0; i < numPeers; ++i) {
    for (int n = 0; n < numNics; ++n) {
      const auto& nicSpec = params[i].h_nicDeviceIbgdaResources[n];
      h_qps.insert(h_qps.end(), nicSpec.qps.begin(), nicSpec.qps.end());
      h_qps.insert(
          h_qps.end(),
          nicSpec.companionQps.begin(),
          nicSpec.companionQps.end());
    }
  }
  err =
      cudaMemcpy(d_allQps, h_qps.data(), totalQpBytes, cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess)
      << "Failed to copy QP arrays to GPU: " << cudaGetErrorString(err);

  // 2. Allocate one contiguous GPU buffer for all NicDeviceIbgdaResources
  // structs:
  //    [peer0_nic0..nicN-1][peer1_nic0..nicN-1]...
  std::size_t totalNicBytes =
      numPeers * numNics * sizeof(NicDeviceIbgdaResources);
  NicDeviceIbgdaResources* d_allNicResources = nullptr;
  err = cudaMalloc(&d_allNicResources, totalNicBytes);
  CHECK(err == cudaSuccess)
      << "Failed to allocate GPU NicDeviceIbgdaResources array: "
      << cudaGetErrorString(err);
  outGpuAllocations.push_back(d_allNicResources);

  std::vector<NicDeviceIbgdaResources> h_nicResources;
  h_nicResources.reserve(numPeers * numNics);
  for (int i = 0; i < numPeers; ++i) {
    for (int n = 0; n < numNics; ++n) {
      // QP pointers for this peer/NIC start at offset:
      //   (i * qpsPerPeer) + (n * kQpKindsPerNic * qpsPerNic) within d_allQps.
      auto* d_mainQps =
          d_allQps + (i * qpsPerPeer) + (n * kQpKindsPerNic * qpsPerNic);
      auto* d_companionQps = d_mainQps + qpsPerNic;
      h_nicResources.push_back(
          NicDeviceIbgdaResources{
              DeviceSpan<doca_gpu_dev_verbs_qp*>(d_mainQps, qpsPerNic),
              DeviceSpan<doca_gpu_dev_verbs_qp*>(d_companionQps, qpsPerNic),
              params[i].h_nicDeviceIbgdaResources[n].sinkLkey,
              params[i].h_nicDeviceIbgdaResources[n].deviceId,
          });
    }
  }
  err = cudaMemcpy(
      d_allNicResources,
      h_nicResources.data(),
      totalNicBytes,
      cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess)
      << "Failed to copy NicDeviceIbgdaResources array to GPU: "
      << cudaGetErrorString(err);

  // 3. Build transport objects pointing into the contiguous
  // NicDeviceIbgdaResources array.
  std::vector<P2pIbgdaTransportDevice> h_transports;
  h_transports.reserve(numPeers);
  for (int i = 0; i < numPeers; ++i) {
    NicDeviceIbgdaResources* d_peerNicResources =
        d_allNicResources + i * numNics;
    h_transports.emplace_back(
        DeviceSpan<NicDeviceIbgdaResources>(d_peerNicResources, numNics),
        params[i].remoteSignalBuf,
        params[i].localSignalBuf,
        params[i].counterBuf,
        params[i].numSignalSlots,
        params[i].numCounterSlots,
        params[i].discardSignalSlot,
        params[i].sendRecvState);
  }

  // 4. Allocate and copy transport objects to GPU.
  P2pIbgdaTransportDevice* gpuPtr = nullptr;
  std::size_t transportSize = numPeers * sizeof(P2pIbgdaTransportDevice);
  err = cudaMalloc(&gpuPtr, transportSize);
  CHECK(err == cudaSuccess) << "Failed to allocate GPU device transports: "
                            << cudaGetErrorString(err);
  outGpuAllocations.push_back(gpuPtr); // track before memcpy for leak safety
  err = cudaMemcpy(
      gpuPtr, h_transports.data(), transportSize, cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess)
      << "Failed to copy device transports to GPU: " << cudaGetErrorString(err);

  return gpuPtr;
}

std::size_t getP2pIbgdaTransportDeviceSize() {
  return sizeof(P2pIbgdaTransportDevice);
}

} // namespace comms::pipes
