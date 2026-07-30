// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibgda/MultipeerIbgdaTransportCuda.cuh"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <cstring>
#include <limits>

#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims {
namespace {

std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, const char* label) {
  CHECK_LE(lhs, std::numeric_limits<std::size_t>::max() - rhs)
      << label << " size overflow";
  return lhs + rhs;
}

std::size_t checkedMul(std::size_t lhs, std::size_t rhs, const char* label) {
  CHECK(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs)
      << label << " size overflow";
  return lhs * rhs;
}

} // namespace

P2pIbgdaTransportDevice* buildDeviceTransportsOnGpu(
    const std::vector<P2pIbgdaTransportBuildParams>& params,
    int numPeers,
    std::vector<void*>& outGpuAllocations) {
  // All peers must have the same shape: same numNics, and same per-NIC QP
  // counts. Take peer 0's layout as canonical and validate the rest.
  CHECK(!params.empty() && !params[0].h_nicDeviceIbgdaResources.empty())
      << "buildDeviceTransportsOnGpu: empty params or zero NICs";
  int numNics = static_cast<int>(params[0].h_nicDeviceIbgdaResources.size());
  int mainQpsPerNic =
      static_cast<int>(params[0].h_nicDeviceIbgdaResources[0].qps.size());
  int companionQpsPerNic = static_cast<int>(
      params[0].h_nicDeviceIbgdaResources[0].companionQps.size());
  for (int i = 0; i < numPeers; ++i) {
    CHECK_EQ(params[i].maxChannels, params[0].maxChannels)
        << "All peers must have the same maxChannels";
    CHECK_EQ(params[i].qpsPerConnection, params[0].qpsPerConnection)
        << "All peers must have the same qpsPerConnection";
    CHECK_EQ(params[i].qpDirectionCount, params[0].qpDirectionCount)
        << "All peers must have the same qpDirectionCount";
    CHECK_EQ(
        static_cast<int>(params[i].h_nicDeviceIbgdaResources.size()), numNics)
        << "All peers must have the same numNics";
    for (int n = 0; n < numNics; ++n) {
      CHECK_EQ(
          static_cast<int>(params[i].h_nicDeviceIbgdaResources[n].qps.size()),
          params[i].maxChannels * params[i].qpDirectionCount *
              params[i].qpsPerConnection)
          << "Main QP count must equal maxChannels * qpDirectionCount * "
             "qpsPerConnection";
      CHECK_EQ(
          static_cast<int>(
              params[i].h_nicDeviceIbgdaResources[n].companionQps.size()),
          params[i].maxChannels * params[i].qpDirectionCount *
              params[i].qpsPerConnection)
          << "Companion QP count must equal maxChannels * qpDirectionCount * "
             "qpsPerConnection";
      CHECK_EQ(
          static_cast<int>(params[i].h_nicDeviceIbgdaResources[n].qps.size()),
          mainQpsPerNic)
          << "All peers' NICs must have the same QP count";
      CHECK_EQ(
          static_cast<int>(
              params[i].h_nicDeviceIbgdaResources[n].companionQps.size()),
          companionQpsPerNic)
          << "All peers' NICs must have the same companion QP count";
    }
  }

  // 1. Allocate one contiguous GPU buffer for all QP pointer arrays.
  //    Layout per peer: [nic0_main][nic0_comp][nic1_main][nic1_comp]...
  //    Total: numPeers * numNics * (mainQpsPerNic + companionQpsPerNic).
  std::size_t qpsPerPeer = static_cast<std::size_t>(numNics) *
      (static_cast<std::size_t>(mainQpsPerNic) + companionQpsPerNic);
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
      auto* d_mainQps = d_allQps + (i * qpsPerPeer) +
          n * (mainQpsPerNic + companionQpsPerNic);
      auto* d_companionQps = d_mainQps + mainQpsPerNic;
      h_nicResources.push_back(
          NicDeviceIbgdaResources{
              DeviceSpan<doca_gpu_dev_verbs_qp*>(d_mainQps, mainQpsPerNic),
              DeviceSpan<doca_gpu_dev_verbs_qp*>(
                  d_companionQps, companionQpsPerNic),
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
  IbLocalChannel* d_allLocalChannels = nullptr;
  std::size_t blockStateBytes = static_cast<std::size_t>(numPeers) *
      params[0].maxChannels * sizeof(IbLocalChannel);
  err = cudaMalloc(&d_allLocalChannels, blockStateBytes);
  CHECK(err == cudaSuccess)
      << "Failed to allocate GPU IB channel state: " << cudaGetErrorString(err);
  outGpuAllocations.push_back(d_allLocalChannels);
  std::size_t totalCompletionSlots = 0;
  for (int i = 0; i < numPeers; ++i) {
    const int pipelineDepth = params[i].channelLayout.pipelineDepth;
    CHECK_GE(pipelineDepth, 0) << "pipelineDepth must not be negative";
    const std::size_t peerCompletionSlots = checkedMul(
        static_cast<std::size_t>(params[i].maxChannels),
        static_cast<std::size_t>(pipelineDepth),
        "send completion slots");
    totalCompletionSlots = checkedAdd(
        totalCompletionSlots, peerCompletionSlots, "send completion slots");
  }
  IbSendCompletionSlot* d_allCompletionSlots = nullptr;
  const std::size_t completionSlotBytes = checkedMul(
      totalCompletionSlots,
      sizeof(IbSendCompletionSlot),
      "send completion slots");
  if (completionSlotBytes != 0) {
    err = cudaMalloc(&d_allCompletionSlots, completionSlotBytes);
    CHECK(err == cudaSuccess)
        << "Failed to allocate GPU send completion slots: "
        << cudaGetErrorString(err);
    outGpuAllocations.push_back(d_allCompletionSlots);
    err = cudaMemset(d_allCompletionSlots, 0, completionSlotBytes);
    CHECK(err == cudaSuccess)
        << "Failed to initialize GPU send completion slots: "
        << cudaGetErrorString(err);
  }
  std::vector<IbLocalChannel> h_localChannels(
      static_cast<std::size_t>(numPeers) * params[0].maxChannels);
  std::size_t completionSlotOffset = 0;
  for (int i = 0; i < numPeers; ++i) {
    for (int channel = 0; channel < params[i].maxChannels; ++channel) {
      const auto channelIndex =
          static_cast<std::size_t>(i) * params[0].maxChannels + channel;
      const std::size_t pipelineDepth =
          static_cast<std::size_t>(params[i].channelLayout.pipelineDepth);
      IbSendCompletionSlot* completionSlots = pipelineDepth == 0
          ? nullptr
          : d_allCompletionSlots + completionSlotOffset;
      h_localChannels[channelIndex] =
          makeIbLocalChannel(params[i].channelLayout, channel, completionSlots);
      completionSlotOffset += pipelineDepth;
    }
  }
  err = cudaMemcpy(
      d_allLocalChannels,
      h_localChannels.data(),
      blockStateBytes,
      cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess) << "Failed to initialize GPU IB channel state: "
                            << cudaGetErrorString(err);

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
        params[i].maxChannels,
        params[i].qpsPerConnection,
        params[i].qpDirectionCount,
        DeviceSpan<IbLocalChannel>(
            d_allLocalChannels + i * params[i].maxChannels,
            params[i].maxChannels),
        params[i].channelLayout);
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

void writeDeviceTransportSlot(
    P2pIbgdaTransportDevice* deviceArray,
    int peerIndex,
    const P2pIbgdaTransportBuildParams& params,
    std::vector<void*>& outGpuAllocations) {
  CHECK(!params.h_nicDeviceIbgdaResources.empty())
      << "writeDeviceTransportSlot needs >= 1 NIC";
  int numNics = static_cast<int>(params.h_nicDeviceIbgdaResources.size());
  int mainQpsPerNic =
      static_cast<int>(params.h_nicDeviceIbgdaResources[0].qps.size());
  int companionQpsPerNic =
      static_cast<int>(params.h_nicDeviceIbgdaResources[0].companionQps.size());
  for (int n = 0; n < numNics; ++n) {
    CHECK_EQ(
        static_cast<int>(params.h_nicDeviceIbgdaResources[n].qps.size()),
        mainQpsPerNic)
        << "All NICs must have the same QP count";
    CHECK_EQ(
        static_cast<int>(
            params.h_nicDeviceIbgdaResources[n].companionQps.size()),
        companionQpsPerNic)
        << "All NICs must have the same companion QP count";
  }

  std::size_t qpsPerPeer = static_cast<std::size_t>(numNics) *
      (static_cast<std::size_t>(mainQpsPerNic) + companionQpsPerNic);
  std::size_t qpBytes = qpsPerPeer * sizeof(doca_gpu_dev_verbs_qp*);
  doca_gpu_dev_verbs_qp** d_qps = nullptr;
  cudaError_t err = cudaMalloc(&d_qps, qpBytes);
  CHECK(err == cudaSuccess) << "Failed to allocate per-peer GPU QP array: "
                            << cudaGetErrorString(err);
  outGpuAllocations.push_back(d_qps);

  std::vector<doca_gpu_dev_verbs_qp*> h_qps;
  h_qps.reserve(qpsPerPeer);
  for (int n = 0; n < numNics; ++n) {
    const auto& nicSpec = params.h_nicDeviceIbgdaResources[n];
    h_qps.insert(h_qps.end(), nicSpec.qps.begin(), nicSpec.qps.end());
    h_qps.insert(
        h_qps.end(), nicSpec.companionQps.begin(), nicSpec.companionQps.end());
  }
  err = cudaMemcpy(d_qps, h_qps.data(), qpBytes, cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess)
      << "Failed to copy per-peer QP array to GPU: " << cudaGetErrorString(err);

  std::size_t nicBytes = numNics * sizeof(NicDeviceIbgdaResources);
  NicDeviceIbgdaResources* d_nicResources = nullptr;
  err = cudaMalloc(&d_nicResources, nicBytes);
  CHECK(err == cudaSuccess)
      << "Failed to allocate per-peer NicDeviceIbgdaResources: "
      << cudaGetErrorString(err);
  outGpuAllocations.push_back(d_nicResources);

  std::vector<NicDeviceIbgdaResources> h_nicResources;
  h_nicResources.reserve(numNics);
  for (int n = 0; n < numNics; ++n) {
    auto* d_mainQps = d_qps + n * (mainQpsPerNic + companionQpsPerNic);
    auto* d_companionQps = d_mainQps + mainQpsPerNic;
    h_nicResources.push_back(
        NicDeviceIbgdaResources{
            DeviceSpan<doca_gpu_dev_verbs_qp*>(d_mainQps, mainQpsPerNic),
            DeviceSpan<doca_gpu_dev_verbs_qp*>(
                d_companionQps, companionQpsPerNic),
            params.h_nicDeviceIbgdaResources[n].sinkLkey,
            params.h_nicDeviceIbgdaResources[n].deviceId,
        });
  }
  err = cudaMemcpy(
      d_nicResources, h_nicResources.data(), nicBytes, cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess)
      << "Failed to copy per-peer NicDeviceIbgdaResources to GPU: "
      << cudaGetErrorString(err);

  IbLocalChannel* d_localChannels = nullptr;
  std::size_t blockStateBytes =
      static_cast<std::size_t>(params.maxChannels) * sizeof(IbLocalChannel);
  err = cudaMalloc(&d_localChannels, blockStateBytes);
  CHECK(err == cudaSuccess) << "Failed to allocate per-peer IB channel state: "
                            << cudaGetErrorString(err);
  outGpuAllocations.push_back(d_localChannels);
  CHECK_GE(params.channelLayout.pipelineDepth, 0)
      << "pipelineDepth must not be negative";
  const std::size_t pipelineDepth =
      static_cast<std::size_t>(params.channelLayout.pipelineDepth);
  const std::size_t completionSlotCount = checkedMul(
      static_cast<std::size_t>(params.maxChannels),
      pipelineDepth,
      "send completion slots");
  IbSendCompletionSlot* d_completionSlots = nullptr;
  const std::size_t completionSlotBytes = checkedMul(
      completionSlotCount,
      sizeof(IbSendCompletionSlot),
      "send completion slots");
  if (completionSlotBytes != 0) {
    err = cudaMalloc(&d_completionSlots, completionSlotBytes);
    CHECK(err == cudaSuccess) << "Failed to allocate per-peer send completion "
                                 "slots: "
                              << cudaGetErrorString(err);
    outGpuAllocations.push_back(d_completionSlots);
    err = cudaMemset(d_completionSlots, 0, completionSlotBytes);
    CHECK(err == cudaSuccess)
        << "Failed to initialize per-peer send completion slots: "
        << cudaGetErrorString(err);
  }
  std::vector<IbLocalChannel> h_localChannels(params.maxChannels);
  for (int channel = 0; channel < params.maxChannels; ++channel) {
    IbSendCompletionSlot* completionSlots = pipelineDepth == 0
        ? nullptr
        : d_completionSlots + static_cast<std::size_t>(channel) * pipelineDepth;
    h_localChannels[channel] =
        makeIbLocalChannel(params.channelLayout, channel, completionSlots);
  }
  err = cudaMemcpy(
      d_localChannels,
      h_localChannels.data(),
      blockStateBytes,
      cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess)
      << "Failed to initialize per-peer IB channel state: "
      << cudaGetErrorString(err);

  P2pIbgdaTransportDevice hostTransport(
      DeviceSpan<NicDeviceIbgdaResources>(d_nicResources, numNics),
      params.remoteSignalBuf,
      params.localSignalBuf,
      params.counterBuf,
      params.numSignalSlots,
      params.numCounterSlots,
      params.maxChannels,
      params.qpsPerConnection,
      params.qpDirectionCount,
      DeviceSpan<IbLocalChannel>(d_localChannels, params.maxChannels),
      params.channelLayout);

  err = cudaMemcpy(
      deviceArray + peerIndex,
      &hostTransport,
      sizeof(P2pIbgdaTransportDevice),
      cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess) << "Failed to copy per-peer device transport slot: "
                            << cudaGetErrorString(err);
}

std::size_t getP2pIbgdaTransportDeviceSize() {
  return sizeof(P2pIbgdaTransportDevice);
}

} // namespace comms::prims
