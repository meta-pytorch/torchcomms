// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/ibgda/MultipeerIbgdaTransportCuda.cuh"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims {
namespace {

std::size_t checkedMul(std::size_t lhs, std::size_t rhs, const char* label) {
  CHECK(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs)
      << label << " size overflow";
  return lhs * rhs;
}

void throwOnCudaError(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(error));
  }
}

struct QpTableLayout {
  std::size_t qpsPerChannel;
  std::size_t qpsPerNic;
  std::size_t qpsPerPeer;
};

QpTableLayout qpTableLayout(const IbgdaFixedDeviceTables& tables) {
  const std::size_t qpsPerChannel = checkedMul(
      static_cast<std::size_t>(tables.qpDirectionCount),
      static_cast<std::size_t>(tables.qpsPerConnection),
      "QP slots per channel");
  const std::size_t qpsPerNic = checkedMul(
      static_cast<std::size_t>(tables.maxChannels),
      qpsPerChannel,
      "QP slots per NIC");
  return QpTableLayout{
      .qpsPerChannel = qpsPerChannel,
      .qpsPerNic = qpsPerNic,
      .qpsPerPeer = checkedMul(
          static_cast<std::size_t>(tables.numNics),
          checkedMul(std::size_t{2}, qpsPerNic, "main and companion QP slots"),
          "QP slots per peer"),
  };
}

doca_gpu_dev_verbs_qp** peerQps(
    const IbgdaFixedDeviceTables& tables,
    const QpTableLayout& layout,
    int peerIndex) {
  return tables.qps + static_cast<std::size_t>(peerIndex) * layout.qpsPerPeer;
}

IbChannel* peerChannels(const IbgdaFixedDeviceTables& tables, int peerIndex) {
  return tables.channels +
      static_cast<std::size_t>(peerIndex) * tables.maxChannels;
}

bool tryCudaMemset(void* ptr, std::size_t bytes, const char* label) noexcept {
  if (bytes == 0) {
    return true;
  }
  const cudaError_t error = cudaMemset(ptr, 0, bytes);
  if (error == cudaSuccess) {
    return true;
  }
  LOG(ERROR) << "Failed to clear fixed IBGDA " << label << ": "
             << cudaGetErrorString(error);
  return false;
}

} // namespace

IbgdaFixedDeviceTables allocateIbgdaFixedDeviceTables(
    int numPeers,
    int numNics,
    int maxChannels,
    int qpsPerConnection,
    int qpDirectionCount,
    int pipelineDepth,
    std::vector<void*>& outGpuAllocations) {
  CHECK_GT(numPeers, 0);
  CHECK_GT(numNics, 0);
  CHECK_GT(maxChannels, 0);
  CHECK_GT(qpsPerConnection, 0);
  CHECK_GT(qpDirectionCount, 0);
  CHECK_GE(pipelineDepth, 0);

  IbgdaFixedDeviceTables tables{
      .numPeers = numPeers,
      .numNics = numNics,
      .maxChannels = maxChannels,
      .qpsPerConnection = qpsPerConnection,
      .qpDirectionCount = qpDirectionCount,
      .pipelineDepth = pipelineDepth,
  };
  const QpTableLayout layout = qpTableLayout(tables);

  const std::size_t qpBytes = checkedMul(
      checkedMul(
          static_cast<std::size_t>(numPeers),
          layout.qpsPerPeer,
          "fixed QP slots"),
      sizeof(doca_gpu_dev_verbs_qp*),
      "fixed QP table");
  cudaError_t err = cudaMalloc(&tables.qps, qpBytes);
  throwOnCudaError(err, "Failed to allocate fixed GPU QP tables");
  outGpuAllocations.push_back(tables.qps);
  err = cudaMemset(tables.qps, 0, qpBytes);
  throwOnCudaError(err, "Failed to zero fixed GPU QP tables");

  const std::size_t nicBytes = checkedMul(
      checkedMul(
          static_cast<std::size_t>(numPeers),
          static_cast<std::size_t>(numNics),
          "fixed NIC resources"),
      sizeof(NicDeviceIbgdaResources),
      "fixed NIC resource table");
  err = cudaMalloc(&tables.nicResources, nicBytes);
  throwOnCudaError(err, "Failed to allocate fixed GPU NIC resource table");
  outGpuAllocations.push_back(tables.nicResources);
  err = cudaMemset(tables.nicResources, 0, nicBytes);
  throwOnCudaError(err, "Failed to zero fixed GPU NIC resource table");

  const std::size_t channelCount = checkedMul(
      static_cast<std::size_t>(numPeers),
      static_cast<std::size_t>(maxChannels),
      "fixed channel descriptors");
  const std::size_t channelBytes = checkedMul(
      channelCount, sizeof(IbChannel), "fixed channel descriptor table");
  err = cudaMalloc(&tables.channels, channelBytes);
  throwOnCudaError(
      err, "Failed to allocate fixed GPU channel descriptor table");
  outGpuAllocations.push_back(tables.channels);
  err = cudaMemset(tables.channels, 0, channelBytes);
  throwOnCudaError(err, "Failed to zero fixed GPU channel descriptor table");

  const std::size_t transportBytes = checkedMul(
      static_cast<std::size_t>(numPeers),
      sizeof(P2pIbgdaTransportDevice),
      "fixed device transport table");
  err = cudaMalloc(&tables.transports, transportBytes);
  throwOnCudaError(err, "Failed to allocate fixed GPU device transport table");
  outGpuAllocations.push_back(tables.transports);
  err = cudaMemset(tables.transports, 0, transportBytes);
  throwOnCudaError(err, "Failed to zero fixed GPU device transport table");

  return tables;
}

P2pIbgdaTransportDevice* ibgdaDeviceSlot(
    P2pIbgdaTransportDevice* transports,
    int peerIndex) {
  return transports + peerIndex;
}

IbSendCompletionSlot* allocateIbgdaCompletionSlots(
    std::size_t numChannels,
    int pipelineDepth,
    std::vector<void*>& outGpuAllocations) {
  CHECK_GE(pipelineDepth, 0);
  const std::size_t completionCount = checkedMul(
      numChannels,
      static_cast<std::size_t>(pipelineDepth),
      "send completion slots");
  const std::size_t completionBytes = checkedMul(
      completionCount, sizeof(IbSendCompletionSlot), "send completion slots");
  if (completionBytes == 0) {
    return nullptr;
  }

  IbSendCompletionSlot* completionSlots = nullptr;
  cudaError_t err = cudaMalloc(&completionSlots, completionBytes);
  throwOnCudaError(err, "Failed to allocate GPU send completion slots");
  outGpuAllocations.push_back(completionSlots);
  err = cudaMemset(completionSlots, 0, completionBytes);
  throwOnCudaError(err, "Failed to zero GPU send completion slots");
  return completionSlots;
}

bool resetIbgdaDeviceSlot(
    const IbgdaFixedDeviceTables& tables,
    int peerIndex) noexcept {
  if (tables.transports == nullptr || tables.qps == nullptr ||
      tables.nicResources == nullptr || tables.channels == nullptr ||
      peerIndex < 0 || peerIndex >= tables.numPeers) {
    LOG(ERROR) << "Invalid fixed IBGDA reset peer=" << peerIndex;
    return false;
  }

  const std::size_t nicBytes = static_cast<std::size_t>(tables.numNics) *
      sizeof(NicDeviceIbgdaResources);
  const bool nicsCleared = tryCudaMemset(
      tables.nicResources +
          static_cast<std::size_t>(peerIndex) * tables.numNics,
      nicBytes,
      "NIC resource entries");
  const bool transportCleared = tryCudaMemset(
      tables.transports + peerIndex,
      sizeof(P2pIbgdaTransportDevice),
      "device transport entry");
  return nicsCleared && transportCleared;
}

void populateIbgdaDeviceRange(
    const IbgdaFixedDeviceTables& tables,
    int peerIndex,
    int beginChannel,
    int endChannel,
    const P2pIbgdaTransportBuildParams& params,
    const std::vector<IbChannel>& rangeChannels) {
  CHECK(tables.transports != nullptr);
  CHECK(tables.qps != nullptr);
  CHECK(tables.nicResources != nullptr);
  CHECK(tables.channels != nullptr);
  CHECK_GE(peerIndex, 0);
  CHECK_LT(peerIndex, tables.numPeers);
  CHECK_GE(beginChannel, 0);
  CHECK_LE(beginChannel, endChannel);
  CHECK_LE(endChannel, tables.maxChannels);
  CHECK_EQ(params.maxChannels, tables.maxChannels);
  CHECK_EQ(params.qpsPerConnection, tables.qpsPerConnection);
  CHECK_EQ(params.qpDirectionCount, tables.qpDirectionCount);
  CHECK_EQ(params.channelLayout.pipelineDepth, tables.pipelineDepth);
  CHECK_EQ(
      rangeChannels.size(),
      static_cast<std::size_t>(endChannel - beginChannel));
  CHECK_EQ(
      static_cast<int>(params.h_nicDeviceIbgdaResources.size()),
      tables.numNics);

  const QpTableLayout layout = qpTableLayout(tables);
  const std::size_t beginQp =
      static_cast<std::size_t>(beginChannel) * layout.qpsPerChannel;
  const std::size_t qpCount =
      static_cast<std::size_t>(endChannel - beginChannel) *
      layout.qpsPerChannel;
  auto* const peerQpTable = peerQps(tables, layout, peerIndex);

  cudaError_t err = cudaSuccess;
  for (int nic = 0; nic < tables.numNics; ++nic) {
    const auto& nicSpec = params.h_nicDeviceIbgdaResources[nic];
    CHECK_EQ(nicSpec.qps.size(), layout.qpsPerNic);
    CHECK_EQ(nicSpec.companionQps.size(), layout.qpsPerNic);
    auto* mainQps =
        peerQpTable + static_cast<std::size_t>(nic) * 2 * layout.qpsPerNic;
    if (qpCount != 0) {
      const std::size_t qpBytes = qpCount * sizeof(doca_gpu_dev_verbs_qp*);
      err = cudaMemcpy(
          mainQps + beginQp,
          nicSpec.qps.data() + beginQp,
          qpBytes,
          cudaMemcpyHostToDevice);
      throwOnCudaError(err, "Failed to populate fixed main QP range");
      err = cudaMemcpy(
          mainQps + layout.qpsPerNic + beginQp,
          nicSpec.companionQps.data() + beginQp,
          qpBytes,
          cudaMemcpyHostToDevice);
      throwOnCudaError(err, "Failed to populate fixed companion QP range");
    }
  }

  auto* const peerChannelTable = peerChannels(tables, peerIndex);
  if (beginChannel != endChannel) {
    err = cudaMemcpy(
        peerChannelTable + beginChannel,
        rangeChannels.data(),
        rangeChannels.size() * sizeof(IbChannel),
        cudaMemcpyHostToDevice);
    throwOnCudaError(err, "Failed to populate fixed channel descriptor range");
  }
}

void publishIbgdaDeviceSlot(
    const IbgdaFixedDeviceTables& tables,
    int peerIndex,
    const P2pIbgdaTransportBuildParams& params) {
  CHECK(tables.transports != nullptr);
  CHECK(tables.qps != nullptr);
  CHECK(tables.nicResources != nullptr);
  CHECK(tables.channels != nullptr);
  CHECK_GE(peerIndex, 0);
  CHECK_LT(peerIndex, tables.numPeers);
  CHECK_EQ(params.maxChannels, tables.maxChannels);
  CHECK_EQ(params.qpsPerConnection, tables.qpsPerConnection);
  CHECK_EQ(params.qpDirectionCount, tables.qpDirectionCount);
  CHECK_EQ(params.channelLayout.pipelineDepth, tables.pipelineDepth);
  CHECK_EQ(
      static_cast<int>(params.h_nicDeviceIbgdaResources.size()),
      tables.numNics);
  CHECK_LE(tables.numNics, kMaxNicsPerGpu);

  const QpTableLayout layout = qpTableLayout(tables);
  CHECK_LE(layout.qpsPerNic, static_cast<std::size_t>(UINT32_MAX));
  const auto qpsPerNicSpan = static_cast<uint32_t>(layout.qpsPerNic);
  auto* const peerQpTable = peerQps(tables, layout, peerIndex);

  std::vector<NicDeviceIbgdaResources> hostNicResources;
  hostNicResources.reserve(tables.numNics);
  for (int nic = 0; nic < tables.numNics; ++nic) {
    const auto& nicSpec = params.h_nicDeviceIbgdaResources[nic];
    CHECK_EQ(nicSpec.qps.size(), layout.qpsPerNic);
    CHECK_EQ(nicSpec.companionQps.size(), layout.qpsPerNic);
    auto* mainQps =
        peerQpTable + static_cast<std::size_t>(nic) * 2 * layout.qpsPerNic;
    hostNicResources.push_back(
        NicDeviceIbgdaResources{
            DeviceSpan<doca_gpu_dev_verbs_qp*>(mainQps, qpsPerNicSpan),
            DeviceSpan<doca_gpu_dev_verbs_qp*>(
                mainQps + layout.qpsPerNic, qpsPerNicSpan),
            nicSpec.sinkLkey,
            nicSpec.deviceId,
        });
  }
  cudaError_t err = cudaMemcpy(
      tables.nicResources +
          static_cast<std::size_t>(peerIndex) * tables.numNics,
      hostNicResources.data(),
      static_cast<std::size_t>(tables.numNics) *
          sizeof(NicDeviceIbgdaResources),
      cudaMemcpyHostToDevice);
  throwOnCudaError(err, "Failed to publish fixed NIC resource entries");

  auto* const peerChannelTable = peerChannels(tables, peerIndex);
  P2pIbgdaTransportDevice hostTransport(
      DeviceSpan<NicDeviceIbgdaResources>(
          tables.nicResources +
              static_cast<std::size_t>(peerIndex) * tables.numNics,
          tables.numNics),
      params.remoteSignalBuf,
      params.localSignalBuf,
      params.counterBuf,
      params.numSignalSlots,
      params.numCounterSlots,
      params.maxChannels,
      params.qpsPerConnection,
      params.qpDirectionCount,
      DeviceSpan<IbChannel>(peerChannelTable, params.maxChannels),
      params.channelLayout);
  err = cudaMemcpy(
      tables.transports + peerIndex,
      &hostTransport,
      sizeof(P2pIbgdaTransportDevice),
      cudaMemcpyHostToDevice);
  throwOnCudaError(err, "Failed to publish fixed device transport slot");
}

bool clearIbgdaDeviceRange(
    const IbgdaFixedDeviceTables& tables,
    int peerIndex,
    int beginChannel,
    int endChannel) noexcept {
  if (tables.qps == nullptr || tables.channels == nullptr || peerIndex < 0 ||
      peerIndex >= tables.numPeers || beginChannel < 0 ||
      beginChannel > endChannel || endChannel > tables.maxChannels) {
    LOG(ERROR) << "Invalid fixed IBGDA clear range: peer=" << peerIndex
               << " begin=" << beginChannel << " end=" << endChannel;
    return false;
  }

  const QpTableLayout layout = qpTableLayout(tables);
  const std::size_t beginQp =
      static_cast<std::size_t>(beginChannel) * layout.qpsPerChannel;
  const std::size_t qpCount =
      static_cast<std::size_t>(endChannel - beginChannel) *
      layout.qpsPerChannel;
  auto* const peerQpTable = peerQps(tables, layout, peerIndex);

  bool ok = true;
  const std::size_t qpBytes = qpCount * sizeof(doca_gpu_dev_verbs_qp*);
  for (int nic = 0; nic < tables.numNics; ++nic) {
    auto* mainQps =
        peerQpTable + static_cast<std::size_t>(nic) * 2 * layout.qpsPerNic;
    ok = tryCudaMemset(mainQps + beginQp, qpBytes, "main QP range") && ok;
    ok = tryCudaMemset(
             mainQps + layout.qpsPerNic + beginQp,
             qpBytes,
             "companion QP range") &&
        ok;
  }

  const std::size_t channelCount =
      static_cast<std::size_t>(endChannel - beginChannel);
  ok = tryCudaMemset(
           peerChannels(tables, peerIndex) + beginChannel,
           channelCount * sizeof(IbChannel),
           "channel descriptor range") &&
      ok;
  return ok;
}

} // namespace comms::prims
