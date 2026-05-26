// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultimemNvlTransport.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "comms/pipes/SignalState.cuh"

namespace comms::pipes {

namespace {

int getCurrentCudaDevice() {
  int cudaDevice = 0;
  const auto err = cudaGetDevice(&cudaDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaGetDevice failed: ") + cudaGetErrorString(err));
  }
  return cudaDevice;
}

std::vector<int> identityRankMap(int nRanks) {
  if (nRanks <= 0) {
    return {};
  }
  std::vector<int> rankMap(static_cast<std::size_t>(nRanks));
  for (int rank = 0; rank < nRanks; ++rank) {
    rankMap[static_cast<std::size_t>(rank)] = rank;
  }
  return rankMap;
}

} // namespace

MultimemNvlTransport::MultimemNvlTransport(
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    int commRank,
    std::vector<int> nvlRankToCommRank,
    const MultimemNvlTransportConfig& config)
    : commRank_(commRank),
      nvlRanks_(static_cast<int>(nvlRankToCommRank.size())),
      nvlRankToCommRank_(std::move(nvlRankToCommRank)),
      cudaDevice_(getCurrentCudaDevice()),
      config_(config) {
  if (nvlRanks_ <= 0) {
    throw std::runtime_error(
        "MultimemNvlTransport: nvlRankToCommRank must be non-empty");
  }
  bool foundCommRank = false;
  for (int rank = 0; rank < nvlRanks_; ++rank) {
    if (nvlRankToCommRank_[static_cast<std::size_t>(rank)] == commRank_) {
      foundCommRank = true;
    }
  }
  if (!foundCommRank) {
    throw std::runtime_error(
        "MultimemNvlTransport: commRank must appear in nvlRankToCommRank");
  }
  if (config_.dataBufferSize == 0) {
    throw std::runtime_error(
        "MultimemNvlTransport: dataBufferSize must be non-zero");
  }
  const uint64_t totalSignalCount =
      static_cast<uint64_t>(config_.userSignalCount) +
      static_cast<uint64_t>(config_.internalSignalCount);
  if (totalSignalCount == 0) {
    throw std::runtime_error(
        "MultimemNvlTransport: at least one signal slot is required");
  }
  if (totalSignalCount >
      static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("MultimemNvlTransport: signalCount too large");
  }

  dataBufferHandler_ = std::make_unique<MultimemHandler>(
      bootstrap,
      commRank_,
      nvlRankToCommRank_,
      cudaDevice_,
      config_.dataBufferSize);
  signalBufferHandler_ = std::make_unique<MultimemHandler>(
      std::move(bootstrap),
      commRank_,
      nvlRankToCommRank_,
      cudaDevice_,
      getSignalBufferSize(static_cast<int>(totalSignalCount)));
}

MultimemNvlTransport::MultimemNvlTransport(
    int nvlRank,
    int nvlRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultimemNvlTransportConfig& config)
    : MultimemNvlTransport(
          std::move(bootstrap),
          nvlRank,
          identityRankMap(nvlRanks),
          config) {}

void MultimemNvlTransport::exchange() {
  if (exchanged_) {
    return;
  }
  dataBufferHandler_->exchange();
  signalBufferHandler_->exchange();
  exchanged_ = true;
}

MultimemNvlTransportDevice MultimemNvlTransport::getDeviceTransport() const {
  if (!exchanged_) {
    throw std::runtime_error(
        "MultimemNvlTransport: exchange() must complete before device use");
  }

  auto* localSignals =
      static_cast<SignalState*>(signalBufferHandler_->getLocalDeviceMemPtr());
  auto* multimemSignals = static_cast<SignalState*>(
      signalBufferHandler_->getMultimemDeviceMemPtr());
  const auto userSignalCount = config_.userSignalCount;
  const auto internalSignalCount = config_.internalSignalCount;

  return MultimemNvlTransportDevice{
      .localData =
          static_cast<char*>(dataBufferHandler_->getLocalDeviceMemPtr()),
      .multimemData =
          static_cast<char*>(dataBufferHandler_->getMultimemDeviceMemPtr()),
      .userLocalSignals =
          DeviceSpan<SignalState>(localSignals, userSignalCount),
      .userMultimemSignals =
          DeviceSpan<SignalState>(multimemSignals, userSignalCount),
      .internalLocalSignals = DeviceSpan<SignalState>(
          localSignals + userSignalCount, internalSignalCount),
      .internalMultimemSignals = DeviceSpan<SignalState>(
          multimemSignals + userSignalCount, internalSignalCount),
      .dataBufferSize = config_.dataBufferSize,
  };
}

std::size_t MultimemNvlTransport::getAllocatedDataBufferSize() const {
  return dataBufferHandler_->getAllocatedSize();
}

std::size_t MultimemNvlTransport::getAllocatedSignalBufferSize() const {
  return signalBufferHandler_->getAllocatedSize();
}

} // namespace comms::pipes
