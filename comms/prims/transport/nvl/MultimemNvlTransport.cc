// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/nvl/MultimemNvlTransport.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "comms/common/BitOps.cuh"
#include "comms/prims/core/SignalState.cuh"

namespace comms::prims {

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

void MultimemNvlTransport::validateRankMap(
    int commRank,
    const std::vector<int>& nvlRankToCommRank) {
  if (nvlRankToCommRank.empty()) {
    throw std::runtime_error(
        "MultimemNvlTransport: nvlRankToCommRank must be non-empty");
  }
  // Single pass: reject negative ranks, duplicates, and require commRank to be
  // present.
  std::unordered_set<int> seen;
  seen.reserve(nvlRankToCommRank.size());
  bool sawCommRank = false;
  for (const int peerCommRank : nvlRankToCommRank) {
    if (peerCommRank < 0) {
      throw std::runtime_error(
          "MultimemNvlTransport: nvlRankToCommRank contains a negative rank");
    }
    if (!seen.insert(peerCommRank).second) {
      throw std::runtime_error(
          "MultimemNvlTransport: nvlRankToCommRank contains duplicate ranks");
    }
    if (peerCommRank == commRank) {
      sawCommRank = true;
    }
  }
  if (!sawCommRank) {
    throw std::runtime_error(
        "MultimemNvlTransport: commRank must appear in nvlRankToCommRank");
  }
}

MultimemNvlTransport::MultimemNvlTransport(
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    int commRank,
    std::vector<int> nvlRankToCommRank,
    const MultimemNvlTransportConfig& config)
    : commRank_(commRank),
      nvlRanks_(static_cast<int>(nvlRankToCommRank.size())),
      nvlRankToCommRank_(std::move(nvlRankToCommRank)),
      config_(config) {
  // Topology validation runs BEFORE cudaGetDevice so the rank-map preconditions
  // are exercisable on CPU-only hosts (see MultimemNvlTransportValidationTest).
  validateRankMap(commRank_, nvlRankToCommRank_);

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

  // commRank presence in the map is already verified by validateRankMap.
  int nvlRank = -1;
  for (int rank = 0; rank < nvlRanks_; ++rank) {
    if (nvlRankToCommRank_[static_cast<std::size_t>(rank)] == commRank_) {
      nvlRank = rank;
      break;
    }
  }

  cudaDevice_ = getCurrentCudaDevice();

  signalRegionOffset_ =
      comms::bitops::alignUp(config_.dataBufferSize, alignof(SignalState));
  const std::size_t signalRegionBytes =
      getSignalBufferSize(static_cast<int>(totalSignalCount));
  const std::size_t combinedSize = signalRegionOffset_ + signalRegionBytes;

  // The GpuMemHandler owns the unicast backing; exchange() adds the multicast
  // overlay over it. Size the allocation to the multicast granularity
  // (alignFloor) so it is bindable into a multicast object. Only multicast is
  // used (no P2P exchangeMemPtrs), so the selfRank/nRanks coordinates here are
  // the NVL-team rank and size.
  const std::size_t alignFloor =
      GpuMemHandler::backingGranularity(cudaDevice_, nvlRanks_);
  combinedHandler_ = std::make_unique<GpuMemHandler>(
      std::move(bootstrap),
      nvlRank,
      nvlRanks_,
      combinedSize,
      GpuMemHandler::detectBestMode(),
      alignFloor);
}

MultimemNvlTransport::MultimemNvlTransport(
    int nvlRank,
    int nvlRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultimemNvlTransportConfig& config)
    : MultimemNvlTransport(
          std::move(bootstrap),
          nvlRank,
          [&]() {
            // Precheck the identity-map contract before delegating so a
            // misuse (out-of-range nvlRank) surfaces as a targeted message
            // instead of the generic "commRank must appear in
            // nvlRankToCommRank" error from validateRankMap.
            if (nvlRank < 0 || nvlRank >= nvlRanks) {
              throw std::runtime_error(
                  "MultimemNvlTransport: nvlRank must be in [0, nvlRanks)");
            }
            return identityRankMap(nvlRanks);
          }(),
          config) {}

void MultimemNvlTransport::exchange() {
  if (exchanged_) {
    return;
  }
  if (broken_) {
    throw std::runtime_error(
        "MultimemNvlTransport::exchange: previous exchange() failed; "
        "rebuild the transport to retry (same-object retry is unsafe after "
        "a partial multicast setup)");
  }
  try {
    combinedHandler_->exchangeMulticast(
        commRank_, nvlRankToCommRank_, cudaDevice_);
  } catch (...) {
    broken_ = true;
    throw;
  }
  exchanged_ = true;
}

MultimemNvlTransportDevice MultimemNvlTransport::getDeviceTransport() const {
  if (!exchanged_) {
    throw std::runtime_error(
        "MultimemNvlTransport: exchange() must complete before device use");
  }

  auto* localBase =
      static_cast<char*>(combinedHandler_->getLocalDeviceMemPtr());
  auto* multimemBase =
      static_cast<char*>(combinedHandler_->getMultimemDeviceMemPtr());
  auto* localSignals =
      reinterpret_cast<SignalState*>(localBase + signalRegionOffset_);
  auto* multimemSignals =
      reinterpret_cast<SignalState*>(multimemBase + signalRegionOffset_);
  const auto userSignalCount = config_.userSignalCount;
  const auto internalSignalCount = config_.internalSignalCount;

  return MultimemNvlTransportDevice{
      .localData = localBase,
      .multimemData = multimemBase,
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
  return config_.dataBufferSize;
}

std::size_t MultimemNvlTransport::getAllocatedSignalBufferSize() const {
  // Report the usable signal region size (SignalState-aligned bytes for the
  // configured user + internal slot counts), not the padded backing tail. The
  // GpuMemHandler rounds the physical allocation up to the multicast backing
  // granularity, so combinedHandler_->getAllocatedSize() - signalRegionOffset_
  // would include trailing padding that is not addressable as SignalState.
  return getSignalBufferSize(
      static_cast<int>(config_.userSignalCount + config_.internalSignalCount));
}

} // namespace comms::prims
