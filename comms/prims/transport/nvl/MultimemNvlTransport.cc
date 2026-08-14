// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/nvl/MultimemNvlTransport.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "comms/prims/core/SignalState.cuh"
#include "comms/utils/checks.h"

namespace comms::prims {

namespace {

constexpr uint64_t kMultimemNvlTransportProtocol = 0x4D4D4E564CULL;
constexpr uint64_t kMultimemNvlTransportProtocolVersion = 4;

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

  const auto validation =
      validate_multimem_nvl_transport_config(config_, nvlRanks_);
  if (!validation) {
    throw std::runtime_error(
        std::string("MultimemNvlTransport: ") +
        std::string(validation.errorMessage));
  }
  dataBufferSize_ = validation.dataBufferSize;
  internalSignalCount_ = validation.internalSignalCount;
  if (config_.pipelineDepth != 0) {
    signalsPerLane_ =
        multimem_staging_signals_per_lane(static_cast<uint32_t>(nvlRanks_));
  }
  // commRank presence in the map is already verified by validateRankMap.
  int nvlRank = -1;
  for (int rank = 0; rank < nvlRanks_; ++rank) {
    if (nvlRankToCommRank_[static_cast<std::size_t>(rank)] == commRank_) {
      nvlRank = rank;
      break;
    }
  }
  // Defensive backstop for the validateRankMap invariant: a -1 here would flow
  // into device-side signal-slot indexing and produce out-of-bounds offsets, so
  // fail loudly rather than corrupt memory if the map ever omits this rank.
  if (nvlRank < 0) {
    throw std::runtime_error(
        "MultimemNvlTransport: commRank not found in nvlRankToCommRank_");
  }
  nvlRank_ = nvlRank;

  static_assert(alignof(SignalState) == detail::kMultimemSignalAlignment);
  static_assert(sizeof(SignalState) == detail::kMultimemSignalStateSize);
  signalRegionOffset_ = validation.signalRegionOffset;
  const std::size_t combinedSize = validation.backingAllocationSize;
  const std::size_t signalRegionBytes = combinedSize - signalRegionOffset_;

  cudaDevice_ = getCurrentCudaDevice();

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

  // Zero the signal region so every rank starts from SignalState{0}. Neither
  // GpuMemHandler nor MultimemHandler zero the backing ("the caller is
  // responsible for zeroing the backing"), and the device arrival-counter
  // barrier reads its counter/epoch slots assuming they are zero-initialized.
  // Write through the local unicast VA -- the multicast overlay does not exist
  // until exchange() -- and synchronize so the zero is materialized before
  // exchange()'s post-map barrier lets any peer signal into this region. This
  // mirrors MultiPeerNvlTransport, which zeroes its signal buffer at
  // construction.
  if (signalRegionBytes != 0) {
    auto* localSignalRegion =
        static_cast<char*>(combinedHandler_->getLocalDeviceMemPtr()) +
        signalRegionOffset_;
    CUDA_CHECK(cudaMemset(localSignalRegion, 0, signalRegionBytes));
    CUDA_CHECK(cudaStreamSynchronize(/*stream=*/0));
  }
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
    const MulticastExchangeContract contract{
        .protocol = kMultimemNvlTransportProtocol,
        .version = kMultimemNvlTransportProtocolVersion,
        .parameters =
            {
                config_.perChannelSize,
                config_.pipelineDepth,
                config_.maxChannels,
                config_.maxBlocks,
                config_.userSignalCount,
            },
    };
    combinedHandler_->exchangeMulticast(
        commRank_, nvlRankToCommRank_, cudaDevice_, contract);
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
  const auto internalSignalCount = internalSignalCount_;

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
      .dataBufferSize = dataBufferSize_,
      .nvlRank = nvlRank_,
      .nvlRanks = nvlRanks_,
      .pipelineDepth = static_cast<uint32_t>(config_.pipelineDepth),
      .maxChannels = static_cast<uint32_t>(config_.maxChannels),
      .signalsPerLane = signalsPerLane_,
  };
}

std::size_t MultimemNvlTransport::getAllocatedDataBufferSize() const {
  return dataBufferSize_;
}

std::size_t MultimemNvlTransport::getAllocatedSignalBufferSize() const {
  // Report the usable signal region size (SignalState-aligned bytes for the
  // configured user + internal slot counts), not the padded backing tail. The
  // GpuMemHandler rounds the physical allocation up to the multicast backing
  // granularity, so combinedHandler_->getAllocatedSize() - signalRegionOffset_
  // would include trailing padding that is not addressable as SignalState.
  return getSignalBufferSize(
      static_cast<int>(config_.userSignalCount + internalSignalCount_));
}

} // namespace comms::prims
