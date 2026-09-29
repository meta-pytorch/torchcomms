// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/nvl/MultimemNvlTransport.h"

#include <algorithm>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "comms/prims/bootstrap/NvlBootstrapAdapter.h"
#include "comms/prims/bootstrap/TeamStatusAgreement.h"
#include "comms/prims/core/SignalState.cuh"
#include "comms/utils/checks.h"
#ifdef __HIP_PLATFORM_AMD__
#include "comms/prims/transport/amd/HipHostCompat.h"
#else
#include "comms/utils/CudaRAII.h"
#endif

namespace comms::prims {

namespace {

constexpr uint64_t kMultimemNvlTransportProtocol = 0x4D4D4E564CULL;
constexpr uint64_t kMultimemNvlTransportProtocolVersion = 6;

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

MultimemNvlTransport::~MultimemNvlTransport() = default;

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
    : nvlRanks_(static_cast<int>(nvlRankToCommRank.size())), config_(config) {
  // Topology validation runs BEFORE cudaGetDevice so the rank-map preconditions
  // are exercisable on CPU-only hosts (see MultimemNvlTransportValidationTest).
  validateRankMap(commRank, nvlRankToCommRank);

  const auto validation =
      validate_multimem_nvl_transport_config(config_, nvlRanks_);
  if (!validation) {
    throw std::runtime_error(
        std::string("MultimemNvlTransport: ") +
        std::string(validation.errorMessage));
  }
  dataBufferSize_ = validation.dataBufferSize;
  internalSignalCount_ = validation.internalSignalCount;
  signalsPerChannel_ = validation.signalsPerChannel;

  // validateRankMap() guarantees that commRank appears exactly once.
  const auto nvlRankIt =
      std::find(nvlRankToCommRank.begin(), nvlRankToCommRank.end(), commRank);
  const int nvlRank =
      static_cast<int>(std::distance(nvlRankToCommRank.begin(), nvlRankIt));
  nvlRank_ = nvlRank;

  static_assert(alignof(SignalState) == detail::kMultimemSignalAlignment);
  static_assert(sizeof(SignalState) == detail::kMultimemSignalStateSize);
  signalRegionOffset_ = validation.signalRegionOffset;
  const std::size_t combinedSize = validation.backingAllocationSize;
  const std::size_t signalRegionBytes = combinedSize - signalRegionOffset_;

  cudaDevice_ = getCurrentCudaDevice();

  // The GpuMemHandler owns the unicast backing; exchange() adds the multicast
  // overlay over it. Size the allocation to the multicast granularity
  // (alignFloor) so it is bindable into a multicast object. The unicast peer
  // exchange and multicast exchange both use NVL-team coordinates.
  const std::size_t alignFloor =
      GpuMemHandler::backingGranularity(cudaDevice_, nvlRanks_);
  nvlBootstrap_ = std::make_shared<NvlBootstrapAdapter>(
      std::move(bootstrap), std::move(nvlRankToCommRank));
  combinedHandler_ = std::make_unique<GpuMemHandler>(
      nvlBootstrap_,
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

std::unique_ptr<meta::comms::DeviceBuffer>
MultimemNvlTransport::exchangeUnicastPeerViews() {
  combinedHandler_->exchangeMemPtrs();

  std::unique_ptr<meta::comms::DeviceBuffer> devicePeerInternalSignals;
  std::exception_ptr localError;
  try {
    std::vector<SignalState*> peerInternalSignals(
        static_cast<std::size_t>(nvlRanks_));
    for (int rank = 0; rank < nvlRanks_; ++rank) {
      auto* peerBase =
          static_cast<char*>(combinedHandler_->getPeerDeviceMemPtr(rank));
      auto* peerSignals =
          reinterpret_cast<SignalState*>(peerBase + signalRegionOffset_);
      peerInternalSignals[static_cast<std::size_t>(rank)] =
          peerSignals + config_.userSignalCount;
    }

    devicePeerInternalSignals = std::make_unique<meta::comms::DeviceBuffer>(
        peerInternalSignals.size() * sizeof(SignalState*));
    FB_CUDACHECKTHROW(cudaMemcpy(
        devicePeerInternalSignals->get(),
        peerInternalSignals.data(),
        peerInternalSignals.size() * sizeof(SignalState*),
        cudaMemcpyHostToDevice));
  } catch (...) {
    localError = std::current_exception();
  }

  std::vector<detail::TeamStatus> status(static_cast<std::size_t>(nvlRanks_));
  detail::allGatherAndAgree(
      *nvlBootstrap_,
      nvlRank_,
      nvlRanks_,
      status,
      localError,
      "MultimemNvlTransport::exchange: construct the unicast peer pointer "
      "table");

  return devicePeerInternalSignals;
}

void MultimemNvlTransport::exchange() {
  if (exchangeState_ == ExchangeState::kReady) {
    return;
  }
  if (exchangeState_ == ExchangeState::kFailed) {
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
                static_cast<uint64_t>(config_.enableUnicastPeerViews),
            },
    };
    combinedHandler_->exchangeMulticast(
        nvlRank_, identityRankMap(nvlRanks_), cudaDevice_, contract);

    if (config_.enableUnicastPeerViews) {
      internalUnicastSignalsByRank_ = exchangeUnicastPeerViews();
    }
  } catch (...) {
    exchangeState_ = ExchangeState::kFailed;
    throw;
  }
  exchangeState_ = ExchangeState::kReady;
}

MultimemNvlTransportDevice MultimemNvlTransport::getDeviceTransport() const {
  if (exchangeState_ != ExchangeState::kReady) {
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
      .maxBlocks = static_cast<uint32_t>(config_.maxBlocks),
      .signalsPerChannel = signalsPerChannel_,
      .internalUnicastSignalsByRank = internalUnicastSignalsByRank_
          ? DeviceSpan<SignalState*>(
                static_cast<SignalState**>(
                    internalUnicastSignalsByRank_->get()),
                static_cast<uint32_t>(nvlRanks_))
          : DeviceSpan<SignalState*>{},
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
