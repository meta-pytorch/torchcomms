// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/prims/memory/GpuMemHandler.h"
#include "comms/prims/transport/nvl/MultimemNvlTransportDevice.cuh"

namespace comms::prims {

struct MultimemNvlTransportConfig {
  // Size in bytes of the multicast staging data window.
  std::size_t dataBufferSize{0};

  // Signal slots exposed through signal(), read_signal(), and
  // wait_signal_until().
  uint32_t userSignalCount{1};

  // Signal slots reserved for transport-internal protocols. These are not
  // addressable through the public signal API.
  uint32_t internalSignalCount{0};
};

namespace detail {

inline std::optional<uint32_t> checked_multimem_internal_signal_count(
    std::size_t dataBufferSize,
    uint32_t userSignalCount,
    std::size_t pipelineDepth,
    std::size_t maxChannels,
    int nvlRanks) {
  if (nvlRanks <= 0 || userSignalCount > std::numeric_limits<int>::max()) {
    return std::nullopt;
  }

  const bool hasPipelineDepth = pipelineDepth != 0;
  const bool hasMaxChannels = maxChannels != 0;
  if (hasPipelineDepth != hasMaxChannels) {
    return std::nullopt;
  }
  if (!hasPipelineDepth) {
    return 0;
  }
  if (pipelineDepth > std::numeric_limits<uint32_t>::max() ||
      maxChannels > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }

  constexpr std::size_t kDataAlignment = 16;
  const std::size_t alignedUnits = dataBufferSize / kDataAlignment;
  const auto ranks = static_cast<std::size_t>(nvlRanks);
  if (maxChannels > alignedUnits ||
      pipelineDepth > alignedUnits / maxChannels ||
      ranks > alignedUnits / maxChannels / pipelineDepth) {
    return std::nullopt;
  }

  const uint64_t ranksWide = static_cast<uint64_t>(nvlRanks);
  const uint64_t signalsPerLane = 2 * ranksWide + 4;
  if (signalsPerLane > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }
  const uint64_t maxTotalSignals =
      static_cast<uint64_t>(std::numeric_limits<int>::max());
  const uint64_t maxInternalSignals = maxTotalSignals - userSignalCount;
  if (maxChannels > maxInternalSignals / signalsPerLane) {
    return std::nullopt;
  }
  const uint64_t signalsPerRound =
      static_cast<uint64_t>(maxChannels) * signalsPerLane;
  if (pipelineDepth > maxInternalSignals / signalsPerRound) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(
      static_cast<uint64_t>(pipelineDepth) * signalsPerRound);
}

} // namespace detail

/**
 * Host-side owner for a copy-based NVL multimem transport.
 *
 * This transport is group-scoped rather than peer-scoped: every rank in the
 * NVL team maps the same multicast object and a private local backing VA.
 * Writers use `multimemData` to broadcast into every rank's `localData` at the
 * same offset; readers consume from their local backing memory after observing
 * a signal. The numeric `multimemData` pointer is a process-local CUDA VA and
 * is not part of the cross-rank protocol.
 *
 * The caller must set the current CUDA device before construction. The
 * transport records that device ordinal and passes it to MultimemHandler; it
 * does not switch CUDA devices internally.
 *
 * `bootstrap` is the global communicator bootstrap. `commRank` is this
 * process's rank in that global communicator. `nvlRankToCommRank` maps each
 * NVLink-local rank to its global communicator rank; the transport uses
 * NVLink-domain bootstrap collectives internally.
 */
class MultimemNvlTransport {
 public:
  MultimemNvlTransport(
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      int commRank,
      std::vector<int> nvlRankToCommRank,
      const MultimemNvlTransportConfig& config);

  // Compatibility constructor for callers that already wrapped `bootstrap` in
  // an NVLink-local adapter. Prefer the global-bootstrap constructor above for
  // new code.
  MultimemNvlTransport(
      int nvlRank,
      int nvlRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultimemNvlTransportConfig& config);

  ~MultimemNvlTransport() = default;

  MultimemNvlTransport(const MultimemNvlTransport&) = delete;
  MultimemNvlTransport& operator=(const MultimemNvlTransport&) = delete;
  MultimemNvlTransport(MultimemNvlTransport&&) = delete;
  MultimemNvlTransport& operator=(MultimemNvlTransport&&) = delete;

  // Runs the host-side multicast exchange and marks the transport ready for
  // device-side use. Throws on failure. After a failed call, the transport is
  // poisoned: subsequent exchange() calls will throw immediately rather than
  // retry, since cuMulticastCreate / cuMemImport / bind failures can leave
  // partial driver state that is unsafe to re-exchange on the same object.
  void exchange();

  MultimemNvlTransportDevice getDeviceTransport() const;

  std::size_t getAllocatedDataBufferSize() const;
  std::size_t getAllocatedSignalBufferSize() const;

  // Mirrors NCCL/NVLS enablement: multicast must be supported on cudaDevice
  // and useful only for teams larger than two ranks. The caller should already
  // have made cudaDevice current.
  static bool isEligible(int nRanks, int cudaDevice) {
    return nRanks > 2 && GpuMemHandler::isMultimemSupported(cudaDevice);
  }

  // Throws std::runtime_error if `nvlRankToCommRank` is empty, contains a
  // negative rank, contains duplicates, or does not contain `commRank`. Exposed
  // as a static so unit tests can exercise the rank-map preconditions without
  // requiring a CUDA device (the constructor calls cudaGetDevice).
  static void validateRankMap(
      int commRank,
      const std::vector<int>& nvlRankToCommRank);

 private:
  const int commRank_{-1};
  const int nvlRanks_{-1};
  // This rank's index within the NVL team (its position in nvlRankToCommRank_).
  // Retained so getDeviceTransport() can expose it to the device staging
  // protocol, which keys its per-rank signal slots on the NVL-local rank.
  int nvlRank_{-1};
  const std::vector<int> nvlRankToCommRank_;
  // Recorded after the rank-map validation in the constructor body so a bad
  // topology fails before cudaGetDevice is consulted (lets tests cover the
  // rank-map preconditions on CPU-only hosts).
  int cudaDevice_{-1};
  const MultimemNvlTransportConfig config_;
  bool exchanged_{false};
  // Set when exchange() throws; subsequent calls throw instead of silently
  // retrying. Multicast object create/import/bind failures can leave partial
  // driver state, so a same-object retry is unsafe — callers must rebuild the
  // transport to recover.
  bool broken_{false};

  // Single GpuMemHandler whose physical allocation contains both the data
  // window and the signal slots back-to-back. Layout: [data | pad-to-128B |
  // signals]. The handler provides the unicast local VA (getLocalDeviceMemPtr)
  // and, via its multicast overlay (exchangeMulticast), the multicast VA
  // (getMultimemDeviceMemPtr) -- one cuMulticastCreate / one set of binds / one
  // MC VA range over one shared physical backing.
  std::unique_ptr<GpuMemHandler> combinedHandler_;
  // Offset of the signal region within combinedHandler_'s allocation. Equals
  // dataBufferSize rounded up to SignalState's 128-byte alignment.
  std::size_t signalRegionOffset_{0};
};

} // namespace comms::prims
