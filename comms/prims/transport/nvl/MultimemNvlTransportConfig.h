// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace comms::prims {

struct MultimemNvlTransportConfigParams {
  // Total data capacity is `perChannelSize * maxChannels`.
  std::size_t perChannelSize{0};

  // Staging pipeline depth. May be zero only when `maxBlocks` is also zero.
  std::size_t pipelineDepth{0};

  // Number of provisioned data channels.
  std::size_t maxChannels{0};

  // Maximum collective launch block count. Transport data and internal
  // staging-signal capacity are provisioned independently by `maxChannels`.
  std::size_t maxBlocks{0};

  // Signal slots exposed through signal(), read_signal(), and
  // wait_signal_until(). This is orthogonal to staging geometry.
  uint32_t userSignalCount{0};
};

struct MultimemNvlTransportConfig {
  constexpr MultimemNvlTransportConfig() = default;

  explicit constexpr MultimemNvlTransportConfig(
      MultimemNvlTransportConfigParams params)
      : perChannelSize(params.perChannelSize),
        pipelineDepth(params.pipelineDepth),
        maxChannels(params.maxChannels),
        maxBlocks(params.maxBlocks),
        userSignalCount(params.userSignalCount) {}

  std::size_t perChannelSize{0};
  std::size_t pipelineDepth{0};
  std::size_t maxChannels{0};
  std::size_t maxBlocks{0};
  uint32_t userSignalCount{0};

  bool operator==(const MultimemNvlTransportConfig&) const = default;
};

constexpr MultimemNvlTransportConfig make_multimem_nvl_transport_config(
    MultimemNvlTransportConfigParams params) {
  return MultimemNvlTransportConfig{params};
}

struct MultimemNvlTransportConfigValidation {
  std::size_t dataBufferSize{0};
  uint32_t signalsPerChannel{0};
  uint32_t internalSignalCount{0};
  std::size_t signalRegionOffset{0};
  std::size_t backingAllocationSize{0};
  std::string_view errorMessage{};

  explicit operator bool() const {
    return errorMessage.empty();
  }
};

namespace detail {
inline constexpr uint64_t kMultimemSignalsPerPeer = 3;
inline constexpr uint64_t kMultimemSignalsPerLane = 4;
inline constexpr std::size_t kMultimemSignalAlignment = 128;
inline constexpr std::size_t kMultimemSignalStateSize = 128;
} // namespace detail

#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
    constexpr uint64_t multimem_staging_signals_per_channel(
        uint64_t nvlRanks,
        uint64_t pipelineDepth) {
  return detail::kMultimemSignalsPerPeer * nvlRanks +
      detail::kMultimemSignalsPerLane * pipelineDepth;
}

// Validate the complete topology-aware config and derive all allocation sizes.
MultimemNvlTransportConfigValidation validate_multimem_nvl_transport_config(
    const MultimemNvlTransportConfig& config,
    int nvlRanks);

} // namespace comms::prims
