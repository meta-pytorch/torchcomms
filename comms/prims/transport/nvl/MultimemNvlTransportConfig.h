// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace comms::prims {

struct MultimemNvlTransportConfig {
  std::size_t dataBufferSize{0};
  uint32_t userSignalCount{0};
  std::size_t pipelineDepth{0};
  std::size_t maxGroups{0};

  bool operator==(const MultimemNvlTransportConfig&) const = default;
};

enum class MultimemNvlTransportConfigError {
  None,
  MissingDataBuffer,
  InvalidRankCount,
  PartialStagingGeometry,
  GeometryOutOfRange,
  InsufficientDataCapacity,
  SignalCountOverflow,
  NoSignalSlots,
};

struct MultimemNvlTransportConfigValidation {
  uint32_t internalSignalCount{0};
  MultimemNvlTransportConfigError error{MultimemNvlTransportConfigError::None};

  explicit operator bool() const {
    return error == MultimemNvlTransportConfigError::None;
  }
};

struct ResolvedMultimemNvlTransportConfig {
  MultimemNvlTransportConfig config;
  uint32_t internalSignalCount{0};
  MultimemNvlTransportConfigError error{MultimemNvlTransportConfigError::None};

  explicit operator bool() const {
    return error == MultimemNvlTransportConfigError::None;
  }
};

#if defined(__CUDACC__) || defined(__HIPCC__)
#define MULTIMEM_HOST_DEVICE __host__ __device__
#else
#define MULTIMEM_HOST_DEVICE
#endif

MULTIMEM_HOST_DEVICE constexpr uint64_t multimem_staging_signals_per_lane_wide(
    uint64_t nvlRanks) {
  return 2 * nvlRanks + 4;
}

MULTIMEM_HOST_DEVICE constexpr uint32_t multimem_staging_signals_per_lane(
    uint32_t nvlRanks) {
  return static_cast<uint32_t>(
      multimem_staging_signals_per_lane_wide(nvlRanks));
}

#undef MULTIMEM_HOST_DEVICE

const char* multimem_nvl_transport_config_error_string(
    MultimemNvlTransportConfigError error);

MultimemNvlTransportConfigValidation validate_multimem_nvl_transport_geometry(
    const MultimemNvlTransportConfig& config);

MultimemNvlTransportConfigValidation validate_multimem_nvl_transport_config(
    const MultimemNvlTransportConfig& config,
    int nvlRanks);

ResolvedMultimemNvlTransportConfig resolve_multimem_nvl_transport_config(
    const std::optional<MultimemNvlTransportConfig>& overrideConfig,
    const MultimemNvlTransportConfig& fallbackConfig,
    std::size_t topologyDataBufferSize,
    int nvlRanks);

} // namespace comms::prims
