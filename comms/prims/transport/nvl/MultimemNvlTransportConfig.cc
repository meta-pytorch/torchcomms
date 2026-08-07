// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/nvl/MultimemNvlTransportConfig.h"

#include <limits>

namespace comms::prims {

const char* multimem_nvl_transport_config_error_string(
    MultimemNvlTransportConfigError error) {
  switch (error) {
    case MultimemNvlTransportConfigError::None:
      return "none";
    case MultimemNvlTransportConfigError::MissingDataBuffer:
      return "data buffer size must be non-zero";
    case MultimemNvlTransportConfigError::InvalidRankCount:
      return "NVL rank count must be positive";
    case MultimemNvlTransportConfigError::PartialStagingGeometry:
      return "pipeline depth and maximum groups must both be zero or non-zero";
    case MultimemNvlTransportConfigError::GeometryOutOfRange:
      return "pipeline depth or maximum groups exceeds UINT32_MAX";
    case MultimemNvlTransportConfigError::InsufficientDataCapacity:
      return "data buffer is too small for the staging geometry";
    case MultimemNvlTransportConfigError::SignalCountOverflow:
      return "signal count exceeds INT_MAX";
    case MultimemNvlTransportConfigError::NoSignalSlots:
      return "at least one signal slot is required";
  }
  return "unknown configuration error";
}

MultimemNvlTransportConfigValidation validate_multimem_nvl_transport_geometry(
    const MultimemNvlTransportConfig& config) {
  if (config.userSignalCount > std::numeric_limits<int>::max()) {
    return {0, MultimemNvlTransportConfigError::SignalCountOverflow};
  }

  const bool hasPipelineDepth = config.pipelineDepth != 0;
  const bool hasMaxGroups = config.maxGroups != 0;
  if (hasPipelineDepth != hasMaxGroups) {
    return {0, MultimemNvlTransportConfigError::PartialStagingGeometry};
  }
  if (!hasPipelineDepth) {
    if (config.userSignalCount == 0) {
      return {0, MultimemNvlTransportConfigError::NoSignalSlots};
    }
    return {};
  }
  if (config.pipelineDepth > std::numeric_limits<uint32_t>::max() ||
      config.maxGroups > std::numeric_limits<uint32_t>::max()) {
    return {0, MultimemNvlTransportConfigError::GeometryOutOfRange};
  }
  return {};
}

MultimemNvlTransportConfigValidation validate_multimem_nvl_transport_config(
    const MultimemNvlTransportConfig& config,
    int nvlRanks) {
  const auto geometry = validate_multimem_nvl_transport_geometry(config);
  if (!geometry) {
    return geometry;
  }
  if (config.dataBufferSize == 0) {
    return {0, MultimemNvlTransportConfigError::MissingDataBuffer};
  }
  if (nvlRanks <= 0) {
    return {0, MultimemNvlTransportConfigError::InvalidRankCount};
  }
  if (config.pipelineDepth == 0) {
    return {};
  }

  constexpr std::size_t kDataAlignment = 16;
  const std::size_t alignedUnits = config.dataBufferSize / kDataAlignment;
  const auto ranks = static_cast<std::size_t>(nvlRanks);
  if (config.maxGroups > alignedUnits ||
      config.pipelineDepth > alignedUnits / config.maxGroups ||
      ranks > alignedUnits / config.maxGroups / config.pipelineDepth) {
    return {0, MultimemNvlTransportConfigError::InsufficientDataCapacity};
  }

  const auto signalsPerLaneWide =
      multimem_staging_signals_per_lane_wide(static_cast<uint64_t>(nvlRanks));
  if (signalsPerLaneWide > std::numeric_limits<uint32_t>::max()) {
    return {0, MultimemNvlTransportConfigError::GeometryOutOfRange};
  }
  const auto maxInternalSignals = static_cast<std::size_t>(
      std::numeric_limits<int>::max() - config.userSignalCount);
  const auto signalsPerLane = static_cast<std::size_t>(signalsPerLaneWide);
  if (config.maxGroups > maxInternalSignals / signalsPerLane) {
    return {0, MultimemNvlTransportConfigError::SignalCountOverflow};
  }
  const auto signalsPerRound = config.maxGroups * signalsPerLane;
  if (config.pipelineDepth > maxInternalSignals / signalsPerRound) {
    return {0, MultimemNvlTransportConfigError::SignalCountOverflow};
  }
  return {
      static_cast<uint32_t>(config.pipelineDepth * signalsPerRound),
      MultimemNvlTransportConfigError::None,
  };
}

ResolvedMultimemNvlTransportConfig resolve_multimem_nvl_transport_config(
    const std::optional<MultimemNvlTransportConfig>& overrideConfig,
    const MultimemNvlTransportConfig& fallbackConfig,
    std::size_t topologyDataBufferSize,
    int nvlRanks) {
  auto config = overrideConfig.value_or(fallbackConfig);
  if (config.dataBufferSize == 0) {
    config.dataBufferSize = topologyDataBufferSize;
  }
  const auto validation =
      validate_multimem_nvl_transport_config(config, nvlRanks);
  return {config, validation.internalSignalCount, validation.error};
}

} // namespace comms::prims
