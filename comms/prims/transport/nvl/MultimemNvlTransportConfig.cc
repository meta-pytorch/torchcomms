// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/nvl/MultimemNvlTransportConfig.h"

#include <limits>

namespace comms::prims {

MultimemNvlTransportConfigValidation validate_multimem_nvl_transport_config(
    const MultimemNvlTransportConfig& config,
    int nvlRanks) {
  if (config.perChannelSize == 0) {
    return {.errorMessage = "per-channel size must be non-zero"};
  }
  if (config.maxChannels == 0) {
    return {.errorMessage = "maximum channels must be non-zero"};
  }
  if (nvlRanks <= 0) {
    return {.errorMessage = "NVL rank count must be positive"};
  }
  if (config.userSignalCount > std::numeric_limits<int>::max()) {
    return {.errorMessage = "signal count exceeds INT_MAX"};
  }

  const bool hasPipelineDepth = config.pipelineDepth != 0;
  const bool hasMaxBlocks = config.maxBlocks != 0;
  if (hasPipelineDepth != hasMaxBlocks) {
    return {
        .errorMessage =
            "pipeline depth and maximum blocks must both be zero or non-zero"};
  }
  if (config.pipelineDepth > std::numeric_limits<uint32_t>::max() ||
      config.maxChannels > std::numeric_limits<uint32_t>::max()) {
    return {.errorMessage = "transport geometry exceeds UINT32_MAX"};
  }
  if (config.maxBlocks > config.maxChannels) {
    return {.errorMessage = "maximum blocks must not exceed maximum channels"};
  }
  if (config.perChannelSize >
      std::numeric_limits<std::size_t>::max() / config.maxChannels) {
    return {
        .errorMessage = "per-channel size times maximum channels overflows"};
  }
  const std::size_t dataBufferSize = config.perChannelSize * config.maxChannels;

  uint32_t internalSignalCount = 0;
  uint32_t signalsPerChannel = 0;
  if (config.pipelineDepth != 0) {
    constexpr std::size_t kDataAlignment = 16;
    if (config.pipelineDepth >
            std::numeric_limits<std::size_t>::max() / kDataAlignment ||
        config.perChannelSize % (config.pipelineDepth * kDataAlignment) != 0) {
      return {
          .errorMessage =
              "per-channel size must be divisible by pipeline depth times 16"};
    }
    const std::size_t alignedUnits = dataBufferSize / kDataAlignment;
    const auto ranks = static_cast<std::size_t>(nvlRanks);
    if (config.maxChannels > alignedUnits ||
        config.pipelineDepth > alignedUnits / config.maxChannels ||
        ranks > alignedUnits / config.maxChannels / config.pipelineDepth) {
      return {
          .errorMessage = "data buffer is too small for the staging geometry"};
    }

    const auto signalsPerChannelWide = multimem_staging_signals_per_channel(
        static_cast<uint64_t>(nvlRanks), config.pipelineDepth);
    if (signalsPerChannelWide > std::numeric_limits<uint32_t>::max()) {
      return {.errorMessage = "transport geometry exceeds UINT32_MAX"};
    }
    const auto maxInternalSignals = static_cast<std::size_t>(
        std::numeric_limits<int>::max() - config.userSignalCount);
    signalsPerChannel = static_cast<uint32_t>(signalsPerChannelWide);
    if (config.maxChannels > maxInternalSignals / signalsPerChannel) {
      return {.errorMessage = "signal count exceeds INT_MAX"};
    }
    internalSignalCount = static_cast<uint32_t>(
        config.maxChannels * static_cast<std::size_t>(signalsPerChannel));
  }

  constexpr auto kSignalAlignment = detail::kMultimemSignalAlignment;
  constexpr auto kSignalStateSize = detail::kMultimemSignalStateSize;
  if (dataBufferSize >
      std::numeric_limits<std::size_t>::max() - (kSignalAlignment - 1)) {
    return {
        .errorMessage = "combined data and signal allocation size overflows"};
  }
  const std::size_t signalRegionOffset =
      ((dataBufferSize + kSignalAlignment - 1) / kSignalAlignment) *
      kSignalAlignment;
  const auto totalSignalCount =
      static_cast<std::size_t>(config.userSignalCount) + internalSignalCount;
  if (totalSignalCount >
      std::numeric_limits<std::size_t>::max() / kSignalStateSize) {
    return {
        .errorMessage = "combined data and signal allocation size overflows"};
  }
  const std::size_t signalRegionBytes = totalSignalCount * kSignalStateSize;
  if (signalRegionOffset >
      std::numeric_limits<std::size_t>::max() - signalRegionBytes) {
    return {
        .errorMessage = "combined data and signal allocation size overflows"};
  }

  return {
      .dataBufferSize = dataBufferSize,
      .signalsPerChannel = signalsPerChannel,
      .internalSignalCount = internalSignalCount,
      .signalRegionOffset = signalRegionOffset,
      .backingAllocationSize = signalRegionOffset + signalRegionBytes,
  };
}

} // namespace comms::prims
