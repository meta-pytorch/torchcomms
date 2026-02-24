// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstring>

#include "comms/pipes/NvmlFabricInfo.h"
#include "comms/pipes/TopologyDiscovery.h"

namespace comms::pipes::tests {

/// Shared helper: build a RankTopologyInfo with the given hostname and CUDA
/// device. Fabric info is left unavailable by default unless explicitly set.
inline RankTopologyInfo make_rank_info(
    const char* hostname,
    int cudaDevice,
    bool fabricAvailable = false,
    const char* clusterUuid = nullptr,
    unsigned int cliqueId = 0) {
  RankTopologyInfo info{};
  std::strncpy(info.hostname, hostname, sizeof(info.hostname) - 1);
  info.cudaDevice = cudaDevice;
  info.fabricInfo.available = fabricAvailable;
  if (clusterUuid) {
    std::memcpy(
        info.fabricInfo.clusterUuid, clusterUuid, NvmlFabricInfo::kUuidLen);
  }
  info.fabricInfo.cliqueId = cliqueId;
  return info;
}

/// Helper: build a RankTopologyInfo with MNNVL fabric info.
/// The 128-bit clusterUuid is set by writing `uuidVal` into both 64-bit halves.
inline RankTopologyInfo make_mnnvl_rank_info(
    const char* hostname,
    int cudaDevice,
    int64_t uuidVal,
    unsigned int cliqueId) {
  RankTopologyInfo info = make_rank_info(hostname, cudaDevice);
  info.fabricInfo.available = true;
  std::memcpy(info.fabricInfo.clusterUuid, &uuidVal, sizeof(uuidVal));
  std::memcpy(
      info.fabricInfo.clusterUuid + sizeof(uuidVal), &uuidVal, sizeof(uuidVal));
  info.fabricInfo.cliqueId = cliqueId;
  return info;
}

} // namespace comms::pipes::tests
