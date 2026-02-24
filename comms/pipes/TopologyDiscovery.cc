// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/TopologyDiscovery.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <unistd.h>

#include <cuda_runtime.h>

#include "comms/pipes/NvmlFabricInfo.h"

namespace comms::pipes {

namespace {

#define CUDA_CHECK(cmd)                                                    \
  do {                                                                     \
    cudaError_t err = (cmd);                                               \
    if (err != cudaSuccess) {                                              \
      throw std::runtime_error(                                            \
          std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + \
          __FILE__ + ":" + std::to_string(__LINE__));                      \
    }                                                                      \
  } while (0)

/// Default LocalInfoFn: gathers hostname, CUDA PCI bus ID, and NVML fabric
/// info from real hardware.
RankTopologyInfo default_local_info(int deviceId) {
  RankTopologyInfo info{};
  info.cudaDevice = deviceId;
  if (gethostname(info.hostname, sizeof(info.hostname)) != 0) {
    throw std::runtime_error(
        std::string("gethostname failed: ") +
        std::strerror(errno)); // NOLINT(facebook-hte-BadCall-strerror)
  }
  char busId[NvmlFabricInfo::kBusIdLen];
  CUDA_CHECK(cudaDeviceGetPCIBusId(busId, NvmlFabricInfo::kBusIdLen, deviceId));
  info.fabricInfo = NvmlFabricInfo::query(busId);
  return info;
}

/// Default PeerAccessFn: queries cudaDeviceCanAccessPeer.
bool default_peer_access(int deviceA, int deviceB) {
  int canAccess = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, deviceA, deviceB));
  return canAccess != 0;
}

} // namespace

TopologyDiscovery::TopologyDiscovery()
    : peerAccessFn_(default_peer_access), localInfoFn_(default_local_info) {}

TopologyDiscovery::TopologyDiscovery(PeerAccessFn peerAccessFn)
    : peerAccessFn_(std::move(peerAccessFn)),
      localInfoFn_(default_local_info) {}

TopologyDiscovery::TopologyDiscovery(
    PeerAccessFn peerAccessFn,
    LocalInfoFn localInfoFn)
    : peerAccessFn_(std::move(peerAccessFn)),
      localInfoFn_(std::move(localInfoFn)) {}

TopologyResult TopologyDiscovery::classify(
    int myRank,
    int nRanks,
    std::vector<RankTopologyInfo>& allInfo) {
  TopologyResult result;
  if (myRank < 0 || myRank >= static_cast<int>(allInfo.size())) {
    throw std::runtime_error(
        "TopologyDiscovery::classify: myRank " + std::to_string(myRank) +
        " out of range [0, " + std::to_string(allInfo.size()) + ")");
  }
  auto& myInfo = allInfo[myRank];
  const auto& peerAccessFn = peerAccessFn_;

  std::vector<int> nvlGroupGlobalRanks;
  nvlGroupGlobalRanks.push_back(myRank);

  for (int r = 0; r < nRanks; ++r) {
    if (r == myRank) {
      continue;
    }

    // Tier 1: MNNVL fabric match (GB200 cross-host NVLink).
    if (myInfo.fabricInfo.available && allInfo[r].fabricInfo.available &&
        sizeof(myInfo.fabricInfo.clusterUuid) >= NvmlFabricInfo::kUuidLen &&
        std::memcmp(
            myInfo.fabricInfo.clusterUuid,
            allInfo[r].fabricInfo.clusterUuid,
            NvmlFabricInfo::kUuidLen) == 0 &&
        myInfo.fabricInfo.cliqueId == allInfo[r].fabricInfo.cliqueId) {
      nvlGroupGlobalRanks.push_back(r);
      continue;
    }

    // Tier 2: Same hostname + peer access check.
    if (peerAccessFn &&
        std::strncmp(
            myInfo.hostname, allInfo[r].hostname, sizeof(myInfo.hostname)) ==
            0) {
      if (peerAccessFn(myInfo.cudaDevice, allInfo[r].cudaDevice)) {
        nvlGroupGlobalRanks.push_back(r);
        continue;
      }
    }
  }

  // Sort NVL group by global rank so that NVL local indices are consistent
  // across all ranks.
  std::sort(nvlGroupGlobalRanks.begin(), nvlGroupGlobalRanks.end());

  for (int i = 0; i < static_cast<int>(nvlGroupGlobalRanks.size()); ++i) {
    int gRank = nvlGroupGlobalRanks[i];
    result.globalToNvlLocal[gRank] = i;
    if (gRank != myRank) {
      result.nvlPeerRanks.push_back(gRank);
    }
  }

  // Store fabric info in the result.
  if (myInfo.fabricInfo.available) {
    std::memcpy(
        result.clusterUuid,
        myInfo.fabricInfo.clusterUuid,
        NvmlFabricInfo::kUuidLen);
    result.cliqueId = myInfo.fabricInfo.cliqueId;
    result.fabricAvailable = true;
  }

  return result;
}

TopologyResult TopologyDiscovery::discover(
    int myRank,
    int nRanks,
    int deviceId,
    ctran::bootstrap::IBootstrap& bootstrap) {
  std::vector<RankTopologyInfo> allInfo(nRanks);

  allInfo[myRank] = localInfoFn_(deviceId);

  bootstrap.allGather(allInfo.data(), sizeof(RankTopologyInfo), myRank, nRanks)
      .get();

  return classify(myRank, nRanks, allInfo);
}

} // namespace comms::pipes
