// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/TopologyDiscovery.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <unistd.h>

#include <cuda_runtime.h>
#include <glog/logging.h>

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

/// Format a 16-byte cluster UUID as "lower64.upper64" hex, matching NCCL's
/// log format.
std::string formatUuid(const char uuid[NvmlFabricInfo::kUuidLen]) {
  uint64_t lo = 0;
  uint64_t hi = 0;
  std::memcpy(&lo, uuid, sizeof(lo));
  std::memcpy(&hi, uuid + sizeof(lo), sizeof(hi));
  std::ostringstream os;
  os << std::hex << lo << "." << hi;
  return os.str();
}

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
    std::vector<RankTopologyInfo>& allInfo,
    const TopologyConfig& topoConfig) {
  TopologyResult result;
  if (myRank < 0 || myRank >= static_cast<int>(allInfo.size())) {
    throw std::runtime_error(
        "TopologyDiscovery::classify: myRank " + std::to_string(myRank) +
        " out of range [0, " + std::to_string(allInfo.size()) + ")");
  }
  auto& myInfo = allInfo[myRank];
  const auto& peerAccessFn = peerAccessFn_;

  // Handle MnnvlMode (following NCCL's NCCL_MNNVL_ENABLE semantics).
  if (topoConfig.mnnvlMode == MnnvlMode::kDisabled) {
    if (myInfo.fabricInfo.available) {
      LOG(INFO) << "TopologyDiscovery: rank " << myRank
                << " MNNVL disabled by config (MnnvlMode::kDisabled),"
                << " ignoring available fabric info";
    }
    myInfo.fabricInfo.available = false;
  } else if (topoConfig.mnnvlMode == MnnvlMode::kEnabled) {
    if (!myInfo.fabricInfo.available) {
      throw std::runtime_error(
          "TopologyDiscovery: MnnvlMode::kEnabled but MNNVL fabric info is"
          " not available on rank " +
          std::to_string(myRank) +
          ". Ensure the system supports Multi-Node NVLink and the Fabric"
          " Manager is running.");
    }
  }
  // MnnvlMode::kAuto — use fabric info if available, no error if not.

  // Apply MNNVL overrides (following NCCL's NCCL_MNNVL_UUID and
  // NCCL_MNNVL_CLIQUE_ID semantics). Only take effect when fabric info is
  // available — on non-MNNVL hardware (H100 and earlier), these fields are
  // irrelevant since NVLink connectivity is determined by same-host +
  // cudaDeviceCanAccessPeer.
  if (myInfo.fabricInfo.available) {
    if (topoConfig.mnnvlUuid.has_value()) {
      std::string oldUuid = formatUuid(myInfo.fabricInfo.clusterUuid);
      int64_t uuid = topoConfig.mnnvlUuid.value();
      static_assert(
          sizeof(myInfo.fabricInfo.clusterUuid) >= 2 * sizeof(uuid),
          "clusterUuid buffer must be at least 16 bytes");
      std::memcpy(myInfo.fabricInfo.clusterUuid, &uuid, sizeof(uuid));
      std::memcpy(
          myInfo.fabricInfo.clusterUuid + sizeof(uuid), &uuid, sizeof(uuid));
      LOG(INFO) << "TopologyDiscovery: rank " << myRank
                << " overriding MNNVL cluster UUID from " << oldUuid << " to "
                << formatUuid(myInfo.fabricInfo.clusterUuid);
    }
    if (topoConfig.mnnvlCliqueId.has_value()) {
      unsigned int oldCliqueId = myInfo.fabricInfo.cliqueId;
      myInfo.fabricInfo.cliqueId =
          static_cast<unsigned int>(topoConfig.mnnvlCliqueId.value());
      LOG(INFO) << "TopologyDiscovery: rank " << myRank
                << " overriding MNNVL clique ID from 0x" << std::hex
                << oldCliqueId << " to 0x" << myInfo.fabricInfo.cliqueId
                << std::dec;
    }
  }

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
    ctran::bootstrap::IBootstrap& bootstrap,
    const TopologyConfig& topoConfig) {
  std::vector<RankTopologyInfo> allInfo(nRanks);

  allInfo[myRank] = localInfoFn_(deviceId);

  bootstrap.allGather(allInfo.data(), sizeof(RankTopologyInfo), myRank, nRanks)
      .get();

  return classify(myRank, nRanks, allInfo, topoConfig);
}

} // namespace comms::pipes
