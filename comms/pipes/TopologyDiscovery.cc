// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/TopologyDiscovery.h"

#include <algorithm>
#include <cerrno>
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

struct RankInfo {
  char hostname[64];
  int cudaDevice;
  NvmlFabricInfo fabricInfo;
};

} // namespace

TopologyResult TopologyDiscovery::discover(
    int myRank,
    int nRanks,
    int deviceId,
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap) {
  TopologyResult result;

  std::vector<RankInfo> allInfo(nRanks);
  auto& myInfo = allInfo[myRank];

  std::memset(&myInfo, 0, sizeof(RankInfo));
  myInfo.cudaDevice = deviceId;
  if (gethostname(myInfo.hostname, sizeof(myInfo.hostname)) != 0) {
    throw std::runtime_error(
        std::string("gethostname failed: ") + std::strerror(errno));
  }

  char busId[NvmlFabricInfo::kBusIdLen];
  CUDA_CHECK(cudaDeviceGetPCIBusId(busId, NvmlFabricInfo::kBusIdLen, deviceId));
  myInfo.fabricInfo = NvmlFabricInfo::query(busId);

  bootstrap->allGather(allInfo.data(), sizeof(RankInfo), myRank, nRanks).get();

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

    // Tier 2: Same hostname → local cudaDeviceCanAccessPeer (H100).
    if (std::strncmp(
            myInfo.hostname, allInfo[r].hostname, sizeof(myInfo.hostname)) ==
        0) {
      int canAccess = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(
          &canAccess, myInfo.cudaDevice, allInfo[r].cudaDevice));
      if (canAccess) {
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

} // namespace comms::pipes
