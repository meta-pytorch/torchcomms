// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/topology/TopologyDiscovery.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include <unistd.h>

#include <cuda_runtime.h> // @manual
#include "comms/prims/memory/NvlMemExchange.h"
#include "comms/prims/topology/NvmlFabricInfo.h"
#include "comms/utils/logger/SpdlogLogger.h"

namespace comms::prims {

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
  if (gethostname(info.hostname, sizeof(info.hostname) - 1) != 0) {
    throw std::runtime_error(
        std::string("gethostname failed: ") +
        std::strerror(errno)); // NOLINT(facebook-hte-BadCall-strerror)
  }
  info.hostname[sizeof(info.hostname) - 1] = '\0';
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

bool default_fabric_handle_access(int deviceId) {
  return selectShareableHandleType(deviceId) == ShareableHandleType::kFabric;
}

CanonicalTopologyPolicyWire makePolicyWire(
    const CanonicalTopologyConfig& config) {
  CanonicalTopologyPolicyWire policy{};
  policy.mnnvlUuid = config.mnnvlUuid.value_or(0);
  policy.mnnvlCliqueId = config.mnnvlCliqueId.value_or(0);
  policy.virtualDomainSize = config.virtualDomainSize;
  policy.mnnvlMode = static_cast<std::uint8_t>(config.mnnvlMode);
  policy.hasMnnvlUuid = config.mnnvlUuid.has_value();
  policy.hasMnnvlCliqueId = config.mnnvlCliqueId.has_value();
  policy.p2pDisable = config.p2pDisable;
  policy.enableNvlFabricDomains = config.enableNvlFabricDomains;
  policy.mnnvlTrunkDisable = config.mnnvlTrunkDisable;
  policy.domainMode = static_cast<std::uint8_t>(config.domainMode);
  return policy;
}

bool samePolicy(
    const CanonicalTopologyPolicyWire& first,
    const CanonicalTopologyPolicyWire& second) {
  return first.mnnvlUuid == second.mnnvlUuid &&
      first.mnnvlCliqueId == second.mnnvlCliqueId &&
      first.virtualDomainSize == second.virtualDomainSize &&
      first.mnnvlMode == second.mnnvlMode &&
      first.hasMnnvlUuid == second.hasMnnvlUuid &&
      first.hasMnnvlCliqueId == second.hasMnnvlCliqueId &&
      first.p2pDisable == second.p2pDisable &&
      first.enableNvlFabricDomains == second.enableNvlFabricDomains &&
      first.mnnvlTrunkDisable == second.mnnvlTrunkDisable &&
      first.domainMode == second.domainMode;
}

std::size_t reachabilityRowBytes(int nRanks) {
  const auto ranks = static_cast<std::size_t>(nRanks);
  return ranks / 8 + (ranks % 8 != 0 ? 1 : 0);
}

std::size_t checkedMultiply(
    std::size_t first,
    std::size_t second,
    std::string_view description) {
  if (second != 0 && first > std::numeric_limits<std::size_t>::max() / second) {
    throw std::length_error(
        "TopologyDiscovery: size overflow for " + std::string{description});
  }
  return first * second;
}

bool packedReachability(
    const std::vector<std::uint8_t>& matrix,
    std::size_t rowBytes,
    int from,
    int to) {
  const auto byte = static_cast<std::size_t>(from) * rowBytes +
      static_cast<std::size_t>(to) / 8;
  const auto bit = static_cast<unsigned int>(to) % 8;
  return (matrix[byte] & static_cast<std::uint8_t>(1U << bit)) != 0;
}

void setPackedReachability(std::uint8_t* row, int rank, bool reachable) {
  const auto byte = static_cast<std::size_t>(rank) / 8;
  const auto bit = static_cast<unsigned int>(rank) % 8;
  const auto mask = static_cast<std::uint8_t>(1U << bit);
  if (reachable) {
    row[byte] |= mask;
  } else {
    row[byte] &= static_cast<std::uint8_t>(~mask);
  }
}

void validateCanonicalPreambles(
    const std::vector<CanonicalTopologyPreamble>& preambles,
    int nRanks,
    int myRank,
    std::string_view localFailure) {
  for (int rank = 0; rank < nRanks; ++rank) {
    const auto& preamble = preambles[static_cast<std::size_t>(rank)];
    if (preamble.magic != kCanonicalTopologyWireMagic ||
        preamble.version != kCanonicalTopologyWireVersion ||
        preamble.headerSize != kCanonicalTopologyPreambleSize ||
        preamble.recordSize != kCanonicalTopologyWireSize ||
        preamble.rank != rank || preamble.reserved != 0) {
      throw std::runtime_error(
          "TopologyDiscovery: incompatible canonical topology preamble at rank " +
          std::to_string(rank));
    }
  }
  for (int rank = 0; rank < nRanks; ++rank) {
    const auto status = preambles[static_cast<std::size_t>(rank)].status;
    if (status == 0) {
      continue;
    }
    if (status > 1) {
      throw std::runtime_error(
          "TopologyDiscovery: invalid canonical topology preamble status at rank " +
          std::to_string(rank));
    }
    if (rank == myRank && !localFailure.empty()) {
      throw std::runtime_error(
          "TopologyDiscovery: local topology preparation failed: " +
          std::string{localFailure});
    }
    throw std::runtime_error(
        "TopologyDiscovery: topology preparation failed at rank " +
        std::to_string(rank));
  }
}

template <std::size_t N>
std::string_view fixedStringView(const std::array<char, N>& value) {
  const auto end = std::find(value.begin(), value.end(), '\0');
  if (end == value.end()) {
    throw std::runtime_error(
        "TopologyDiscovery: gathered string is not NUL-terminated");
  }
  return {value.data(), static_cast<std::size_t>(end - value.begin())};
}

template <std::size_t N>
void copyFixedString(std::array<char, N>& destination, std::string_view value) {
  if (value.size() >= N) {
    throw std::invalid_argument(
        "TopologyDiscovery: topology string exceeds wire capacity");
  }
  std::copy(value.begin(), value.end(), destination.begin());
  destination[value.size()] = '\0';
}

std::string_view legacyHostname(const RankTopologyInfo& info) {
  const auto* end = static_cast<const char*>(
      std::memchr(info.hostname, '\0', sizeof(info.hostname)));
  if (end == nullptr) {
    throw std::runtime_error(
        "TopologyDiscovery: local hostname is not NUL-terminated");
  }
  return {info.hostname, static_cast<std::size_t>(end - info.hostname)};
}

void validateCanonicalConfig(
    int nRanks,
    const CanonicalTopologyConfig& config) {
  const auto mode = static_cast<int>(config.mnnvlMode);
  if (mode < static_cast<int>(MnnvlMode::kDisabled) ||
      mode > static_cast<int>(MnnvlMode::kAuto)) {
    throw std::invalid_argument("TopologyDiscovery: invalid MNNVL mode");
  }
  const auto domainMode = static_cast<int>(config.domainMode);
  if (domainMode < static_cast<int>(TopologyDomainMode::kSystem) ||
      domainMode > static_cast<int>(TopologyDomainMode::kVirtual)) {
    throw std::invalid_argument("TopologyDiscovery: invalid domain mode");
  }
  if (config.mnnvlCliqueId.has_value() && *config.mnnvlCliqueId < 0) {
    throw std::invalid_argument(
        "TopologyDiscovery: MNNVL clique ID must be non-negative");
  }
  const auto validateLocalString = [](std::string_view value,
                                      std::string_view name) {
    if (value.size() >= kCanonicalTopologyNameLength) {
      throw std::invalid_argument(
          "TopologyDiscovery: " + std::string{name} + " exceeds wire capacity");
    }
  };
  validateLocalString(config.localHostname, "hostname");
  validateLocalString(config.localZone, "zone");
  validateLocalString(config.localDc, "data center");
  validateLocalString(config.localDeviceRack, "device rack");
  if (config.localPid != -1 && config.localPid <= 0) {
    throw std::invalid_argument(
        "TopologyDiscovery: process ID must be positive or unspecified");
  }
  if (config.domainMode == TopologyDomainMode::kVirtual) {
    if (config.virtualDomainSize <= 0 ||
        nRanks % config.virtualDomainSize != 0) {
      throw std::invalid_argument(
          "TopologyDiscovery: virtual domain size must be positive and divide communicator size");
    }
  } else if (config.virtualDomainSize != 0) {
    throw std::invalid_argument(
        "TopologyDiscovery: virtual domain size requires virtual mode");
  }
}

void validateCanonicalRankInfo(
    const std::vector<CanonicalRankTopologyInfo>& rankInfo,
    int nRanks,
    const CanonicalTopologyPolicyWire& expectedPolicy) {
  if (rankInfo.size() != static_cast<std::size_t>(nRanks)) {
    throw std::invalid_argument(
        "TopologyDiscovery: gathered rank count does not match communicator size");
  }
  for (int rank = 0; rank < nRanks; ++rank) {
    const auto& info = rankInfo[static_cast<std::size_t>(rank)];
    if (info.magic != kCanonicalTopologyWireMagic ||
        info.version != kCanonicalTopologyWireVersion ||
        info.recordSize != kCanonicalTopologyWireSize) {
      throw std::runtime_error(
          "TopologyDiscovery: incompatible canonical topology wire record at rank " +
          std::to_string(rank));
    }
    if (info.rank != rank) {
      throw std::runtime_error(
          "TopologyDiscovery: gathered rank identity mismatch at slot " +
          std::to_string(rank));
    }
    if (info.cudaDevice < 0) {
      throw std::runtime_error(
          "TopologyDiscovery: gathered invalid CUDA device at rank " +
          std::to_string(rank));
    }
    if (info.pid <= 0) {
      throw std::runtime_error(
          "TopologyDiscovery: gathered invalid process ID at rank " +
          std::to_string(rank));
    }
    if (info.fabricInfoAvailable > 1 || info.fabricHandleAvailable > 1) {
      throw std::runtime_error(
          "TopologyDiscovery: gathered invalid capability flag at rank " +
          std::to_string(rank));
    }
    if (info.fabricHandleAvailable && !info.fabricInfoAvailable) {
      throw std::runtime_error(
          "TopologyDiscovery: FABRIC handle readiness requires fabric topology at rank " +
          std::to_string(rank));
    }
    if (info.reserved[0] != 0 || info.reserved[1] != 0 ||
        info.policy.reserved != 0 ||
        std::any_of(
            info.metadataReserved.begin(),
            info.metadataReserved.end(),
            [](std::uint8_t value) { return value != 0; })) {
      throw std::runtime_error(
          "TopologyDiscovery: gathered nonzero reserved wire field at rank " +
          std::to_string(rank));
    }
    if (!samePolicy(info.policy, expectedPolicy)) {
      throw std::runtime_error(
          "TopologyDiscovery: ranks disagree on canonical topology policy");
    }
    if (fixedStringView(info.hostname).empty()) {
      throw std::runtime_error(
          "TopologyDiscovery: gathered hostname is empty at rank " +
          std::to_string(rank));
    }
    (void)fixedStringView(info.deviceRack);
    (void)fixedStringView(info.zone);
    (void)fixedStringView(info.dc);
  }
}

void normalizeFabricOverrides(
    std::vector<CanonicalRankTopologyInfo>& rankInfo,
    const CanonicalTopologyConfig& config) {
  for (auto& info : rankInfo) {
    if (!info.fabricInfoAvailable) {
      continue;
    }
    if (config.mnnvlUuid.has_value()) {
      const auto uuid = *config.mnnvlUuid;
      const auto uuidBytes =
          std::bit_cast<std::array<char, sizeof(uuid)>>(uuid);
      std::copy(uuidBytes.begin(), uuidBytes.end(), info.clusterUuid.begin());
      std::copy(
          uuidBytes.begin(),
          uuidBytes.end(),
          info.clusterUuid.begin() + sizeof(uuid));
    }
    if (config.mnnvlCliqueId.has_value()) {
      info.cliqueId = static_cast<std::uint32_t>(*config.mnnvlCliqueId);
    }
  }
}

void validatePeerReachability(
    const std::vector<CanonicalRankTopologyInfo>& rankInfo,
    const std::vector<std::uint8_t>& peerReachability) {
  const int nRanks = static_cast<int>(rankInfo.size());
  std::vector<std::string_view> hostnames;
  hostnames.reserve(rankInfo.size());
  for (const auto& info : rankInfo) {
    hostnames.push_back(fixedStringView(info.hostname));
  }
  const auto rowBytes = reachabilityRowBytes(nRanks);
  const auto expectedSize = checkedMultiply(
      static_cast<std::size_t>(nRanks), rowBytes, "reachability matrix");
  if (peerReachability.size() != expectedSize) {
    throw std::invalid_argument(
        "TopologyDiscovery: packed peer reachability matrix has the wrong size");
  }
  if (nRanks % 8 != 0) {
    const auto usedBits = static_cast<unsigned int>(nRanks % 8);
    const auto unusedMask = static_cast<std::uint8_t>(0xffU << usedBits);
    for (int rank = 0; rank < nRanks; ++rank) {
      if ((peerReachability
               [static_cast<std::size_t>(rank) * rowBytes + rowBytes - 1] &
           unusedMask) != 0) {
        throw std::invalid_argument(
            "TopologyDiscovery: packed peer reachability has nonzero padding bits");
      }
    }
  }
  const auto reachable = [&](int from, int to) {
    return packedReachability(peerReachability, rowBytes, from, to);
  };

  for (int first = 0; first < nRanks; ++first) {
    if (!reachable(first, first)) {
      throw std::runtime_error(
          "TopologyDiscovery: peer reachability diagonal must contain self");
    }
    for (int second = first + 1; second < nRanks; ++second) {
      if (reachable(first, second) != reachable(second, first)) {
        throw std::runtime_error(
            "TopologyDiscovery: peer reachability is asymmetric");
      }
      if (hostnames.at(static_cast<std::size_t>(first)) !=
              hostnames.at(static_cast<std::size_t>(second)) &&
          reachable(first, second)) {
        throw std::runtime_error(
            "TopologyDiscovery: peer reachability crosses physical hosts");
      }
    }
  }
}

template <typename AdjacentFn>
std::vector<std::vector<int>> buildValidatedDomains(
    int nRanks,
    AdjacentFn adjacent) {
  std::vector<int> parent(static_cast<std::size_t>(nRanks));
  for (int rank = 0; rank < nRanks; ++rank) {
    parent[rank] = rank;
  }
  const auto findRoot = [&parent](int rank) {
    int root = rank;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[rank] != rank) {
      const int next = parent[rank];
      parent[rank] = root;
      rank = next;
    }
    return root;
  };
  for (int first = 0; first < nRanks; ++first) {
    for (int second = first + 1; second < nRanks; ++second) {
      const bool firstToSecond = adjacent(first, second);
      const bool secondToFirst = adjacent(second, first);
      if (firstToSecond != secondToFirst) {
        throw std::runtime_error(
            "TopologyDiscovery: canonical NVL relation is asymmetric");
      }
      if (!firstToSecond) {
        continue;
      }
      const int firstRoot = findRoot(first);
      const int secondRoot = findRoot(second);
      if (firstRoot != secondRoot) {
        parent[std::max(firstRoot, secondRoot)] =
            std::min(firstRoot, secondRoot);
      }
    }
  }

  std::vector<int> domainByRoot(static_cast<std::size_t>(nRanks), -1);
  std::vector<std::vector<int>> domains;
  for (int rank = 0; rank < nRanks; ++rank) {
    const int root = findRoot(rank);
    auto& domainIndex = domainByRoot.at(static_cast<std::size_t>(root));
    if (domainIndex == -1) {
      domainIndex = static_cast<int>(domains.size());
      domains.emplace_back();
    }
    domains.at(static_cast<std::size_t>(domainIndex)).push_back(rank);
  }
  for (const auto& domain : domains) {
    for (int first : domain) {
      for (int second : domain) {
        if (!adjacent(first, second)) {
          throw std::runtime_error(
              "TopologyDiscovery: canonical NVL relation is not transitive");
        }
      }
    }
  }
  return domains;
}

std::vector<std::vector<int>> buildFabricDomains(
    const std::vector<CanonicalRankTopologyInfo>& rankInfo,
    const std::vector<std::uint8_t>& peerReachability,
    bool trunkDisable) {
  const int nRanks = static_cast<int>(rankInfo.size());
  std::vector<std::string_view> hostnames;
  std::vector<std::string_view> deviceRacks;
  hostnames.reserve(rankInfo.size());
  deviceRacks.reserve(rankInfo.size());
  for (const auto& info : rankInfo) {
    hostnames.push_back(fixedStringView(info.hostname));
    deviceRacks.push_back(fixedStringView(info.deviceRack));
  }
  const auto rowBytes = reachabilityRowBytes(nRanks);
  const auto adjacent = [&](int first, int second) {
    if (first == second) {
      return true;
    }
    const auto& firstInfo = rankInfo[static_cast<std::size_t>(first)];
    const auto& secondInfo = rankInfo[static_cast<std::size_t>(second)];
    if (hostnames[first] == hostnames[second] &&
        packedReachability(peerReachability, rowBytes, first, second)) {
      return true;
    }
    const bool sameFabric = firstInfo.cliqueId == secondInfo.cliqueId &&
        firstInfo.clusterUuid == secondInfo.clusterUuid;
    if (!sameFabric) {
      return false;
    }
    if (!trunkDisable) {
      return true;
    }
    const auto firstRack = deviceRacks[first];
    const auto secondRack = deviceRacks[second];
    return !firstRack.empty() && !secondRack.empty() && firstRack == secondRack;
  };
  return buildValidatedDomains(nRanks, adjacent);
}

std::vector<std::vector<int>> buildPhysicalDomains(
    const std::vector<CanonicalRankTopologyInfo>& rankInfo,
    const std::vector<std::uint8_t>& peerReachability) {
  const int nRanks = static_cast<int>(rankInfo.size());
  std::vector<std::string_view> hostnames;
  hostnames.reserve(rankInfo.size());
  for (const auto& info : rankInfo) {
    hostnames.push_back(fixedStringView(info.hostname));
  }
  const auto rowBytes = reachabilityRowBytes(nRanks);
  const auto adjacent = [&](int first, int second) {
    if (first == second) {
      return true;
    }
    return hostnames[first] == hostnames[second] &&
        packedReachability(peerReachability, rowBytes, first, second);
  };
  return buildValidatedDomains(nRanks, adjacent);
}

std::vector<std::vector<int>> buildSingletonDomains(int nRanks) {
  std::vector<std::vector<int>> domains;
  domains.reserve(static_cast<std::size_t>(nRanks));
  for (int rank = 0; rank < nRanks; ++rank) {
    domains.push_back({rank});
  }
  return domains;
}

std::vector<std::vector<int>> applyDomainMode(
    const std::vector<std::vector<int>>& baseDomains,
    int nRanks,
    const CanonicalTopologyConfig& config) {
  if (config.domainMode == TopologyDomainMode::kSystem) {
    return baseDomains;
  }
  if (config.domainMode == TopologyDomainMode::kNoLocal) {
    return buildSingletonDomains(nRanks);
  }

  std::vector<int> baseDomainByRank(static_cast<std::size_t>(nRanks), -1);
  for (std::size_t domain = 0; domain < baseDomains.size(); ++domain) {
    for (int rank : baseDomains[domain]) {
      baseDomainByRank[rank] = static_cast<int>(domain);
    }
  }
  std::vector<std::vector<int>> domains;
  for (int first = 0; first < nRanks; first += config.virtualDomainSize) {
    const int baseDomain = baseDomainByRank[first];
    auto& domain = domains.emplace_back();
    for (int rank = first; rank < first + config.virtualDomainSize; ++rank) {
      if (baseDomainByRank[rank] != baseDomain) {
        throw std::runtime_error(
            "TopologyDiscovery: virtual domain crosses a physical NVL domain");
      }
      domain.push_back(rank);
    }
  }
  return domains;
}

TopologyResult buildLocalMptTopology(
    int myRank,
    const std::vector<std::vector<int>>& ranksByDomain,
    const std::vector<CanonicalRankTopologyInfo>& rankInfo,
    bool fabricActive) {
  TopologyResult result;
  const std::vector<int>* localDomain = nullptr;
  for (const auto& domain : ranksByDomain) {
    if (std::find(domain.begin(), domain.end(), myRank) != domain.end()) {
      localDomain = &domain;
      break;
    }
  }
  if (localDomain == nullptr) {
    throw std::logic_error(
        "TopologyDiscovery: current rank is missing from canonical domains");
  }
  for (std::size_t localRank = 0; localRank < localDomain->size();
       ++localRank) {
    const int globalRank = (*localDomain)[localRank];
    result.globalToNvlLocal.emplace(globalRank, static_cast<int>(localRank));
    if (globalRank != myRank) {
      result.nvlPeerRanks.push_back(globalRank);
    }
  }
  if (fabricActive) {
    const auto& localInfo = rankInfo[static_cast<std::size_t>(myRank)];
    std::copy(
        localInfo.clusterUuid.begin(),
        localInfo.clusterUuid.end(),
        result.clusterUuid);
    result.cliqueId = localInfo.cliqueId;
    result.fabricAvailable = true;
  }
  return result;
}

} // namespace

TopologyDiscovery::TopologyDiscovery()
    : peerAccessFn_(default_peer_access),
      localInfoFn_(default_local_info),
      fabricHandleAccessFn_(default_fabric_handle_access) {}

TopologyDiscovery::TopologyDiscovery(PeerAccessFn peerAccessFn)
    : peerAccessFn_(std::move(peerAccessFn)),
      localInfoFn_(default_local_info),
      fabricHandleAccessFn_(default_fabric_handle_access) {}

TopologyDiscovery::TopologyDiscovery(
    PeerAccessFn peerAccessFn,
    LocalInfoFn localInfoFn)
    : peerAccessFn_(std::move(peerAccessFn)),
      localInfoFn_(std::move(localInfoFn)),
      fabricHandleAccessFn_(default_fabric_handle_access) {}

TopologyDiscovery::TopologyDiscovery(
    PeerAccessFn peerAccessFn,
    LocalInfoFn localInfoFn,
    FabricHandleAccessFn fabricHandleAccessFn)
    : peerAccessFn_(std::move(peerAccessFn)),
      localInfoFn_(std::move(localInfoFn)),
      fabricHandleAccessFn_(std::move(fabricHandleAccessFn)) {}

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

  std::optional<std::unordered_set<int>> logicalNvlRankSet;
  if (topoConfig.logicalNvlRanks.has_value()) {
    logicalNvlRankSet.emplace();
    for (int rank : *topoConfig.logicalNvlRanks) {
      if (rank < 0 || rank >= nRanks) {
        throw std::runtime_error(
            "TopologyDiscovery::classify: logicalNvlRanks contains rank " +
            std::to_string(rank) + " outside [0, " + std::to_string(nRanks) +
            ")");
      }
      logicalNvlRankSet->insert(rank);
    }
    if (logicalNvlRankSet->count(myRank) == 0) {
      throw std::runtime_error(
          "TopologyDiscovery::classify: logicalNvlRanks must include myRank " +
          std::to_string(myRank));
    }
  }

  auto isInLogicalNvlGroup = [&logicalNvlRankSet](int rank) {
    return !logicalNvlRankSet.has_value() || logicalNvlRankSet->count(rank) > 0;
  };

  // Handle MnnvlMode (following NCCL's NCCL_MNNVL_ENABLE semantics).
  // Env vars (NCCL_MNNVL_ENABLE, NCCL_P2P_DISABLE) are read by the caller
  // (e.g. PrimsConfig) and passed via TopologyConfig fields.
  if (topoConfig.mnnvlMode == MnnvlMode::kDisabled) {
    if (myInfo.fabricInfo.available) {
      COMMS_LOG(
          DBG,
          "TopologyDiscovery: rank {} MNNVL disabled by config (MnnvlMode::kDisabled), ignoring available fabric info",
          myRank);
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
      const auto uuidBytes =
          std::bit_cast<std::array<char, sizeof(uuid)>>(uuid);
      std::copy(
          uuidBytes.begin(), uuidBytes.end(), myInfo.fabricInfo.clusterUuid);
      std::copy(
          uuidBytes.begin(),
          uuidBytes.end(),
          myInfo.fabricInfo.clusterUuid + sizeof(uuid));
      COMMS_LOG(
          DBG,
          "TopologyDiscovery: rank {} overriding MNNVL cluster UUID from {} to {}",
          myRank,
          oldUuid,
          formatUuid(myInfo.fabricInfo.clusterUuid));
    }
    if (topoConfig.mnnvlCliqueId.has_value()) {
      unsigned int oldCliqueId = myInfo.fabricInfo.cliqueId;
      myInfo.fabricInfo.cliqueId =
          static_cast<unsigned int>(topoConfig.mnnvlCliqueId.value());
      COMMS_LOG(
          DBG,
          "TopologyDiscovery: rank {} overriding MNNVL clique ID from {:#x} to {:#x}",
          myRank,
          oldCliqueId,
          myInfo.fabricInfo.cliqueId);
    }
  }

  std::vector<int> nvlGroupGlobalRanks;
  nvlGroupGlobalRanks.push_back(myRank);

  for (int r = 0; r < nRanks; ++r) {
    if (r == myRank) {
      continue;
    }
    if (!isInLogicalNvlGroup(r)) {
      continue;
    }

    // Tier 1: MNNVL fabric match (GB200 cross-host NVLink).
    // Skipped when p2pDisable is true (NCCL_P2P_DISABLE=1 disables all
    // NVLink connectivity, matching NCCL's PATH_LOC semantics).
    if (!topoConfig.p2pDisable && myInfo.fabricInfo.available &&
        allInfo[r].fabricInfo.available &&
        sizeof(myInfo.fabricInfo.clusterUuid) >= NvmlFabricInfo::kUuidLen &&
        std::equal(
            myInfo.fabricInfo.clusterUuid,
            myInfo.fabricInfo.clusterUuid + NvmlFabricInfo::kUuidLen,
            allInfo[r].fabricInfo.clusterUuid) &&
        myInfo.fabricInfo.cliqueId == allInfo[r].fabricInfo.cliqueId) {
      nvlGroupGlobalRanks.push_back(r);
      continue;
    }

    // Tier 2: Same hostname + peer access check.
    if (!topoConfig.p2pDisable && peerAccessFn &&
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

  COMMS_LOG(
      DBG,
      "TopologyDiscovery: rank {} classified {} NVL peers from {} total{}{}{}",
      myRank,
      result.nvlPeerRanks.size(),
      nRanks - 1,
      topoConfig.p2pDisable ? " (p2p disabled)" : "",
      myInfo.fabricInfo.available ? " (MNNVL)" : "",
      logicalNvlRankSet.has_value() ? " (logical group constrained)" : "");

  // Store fabric info in the result.
  if (myInfo.fabricInfo.available) {
    std::copy_n(
        myInfo.fabricInfo.clusterUuid,
        NvmlFabricInfo::kUuidLen,
        result.clusterUuid);
    result.cliqueId = myInfo.fabricInfo.cliqueId;
    result.fabricAvailable = true;
  }

  return result;
}

TopologyResult TopologyDiscovery::discover(
    int myRank,
    int nRanks,
    int deviceId,
    meta::comms::IBootstrap& bootstrap,
    const TopologyConfig& topoConfig) {
  std::vector<RankTopologyInfo> allInfo(nRanks);

  allInfo[myRank] = localInfoFn_(deviceId);

  auto result =
      bootstrap
          .allGather(allInfo.data(), sizeof(RankTopologyInfo), myRank, nRanks)
          .get();
  if (result != 0) {
    throw std::runtime_error("TopologyDiscovery::discover allGather failed");
  }

  return classify(myRank, nRanks, allInfo, topoConfig);
}

CanonicalTopologyResult TopologyDiscovery::classifyCanonical(
    int myRank,
    int nRanks,
    std::vector<CanonicalRankTopologyInfo> rankInfo,
    const std::vector<std::uint8_t>& peerReachability,
    const CanonicalTopologyConfig& topoConfig) {
  if (nRanks <= 0 || myRank < 0 || myRank >= nRanks) {
    throw std::invalid_argument(
        "TopologyDiscovery: invalid canonical communicator rank or size");
  }
  validateCanonicalConfig(nRanks, topoConfig);
  const auto expectedPolicy = makePolicyWire(topoConfig);
  validateCanonicalRankInfo(rankInfo, nRanks, expectedPolicy);
  validatePeerReachability(rankInfo, peerReachability);

  normalizeFabricOverrides(rankInfo, topoConfig);
  const bool fabricRequested = topoConfig.enableNvlFabricDomains &&
      topoConfig.mnnvlMode != MnnvlMode::kDisabled && !topoConfig.p2pDisable;
  const bool allRanksFabricReady =
      std::all_of(rankInfo.begin(), rankInfo.end(), [](const auto& info) {
        return info.fabricInfoAvailable && info.fabricHandleAvailable;
      });
  const bool anyRankHasFabricInfo =
      std::any_of(rankInfo.begin(), rankInfo.end(), [](const auto& info) {
        return info.fabricInfoAvailable;
      });
  if (fabricRequested && topoConfig.mnnvlMode == MnnvlMode::kEnabled &&
      !allRanksFabricReady) {
    throw std::runtime_error(
        "TopologyDiscovery: MNNVL is required but the communicator is not uniformly FABRIC-ready");
  }
  const bool fabricActive = fabricRequested && allRanksFabricReady;
  if (myRank == 0 && fabricRequested && anyRankHasFabricInfo &&
      topoConfig.mnnvlMode == MnnvlMode::kAuto && !allRanksFabricReady) {
    COMMS_LOG(
        WARN,
        "TopologyDiscovery: communicator is not uniformly FABRIC-ready; falling back to physical NVL domains");
  }

  std::vector<std::vector<int>> baseDomains;
  if (topoConfig.p2pDisable ||
      topoConfig.domainMode == TopologyDomainMode::kNoLocal) {
    baseDomains = buildSingletonDomains(nRanks);
  } else if (fabricActive) {
    baseDomains = buildFabricDomains(
        rankInfo, peerReachability, topoConfig.mnnvlTrunkDisable);
  } else {
    baseDomains = buildPhysicalDomains(rankInfo, peerReachability);
  }

  auto ranksByDomain = applyDomainMode(baseDomains, nRanks, topoConfig);
  const bool localFabricActive =
      fabricActive && topoConfig.domainMode != TopologyDomainMode::kNoLocal;
  auto mptTopology =
      buildLocalMptTopology(myRank, ranksByDomain, rankInfo, localFabricActive);

  COMMS_LOG(
      DBG,
      "TopologyDiscovery: rank {} canonicalized {} ranks into {} domains (local domain size {}, fabric {})",
      myRank,
      nRanks,
      ranksByDomain.size(),
      mptTopology.globalToNvlLocal.size(),
      localFabricActive ? "enabled" : "disabled");

  return CanonicalTopologyResult{
      .rankInfo = std::move(rankInfo),
      .ranksByDomain = std::move(ranksByDomain),
      .mptTopology = std::move(mptTopology),
      .fabricActive = fabricActive,
  };
}

CanonicalTopologyResult TopologyDiscovery::discoverCanonical(
    int myRank,
    int nRanks,
    int deviceId,
    meta::comms::IBootstrap& bootstrap,
    const CanonicalTopologyConfig& topoConfig) {
  if (nRanks <= 0 || myRank < 0 || myRank >= nRanks) {
    throw std::invalid_argument(
        "TopologyDiscovery: invalid canonical communicator rank or size");
  }
  CanonicalTopologyPreamble localPreamble;
  localPreamble.rank = myRank;
  CanonicalRankTopologyInfo localWire;
  std::vector<CanonicalTopologyPreamble> preambles(
      static_cast<std::size_t>(nRanks));
  std::vector<CanonicalRankTopologyInfo> rankInfo;
  std::vector<std::uint8_t> reachabilityWire;
  std::size_t rowBytes{0};
  std::size_t rowWireBytes{0};
  std::string localPreparationFailure;
  try {
    // Allocate every buffer needed by a later collective before publishing a
    // successful preamble. A rank-local allocation failure must be reported by
    // the preamble rather than skipping a collective that peers will enter.
    rankInfo.resize(static_cast<std::size_t>(nRanks));
    rowBytes = reachabilityRowBytes(nRanks);
    if (rowBytes >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::length_error(
          "TopologyDiscovery: reachability row exceeds bootstrap size limit");
    }
    rowWireBytes = rowBytes + 1;
    const auto rowMatrixBytes = checkedMultiply(
        static_cast<std::size_t>(nRanks),
        rowWireBytes,
        "reachability wire matrix");
    reachabilityWire.resize(rowMatrixBytes, 0);

    validateCanonicalConfig(nRanks, topoConfig);
    const auto localInfo = localInfoFn_(deviceId);
    localWire.rank = myRank;
    localWire.cudaDevice = localInfo.cudaDevice;
    if (localWire.cudaDevice < 0) {
      throw std::runtime_error(
          "local topology returned an invalid CUDA device");
    }
    const auto processId = topoConfig.localPid != -1
        ? static_cast<std::int64_t>(topoConfig.localPid)
        : static_cast<std::int64_t>(getpid());
    if (processId <= 0 ||
        processId > std::numeric_limits<std::int32_t>::max()) {
      throw std::runtime_error("local topology returned an invalid process ID");
    }
    localWire.pid = static_cast<std::int32_t>(processId);
    copyFixedString(
        localWire.hostname,
        topoConfig.localHostname.empty()
            ? legacyHostname(localInfo)
            : std::string_view{topoConfig.localHostname});
    copyFixedString(localWire.deviceRack, topoConfig.localDeviceRack);
    copyFixedString(localWire.zone, topoConfig.localZone);
    copyFixedString(localWire.dc, topoConfig.localDc);
    if (localInfo.fabricInfo.available) {
      std::copy_n(
          localInfo.fabricInfo.clusterUuid,
          NvmlFabricInfo::kUuidLen,
          localWire.clusterUuid.begin());
      localWire.cliqueId = localInfo.fabricInfo.cliqueId;
      localWire.fabricInfoAvailable = 1;
      if (topoConfig.enableNvlFabricDomains &&
          topoConfig.mnnvlMode != MnnvlMode::kDisabled &&
          fabricHandleAccessFn_ &&
          fabricHandleAccessFn_(localWire.cudaDevice)) {
        localWire.fabricHandleAvailable = 1;
      }
    }
    localWire.policy = makePolicyWire(topoConfig);
    rankInfo[static_cast<std::size_t>(myRank)] = localWire;
  } catch (const std::exception& error) {
    localPreamble.status = 1;
    localPreparationFailure = error.what();
  } catch (...) {
    localPreamble.status = 1;
    localPreparationFailure = "non-standard local preparation failure";
  }

  preambles[static_cast<std::size_t>(myRank)] = localPreamble;
  const int preambleGatherResult =
      bootstrap
          .allGather(
              preambles.data(), kCanonicalTopologyPreambleSize, myRank, nRanks)
          .get();
  // IBootstrap cannot collectively agree on a failed collective's status.
  if (preambleGatherResult != 0) {
    throw std::runtime_error(
        "TopologyDiscovery::discoverCanonical preamble allGather failed");
  }
  validateCanonicalPreambles(
      preambles, nRanks, myRank, localPreparationFailure);

  const int topologyGatherResult =
      bootstrap
          .allGather(
              rankInfo.data(), kCanonicalTopologyWireSize, myRank, nRanks)
          .get();
  // IBootstrap cannot collectively agree on a failed collective's status.
  if (topologyGatherResult != 0) {
    throw std::runtime_error(
        "TopologyDiscovery::discoverCanonical topology allGather failed");
  }
  validateCanonicalRankInfo(rankInfo, nRanks, makePolicyWire(topoConfig));

  auto* localRow =
      reachabilityWire.data() + static_cast<std::size_t>(myRank) * rowWireBytes;
  auto* localReachability = localRow + 1;
  setPackedReachability(localReachability, myRank, true);

  std::string localReachabilityFailure;
  try {
    if (!topoConfig.p2pDisable &&
        topoConfig.domainMode != TopologyDomainMode::kNoLocal &&
        peerAccessFn_) {
      const auto localHostname =
          fixedStringView(rankInfo[static_cast<std::size_t>(myRank)].hostname);
      for (int peer = 0; peer < nRanks; ++peer) {
        if (peer == myRank ||
            fixedStringView(
                rankInfo[static_cast<std::size_t>(peer)].hostname) !=
                localHostname) {
          continue;
        }
        setPackedReachability(
            localReachability,
            peer,
            peerAccessFn_(
                rankInfo[static_cast<std::size_t>(myRank)].cudaDevice,
                rankInfo[static_cast<std::size_t>(peer)].cudaDevice));
      }
    }
  } catch (const std::exception& error) {
    localRow[0] = 1;
    localReachabilityFailure = error.what();
  } catch (...) {
    localRow[0] = 1;
    localReachabilityFailure = "non-standard peer reachability failure";
  }

  const int reachabilityGatherResult = bootstrap
                                           .allGather(
                                               reachabilityWire.data(),
                                               static_cast<int>(rowWireBytes),
                                               myRank,
                                               nRanks)
                                           .get();
  // IBootstrap cannot collectively agree on a failed collective's status.
  if (reachabilityGatherResult != 0) {
    throw std::runtime_error(
        "TopologyDiscovery::discoverCanonical reachability allGather failed");
  }

  for (int rank = 0; rank < nRanks; ++rank) {
    const auto status =
        reachabilityWire[static_cast<std::size_t>(rank) * rowWireBytes];
    if (status == 0) {
      continue;
    }
    if (status > 1) {
      throw std::runtime_error(
          "TopologyDiscovery: invalid reachability status at rank " +
          std::to_string(rank));
    }
    if (rank == myRank && !localReachabilityFailure.empty()) {
      throw std::runtime_error(
          "TopologyDiscovery: local peer-access query failed: " +
          localReachabilityFailure);
    }
    throw std::runtime_error(
        "TopologyDiscovery: peer-access query failed at rank " +
        std::to_string(rank));
  }

  const auto packedMatrixBytes = checkedMultiply(
      static_cast<std::size_t>(nRanks), rowBytes, "reachability matrix");
  std::vector<std::uint8_t> peerReachability(packedMatrixBytes, 0);
  for (int rank = 0; rank < nRanks; ++rank) {
    const auto wireOffset = static_cast<std::size_t>(rank) * rowWireBytes + 1;
    const auto matrixOffset = static_cast<std::size_t>(rank) * rowBytes;
    std::copy_n(
        reachabilityWire.begin() + wireOffset,
        rowBytes,
        peerReachability.begin() + matrixOffset);
  }

  return classifyCanonical(
      myRank, nRanks, std::move(rankInfo), peerReachability, topoConfig);
}

} // namespace comms::prims
