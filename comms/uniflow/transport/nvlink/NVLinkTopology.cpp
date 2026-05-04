// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/transport/nvlink/NVLinkTopology.h"

#include <unistd.h>

#include <cstring>
#include <fstream>
#include <string>

namespace uniflow {

// ---------------------------------------------------------------------------
// Host hash — follows NCCL's getHostHash pattern (hostname + boot_id)
// ---------------------------------------------------------------------------

namespace {

uint64_t fnvHash(const void* data, size_t len) {
  uint64_t hash = 0xcbf29ce484222325ULL; // FNV-1a offset basis
  auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < len; ++i) {
    hash ^= bytes[i];
    hash *= 0x100000001b3ULL; // FNV-1a prime
  }
  return hash;
}

} // namespace

uint64_t getHostHash() {
  static uint64_t hash = [] {
    char hostname[256] = {};
    gethostname(hostname, sizeof(hostname) - 1);

    std::string hostId(hostname);

    // Append boot_id for container awareness — each Linux namespace gets a
    // unique boot_id, so containers on the same physical host hash differently.
    std::ifstream bootIdFile("/proc/sys/kernel/random/boot_id");
    if (bootIdFile) {
      std::string bootId;
      std::getline(bootIdFile, bootId);
      hostId += bootId;
    }

    return fnvHash(hostId.data(), hostId.size());
  }();
  return hash;
}

// ---------------------------------------------------------------------------
// NVLinkTopology
// ---------------------------------------------------------------------------

std::vector<uint8_t> NVLinkTopology::serialize() const {
  std::vector<uint8_t> buf(sizeof(*this));
  std::memcpy(buf.data(), this, sizeof(*this));
  return buf;
}

Result<NVLinkTopology> NVLinkTopology::deserialize(
    std::span<const uint8_t> data) {
  if (data.size() != sizeof(NVLinkTopology)) {
    return Err(
        ErrCode::InvalidArgument,
        "NVLinkTopology: expected " + std::to_string(sizeof(NVLinkTopology)) +
            " bytes, got " + std::to_string(data.size()));
  }
  NVLinkTopology topo;
  std::memcpy(&topo, data.data(), sizeof(topo));
  return topo;
}

bool NVLinkTopology::sameDomain(const NVLinkTopology& other) const noexcept {
  return clusterId == other.clusterId && cliqueId == other.cliqueId;
}

bool NVLinkTopology::sameHost(const NVLinkTopology& other) const noexcept {
  return hostHash != 0 && hostHash == other.hostHash;
}

bool isZeroUUid(const std::array<uint8_t, NVLinkTopology::clusterIdLen> uuid) {
  static constexpr std::array<uint8_t, NVLinkTopology::clusterIdLen> zero{};
  return uuid == zero;
}

// ---------------------------------------------------------------------------
// NVLinkTopologyCache
// ---------------------------------------------------------------------------

NVLinkTopologyCache::NVLinkTopologyCache(NvmlApi* api) {
  static NvmlApi defaultApi;
  if (api == nullptr) {
    api = &defaultApi;
  }
  const auto count = api->deviceCount();
  if (count.hasError()) {
    initStatus_ = count.error();
    return;
  }

  uint64_t hostHash = getHostHash();

  for (int dev = 0; dev < count.value(); ++dev) {
    auto info = api->deviceInfo(dev);
    if (info.hasError()) {
      continue;
    }

    NVLinkTopology topology;
    topology.cudaDeviceId = dev;
    topology.hostHash = hostHash;

    nvmlGpuFabricInfoV_t fabricInfo{};
    Status res =
        api->nvmlDeviceGetGpuFabricInfoV(info.value().handle, &fabricInfo);
    if (res.hasValue() && fabricInfo.state == NVML_GPU_FABRIC_STATE_COMPLETED &&
        fabricInfo.status == NVML_SUCCESS) {
      std::memcpy(
          topology.clusterId.data(),
          fabricInfo.clusterUuid,
          NVLinkTopology::clusterIdLen);
      topology.cliqueId = fabricInfo.cliqueId;
    }
    // If fabric info is unavailable, clusterId stays zero (intra-node only).

    topologies_[dev].emplace(std::move(topology));
  }
  initStatus_ = Ok();
}

NVLinkTopologyCache& NVLinkTopologyCache::instance(NvmlApi* api) {
  static NVLinkTopologyCache cache(api);
  return cache;
}

Status NVLinkTopologyCache::available() {
  return initStatus_;
}

const std::optional<NVLinkTopology>& NVLinkTopologyCache::getTopology(
    int deviceId) {
  return topologies_.at(deviceId);
}

} // namespace uniflow
