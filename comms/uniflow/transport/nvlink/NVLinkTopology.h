// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "comms/uniflow/Result.h"
#include "comms/uniflow/drivers/Constants.h"
#include "comms/uniflow/drivers/nvml/NvmlApi.h"

namespace uniflow {

/// Serializable topology descriptor for NVLink clique membership.
/// Two GPUs can communicate over NVLink if they share the same
/// clusterUuid AND cliqueId (same NVLink fabric clique), OR if they
/// are on the same host (same hostHash) and support P2P access.
struct __attribute__((packed)) NVLinkTopology {
  static constexpr size_t clusterIdLen = NVML_GPU_FABRIC_UUID_LEN; // 16

  uint16_t version{0};
  std::array<uint8_t, clusterIdLen> clusterId{};
  uint32_t cliqueId{0};
  int32_t cudaDeviceId{-1};
  uint64_t hostHash{0};

  /// Serialize to opaque bytes for exchange via getTopology()/canConnect().
  std::vector<uint8_t> serialize() const;

  /// Deserialize from opaque bytes. Returns error if data is malformed.
  static Result<NVLinkTopology> deserialize(std::span<const uint8_t> data);

  /// Returns true if this topology is in the same NVLink clique as @p other.
  bool sameDomain(const NVLinkTopology& other) const noexcept;

  /// Returns true if this topology is on the same host as @p other.
  bool sameHost(const NVLinkTopology& other) const noexcept;
};

static_assert(sizeof(NVLinkTopology) == 34);

/// Compute a hash identifying this host. Uses hostname + boot_id for
/// container awareness (following the NCCL getHostHash pattern).
/// Computed once per process and cached.
uint64_t getHostHash();

bool isZeroUUid(const std::array<uint8_t, NVLinkTopology::clusterIdLen> uuid);

class NVLinkTopologyCache {
 public:
  ~NVLinkTopologyCache() = default;

  static NVLinkTopologyCache& instance(NvmlApi* api = nullptr);

  Status available();

  const std::optional<NVLinkTopology>& getTopology(int deviceId);

 private:
  explicit NVLinkTopologyCache(NvmlApi* api);

  std::array<std::optional<NVLinkTopology>, kMaxDevices> topologies_{};
  Status initStatus_{ErrCode::InvalidArgument};
};

} // namespace uniflow
