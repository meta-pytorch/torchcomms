// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>

#include "comms/uniflow/drivers/ibverbs/IbvApi.h"

namespace uniflow {

/*
 * Per-NIC RDMA resources with RAII lifetime management.
 *
 * Full-init constructor: opens device, allocates PD, queries port/GID.
 * Destructor deallocates PD and closes device.
 *
 * Non-copyable, move-only. Shared across factory, transports, and slab pool
 * via shared_ptr<vector<NicResources>>.
 */
struct NicResources {
  ibv_context* ctx{nullptr}; /* Opened device context. */
  ibv_pd* pd{nullptr}; /* Protection domain on this device. */
  uint16_t lid{0}; /* Local identifier (IB fabrics). */
  ibv_gid gid{}; /* GID for RoCE / IB with GRH. */
  uint8_t gidIndex{3}; /* Resolved GID table index used for this NIC. */
  ibv_mtu mtu{IBV_MTU_4096}; /* Active MTU from port query. */
  int linkLayer{IBV_LINK_LAYER_ETHERNET}; /* IB or Ethernet (RoCE). */
  uint8_t portNum{1}; /* Physical port number on the HCA. */
  bool dmaBufSupported{false}; /* Kernel supports DMA-BUF MR registration. */
  int numaNode{-1}; /* NUMA node of the NIC, from Topology (-1 = unknown). */
  bool dataDirect{false}; /* NIC exposes a data-direct sysfs path (mlx5 DD). */

  /*
   * Full-init constructor.
   *
   * @param configuredGidIndex  Forced RoCE GID index, or -1 to auto-select a
   *                            RoCEv2 entry by scanning the GID table (falls
   *                            back to index 3 when ibv_query_gid_ex is
   *                            unavailable or no RoCEv2 entry is found).
   */
  NicResources(
      ibv_device* device,
      std::shared_ptr<IbvApi> api,
      int numaNode = -1,
      int configuredGidIndex = -1,
      std::optional<uint8_t> port = std::nullopt);

  ~NicResources();

  NicResources(const NicResources&) = delete;
  NicResources& operator=(const NicResources&) = delete;
  NicResources(NicResources&& other) noexcept;
  NicResources& operator=(NicResources&&) = delete;

 private:
  uint8_t findActivePort() const;
  /// Resolve the RoCE GID table index for an Ethernet port. Returns the
  /// configured index when >= 0; otherwise scans for a RoCEv2 entry (preferring
  /// IPv4-mapped); falls back to index 3.
  uint8_t resolveGidIndex(int configuredGidIndex, const ibv_port_attr& portAttr)
      const;
  void cleanup();
  std::shared_ptr<IbvApi> ibvApi{nullptr};
};

/// True if the mlx5 Data-Direct sysfs path (from
/// mlx5dv_get_data_direct_sysfs_path) is in the same PCIe domain as the GPU at
/// gpuPciBusId (nvidia-smi / cudaDeviceGetPCIBusId format, e.g.
/// "00000008:06:00.0"). mlx5 Data-Direct MR registration only succeeds when the
/// NIC's data-direct interface shares the GPU's PCIe domain; the NIC's own
/// physical PCIe location does NOT imply the same data-direct domain.
bool dataDirectDomainMatchesGpu(
    const std::string& ddSysfsPath,
    const std::string& gpuPciBusId);

/// From candidateNics (device names), returns those whose mlx5 Data-Direct
/// interface shares the GPU's PCIe domain (gpuPciBusId). Each candidate is
/// opened to probe its data-direct sysfs path; candidates without a DD path, or
/// in a different domain, are excluded. Input order is preserved. This is the
/// NIC selection required for Data-Direct RDMA.
std::vector<std::string> selectDataDirectNicsForGpu(
    IbvApi& ibvApi,
    const std::vector<std::string>& candidateNics,
    const std::string& gpuPciBusId);

} // namespace uniflow
