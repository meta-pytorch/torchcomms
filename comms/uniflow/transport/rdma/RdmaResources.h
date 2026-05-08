// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

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
  ibv_mtu mtu{IBV_MTU_4096}; /* Active MTU from port query. */
  int linkLayer{IBV_LINK_LAYER_ETHERNET}; /* IB or Ethernet (RoCE). */
  uint8_t portNum{1}; /* Physical port number on the HCA. */
  bool dmaBufSupported{false}; /* Kernel supports DMA-BUF MR registration. */

  NicResources(
      ibv_device* device,
      std::shared_ptr<IbvApi> api,
      uint8_t gidIndex = 3,
      std::optional<uint8_t> port = std::nullopt);

  ~NicResources();

  NicResources(const NicResources&) = delete;
  NicResources& operator=(const NicResources&) = delete;
  NicResources(NicResources&& other) noexcept;
  NicResources& operator=(NicResources&&) = delete;

 private:
  uint8_t findActivePort() const;
  std::shared_ptr<IbvApi> ibvApi{nullptr};
};

} // namespace uniflow
