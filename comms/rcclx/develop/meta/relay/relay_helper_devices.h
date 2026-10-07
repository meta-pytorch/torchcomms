/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "nccl.h"

namespace rcclx::relay {

// Helper GPUs for collectives that stage traffic in other GPUs' HBM with no
// process on them: visible devices outside comm that this rank's device can
// reach, filtered by NCCL_RELAY_HELPER_DEVICES (comma-separated PCI bus IDs;
// unset means any). Collective over comm: returns the devices every rank
// found, in PCI bus ID order, capped at maxHelpers, as this process's device
// ordinals (they may differ across processes).
ncclResult_t agreeRelayHelperDevices(
    ncclComm_t comm,
    int maxHelpers,
    std::vector<int>* devices);

// Enables peer access from the current device to `device`, treating "already
// enabled" as success and clearing HIP's sticky error for it. Remote loads and
// stores through IPC mappings take a slow path (~100x) without it, and
// hipIpcMemLazyEnablePeerAccess alone does not enable it.
void enableRelayPeerAccess(int device);

// Makes `device` current for its scope and restores the caller's device.
class RelayDeviceGuard {
 public:
  explicit RelayDeviceGuard(int device);
  ~RelayDeviceGuard();
  RelayDeviceGuard(const RelayDeviceGuard&) = delete;
  RelayDeviceGuard& operator=(const RelayDeviceGuard&) = delete;

 private:
  int previous_{-1};
};

} // namespace rcclx::relay
