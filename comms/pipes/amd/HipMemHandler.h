// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// HipMemHandler — AMD GPU memory sharing via HIP IPC
// =============================================================================
//
// AMD equivalent of GpuMemHandler. Provides GPU memory allocation and
// cross-process sharing using hipIpcGetMemHandle / hipIpcOpenMemHandle.
//
// Same public API as GpuMemHandler:
//   - Constructor allocates GPU memory
//   - exchangeMemPtrs() collectively exchanges IPC handles
//   - getLocalDeviceMemPtr() / getPeerDeviceMemPtr() for access

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"

namespace comms::pipes {

class HipMemHandler {
 public:
  HipMemHandler(
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      int32_t selfRank,
      int32_t nRanks,
      size_t size);

  // Filtered mode: only open IPC handles for localPeerRanks.
  // AllGather still involves all nRanks for coordination, but
  // hipIpcOpenMemHandle is skipped for non-local peers.
  // getPeerDeviceMemPtr() returns nullptr for non-local peers.
  HipMemHandler(
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      int32_t selfRank,
      int32_t nRanks,
      size_t size,
      const std::vector<int32_t>& localPeerRanks);

  ~HipMemHandler();

  HipMemHandler(const HipMemHandler&) = delete;
  HipMemHandler& operator=(const HipMemHandler&) = delete;

  // Collective: all ranks must call
  void exchangeMemPtrs();

  void* getLocalDeviceMemPtr() const;
  void* getPeerDeviceMemPtr(int32_t rank) const;

  size_t getAllocatedSize() const {
    return allocatedSize_;
  }

 private:
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  const int32_t selfRank_;
  const int32_t nRanks_;

  size_t allocatedSize_{0};
  bool exchanged_{false};
  bool filtered_{false};
  std::vector<int32_t> localPeerRanks_;

  void* localPtr_{nullptr};
  hipIpcMemHandle_t localHandle_{};
  std::vector<void*> peerPtrs_;
};

} // namespace comms::pipes
