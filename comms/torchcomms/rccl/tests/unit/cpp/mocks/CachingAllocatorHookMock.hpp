// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>
#include <unordered_set>

#include "comms/torchcomms/rccl/TorchCommRCCLCCA.hpp"

namespace torch {
namespace comms {
namespace test {

class CachingAllocatorHookMock : public CachingAllocatorHookImpl {
 public:
  MOCK_METHOD(
      void,
      regDeregMem,
      (const c10::hip::HIPCachingAllocator::TraceEntry& entry),
      (override));
  MOCK_METHOD(void, registerComm, (TorchCommRCCL * comm), (override));
  MOCK_METHOD(void, deregisterComm, (TorchCommRCCL * comm), (override));
  MOCK_METHOD(void, clear, (), (override));

  // Helper methods for testing
  void setupDefaultBehaviors();

  bool isCommRegistered(TorchCommRCCL* comm) override;

  /**
   * Reset all mock expectations and call counts.
   */
  void reset();

 private:
  std::unordered_set<void*> registered_comms_;
};

} // namespace test
} // namespace comms
} // namespace torch
