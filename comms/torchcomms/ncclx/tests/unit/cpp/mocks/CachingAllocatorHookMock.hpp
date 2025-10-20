// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>
#include "comms/torchcomms/ncclx/TorchCommNCCLXCCA.hpp"

namespace torch {
namespace comms {
namespace test {

/**
 * Mock implementation of CachingAllocatorHookImpl using Google Mock.
 * This class provides mock implementations of all CachingAllocatorHookImpl
 * operations for testing purposes.
 */
class CachingAllocatorHookMock : public CachingAllocatorHookImpl {
 public:
  CachingAllocatorHookMock() = default;
  virtual ~CachingAllocatorHookMock() override = default;

  MOCK_METHOD(
      void,
      regDeregMem,
      (const c10::cuda::CUDACachingAllocator::TraceEntry& te),
      (override));
  MOCK_METHOD(void, registerComm, (TorchCommNCCLX * comm), (override));
  MOCK_METHOD(void, deregisterComm, (TorchCommNCCLX * comm), (override));
  MOCK_METHOD(void, clear, (), (override));

  /**
   * Set up default behaviors for common operations.
   * This method configures the mock to provide reasonable default behaviors.
   */
  void setupDefaultBehaviors();

  /**
   * Reset all mock expectations and call counts.
   */
  void reset();

  /**
   * Check if a communicator is registered with this hook.
   * @param comm Pointer to the communicator to check
   * @return true if the communicator is registered, false otherwise
   */
  bool isCommRegistered(TorchCommNCCLX* comm) override;

 private:
  std::set<TorchCommNCCLX*> registered_comms_;
};

} // namespace test
} // namespace comms
} // namespace torch
