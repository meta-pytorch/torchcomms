// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>
#include "comms/torchcomms/ncclx/TorchCommNCCLXCCA.hpp"

namespace torch::comms::test {

/**
 * Mock implementation of CachingAllocatorHookImpl using Google Mock.
 * This class provides mock implementations of all CachingAllocatorHookImpl
 * operations for testing purposes.
 * Note: Inherits from base interface to avoid CUDA initialization in Default.
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
  MOCK_METHOD(void, registerMemPreHook, (), (override));
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

  /**
   * Check if registerMemPreHook was called.
   * @return true if registerMemPreHook was called, false otherwise
   */
  bool isMemRegisteredCalled();

 private:
  std::set<TorchCommNCCLX*> registered_comms_;
  bool mem_pre_hook_registered_ = false;
};

} // namespace torch::comms::test
