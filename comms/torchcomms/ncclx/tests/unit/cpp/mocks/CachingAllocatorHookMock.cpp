// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CachingAllocatorHookMock.hpp"

using ::testing::_;
using ::testing::Return;

namespace torch::comms::test {

void CachingAllocatorHookMock::setupDefaultBehaviors() {
  // Set up default behavior for regDeregMem (no-op by default)
  ON_CALL(*this, regDeregMem(_)).WillByDefault(Return());

  // Set up default behavior for registerMemPreHook - set our local flag
  ON_CALL(*this, registerMemPreHook()).WillByDefault([this]() {
    mem_pre_hook_registered_ = true;
  });

  // Call registerMemPreHook to simulate what DefaultCachingAllocatorHookImpl
  // constructor does
  registerMemPreHook();
}

void CachingAllocatorHookMock::reset() {
  // Clear all expectations and call counts
  ::testing::Mock::VerifyAndClearExpectations(this);

  // Reset the mem pre hook flag
  mem_pre_hook_registered_ = false;

  // Re-setup default behaviors after reset
  setupDefaultBehaviors();
}

bool CachingAllocatorHookMock::isMemPreHookRegistered() {
  return mem_pre_hook_registered_;
}

} // namespace torch::comms::test
