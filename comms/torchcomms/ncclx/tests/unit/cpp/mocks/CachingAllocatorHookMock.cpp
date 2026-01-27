// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CachingAllocatorHookMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;

namespace torch::comms::test {

void CachingAllocatorHookMock::setupDefaultBehaviors() {
  // Set up default behavior for registerComm
  ON_CALL(*this, registerComm(_)).WillByDefault([this](TorchCommNCCLX* comm) {
    registered_comms_.insert(comm);
  });

  // Set up default behavior for deregisterComm
  ON_CALL(*this, deregisterComm(_)).WillByDefault([this](TorchCommNCCLX* comm) {
    registered_comms_.erase(comm);
  });

  // Set up default behavior for regDeregMem (no-op by default)
  ON_CALL(*this, regDeregMem(_)).WillByDefault(Return());

  // Set up default behavior for clear
  ON_CALL(*this, clear()).WillByDefault([this]() {
    registered_comms_.clear();
  });
}

void CachingAllocatorHookMock::reset() {
  // Clear all expectations and call counts
  ::testing::Mock::VerifyAndClearExpectations(this);

  // Clear the registered communicators set
  registered_comms_.clear();

  // Re-setup default behaviors after reset
  setupDefaultBehaviors();
}

bool CachingAllocatorHookMock::isCommRegistered(TorchCommNCCLX* comm) {
  return registered_comms_.find(comm) != registered_comms_.end();
}

} // namespace torch::comms::test
