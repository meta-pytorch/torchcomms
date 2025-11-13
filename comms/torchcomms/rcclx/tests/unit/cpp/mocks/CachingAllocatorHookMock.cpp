// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rcclx/tests/unit/cpp/mocks/CachingAllocatorHookMock.hpp"

using ::testing::_;
using ::testing::Return;

namespace torch {
namespace comms {
namespace test {

void CachingAllocatorHookMock::setupDefaultBehaviors() {
  ON_CALL(*this, regDeregMem(_)).WillByDefault(Return());

  ON_CALL(*this, registerComm(_)).WillByDefault([this](TorchCommRCCLX* comm) {
    registered_comms_.insert(comm);
  });

  ON_CALL(*this, deregisterComm(_)).WillByDefault([this](TorchCommRCCLX* comm) {
    registered_comms_.erase(comm);
  });

  ON_CALL(*this, clear()).WillByDefault([this]() {
    registered_comms_.clear();
  });
}

bool CachingAllocatorHookMock::isCommRegistered(TorchCommRCCLX* comm) {
  return registered_comms_.find(comm) != registered_comms_.end();
}

} // namespace test
} // namespace comms
} // namespace torch
