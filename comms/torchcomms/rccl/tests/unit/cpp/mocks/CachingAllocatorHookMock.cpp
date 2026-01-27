// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rccl/tests/unit/cpp/mocks/CachingAllocatorHookMock.hpp"

using ::testing::_;
using ::testing::Return;

namespace torch::comms::test {

void CachingAllocatorHookMock::setupDefaultBehaviors() {
  ON_CALL(*this, regDeregMem(_)).WillByDefault(Return());

  ON_CALL(*this, registerComm(_)).WillByDefault([this](TorchCommRCCL* comm) {
    registered_comms_.insert(comm);
  });

  ON_CALL(*this, deregisterComm(_)).WillByDefault([this](TorchCommRCCL* comm) {
    registered_comms_.erase(comm);
  });

  ON_CALL(*this, clear()).WillByDefault([this]() {
    registered_comms_.clear();
  });
}

bool CachingAllocatorHookMock::isCommRegistered(TorchCommRCCL* comm) {
  return registered_comms_.find(comm) != registered_comms_.end();
}

} // namespace torch::comms::test
