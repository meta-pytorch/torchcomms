// Copyright (c) Meta Platforms, Inc. and affiliates.

#define __HIP_PLATFORM_AMD__
#include "comms/torchcomms/rccl/tests/unit/cpp/mocks/RcclMock.hpp"

#include <rccl.h> // @manual

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch {
namespace comms {
namespace test {

void RcclMock::setupDefaultBehaviors() {
  // Default communicator operations
  ON_CALL(*this, getUniqueId(_)).WillByDefault([](ncclUniqueId* uniqueId) {
    memset(uniqueId, 0x42, sizeof(ncclUniqueId));
    return ncclSuccess;
  });

  ON_CALL(*this, commInitRankConfig(_, _, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  ON_CALL(*this, commDestroy(_)).WillByDefault(Return(ncclSuccess));
  ON_CALL(*this, commAbort(_)).WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, commSplit(_, _, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<3>(reinterpret_cast<ncclComm_t>(0x4000)),
          Return(ncclSuccess)));

  ON_CALL(*this, commCount(_, _))
      .WillByDefault(DoAll(SetArgPointee<1>(2), Return(ncclSuccess)));

  ON_CALL(*this, commUserRank(_, _))
      .WillByDefault(DoAll(SetArgPointee<1>(0), Return(ncclSuccess)));

  ON_CALL(*this, commGetAsyncError(_, _))
      .WillByDefault(DoAll(SetArgPointee<1>(ncclSuccess), Return(ncclSuccess)));

  // Default memory registration operations
  ON_CALL(*this, commRegister(_, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<3>(reinterpret_cast<void*>(0x5000)),
          Return(ncclSuccess)));

  ON_CALL(*this, commDeregister(_, _)).WillByDefault(Return(ncclSuccess));

  // Default collective operations
  ON_CALL(*this, allReduce(_, _, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, broadcast(_, _, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, reduce(_, _, _, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, allGather(_, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, reduceScatter(_, _, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));

  // Default point-to-point operations
  ON_CALL(*this, send(_, _, _, _, _, _)).WillByDefault(Return(ncclSuccess));
  ON_CALL(*this, recv(_, _, _, _, _, _)).WillByDefault(Return(ncclSuccess));

  // Default group operations
  ON_CALL(*this, groupStart()).WillByDefault(Return(ncclSuccess));
  ON_CALL(*this, groupEnd()).WillByDefault(Return(ncclSuccess));

  // Default error handling
  ON_CALL(*this, getErrorString(_)).WillByDefault(Return("Success"));
}

} // namespace test
} // namespace comms
} // namespace torch
