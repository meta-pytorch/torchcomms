// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "NcclxMock.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch {
namespace comms {
namespace test {

void NcclxMock::setupDefaultBehaviors() {
  // Error handling - return default error strings
  ON_CALL(*this, getErrorString(_))
      .WillByDefault(Return("mock nccl error string"));

  // Unique ID generation - return success and set a mock unique ID
  ON_CALL(*this, getUniqueId(_))
      .WillByDefault(
          DoAll(SetArgPointee<0>(ncclUniqueId{}), Return(ncclSuccess)));

  // Communicator management - return success by default
  ON_CALL(*this, commInitRankConfig(_, _, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<ncclComm_t>(0x3000)),
          Return(ncclSuccess)));

  ON_CALL(*this, commDestroy(_)).WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, commAbort(_)).WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, commGetAsyncError(_, _))
      .WillByDefault(DoAll(SetArgPointee<1>(ncclSuccess), Return(ncclSuccess)));

  ON_CALL(*this, commSplit(_, _, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<3>(reinterpret_cast<ncclComm_t>(0x4000)),
          Return(ncclSuccess)));

  // Memory registration - return success by default
  ON_CALL(*this, commRegister(_, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<3>(reinterpret_cast<void*>(0x5000)),
          Return(ncclSuccess)));

  ON_CALL(*this, commDeregister(_, _)).WillByDefault(Return(ncclSuccess));

  // Point-to-point operations - return success by default
  ON_CALL(*this, send(_, _, _, _, _, _)).WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, recv(_, _, _, _, _, _)).WillByDefault(Return(ncclSuccess));

  // Collective operations - return success by default
  ON_CALL(*this, broadcast(_, _, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, bcast(_, _, _, _, _, _)).WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, allReduce(_, _, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, reduce(_, _, _, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, allGather(_, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, reduceScatter(_, _, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, winAllocate(_, _, _, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<3>(reinterpret_cast<NcclxWindow>(0x5000)),
          Return(ncclSuccess)));
  ON_CALL(*this, winFree(_, _)).WillByDefault(Return(ncclSuccess));
  ON_CALL(*this, winPut(_, _, _, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));
  ON_CALL(*this, winSharedQuery(_, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<3>(reinterpret_cast<void*>(0x5000)),
          Return(ncclSuccess)));
  ON_CALL(*this, winSignal(_, _, _)).WillByDefault(Return(ncclSuccess));
  ON_CALL(*this, winWaitSignal(_, _, _)).WillByDefault(Return(ncclSuccess));

  // Group operations - return success by default
  ON_CALL(*this, groupStart()).WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, groupEnd()).WillByDefault(Return(ncclSuccess));

  ON_CALL(*this, commUserRank(_, _))
      .WillByDefault(DoAll(SetArgPointee<1>(0), Return(ncclSuccess)));
  ON_CALL(*this, commCount(_, _))
      .WillByDefault(DoAll(SetArgPointee<1>(1), Return(ncclSuccess)));

  ON_CALL(*this, redOpCreatePreMulSum(_, _, _, _, _))
      .WillByDefault(Return(ncclSuccess));
  ON_CALL(*this, redOpDestroy(_, _)).WillByDefault(Return(ncclSuccess));
}

void NcclxMock::reset() {
  // Clear all expectations and call counts
  ::testing::Mock::VerifyAndClearExpectations(this);

  // Re-setup default behaviors after reset
  setupDefaultBehaviors();
}

} // namespace test
} // namespace comms
} // namespace torch
