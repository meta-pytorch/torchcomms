// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CudaMock.hpp"
#include <cstring>

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch {
namespace comms {
namespace test {

void CudaMock::setupDefaultBehaviors() {
  // Device management - return success by default
  ON_CALL(*this, setDevice(_)).WillByDefault(Return(cudaSuccess));

  ON_CALL(*this, getDeviceCount(_))
      .WillByDefault(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

  ON_CALL(*this, getDeviceProperties(_, _))
      .WillByDefault(
          DoAll(SetArgPointee<0>(cudaDeviceProp{}), Return(cudaSuccess)));

  ON_CALL(*this, memGetInfo(_, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(std::numeric_limits<int>::max()),
          SetArgPointee<1>(std::numeric_limits<int>::max()),
          Return(cudaSuccess)));

  ON_CALL(*this, getDeviceCount(_))
      .WillByDefault(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

  // Stream management - return success by default
  ON_CALL(*this, getStreamPriorityRange(_, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(10), SetArgPointee<1>(-10), Return(cudaSuccess)));

  ON_CALL(*this, streamCreateWithPriority(_, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaStream_t>(0x1)),
          Return(cudaSuccess)));

  ON_CALL(*this, streamDestroy(_)).WillByDefault(Return(cudaSuccess));

  ON_CALL(*this, streamWaitEvent(_, _, _)).WillByDefault(Return(cudaSuccess));

  ON_CALL(*this, getCurrentCUDAStream(_))
      .WillByDefault(Return(reinterpret_cast<cudaStream_t>(0x1)));

  ON_CALL(*this, streamIsCapturing(_, _))
      .WillByDefault(DoAll(
          SetArgPointee<1>(cudaStreamCaptureStatusNone), Return(cudaSuccess)));

  ON_CALL(*this, streamGetCaptureInfo(_, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<1>(cudaStreamCaptureStatusNone),
          SetArgPointee<2>(0ULL),
          Return(cudaSuccess)));

  // CUDA Graph and User Object management - return success by default
  ON_CALL(*this, userObjectCreate(_, _, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaUserObject_t>(0x3000)),
          Return(cudaSuccess)));

  ON_CALL(*this, graphRetainUserObject(_, _, _, _))
      .WillByDefault(Return(cudaSuccess));

  ON_CALL(*this, streamGetCaptureInfo_v2(_, _, _, _, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<1>(cudaStreamCaptureStatusNone),
          SetArgPointee<2>(0ULL),
          SetArgPointee<3>(reinterpret_cast<cudaGraph_t>(0x4000)),
          Return(cudaSuccess)));

  ON_CALL(*this, threadExchangeStreamCaptureMode(_))
      .WillByDefault(Return(cudaSuccess));

  // Memory management - return success by default
  ON_CALL(*this, malloc(_, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<void*>(0x1000)),
          Return(cudaSuccess)));

  ON_CALL(*this, free(_)).WillByDefault(Return(cudaSuccess));

  ON_CALL(*this, memcpyAsync(_, _, _, _, _)).WillByDefault(Return(cudaSuccess));

  // Event management - return success by default
  ON_CALL(*this, eventCreate(_))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0x2000)),
          Return(cudaSuccess)));

  ON_CALL(*this, eventCreateWithFlags(_, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<cudaEvent_t>(0x4000)),
          Return(cudaSuccess)));

  ON_CALL(*this, eventDestroy(_)).WillByDefault(Return(cudaSuccess));

  ON_CALL(*this, eventRecord(_, _)).WillByDefault(Return(cudaSuccess));

  ON_CALL(*this, eventQuery(_)).WillByDefault(Return(cudaSuccess));

  // Error handling - return default error strings
  ON_CALL(*this, getErrorString(_)).WillByDefault(Return("mock error string"));
}

void CudaMock::reset() {
  // Clear all expectations and call counts
  ::testing::Mock::VerifyAndClearExpectations(this);

  // Re-setup default behaviors after reset
  setupDefaultBehaviors();
}

} // namespace test
} // namespace comms
} // namespace torch
