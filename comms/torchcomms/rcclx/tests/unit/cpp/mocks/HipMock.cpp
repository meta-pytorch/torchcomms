// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rcclx/tests/unit/cpp/mocks/HipMock.hpp"

#include <hip/hip_runtime.h>

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace torch {
namespace comms {
namespace test {

void HipMock::setupDefaultBehaviors() {
  // Default device operations
  ON_CALL(*this, setDevice(_)).WillByDefault(Return(hipSuccess));

  ON_CALL(*this, getDeviceCount(_))
      .WillByDefault(DoAll(SetArgPointee<0>(1), Return(hipSuccess)));

  ON_CALL(*this, streamCreateWithPriority(_, _, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<hipStream_t>(0x1000)),
          Return(hipSuccess)));
  ON_CALL(*this, streamDestroy(_)).WillByDefault(Return(hipSuccess));
  ON_CALL(*this, streamSynchronize(_)).WillByDefault(Return(hipSuccess));

  ON_CALL(*this, threadExchangeStreamCaptureMode(_))
      .WillByDefault(Return(hipSuccess));

  // Default event operations
  ON_CALL(*this, eventCreate(_))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<hipEvent_t>(0x2000)),
          Return(hipSuccess)));

  ON_CALL(*this, eventDestroy(_)).WillByDefault(Return(hipSuccess));
  ON_CALL(*this, eventRecord(_, _)).WillByDefault(Return(hipSuccess));
  ON_CALL(*this, eventQuery(_)).WillByDefault(Return(hipSuccess));

  // Default memory operations
  ON_CALL(*this, malloc(_, _))
      .WillByDefault(DoAll(
          SetArgPointee<0>(reinterpret_cast<void*>(0x3000)),
          Return(hipSuccess)));

  ON_CALL(*this, free(_)).WillByDefault(Return(hipSuccess));

  ON_CALL(*this, memcpyAsync(_, _, _, _, _)).WillByDefault(Return(hipSuccess));

  // Default error handling
  ON_CALL(*this, getErrorString(_)).WillByDefault(Return("Success"));

  // Default priority range
  ON_CALL(*this, getStreamPriorityRange(_, _))
      .WillByDefault(
          DoAll(SetArgPointee<0>(0), SetArgPointee<1>(-1), Return(hipSuccess)));

  // Default device properties
  ON_CALL(*this, getDeviceProperties(_, _))
      .WillByDefault([](hipDeviceProp_t* prop, int /*device*/) {
        memset(prop, 0, sizeof(hipDeviceProp_t));
        strcpy(prop->name, "Mock HIP Device");
        prop->totalGlobalMem = 8ULL * 1024 * 1024 * 1024; // 8GB
        prop->multiProcessorCount = 64;
        prop->maxThreadsPerBlock = 1024;
        prop->warpSize = 64;
        return hipSuccess;
      });
}

} // namespace test
} // namespace comms
} // namespace torch
