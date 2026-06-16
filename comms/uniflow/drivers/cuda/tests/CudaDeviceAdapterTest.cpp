// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaDeviceAdapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"

namespace uniflow {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

namespace {

constexpr size_t kSize = 4096;

} // namespace

TEST(CudaDeviceAdapterTest, PinnedHostAllocPassesMappedPortableFlags) {
  // pinnedHostAlloc must request mapped + portable so the returned host
  // pointer is usable for both DMA from a NIC and host-mapped device
  // access from any CUDA context. Verifying the flag argument is the
  // observable contract that callers (RdmaSlabPool, RdmaTransport) rely
  // on -- a bare delegation/round-trip check would not catch a regression
  // in those flags.
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int sentinel = 0;
  EXPECT_CALL(
      *mock, hostAlloc(kSize, cudaHostAllocMapped | cudaHostAllocPortable))
      .WillOnce(Return(Result<void*>(&sentinel)));

  CudaDeviceAdapter adapter(mock);

  auto res = adapter.pinnedHostAlloc(kSize);
  ASSERT_FALSE(res.hasError());
}

TEST(CudaDeviceAdapterTest, PinnedHostAllocPropagatesError) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  EXPECT_CALL(*mock, hostAlloc(kSize, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "oom")));

  CudaDeviceAdapter adapter(mock);

  auto res = adapter.pinnedHostAlloc(kSize);
  ASSERT_TRUE(res.hasError());
  EXPECT_EQ(res.error().code(), ErrCode::DriverError);
}

TEST(CudaDeviceAdapterTest, PinnedHostFreeDelegatesToCudaApi) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int sentinel = 0;
  void* fakePtr = &sentinel;
  EXPECT_CALL(*mock, hostFree(fakePtr)).WillOnce(Return(Ok()));

  CudaDeviceAdapter adapter(mock);

  EXPECT_FALSE(adapter.pinnedHostFree(fakePtr).hasError());
}

} // namespace uniflow
