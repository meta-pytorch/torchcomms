// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaDeviceAdapter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/drivers/cuda/mock/MockCudaDriverApi.h"

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

TEST(CudaDeviceAdapterTest, IsDmaBuffSupportedDelegatesToDriverApi) {
  auto cudaApi = std::make_shared<NiceMock<MockCudaApi>>();
  auto driverApi = std::make_shared<NiceMock<MockCudaDriverApi>>();
  EXPECT_CALL(*driverApi, isDmaBufSupported(3)).WillOnce(Return(true));

  CudaDeviceAdapter adapter(cudaApi, driverApi);

  auto res = adapter.isDmaBuffSupported(3);
  ASSERT_FALSE(res.hasError());
  EXPECT_TRUE(res.value());
}

TEST(CudaDeviceAdapterTest, ExportDmaBuffPopulatesFdOffsetAndIova) {
  auto cudaApi = std::make_shared<NiceMock<MockCudaApi>>();
  auto driverApi = std::make_shared<NiceMock<MockCudaDriverApi>>();
  constexpr int kFakeFd = 42;
  EXPECT_CALL(
      *driverApi,
      cuMemGetHandleForAddressRange(
          _, _, _, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0))
      .WillOnce([&](void* handle,
                    CUdeviceptr,
                    size_t,
                    CUmemRangeHandleType,
                    unsigned long long) {
        *static_cast<int*>(handle) = kFakeFd;
        return Ok();
      });

  CudaDeviceAdapter adapter(cudaApi, driverApi);

  // Use a real heap-aligned allocation as the base, then offset into it to
  // get a non-page-aligned pointer. posix_memalign + free keeps the
  // allocate/free pair symmetric (mixing aligned `new[]` with default
  // `delete[]` would be UB).
  const size_t pageSize = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  void* backingRaw = nullptr;
  ASSERT_EQ(::posix_memalign(&backingRaw, pageSize, pageSize * 2), 0);
  std::unique_ptr<void, decltype(&::free)> backing(backingRaw, &::free);
  void* ptr = static_cast<std::byte*>(backing.get()) + 128;

  auto res = adapter.exportDmaBuff(/*deviceId=*/0, ptr, /*len=*/4096);
  ASSERT_FALSE(res.hasError());
  EXPECT_EQ(res->fd, kFakeFd);
  EXPECT_EQ(res->offset, 128u);
  EXPECT_EQ(res->len, 4096u);
  EXPECT_EQ(res->iova, reinterpret_cast<uint64_t>(ptr));
}

TEST(CudaDeviceAdapterTest, ExportDmaBuffRejectsNullPtr) {
  auto cudaApi = std::make_shared<NiceMock<MockCudaApi>>();
  auto driverApi = std::make_shared<NiceMock<MockCudaDriverApi>>();
  CudaDeviceAdapter adapter(cudaApi, driverApi);

  auto res = adapter.exportDmaBuff(/*deviceId=*/0, nullptr, /*len=*/4096);
  EXPECT_TRUE(res.hasError());
}

TEST(CudaDeviceAdapterTest, CloseDmaBuffIsNoopForUnsetFd) {
  auto cudaApi = std::make_shared<NiceMock<MockCudaApi>>();
  auto driverApi = std::make_shared<NiceMock<MockCudaDriverApi>>();
  CudaDeviceAdapter adapter(cudaApi, driverApi);

  DmaBuff buff;
  EXPECT_FALSE(adapter.closeDmaBuff(buff).hasError());
  EXPECT_EQ(buff.fd, -1);
}

} // namespace uniflow
