// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDevicePtr.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy
#include <unistd.h>

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

// The methods under test exist only in the AMD build of CudaDriverApi.
#if defined(__HIP_PLATFORM_AMD__)

namespace uniflow {
namespace {

constexpr int kDevice = 0;

CUmemLocation deviceLocation(int device) {
  CUmemLocation location{};
  location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  location.id = device;
  return location;
}

// One physical chunk shareable as a POSIX fd, mapped read-write for the
// device. The destructor unwinds whatever alloc() set up.
class VmmChunk {
 public:
  explicit VmmChunk(CudaDriverApi& driver) : driver_{driver} {}

  ~VmmChunk() {
    if (mapped_) {
      (void)driver_.cuMemUnmap(ptr_, size_);
    }
    if (reserved_) {
      (void)driver_.cuMemAddressFree(ptr_, size_);
    }
    if (created_) {
      (void)driver_.cuMemRelease(handle_);
    }
  }

  VmmChunk(const VmmChunk&) = delete;
  VmmChunk& operator=(const VmmChunk&) = delete;

  Status alloc(int device) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location = deviceLocation(device);
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CHECK_EXPR(driver_.cuMemGetAllocationGranularity(
        &size_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CHECK_EXPR(driver_.cuMemCreate(&handle_, size_, &prop, 0));
    created_ = true;
    CHECK_EXPR(
        driver_.cuMemAddressReserve(&ptr_, size_, size_, CUdeviceptr{}, 0));
    reserved_ = true;
    CHECK_EXPR(driver_.cuMemMap(ptr_, size_, 0, handle_, 0));
    mapped_ = true;
    CUmemAccessDesc desc{};
    desc.location = deviceLocation(device);
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return driver_.cuMemSetAccess(ptr_, size_, &desc, 1);
  }

  CUdeviceptr ptr() const {
    return ptr_;
  }

  CUmemGenericAllocationHandle handle() const {
    return handle_;
  }

 private:
  CudaDriverApi& driver_;
  CUmemGenericAllocationHandle handle_{};
  CUdeviceptr ptr_{};
  size_t size_{0};
  bool created_{false};
  bool reserved_{false};
  bool mapped_{false};
};

class CudaDriverApiVmmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "no GPU available";
    }
    ASSERT_EQ(cudaSetDevice(kDevice), cudaSuccess);
    // Each case observes only the last error its own calls leave behind.
    (void)cudaGetLastError();
  }

  CudaDriverApi driver_;
};

TEST_F(CudaDriverApiVmmTest, GetAccessReportsGrantedFlags) {
  VmmChunk chunk(driver_);
  const Status allocated = chunk.alloc(kDevice);
  ASSERT_FALSE(allocated.hasError()) << allocated.error().message();

  unsigned long long flags = 0;
  const CUmemLocation location = deviceLocation(kDevice);
  const Status access = driver_.cuMemGetAccess(&flags, &location, chunk.ptr());

  ASSERT_FALSE(access.hasError()) << access.error().message();
  EXPECT_EQ(
      flags,
      static_cast<unsigned long long>(CU_MEM_ACCESS_FLAGS_PROT_READWRITE));
}

// Imports run one after the other, never holding two at once. On AMD the
// first resolves the osHandle convention and the second takes the cached one.
TEST_F(CudaDriverApiVmmTest, ImportsExportedChunkTwiceInTurn) {
  VmmChunk chunk(driver_);
  const Status allocated = chunk.alloc(kDevice);
  ASSERT_FALSE(allocated.hasError()) << allocated.error().message();
  int fd = -1;
  const Status exported = driver_.cuMemExportToShareableHandle(
      &fd, chunk.handle(), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  ASSERT_FALSE(exported.hasError()) << exported.error().message();
  const std::unique_ptr<int, void (*)(int*)> fdGuard(
      &fd, [](int* f) { ::close(*f); });

  for (const char* round : {"first import", "second import"}) {
    SCOPED_TRACE(round);
    CUmemGenericAllocationHandle imported{};
    const Status importedStatus = driver_.importPosixFd(&imported, fd);
    ASSERT_FALSE(importedStatus.hasError()) << importedStatus.error().message();
    // A by-value runtime rejects the pointer attempt first; that failure must
    // not stay behind as the last error.
    EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
    const Status released = driver_.cuMemRelease(imported);
    ASSERT_FALSE(released.hasError()) << released.error().message();
  }
}

// HIP records a failed call as the runtime's last error; CUDA does not, so only
// AMD needs the scrub the cases below pin.

// cudaMalloc'd memory, freed at scope exit.
using DeviceBuffer = std::unique_ptr<void, void (*)(void*)>;

DeviceBuffer mallocDevice(size_t bytes) {
  void* ptr = nullptr;
  if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
    return DeviceBuffer(nullptr, [](void*) {});
  }
  return DeviceBuffer(ptr, [](void* p) { (void)cudaFree(p); });
}

Status getAccess(CudaDriverApi& driver, const DeviceBuffer& buffer) {
  unsigned long long flags = 0;
  const CUmemLocation location = deviceLocation(kDevice);
  return driver.cuMemGetAccess(
      &flags, &location, toDevicePtr(reinterpret_cast<uint64_t>(buffer.get())));
}

// The P2P classifier routes cudaMalloc'd segments to IPC on this miss, which
// must not fail the caller's next launch check.
TEST_F(CudaDriverApiVmmTest, GetAccessMissLeavesNoLastError) {
  const DeviceBuffer buffer = mallocDevice(4096);
  ASSERT_NE(buffer, nullptr);

  const Status access = getAccess(driver_, buffer);

  EXPECT_TRUE(access.hasError());
  EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

TEST_F(CudaDriverApiVmmTest, GetAccessMissKeepsPendingError) {
  const DeviceBuffer buffer = mallocDevice(4096);
  ASSERT_NE(buffer, nullptr);
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  ASSERT_NE(cudaSetDevice(count), cudaSuccess);
  ASSERT_NE(cudaPeekAtLastError(), cudaSuccess);

  const Status access = getAccess(driver_, buffer);

  EXPECT_TRUE(access.hasError());
  // HIP may replace the pending code with the miss's own, so only that an
  // error is still pending is checked.
  EXPECT_NE(cudaPeekAtLastError(), cudaSuccess);
  (void)cudaGetLastError();
}

// The P2P transport falls back to IPC after a VMM export failure, so a failed
// IPC export must not fail the caller's next launch check either.
TEST_F(CudaDriverApiVmmTest, IpcGetMissLeavesNoLastError) {
  CudaApi api;
  int host = 0;

  EXPECT_TRUE(api.ipcGetMemHandle(&host).hasError());
  EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
}

TEST_F(CudaDriverApiVmmTest, IpcGetMissKeepsPendingError) {
  CudaApi api;
  int host = 0;
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  ASSERT_NE(cudaSetDevice(count), cudaSuccess);
  ASSERT_NE(cudaPeekAtLastError(), cudaSuccess);

  EXPECT_TRUE(api.ipcGetMemHandle(&host).hasError());
  EXPECT_NE(cudaPeekAtLastError(), cudaSuccess);
  (void)cudaGetLastError();
}

} // namespace
} // namespace uniflow

#endif
