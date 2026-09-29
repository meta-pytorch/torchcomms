// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <unistd.h>

#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/memory/CuMemMapping.h"
#include "comms/prims/memory/NvlMemExchange.h"
#include "comms/prims/platform/CudaDriverLazy.h"

namespace comms::prims::tests {
namespace {

bool vmmDevice(CUdevice& cuDev, int& cudaDevice) {
#if CUDART_VERSION < 12030
  (void)cuDev;
  (void)cudaDevice;
  return false;
#else
  if (cuda_driver_lazy_init() != 0) {
    return false;
  }
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
    return false;
  }
  cudaDevice = 0;
  if (cudaSetDevice(cudaDevice) != cudaSuccess) {
    return false;
  }
  return pfn_cuDeviceGet(&cuDev, cudaDevice) == CUDA_SUCCESS;
#endif
}

void* asPtr(CUdeviceptr p) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer type
  return reinterpret_cast<void*>(p);
}

} // namespace

TEST(NvlMemExchangeTest, ToCudaHandleTypeMapsAndThrows) {
#if CUDART_VERSION >= 12030
  EXPECT_EQ(
      toCudaHandleType(ShareableHandleType::kFabric),
      CU_MEM_HANDLE_TYPE_FABRIC);
  EXPECT_EQ(
      toCudaHandleType(ShareableHandleType::kPosixFd),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  EXPECT_THROW(
      toCudaHandleType(ShareableHandleType::kUnsupported), std::runtime_error);
#else
  GTEST_SKIP() << "CUDA 12.3+ required";
#endif
}

TEST(NvlMemExchangeTest, MakeVmmAllocationPropFields) {
  CUdevice cuDev = 0;
  int cudaDevice = 0;
  if (!vmmDevice(cuDev, cudaDevice)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }
#if CUDART_VERSION >= 12030
  const unsigned int mask = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  auto prop = makeVmmAllocationProp(cuDev, mask);
  EXPECT_EQ(prop.type, CU_MEM_ALLOCATION_TYPE_PINNED);
  EXPECT_EQ(prop.location.type, CU_MEM_LOCATION_TYPE_DEVICE);
  EXPECT_EQ(prop.location.id, cuDev);
  EXPECT_EQ(static_cast<unsigned int>(prop.requestedHandleTypes), mask);
#endif
}

TEST(NvlMemExchangeTest, SelectShareableHandleTypeReturnsSupported) {
  CUdevice cuDev = 0;
  int cudaDevice = 0;
  if (!vmmDevice(cuDev, cudaDevice)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }
  const auto type = selectShareableHandleType(cudaDevice);
  // On a Hopper+ GPU this is fabric or POSIX FD; on older GPUs it may be
  // unsupported, in which case there is nothing to assert.
  if (type == ShareableHandleType::kUnsupported) {
    GTEST_SKIP() << "no shareable handle type supported on this device";
  }
  EXPECT_TRUE(
      type == ShareableHandleType::kFabric ||
      type == ShareableHandleType::kPosixFd);
}

TEST(NvlMemExchangeTest, PosixFdExportImportRoundTripSharesMemory) {
  CUdevice cuDev = 0;
  int cudaDevice = 0;
  if (!vmmDevice(cuDev, cudaDevice)) {
    GTEST_SKIP() << "VMM / CUDA 12.3+ unavailable";
  }
#if CUDART_VERSION >= 12030
  std::shared_ptr<CuMemAllocation> original = CuMemAllocation::create(
      cuDev, /*size=*/4096, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);

  // Export then re-import within this process (the importer duplicates the
  // exporter's fd via pidfd_getfd). Skip if the sandbox forbids pidfd_getfd.
  ShareableHandle shareable;
  CUmemGenericAllocationHandle importedHandle = 0;
  try {
    shareable = exportShareableHandle(
        original->handle(), ShareableHandleType::kPosixFd);
    importedHandle = importShareableHandle(shareable);
  } catch (const std::runtime_error& ex) {
    if (shareable.type == ShareableHandleType::kPosixFd && shareable.fd >= 0) {
      close(shareable.fd);
    }
    GTEST_SKIP() << "posix-fd export/import unavailable: " << ex.what();
  }
  if (shareable.fd >= 0) {
    close(shareable.fd);
  }

  std::shared_ptr<CuMemAllocation> imported =
      CuMemAllocation::adopt(importedHandle, cuDev, original->size());

  auto mOrig = CuMemMapping::overAllocation(
      original, original->size(), original->granularity());
  auto mImported = CuMemMapping::overAllocation(
      imported, imported->size(), imported->granularity());

  ASSERT_EQ(
      cudaMemset(asPtr(mOrig.devicePtr()), 0x3D, original->size()),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> host(original->size(), 0);
  ASSERT_EQ(
      cudaMemcpy(
          host.data(),
          asPtr(mImported.devicePtr()),
          original->size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (auto b : host) {
    ASSERT_EQ(b, 0x3D)
        << "imported handle must map the same physical memory as the exporter";
  }
#endif
}

} // namespace comms::prims::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
