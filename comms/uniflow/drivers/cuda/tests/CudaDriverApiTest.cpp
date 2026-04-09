// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"

#include <gtest/gtest.h>

namespace uniflow {

class CudaDriverApiTest : public ::testing::Test {
 protected:
  CudaDriverApi api;
};

TEST_F(CudaDriverApiTest, InitSucceeds) {
  Status s = api.init();
  ASSERT_FALSE(s.hasError()) << s.error().message();
}

TEST_F(CudaDriverApiTest, DeviceGetValid) {
  CUdevice device;
  Status s = api.cuDeviceGet(&device, 0);
  ASSERT_FALSE(s.hasError()) << s.error().message();
}

TEST_F(CudaDriverApiTest, DeviceGetAttributeComputeCapability) {
  CUdevice device;
  Status s = api.cuDeviceGet(&device, 0);
  ASSERT_FALSE(s.hasError()) << s.error().message();

  int major = -1;
  s = api.cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  ASSERT_FALSE(s.hasError()) << s.error().message();
  EXPECT_GT(major, 0);

  int minor = -1;
  s = api.cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  ASSERT_FALSE(s.hasError()) << s.error().message();
  EXPECT_GE(minor, 0);
}

TEST_F(CudaDriverApiTest, GetErrorStringForSuccess) {
  const char* str = nullptr;
  Status s = api.cuGetErrorString(CUDA_SUCCESS, &str);
  ASSERT_FALSE(s.hasError()) << s.error().message();
  EXPECT_NE(str, nullptr);
}

TEST_F(CudaDriverApiTest, GetErrorNameForSuccess) {
  const char* name = nullptr;
  Status s = api.cuGetErrorName(CUDA_SUCCESS, &name);
  ASSERT_FALSE(s.hasError()) << s.error().message();
  EXPECT_NE(name, nullptr);
}

TEST_F(CudaDriverApiTest, GetErrorStringForInvalidValue) {
  const char* str = nullptr;
  Status s = api.cuGetErrorString(CUDA_ERROR_INVALID_VALUE, &str);
  ASSERT_FALSE(s.hasError()) << s.error().message();
  EXPECT_NE(str, nullptr);
}

TEST_F(CudaDriverApiTest, GetErrorNameForInvalidValue) {
  const char* name = nullptr;
  Status s = api.cuGetErrorName(CUDA_ERROR_INVALID_VALUE, &name);
  ASSERT_FALSE(s.hasError()) << s.error().message();
  EXPECT_NE(name, nullptr);
}

TEST_F(CudaDriverApiTest, MemGetAllocationGranularity) {
  CUdevice device;
  Status s = api.cuDeviceGet(&device, 0);
  ASSERT_FALSE(s.hasError()) << s.error().message();

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  size_t granularity = 0;
  s = api.cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  ASSERT_FALSE(s.hasError()) << s.error().message();
  EXPECT_GT(granularity, 0u);
}

TEST_F(CudaDriverApiTest, VmmAllocateAndFreeRoundTrip) {
  CUdevice device;
  Status s = api.cuDeviceGet(&device, 0);
  ASSERT_FALSE(s.hasError()) << s.error().message();

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  size_t granularity = 0;
  s = api.cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  ASSERT_FALSE(s.hasError()) << s.error().message();

  // Create a physical allocation.
  CUmemGenericAllocationHandle handle;
  s = api.cuMemCreate(&handle, granularity, &prop, 0);
  ASSERT_FALSE(s.hasError()) << s.error().message();

  // Reserve a virtual address range.
  CUdeviceptr ptr = 0;
  s = api.cuMemAddressReserve(&ptr, granularity, granularity, 0, 0);
  ASSERT_FALSE(s.hasError()) << s.error().message();
  EXPECT_NE(ptr, 0u);

  // Map physical memory to virtual address.
  s = api.cuMemMap(ptr, granularity, 0, handle, 0);
  ASSERT_FALSE(s.hasError()) << s.error().message();

  // Set access permissions.
  CUmemAccessDesc accessDesc{};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  s = api.cuMemSetAccess(ptr, granularity, &accessDesc, 1);
  ASSERT_FALSE(s.hasError()) << s.error().message();

  // Tear down: unmap, free address, release allocation.
  s = api.cuMemUnmap(ptr, granularity);
  ASSERT_FALSE(s.hasError()) << s.error().message();

  s = api.cuMemAddressFree(ptr, granularity);
  ASSERT_FALSE(s.hasError()) << s.error().message();

  s = api.cuMemRelease(handle);
  ASSERT_FALSE(s.hasError()) << s.error().message();
}

} // namespace uniflow
