// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaApi.h"

#include <gtest/gtest.h>

#include <memory>

namespace uniflow {
namespace {

// The IPC / arch wrappers are thin pass-throughs to the runtime, so the real
// coverage is a GPU round-trip. Skip when no device is present so the target
// still builds/links and passes on CPU-only hosts, and runs for real on GPU CI.
bool hasGpu(CudaApi& api) {
  auto count = api.getDeviceCount();
  return count.hasValue() && count.value() > 0;
}

// Pins the neutral wire size to the driver ABI. The same invariant is enforced
// at compile time in CudaApi.cpp; this gives a CPU-runnable guard too.
TEST(CudaApiTest, IpcMemHandleSizeMatchesAbi) {
  static_assert(CudaApi::kIpcMemHandleSize == 64);
  EXPECT_EQ(sizeof(CudaApi::IpcMemHandle), CudaApi::kIpcMemHandleSize);
}

TEST(CudaApiTest, GetDeviceArchReturnsNonEmpty) {
  CudaApi api;
  if (!hasGpu(api)) {
    GTEST_SKIP() << "no GPU available";
  }
  ASSERT_FALSE(api.setDevice(0).hasError());

  auto arch = api.getDeviceArch(0);
  ASSERT_FALSE(arch.hasError());
  EXPECT_FALSE(arch.value().empty());
}

TEST(CudaApiTest, IpcGetMemHandleReturnsNonZeroHandle) {
  CudaApi api;
  if (!hasGpu(api)) {
    GTEST_SKIP() << "no GPU available";
  }
  ASSERT_FALSE(api.setDevice(0).hasError());

  void* devPtr = nullptr;
  ASSERT_EQ(cudaMalloc(&devPtr, 4096), cudaSuccess);
  // Free the device buffer regardless of how the assertions below exit, so a
  // failed assertion does not leak the GPU allocation.
  std::unique_ptr<void, void (*)(void*)> devGuard(devPtr, [](void* p) {
    if (p != nullptr) {
      (void)cudaFree(p);
    }
  });

  auto handle = api.ipcGetMemHandle(devPtr);
  ASSERT_FALSE(handle.hasError());
  // A valid exported IPC handle is not all-zero.
  const CudaApi::IpcMemHandle zero{};
  EXPECT_NE(handle.value(), zero);
}

} // namespace
} // namespace uniflow
