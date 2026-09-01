// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaApi.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

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

TEST(CudaApiTest, EventSynchronizeReturnsOnlyOnceTheEventCompleted) {
  CudaApi api;
  if (!hasGpu(api)) {
    GTEST_SKIP() << "no GPU available";
  }
  ASSERT_FALSE(api.setDevice(0).hasError());

  void* devPtr = nullptr;
  ASSERT_EQ(cudaMalloc(&devPtr, 4096), cudaSuccess);
  std::unique_ptr<void, void (*)(void*)> devGuard(devPtr, [](void* p) {
    if (p != nullptr) {
      (void)cudaFree(p);
    }
  });
  std::vector<uint8_t> host(4096, 0);

  cudaEvent_t event{};
  ASSERT_FALSE(api.eventCreate(&event).hasError());

  ASSERT_FALSE(api.memcpyAsync(
                      host.data(),
                      devPtr,
                      host.size(),
                      cudaMemcpyDeviceToHost,
                      /*stream=*/nullptr)
                   .hasError());
  ASSERT_FALSE(api.eventRecord(event, /*stream=*/nullptr).hasError());

  EXPECT_FALSE(api.eventSynchronize(event).hasError());
  // The point of the blocking wait: the copy is done by the time it returns, so
  // a caller does not have to poll to know that.
  auto done = api.eventQuery(event);
  ASSERT_FALSE(done.hasError());
  EXPECT_TRUE(done.value());

  EXPECT_FALSE(api.eventDestroy(event).hasError());
}

} // namespace
} // namespace uniflow
