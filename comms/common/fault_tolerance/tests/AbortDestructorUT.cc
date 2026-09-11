// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "comms/common/fault_tolerance/Abort.h"

namespace comms::fault_tolerance::testing {
namespace {

void expectPrimaryContextActive(int deviceOrdinal, bool expected) {
  void* libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  ASSERT_NE(libcuda, nullptr) << dlerror();
  using DeviceGetFn = int (*)(int*, int);
  using CtxGetStateFn = int (*)(int, unsigned int*, int*);
  auto deviceGet = reinterpret_cast<DeviceGetFn>(dlsym(libcuda, "cuDeviceGet"));
  auto ctxGetState = reinterpret_cast<CtxGetStateFn>(
      dlsym(libcuda, "cuDevicePrimaryCtxGetState"));
  ASSERT_NE(deviceGet, nullptr);
  ASSERT_NE(ctxGetState, nullptr);
  int device = 0;
  ASSERT_EQ(deviceGet(&device, deviceOrdinal), 0);
  unsigned int flags = 0;
  int active = 0;
  ASSERT_EQ(ctxGetState(device, &flags, &active), 0);
  EXPECT_EQ(active != 0, expected) << "device " << deviceOrdinal;
}

} // namespace

// Own binary on purpose: sibling suites leave a context on device 0 behind,
// which would mask the leak checked here.
TEST(AbortDestructorTest, unboundThreadDoesNotCreateForeignContext) {
  int deviceCount = 0;
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "needs two devices";
  }
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  auto abort = std::make_unique<Abort>(/*enabled=*/true);
  expectPrimaryContextActive(0, false);

  std::thread([abort = std::move(abort)]() mutable { abort.reset(); }).join();

  expectPrimaryContextActive(0, false);
  expectPrimaryContextActive(1, true);
}

TEST(AbortDestructorTest, boundThreadKeepsItsOwnDevice) {
  int deviceCount = 0;
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  if (deviceCount < 3) {
    GTEST_SKIP() << "needs three devices";
  }
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  auto abort = std::make_unique<Abort>(/*enabled=*/true);

  int deviceAfter = -1;
  std::thread([abort = std::move(abort), &deviceAfter]() mutable {
    EXPECT_EQ(cudaSetDevice(2), cudaSuccess);
    abort.reset();
    EXPECT_EQ(cudaGetDevice(&deviceAfter), cudaSuccess);
  }).join();

  EXPECT_EQ(deviceAfter, 2);
  expectPrimaryContextActive(0, false);
}

} // namespace comms::fault_tolerance::testing
