// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Own binary: sibling suites leave a primary context on device 0 behind, which
// would mask the foreign context checked here.

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <dlfcn.h>

#include <chrono>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/cvars/nccl_cvars.h"

using meta::comms::colltrace::CollTrace;
using meta::comms::colltrace::CollTraceConfig;
using meta::comms::colltrace::ICollTracePlugin;

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

// A communicator on device 1 brings graph tracing up on a fresh thread. That
// thread must not touch device 0, the fresh-thread default, before the setup
// function binds it to device 1.
TEST(GraphCollTraceWarmupDeviceTest, AsyncWarmupStaysOnCommunicatorDevice) {
  int deviceCount = 0;
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "needs two devices";
  }
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  if (!meta::comms::colltrace::graphColltraceSupported(
          "GraphCollTraceWarmupDeviceTest")) {
    GTEST_SKIP() << "graph colltrace unsupported on this device (needs sm_90+)";
  }
  expectPrimaryContextActive(0, false);

  EnvRAII<bool> cvarGuard(NCCL_COLLTRACE_TRACE_CUDA_GRAPH, true);
  EnvRAII<bool> asyncWarmupGuard(NCCL_COLLTRACE_ASYNC_GRAPH_WARMUP, true);
  {
    CollTrace colltrace(
        CollTraceConfig{.maxCheckCancelInterval = std::chrono::milliseconds{1}},
        CommLogData{},
        []() -> meta::comms::CommsMaybeVoid {
          // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
          cudaSetDevice(1);
          return folly::unit;
        },
        std::vector<std::unique_ptr<ICollTracePlugin>>{});
    // The destructor joins the warmup thread.
  }

  expectPrimaryContextActive(0, false);
  expectPrimaryContextActive(1, true);
}

} // namespace
