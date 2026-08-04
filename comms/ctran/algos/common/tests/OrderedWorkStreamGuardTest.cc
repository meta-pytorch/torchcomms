// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <chrono>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/common/OrderedWorkStreamGuard.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace {

using ctran::algos::OrderedWorkStreamGuard;
using ctran::utils::cudagraph::StreamCaptureInfo;

static_assert(std::is_nothrow_destructible_v<OrderedWorkStreamGuard>);

void delayCallback(void* data) {
  const auto delay = *static_cast<const std::chrono::milliseconds*>(data);
  std::this_thread::sleep_for(delay);
}

StreamCaptureInfo captureInfo(cudaStream_t stream) {
  StreamCaptureInfo info{};
  CUDACHECK_TEST(ctran::utils::cudagraph::getStreamCaptureInfo(stream, info));
  return info;
}

// Parameterized over (synchronizeEagerAfterCapturedWork,
// NCCL_CTRAN_GRAPH_MIXING_SUPPORT). The first is a per-consumer construction
// argument (GPE passes true, Prims false); the second is a global perf cvar.
// Both are covered explicitly because their interaction determines whether the
// captured-to-eager barrier is effective.
class OrderedWorkStreamGuardTest
    : public ::testing::TestWithParam<std::tuple<bool, int>> {
 protected:
  bool syncEagerAfterCaptured() const {
    return std::get<0>(GetParam());
  }
  int mixingSupport() const {
    return std::get<1>(GetParam());
  }

  void SetUp() override {
    // Without ncclCvarInit() the cvar globals keep their zero-initialized
    // values, so NCCL_CTRAN_GRAPH_MIXING_SUPPORT would read 0 regardless of its
    // documented default of 1 (the default is the env2num fallback applied
    // during init, not a static initializer). Set it explicitly: the guard
    // latches it in init() below, so this must happen first.
    setenv(
        "NCCL_CTRAN_GRAPH_MIXING_SUPPORT",
        std::to_string(mixingSupport()).c_str(),
        1);
    ncclCvarInit();
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&streamA_, cudaStreamNonBlocking));
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&streamB_, cudaStreamNonBlocking));
    guard_.init(logMetaData_, syncEagerAfterCaptured());
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(streamA_));
    CUDACHECK_TEST(cudaStreamDestroy(streamB_));
    unsetenv("NCCL_CTRAN_GRAPH_MIXING_SUPPORT");
    ncclCvarInit();
  }

  CommLogData logMetaData_{};
  OrderedWorkStreamGuard guard_;
  cudaStream_t streamA_{};
  cudaStream_t streamB_{};
};

TEST_P(OrderedWorkStreamGuardTest, OrdersEagerWorkAcrossStreams) {
  int* value{};
  CUDACHECK_TEST(cudaMalloc(&value, sizeof(*value)));

  std::chrono::milliseconds delay{200};
  auto first = guard_.acquire(streamA_, captureInfo(streamA_));
  ASSERT_EQ(first.status(), commSuccess);
  CUDACHECK_TEST(cudaLaunchHostFunc(streamA_, delayCallback, &delay));
  CUDACHECK_TEST(cudaMemsetAsync(value, 1, sizeof(*value), streamA_));
  ASSERT_EQ(first.release(), commSuccess);

  auto second = guard_.acquire(streamB_, captureInfo(streamB_));
  ASSERT_EQ(second.status(), commSuccess);
  CUDACHECK_TEST(cudaMemsetAsync(value, 2, sizeof(*value), streamB_));
  ASSERT_EQ(second.release(), commSuccess);

  CUDACHECK_TEST(cudaStreamSynchronize(streamB_));
  int result{};
  CUDACHECK_TEST(
      cudaMemcpy(&result, value, sizeof(result), cudaMemcpyDeviceToHost));
  EXPECT_EQ(result, 0x02020202);
  CUDACHECK_TEST(cudaFree(value));
}

// The guard's captured-to-eager policy, across both dimensions.
//
// synchronizeEagerAfterCapturedWork=true asks for a host barrier after captured
// work; false asks only for GPU-side ordering. The barrier is effective only at
// mixing=1, where a graph EVENT_RECORD node records execModeSyncEvent_ at
// replay. At mixing=0 the captured fence is an absorbed plain record on a
// separate event, so no node records execModeSyncEvent_ and the barrier has
// nothing to wait on -- the documented mixing=0 caveat. Either way the eager
// acquire must succeed rather than fail on a capture-bound event.
TEST_P(OrderedWorkStreamGuardTest, AppliesCapturedToEagerPolicy) {
  cudaGraph_t graph{};
  cudaGraphExec_t graphExec{};
  std::chrono::milliseconds delay{500};

  CUDACHECK_TEST(cudaStreamBeginCapture(streamA_, cudaStreamCaptureModeGlobal));
  auto captured = guard_.acquire(streamA_, captureInfo(streamA_));
  ASSERT_EQ(captured.status(), commSuccess);
  CUDACHECK_TEST(cudaLaunchHostFunc(streamA_, delayCallback, &delay));
  ASSERT_EQ(captured.release(), commSuccess);
  CUDACHECK_TEST(cudaStreamEndCapture(streamA_, &graph));
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  CUDACHECK_TEST(cudaGraphLaunch(graphExec, streamA_));

  // The eager acquire must not fail on a capture-bound event, whichever policy
  // is in force.
  auto eager = guard_.acquire(streamB_, captureInfo(streamB_));
  ASSERT_EQ(eager.status(), commSuccess);
  const cudaError_t replayStatus = cudaStreamQuery(streamA_);
  if (syncEagerAfterCaptured() && mixingSupport() != 0) {
    EXPECT_EQ(replayStatus, cudaSuccess)
        << "host sync must have waited for the replay to complete";
  } else {
    EXPECT_EQ(replayStatus, cudaErrorNotReady)
        << "without an effective host barrier the replay stays in flight";
  }
  ASSERT_EQ(eager.release(), commSuccess);

  CUDACHECK_TEST(cudaStreamSynchronize(streamB_));
  CUDACHECK_TEST(cudaStreamSynchronize(streamA_));
  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
}

TEST_P(OrderedWorkStreamGuardTest, PropagatesPoisonedError) {
  auto first = guard_.acquire(streamA_, captureInfo(streamA_));
  ASSERT_EQ(first.status(), commSuccess);
  ASSERT_EQ(first.release(), commSuccess);

  cudaStream_t staleStream{};
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&staleStream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaStreamDestroy(staleStream));

  auto failed = guard_.acquire(staleStream, captureInfo(streamB_));
  ASSERT_EQ(failed.status(), commUnhandledCudaError);
  ASSERT_EQ(failed.release(), commUnhandledCudaError);

  auto poisoned = guard_.acquire(streamB_, captureInfo(streamB_));
  EXPECT_EQ(poisoned.status(), commUnhandledCudaError);
}

TEST_P(OrderedWorkStreamGuardTest, DoubleInitAborts) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
      guard_.init(logMetaData_, syncEagerAfterCaptured()), "initialized twice");
}

INSTANTIATE_TEST_SUITE_P(
    CapturedToEagerPolicy,
    OrderedWorkStreamGuardTest,
    ::testing::Combine(::testing::Bool(), ::testing::Values(0, 1)),
    [](const ::testing::TestParamInfo<std::tuple<bool, int>>& info) {
      return std::string(
                 std::get<0>(info.param) ? "SyncEager" : "NoSyncEager") +
          "Mixing" + std::to_string(std::get<1>(info.param));
    });

} // namespace
