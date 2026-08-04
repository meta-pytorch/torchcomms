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
    CUDACHECK_TEST(cudaMalloc(&buf_, kBufBytes));
    guard_.init(logMetaData_, syncEagerAfterCaptured());
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaFree(buf_));
    CUDACHECK_TEST(cudaStreamDestroy(streamA_));
    CUDACHECK_TEST(cudaStreamDestroy(streamB_));
    unsetenv("NCCL_CTRAN_GRAPH_MIXING_SUPPORT");
    ncclCvarInit();
  }

  static constexpr size_t kBufBytes = sizeof(int);

  CommLogData logMetaData_{};
  OrderedWorkStreamGuard guard_;
  cudaStream_t streamA_{};
  cudaStream_t streamB_{};
  int* buf_{};
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

// Two captures in sequence through the same guard, each replayed, with an eager
// acquire after each. This is the `isNewCapture` path: on the second capture's
// first acquire, `lastRecordNode_` (mixing=1) and `captureFenceEvent_`
// (mixing=0) both belong to graph 1, so neither can carry ordering into graph
// 2 -- mixing=1 falls back to an external event wait and mixing=0 to nothing.
//
// The regression this guards against is state carried between captures: a
// stale `lastCaptureId_`, a `lastRecordNode_` from the wrong graph, or a
// capture-bound event leaking into the eager path. Every acquire/release must
// still succeed and the guard must not latch an error, since `doAcquire()`
// returns `error_` on all later calls once poisoned.
TEST_P(OrderedWorkStreamGuardTest, HandlesSerialCaptures) {
  std::chrono::milliseconds delay{100};

  auto captureAndReplay = [&]() {
    cudaGraph_t graph{};
    cudaGraphExec_t graphExec{};
    CUDACHECK_TEST(
        cudaStreamBeginCapture(streamA_, cudaStreamCaptureModeGlobal));
    auto captured = guard_.acquire(streamA_, captureInfo(streamA_));
    ASSERT_EQ(captured.status(), commSuccess);
    CUDACHECK_TEST(cudaLaunchHostFunc(streamA_, delayCallback, &delay));
    ASSERT_EQ(captured.release(), commSuccess);
    CUDACHECK_TEST(cudaStreamEndCapture(streamA_, &graph));
    CUDACHECK_TEST(
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CUDACHECK_TEST(cudaGraphLaunch(graphExec, streamA_));
    CUDACHECK_TEST(cudaStreamSynchronize(streamA_));
    CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
    CUDACHECK_TEST(cudaGraphDestroy(graph));
  };

  // Graph 1, then an eager acquire that consumes its fence.
  captureAndReplay();
  {
    auto eager = guard_.acquire(streamB_, captureInfo(streamB_));
    EXPECT_EQ(eager.status(), commSuccess) << "eager acquire after capture 1";
    EXPECT_EQ(eager.release(), commSuccess);
  }
  CUDACHECK_TEST(cudaStreamSynchronize(streamB_));

  // Graph 2 is a distinct capture id, so its first acquire takes the
  // isNewCapture path with graph 1's state still recorded in the guard.
  captureAndReplay();
  {
    auto eager = guard_.acquire(streamB_, captureInfo(streamB_));
    EXPECT_EQ(eager.status(), commSuccess) << "eager acquire after capture 2";
    EXPECT_EQ(eager.release(), commSuccess);
  }
  CUDACHECK_TEST(cudaStreamSynchronize(streamB_));

  // The guard latches errors, so a clean acquire here proves none of the
  // transitions above poisoned it.
  auto final = guard_.acquire(streamB_, captureInfo(streamB_));
  EXPECT_EQ(final.status(), commSuccess) << "guard must not have latched";
  EXPECT_EQ(final.release(), commSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streamB_));
}

// Serial captures where each capture contains two submits, so the second
// submit takes the intra-capture path (`!isNewCapture`: the
// cudaStreamUpdateCaptureDependencies re-link at mixing=1, the folded
// dependency-edge wait at mixing=0) and the next capture's first submit then
// takes the isNewCapture path with that state in place. Verifies the guard
// transitions cleanly between the two in-capture branches across a capture
// boundary, and that both graphs stay independently replayable.
TEST_P(OrderedWorkStreamGuardTest, HandlesSerialCapturesWithMultipleSubmits) {
  auto captureTwoSubmits = [&](cudaGraph_t* graph) {
    CUDACHECK_TEST(
        cudaStreamBeginCapture(streamA_, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < 2; ++i) {
      auto scope = guard_.acquire(streamA_, captureInfo(streamA_));
      ASSERT_EQ(scope.status(), commSuccess) << "submit " << i;
      CUDACHECK_TEST(cudaMemsetAsync(buf_, i + 1, kBufBytes, streamA_));
      ASSERT_EQ(scope.release(), commSuccess) << "submit " << i;
    }
    CUDACHECK_TEST(cudaStreamEndCapture(streamA_, graph));
    ASSERT_NE(*graph, nullptr);
  };

  cudaGraph_t graph1{};
  cudaGraph_t graph2{};
  captureTwoSubmits(&graph1);
  captureTwoSubmits(&graph2);

  // Both graphs must instantiate and replay; a dependency on a node from the
  // other graph would fail here rather than at capture time.
  for (cudaGraph_t graph : {graph1, graph2}) {
    cudaGraphExec_t graphExec{};
    CUDACHECK_TEST(
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CUDACHECK_TEST(cudaGraphLaunch(graphExec, streamA_));
    CUDACHECK_TEST(cudaStreamSynchronize(streamA_));
    CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  }

  auto eager = guard_.acquire(streamB_, captureInfo(streamB_));
  EXPECT_EQ(eager.status(), commSuccess) << "guard must not have latched";
  EXPECT_EQ(eager.release(), commSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(streamB_));

  CUDACHECK_TEST(cudaGraphDestroy(graph1));
  CUDACHECK_TEST(cudaGraphDestroy(graph2));
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
