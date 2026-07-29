// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/common/OrderedWorkStreamGuard.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"

namespace {

using ctran::utils::OrderedWorkStreamGuard;
using ctran::utils::cudagraph::StreamCaptureInfo;

void CUDART_CB delayCallback(void* data) {
  const auto delay = *static_cast<const std::chrono::milliseconds*>(data);
  std::this_thread::sleep_for(delay);
}

StreamCaptureInfo captureInfo(cudaStream_t stream) {
  StreamCaptureInfo info{};
  CUDACHECK_TEST(ctran::utils::cudagraph::getStreamCaptureInfo(stream, info));
  return info;
}

class OrderedWorkStreamGuardTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&streamA_, cudaStreamNonBlocking));
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&streamB_, cudaStreamNonBlocking));
    guard_.init(logMetaData_, GetParam());
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(streamA_));
    CUDACHECK_TEST(cudaStreamDestroy(streamB_));
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

  auto eager = guard_.acquire(streamB_, captureInfo(streamB_));
  ASSERT_EQ(eager.status(), commSuccess);
  const cudaError_t replayStatus = cudaStreamQuery(streamA_);
  if (GetParam()) {
    EXPECT_EQ(replayStatus, cudaSuccess);
  } else {
    EXPECT_EQ(replayStatus, cudaErrorNotReady);
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
  EXPECT_DEATH(guard_.init(logMetaData_, GetParam()), "initialized twice");
}

INSTANTIATE_TEST_SUITE_P(
    CapturedToEagerPolicy,
    OrderedWorkStreamGuardTest,
    ::testing::Bool());

} // namespace
