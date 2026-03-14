// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/CudaRAII.h"

#include <gtest/gtest.h>

using namespace meta::comms;

TEST(StreamCaptureModeGuardTest, StandaloneRestoresMode) {
  cudaStreamCaptureMode before = cudaStreamCaptureModeGlobal;
  ASSERT_EQ(cudaThreadExchangeStreamCaptureMode(&before), cudaSuccess);
  ASSERT_EQ(cudaThreadExchangeStreamCaptureMode(&before), cudaSuccess);

  {
    StreamCaptureModeGuard guard{cudaStreamCaptureModeRelaxed};
    cudaStreamCaptureMode during = cudaStreamCaptureModeGlobal;
    ASSERT_EQ(cudaThreadExchangeStreamCaptureMode(&during), cudaSuccess);
    EXPECT_EQ(during, cudaStreamCaptureModeRelaxed);
    ASSERT_EQ(cudaThreadExchangeStreamCaptureMode(&during), cudaSuccess);
  }

  cudaStreamCaptureMode after = cudaStreamCaptureModeRelaxed;
  ASSERT_EQ(cudaThreadExchangeStreamCaptureMode(&after), cudaSuccess);
  EXPECT_EQ(after, before);
  ASSERT_EQ(cudaThreadExchangeStreamCaptureMode(&after), cudaSuccess);
}

struct MockCudaApi {
  int exchangeCallCount{0};
  cudaStreamCaptureMode lastRequestedMode{cudaStreamCaptureModeGlobal};

  cudaError_t threadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) {
    ++exchangeCallCount;
    lastRequestedMode = *mode;
    std::swap(*mode, storedMode_);
    return cudaSuccess;
  }

 private:
  cudaStreamCaptureMode storedMode_{cudaStreamCaptureModeGlobal};
};

TEST(StreamCaptureModeGuardTest, TemplateConstructorCallsApi) {
  MockCudaApi mock;
  ASSERT_EQ(mock.exchangeCallCount, 0);

  {
    StreamCaptureModeGuard guard{&mock, cudaStreamCaptureModeRelaxed};
    EXPECT_EQ(mock.exchangeCallCount, 1);
    EXPECT_EQ(mock.lastRequestedMode, cudaStreamCaptureModeRelaxed);
  }

  EXPECT_EQ(mock.exchangeCallCount, 2);
  EXPECT_EQ(mock.lastRequestedMode, cudaStreamCaptureModeGlobal);
}
