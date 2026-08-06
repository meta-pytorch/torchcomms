// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "comms/prims/tests/P2pIbTransportDeviceAbortTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims {

TEST(P2pIbTransportDeviceAbortTest, WrapperWaitSignalAcceptsAbortDevice) {
  CUDACHECK_TEST(cudaSetDevice(0));
  test::launchIbWrapperWaitSignalAbortCompileCheck();
  CUDACHECK_TEST(cudaDeviceSynchronize());
}

TEST(P2pIbTransportDeviceAbortTest, IbrcWaitSignalAcceptsDisabledAbortDevice) {
  CUDACHECK_TEST(cudaSetDevice(0));

  uint64_t* signal = nullptr;
  bool* success = nullptr;
  CUDACHECK_TEST(cudaMalloc(&signal, sizeof(uint64_t)));
  CUDACHECK_TEST(cudaMalloc(&success, sizeof(bool)));
  CUDACHECK_TEST(cudaMemset(signal, 0, sizeof(uint64_t)));
  CUDACHECK_TEST(cudaMemset(success, 0, sizeof(bool)));

  test::launchIbrcWaitSignalWithDisabledAbort(signal, success);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  bool hostSuccess = false;
  CUDACHECK_TEST(
      cudaMemcpy(&hostSuccess, success, sizeof(bool), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(hostSuccess);

  CUDACHECK_TEST(cudaFree(success));
  CUDACHECK_TEST(cudaFree(signal));
}

} // namespace comms::prims
