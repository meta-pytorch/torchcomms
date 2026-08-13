// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/prims/tests/P2pIbTransportDeviceAbortTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims {

namespace {

void runDisabledAbortWait(void (*launcher)(uint64_t*, bool*)) {
  CUDACHECK_TEST(cudaSetDevice(0));

  uint64_t* signal = nullptr;
  bool* success = nullptr;
  CUDACHECK_TEST(cudaMalloc(&signal, sizeof(uint64_t)));
  CUDACHECK_TEST(cudaMalloc(&success, sizeof(bool)));
  CUDACHECK_TEST(cudaMemset(signal, 0, sizeof(uint64_t)));
  CUDACHECK_TEST(cudaMemset(success, 0, sizeof(bool)));

  launcher(signal, success);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  bool hostSuccess = false;
  CUDACHECK_TEST(
      cudaMemcpy(&hostSuccess, success, sizeof(bool), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(hostSuccess);

  CUDACHECK_TEST(cudaFree(success));
  CUDACHECK_TEST(cudaFree(signal));
}

void runPreAbortedSkipWait(
    void (*launcher)(uint64_t*, bool*, comms::fault_tolerance::AbortDevice)) {
  CUDACHECK_TEST(cudaSetDevice(0));

  uint64_t* signal = nullptr;
  bool* success = nullptr;
  CUDACHECK_TEST(cudaMalloc(&signal, sizeof(uint64_t)));
  CUDACHECK_TEST(cudaMalloc(&success, sizeof(bool)));
  CUDACHECK_TEST(cudaMemset(signal, 0, sizeof(uint64_t)));
  CUDACHECK_TEST(cudaMemset(success, 0, sizeof(bool)));

  comms::fault_tolerance::Abort abort(/*enabled=*/true);
  abort.setAbort();
  launcher(signal, success, abort.getDeviceHandle());
  CUDACHECK_TEST(cudaDeviceSynchronize());

  bool hostSuccess = false;
  CUDACHECK_TEST(
      cudaMemcpy(&hostSuccess, success, sizeof(bool), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(hostSuccess);

  CUDACHECK_TEST(cudaFree(success));
  CUDACHECK_TEST(cudaFree(signal));
}

} // namespace

TEST(P2pIbTransportDeviceAbortTest, WrapperWaitSignalAcceptsAbortDevice) {
  runDisabledAbortWait(test::launchIbWrapperWaitSignalAbortCompileCheck);
}

TEST(P2pIbTransportDeviceAbortTest, IbrcWaitSignalAcceptsDisabledAbortDevice) {
  runDisabledAbortWait(test::launchIbrcWaitSignalWithDisabledAbort);
}

TEST(P2pIbTransportDeviceAbortTest, WrapperWaitSignalSkipsWhenPreAborted) {
  runPreAbortedSkipWait(test::launchIbWrapperWaitSignalWithPreAbortedSkip);
}

TEST(P2pIbTransportDeviceAbortTest, IbrcWaitSignalSkipsWhenPreAborted) {
  runPreAbortedSkipWait(test::launchIbrcWaitSignalWithPreAbortedSkip);
}

} // namespace comms::prims
