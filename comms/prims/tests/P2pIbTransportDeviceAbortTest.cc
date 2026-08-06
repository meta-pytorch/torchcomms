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

} // namespace comms::prims
