// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/prims/tests/MultimemNvlSignalTrapTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

#ifndef NVL_SIGNAL_TRAP_CASE
#error "NVL_SIGNAL_TRAP_CASE must select one isolated trap case"
#endif

namespace comms::prims::tests {

TEST(MultimemNvlSignalTrapTest, InvalidGeometryTraps) {
  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    GTEST_SKIP() << "No CUDA devices available";
  }
  CUDACHECK_TEST(cudaSetDevice(0));

  test::launchNvlSignalTrap(
      static_cast<test::NvlSignalTrapCase>(NVL_SIGNAL_TRAP_CASE));
  const auto error = cudaDeviceSynchronize();
  EXPECT_TRUE(
      error == cudaErrorIllegalInstruction || error == cudaErrorAssert ||
      error == cudaErrorLaunchFailure)
      << cudaGetErrorString(error);
  EXPECT_EQ(cudaDeviceReset(), cudaSuccess);
}

} // namespace comms::prims::tests
