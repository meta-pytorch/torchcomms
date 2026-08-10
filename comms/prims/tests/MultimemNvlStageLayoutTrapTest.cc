// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/prims/tests/MultimemNvlStageLayoutTrapTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

#ifndef STAGE_LAYOUT_TRAP_CASE
#error "STAGE_LAYOUT_TRAP_CASE must select one isolated trap case"
#endif

namespace comms::prims::tests {

TEST(MultimemNvlStageLayoutTrapTest, InvalidGeometryTraps) {
  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    GTEST_SKIP() << "No CUDA devices available";
  }
  CUDACHECK_TEST(cudaSetDevice(0));

  test::launchStageLayoutTrap(
      static_cast<test::StageLayoutTrapCase>(STAGE_LAYOUT_TRAP_CASE));
  const auto error = cudaDeviceSynchronize();
  EXPECT_TRUE(
      error == cudaErrorIllegalInstruction || error == cudaErrorAssert ||
      error == cudaErrorLaunchFailure)
      << cudaGetErrorString(error);

  EXPECT_EQ(cudaDeviceReset(), cudaSuccess);
}

} // namespace comms::prims::tests
