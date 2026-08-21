// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <string>

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

  constexpr auto testCase =
      static_cast<test::NvlSignalTrapCase>(NVL_SIGNAL_TRAP_CASE);
  if constexpr (testCase == test::NvlSignalTrapCase::WaitTimeout) {
    testing::internal::CaptureStdout();
  }
  test::launchNvlSignalTrap(testCase);
  const auto error = cudaDeviceSynchronize();
  if constexpr (testCase == test::NvlSignalTrapCase::WaitTimeout) {
    const auto output = testing::internal::GetCapturedStdout();
    EXPECT_NE(
        output.find("CUDA ABORT ERROR: NVL signal wait for sequence=1"),
        std::string::npos)
        << output;
    EXPECT_NE(output.find("MultimemNvlSignal.cuh:"), std::string::npos)
        << output;
  }
  EXPECT_TRUE(
      error == cudaErrorIllegalInstruction || error == cudaErrorAssert ||
      error == cudaErrorLaunchFailure)
      << cudaGetErrorString(error);
  EXPECT_EQ(cudaDeviceReset(), cudaSuccess);
}

} // namespace comms::prims::tests
