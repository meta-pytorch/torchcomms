// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>

#include "comms/prims/tests/P2pNvlProgressTrapTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

#ifndef NVL_PROGRESS_TRAP_CASE
#error "NVL_PROGRESS_TRAP_CASE must select one isolated trap case"
#endif

namespace comms::prims::tests {
namespace {

/*
 * The device-side diagnostic this case must emit. Asserting on it is what makes
 * the target prove *why* the process trapped: the CUDA status alone is generic,
 * so any unrelated kernel fault would satisfy it and the target would pass for
 * the wrong reason.
 *
 * The runtime flushes device printf on synchronize, including on the trap path,
 * so the message reaches stdout before launchNvlProgressTrap() returns.
 */
constexpr const char* kExpectedDiagnostic =
#if NVL_PROGRESS_TRAP_CASE == 0
    "progress_send_once observed abort";
#elif NVL_PROGRESS_TRAP_CASE == 1
    "init_send_progress: channel 0 already has an in-flight send";
#elif NVL_PROGRESS_TRAP_CASE == 2
    "init_send_progress: progress storage not configured";
#else
#error "No expected diagnostic for this NVL_PROGRESS_TRAP_CASE"
#endif

} // namespace

TEST(P2pNvlProgressTrapTest, ProgressMisuseTraps) {
  int deviceCount = 0;
  CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    GTEST_SKIP() << "No CUDA devices available";
  }
  CUDACHECK_TEST(cudaSetDevice(0));

  // The helper synchronizes internally: the `Abort` backing the kernel's device
  // handle is owned there, so waiting out here would read freed state. Capture
  // spans the launch and that synchronize, which is when the runtime flushes
  // the device printf.
  ::testing::internal::CaptureStdout();
  const auto error = test::launchNvlProgressTrap(
      static_cast<test::NvlProgressTrapCase>(NVL_PROGRESS_TRAP_CASE));
  const std::string deviceOutput = ::testing::internal::GetCapturedStdout();
  // Re-emit so the diagnostic still appears in the test log on success.
  std::cerr << deviceOutput;

  EXPECT_NE(deviceOutput.find(kExpectedDiagnostic), std::string::npos)
      << "trapped without the expected diagnostic \"" << kExpectedDiagnostic
      << "\"; this case may be trapping for an unintended reason";
  EXPECT_TRUE(
      error == cudaErrorIllegalInstruction || error == cudaErrorAssert ||
      error == cudaErrorLaunchFailure)
      << cudaGetErrorString(error);
  EXPECT_EQ(cudaDeviceReset(), cudaSuccess);
}

} // namespace comms::prims::tests
