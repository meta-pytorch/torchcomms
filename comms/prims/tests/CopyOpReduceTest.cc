// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "comms/prims/tests/CopyOpReduceTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

namespace comms::prims::test {
namespace {

constexpr std::size_t kGuardElems = 4;
constexpr std::size_t kOffsetElems = 4;
constexpr float kGuardValue = -9876.5f;

struct PolicyParam {
  CopyOpReducePolicy policy;
  const char* name;
};

class CopyOpReduceTest : public ::testing::TestWithParam<PolicyParam> {};

TEST_P(CopyOpReduceTest, ProducesExpectedSumAndPreservesGuards) {
  cudaDeviceProp properties{};
  CUDACHECK_TEST(cudaGetDeviceProperties(&properties, 0));
  if (GetParam().policy == CopyOpReducePolicy::CpAsyncSmemReduce &&
      properties.major < 8) {
    GTEST_SKIP() << "cp.async requires SM80 or newer";
  }

  const std::vector<std::size_t> elementCounts = {
      3,
      8191,
      8192,
      8195,
      3 * 8192 + 5,
      512 * 1024 / sizeof(float),
  };

  for (const std::size_t nelems : elementCounts) {
    std::vector<float> staging(nelems);
    std::vector<float> local(kOffsetElems + nelems);
    for (std::size_t i = 0; i < nelems; ++i) {
      staging[i] = 0.25f * static_cast<float>(i % 29) - 3.5f;
      local[kOffsetElems + i] = -0.125f * static_cast<float>(i % 17) + 2.25f;
    }
    const std::vector<float> originalStaging = staging;
    const std::vector<float> originalLocal = local;
    std::vector<float> output(kGuardElems + nelems + kGuardElems, kGuardValue);

    meta::comms::DeviceBuffer stagingDevice(nelems * sizeof(float));
    meta::comms::DeviceBuffer localDevice(local.size() * sizeof(float));
    meta::comms::DeviceBuffer outputDevice(output.size() * sizeof(float));
    CUDACHECK_TEST(cudaMemcpy(
        stagingDevice.get(),
        staging.data(),
        nelems * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        localDevice.get(),
        local.data(),
        local.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        outputDevice.get(),
        output.data(),
        output.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    launchCopyOpReduce(
        GetParam().policy,
        static_cast<float*>(outputDevice.get()) + kGuardElems,
        static_cast<const float*>(stagingDevice.get()),
        static_cast<const float*>(localDevice.get()),
        kOffsetElems * sizeof(float),
        nelems * sizeof(float));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    CUDACHECK_TEST(cudaMemcpy(
        output.data(),
        outputDevice.get(),
        output.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < nelems; ++i) {
      EXPECT_FLOAT_EQ(
          output[kGuardElems + i], staging[i] + local[kOffsetElems + i])
          << "policy=" << GetParam().name << " nelems=" << nelems
          << " index=" << i;
    }
    for (std::size_t i = 0; i < kGuardElems; ++i) {
      EXPECT_EQ(output[i], kGuardValue);
      EXPECT_EQ(output[kGuardElems + nelems + i], kGuardValue);
    }

    CUDACHECK_TEST(cudaMemcpy(
        staging.data(),
        stagingDevice.get(),
        nelems * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(
        local.data(),
        localDevice.get(),
        local.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(staging, originalStaging);
    EXPECT_EQ(local, originalLocal);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Policies,
    CopyOpReduceTest,
    ::testing::Values(
        PolicyParam{CopyOpReducePolicy::TileReduce, "TileReduce"},
        PolicyParam{CopyOpReducePolicy::TileReduceStaged, "TileReduceStaged"},
        PolicyParam{
            CopyOpReducePolicy::CpAsyncSmemReduce,
            "CpAsyncSmemReduce"}),
    [](const ::testing::TestParamInfo<PolicyParam>& info) {
      return std::string(info.param.name);
    });

} // namespace
} // namespace comms::prims::test
