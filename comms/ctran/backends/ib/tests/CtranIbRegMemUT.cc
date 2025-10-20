// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual

class CtranIbRegMemTest : public ::testing::Test {
 public:
  CtranIbRegMemTest() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_DEBUG", "INFO", 1);
    ncclCvarInit();
    ctran::logging::initCtranLogging(true /*alwaysInit*/);
  }
};

class CtranIbCpuRegMemTestParam
    : public CtranIbRegMemTest,
      public ::testing::WithParamInterface<std::tuple<size_t>> {};

TEST_P(CtranIbCpuRegMemTestParam, CpuTest) {
  auto [bufSize] = GetParam();

  char* hostMem = new char[bufSize];
  void* ibRegElem = nullptr;
  ASSERT_EQ(CtranIb::regMem(hostMem, bufSize, 0, &ibRegElem), commSuccess);
  ASSERT_NE(ibRegElem, nullptr);

  ASSERT_EQ(CtranIb::deregMem(ibRegElem), commSuccess);
  delete[] hostMem;
}

class CtranIbGpuRegMemTestParam
    : public CtranIbRegMemTest,
      public ::testing::WithParamInterface<std::tuple<MemAllocType, size_t>> {};

TEST_P(CtranIbGpuRegMemTestParam, GpuTest) {
  auto [memType, bufSize] = GetParam();
  void* ibRegElem = nullptr;
  std::vector<TestMemSegment> segments;

  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto buf = commMemAlloc(bufSize, memType, segments);
  ASSERT_NE(buf, nullptr);

  ASSERT_EQ(CtranIb::regMem(buf, bufSize, 0, &ibRegElem), commSuccess);
  ASSERT_NE(ibRegElem, nullptr);

  ASSERT_EQ(CtranIb::deregMem(ibRegElem), commSuccess);
  commMemFree(buf, bufSize, memType);
}

class CtranIbCumemDistjointRegMemTestParam
    : public CtranIbRegMemTest,
      public ::testing::WithParamInterface<std::tuple<size_t>> {};
TEST_P(CtranIbCumemDistjointRegMemTestParam, InvalidCumemDistjointTest) {
  auto [bufSize] = GetParam();
  void* ibRegElem = nullptr;
  std::vector<TestMemSegment> segments;

  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto buf = commMemAlloc(bufSize, kCuMemAllocDisjoint, segments);
  ASSERT_NE(buf, nullptr);

  ASSERT_EQ(CtranIb::regMem(buf, bufSize, 0, &ibRegElem), commInvalidUsage);
  ASSERT_EQ(ibRegElem, nullptr);

  commMemFree(buf, bufSize, kCuMemAllocDisjoint);
}

namespace {
auto testSizes = testing::Values(
    // small, unaligned
    1UL,
    99UL,
    4096UL,
    4097UL,
    8192UL,
    15555UL, // unaligned single segment
    2097152UL + 4096, // partial of 2 segments each with 2MB, page aligned
    1073741824UL // large
);

auto testInvalidSizes = testing::Values(
    2097152UL + 16, // partial of 2 segments each with 2MB, 16B aligned
    2097152UL + 4097 // partial of 2 segments each with 2MB, unligned
);
} // namespace

INSTANTIATE_TEST_SUITE_P(
    CtranIbRegMemTest,
    CtranIbCpuRegMemTestParam,
    testSizes,
    [&](const testing::TestParamInfo<CtranIbCpuRegMemTestParam::ParamType>&
            info) { return "size" + std::to_string(std::get<0>(info.param)); });

INSTANTIATE_TEST_SUITE_P(
    CtranIbRegMemTest,
    CtranIbGpuRegMemTestParam,
    ::testing::Combine(
        testing::Values(kMemCudaMalloc, kMemHostManaged, kCuMemAllocDisjoint),
        testSizes),
    [&](const testing::TestParamInfo<CtranIbGpuRegMemTestParam::ParamType>&
            info) {
      MemAllocType memAllocType = std::get<0>(info.param);

      return testMemAllocTypeToStr(memAllocType) + "_size" +
          std::to_string(std::get<1>(info.param));
    });

INSTANTIATE_TEST_SUITE_P(
    CtranIbRegMemTest,
    CtranIbCumemDistjointRegMemTestParam,
    testInvalidSizes,
    [&](const testing::TestParamInfo<
        CtranIbCumemDistjointRegMemTestParam::ParamType>& info) {
      return "size" + std::to_string(std::get<0>(info.param));
    });
