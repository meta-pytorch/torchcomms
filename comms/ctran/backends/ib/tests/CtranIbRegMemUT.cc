// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/tests/CtranTestUtils.h"
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

  void expectImportMemFailureDoesNotModifyOutputs(
      const int configuredDeviceCount,
      const int receivedKeyCount) {
    EnvRAII devicesPerRank(
        NCCL_CTRAN_IB_DEVICES_PER_RANK, configuredDeviceCount);
    ControlMsg msg(ControlMsgType::IB_EXPORT_MEM);
    msg.ibDesc.nKeys = receivedKeyCount;

    int originalBuf{};
    void* buf = &originalBuf;
    CtranIbRemoteAccessKey key{};
    key.nKeys = 7;
    key.rkeys.fill(11);
    const auto originalKey = key;

    EXPECT_EQ(CtranIb::importMem(&buf, &key, msg), commInvalidArgument);
    EXPECT_EQ(buf, &originalBuf);
    EXPECT_EQ(key.nKeys, originalKey.nKeys);
    EXPECT_EQ(key.rkeys, originalKey.rkeys);
  }
};

TEST_F(CtranIbRegMemTest, ExportMemRejectsNullRegistrationHandle) {
  EnvRAII devicesPerRank(NCCL_CTRAN_IB_DEVICES_PER_RANK, 1);
  ControlMsg msg;
  EXPECT_EQ(CtranIb::exportMem(nullptr, nullptr, msg), commInvalidArgument);
  EXPECT_EQ(msg.type, ControlMsgType::UNSPECIFIED);
}

TEST_F(CtranIbRegMemTest, ExportMemRejectsShortRegistration) {
  EnvRAII devicesPerRank(NCCL_CTRAN_IB_DEVICES_PER_RANK, 1);
  std::vector<ibverbx::IbvMr> mrs;
  ControlMsg msg;
  EXPECT_EQ(CtranIb::exportMem(nullptr, &mrs, msg), commInvalidArgument);
  EXPECT_EQ(msg.type, ControlMsgType::UNSPECIFIED);
}

TEST_F(CtranIbRegMemTest, ExportMemRejectsNullMemoryRegion) {
  EnvRAII devicesPerRank(NCCL_CTRAN_IB_DEVICES_PER_RANK, 1);
  constexpr size_t bufSize = 8192;
  auto hostMem = std::make_unique<char[]>(bufSize);
  void* ibRegElem = nullptr;
  try {
    ASSERT_EQ(
        CtranIb::regMem(hostMem.get(), bufSize, 0, &ibRegElem), commSuccess);
  } catch (const std::bad_alloc&) {
    GTEST_SKIP() << "IB backend not enabled. Skip test";
  }

  auto* mrs = reinterpret_cast<std::vector<ibverbx::IbvMr>*>(ibRegElem);
  ASSERT_EQ(mrs->size(), 1);
  auto movedMr = std::move(mrs->front());
  EXPECT_NE(movedMr.mr(), nullptr);
  EXPECT_EQ(mrs->front().mr(), nullptr);
  ControlMsg msg;
  EXPECT_EQ(CtranIb::exportMem(nullptr, ibRegElem, msg), commInvalidArgument);
  EXPECT_EQ(msg.type, ControlMsgType::UNSPECIFIED);

  mrs->front() = std::move(movedMr);
  EXPECT_EQ(CtranIb::deregMem(ibRegElem), commSuccess);
}

TEST_F(CtranIbRegMemTest, ExportMemRejectsNonpositiveDeviceCount) {
  EnvRAII devicesPerRank(NCCL_CTRAN_IB_DEVICES_PER_RANK, 0);
  std::vector<ibverbx::IbvMr> mrs;
  ControlMsg msg;
  EXPECT_EQ(CtranIb::exportMem(nullptr, &mrs, msg), commInvalidArgument);
  EXPECT_EQ(msg.type, ControlMsgType::UNSPECIFIED);
}

TEST_F(CtranIbRegMemTest, ExportMemRejectsTooManyDevices) {
  EnvRAII devicesPerRank(
      NCCL_CTRAN_IB_DEVICES_PER_RANK, CTRAN_MAX_IB_DEVICES_PER_RANK + 1);
  std::vector<ibverbx::IbvMr> mrs;
  ControlMsg msg;
  EXPECT_EQ(CtranIb::exportMem(nullptr, &mrs, msg), commInvalidArgument);
  EXPECT_EQ(msg.type, ControlMsgType::UNSPECIFIED);
}

TEST_F(CtranIbRegMemTest, GetRemoteAccessKeyRejectsNullRegistrationHandle) {
  EnvRAII devicesPerRank(NCCL_CTRAN_IB_DEVICES_PER_RANK, 1);
  EXPECT_THROW(CtranIb::getRemoteAccessKey(nullptr), ctran::utils::Exception);
}

TEST_F(CtranIbRegMemTest, ImportMemRejectsTooManyDevices) {
  expectImportMemFailureDoesNotModifyOutputs(
      CTRAN_MAX_IB_DEVICES_PER_RANK + 1, CTRAN_MAX_IB_DEVICES_PER_RANK);
}

TEST_F(CtranIbRegMemTest, ImportMemRejectsMismatchedKeyCount) {
  expectImportMemFailureDoesNotModifyOutputs(1, 2);
}

TEST_F(CtranIbRegMemTest, ImportMemRejectsNullOutputs) {
  EnvRAII devicesPerRank(NCCL_CTRAN_IB_DEVICES_PER_RANK, 1);
  ControlMsg msg(ControlMsgType::IB_EXPORT_MEM);
  msg.ibDesc.nKeys = 1;

  int originalBuf{};
  void* buf = &originalBuf;
  CtranIbRemoteAccessKey key{};
  key.nKeys = 7;
  const auto originalKey = key;

  EXPECT_EQ(CtranIb::importMem(nullptr, &key, msg), commInvalidArgument);
  EXPECT_EQ(key.nKeys, originalKey.nKeys);
  EXPECT_EQ(key.rkeys, originalKey.rkeys);
  EXPECT_EQ(CtranIb::importMem(&buf, nullptr, msg), commInvalidArgument);
  EXPECT_EQ(buf, &originalBuf);
}

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

  auto buf = ctran::commMemAlloc(bufSize, memType, segments);
  ASSERT_NE(buf, nullptr);

  ASSERT_EQ(CtranIb::regMem(buf, bufSize, 0, &ibRegElem), commSuccess);
  ASSERT_NE(ibRegElem, nullptr);

  ASSERT_EQ(CtranIb::deregMem(ibRegElem), commSuccess);
  ctran::commMemFree(buf, bufSize, memType);
}

class CtranIbCumemDistjointRegMemTestParam
    : public CtranIbRegMemTest,
      public ::testing::WithParamInterface<std::tuple<size_t>> {};
TEST_P(CtranIbCumemDistjointRegMemTestParam, InvalidCumemDistjointTest) {
  auto [bufSize] = GetParam();
  void* ibRegElem = nullptr;
  std::vector<TestMemSegment> segments;

  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto buf = ctran::commMemAlloc(bufSize, kCuMemAllocDisjoint, segments);
  ASSERT_NE(buf, nullptr);

  ASSERT_EQ(CtranIb::regMem(buf, bufSize, 0, &ibRegElem), commInvalidUsage);
  ASSERT_EQ(ibRegElem, nullptr);

  ctran::commMemFree(buf, bufSize, kCuMemAllocDisjoint);
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
