// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <nccl.h>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/mapper/CtranMapperRegMem.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

class CommRegisterTest : public NcclxBaseTest {
 public:
  CommRegisterTest() = default;

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    NcclxBaseTest::SetUp();
  }

  void TearDown() override {
    NcclxBaseTest::TearDown();
  }
};

class CommRegisterTestParam
    : public CommRegisterTest,
      public ::testing::WithParamInterface<
          std::tuple<size_t, MemAllocType, enum NCCL_CTRAN_REGISTER>> {};

TEST_P(CommRegisterTestParam, RegularUsage) {
  const auto& [nbytes, memType, ctranRegist] = GetParam();
  NcclCommRAII comm(globalRank, numRanks, localRank);
  EnvRAII env(NCCL_CTRAN_REGISTER, ctranRegist);
  EnvRAII env2(NCCL_LOCAL_REGISTER, (int64_t)0);

  if ((memType == kMemNcclMemAlloc || memType == kCuMemAllocDisjoint) &&
      ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  std::vector<TestMemSegment> segments;
  std::vector<void*> segHandles;
  void* buf = testAllocBuf(nbytes, memType, segments);
  ASSERT_NE(buf, nullptr);

  auto res = ncclSuccess;
  for (auto& segment : segments) {
    void* handle = nullptr;
    res = ncclCommRegister(comm, segment.ptr, segment.size, &handle);
    segHandles.push_back(handle);
    ASSERT_EQ(res, ncclSuccess);
    ASSERT_NE(handle, nullptr);
  }

  if (res == ncclSuccess) {
    for (auto& handle : segHandles) {
      res = ncclCommDeregister(comm, handle);
      ASSERT_EQ(res, ncclSuccess);
    }
  }

  // Cleanup cached segments in global cache before we test memory leak
  auto regCache = CtranMapperRegCache::getInstance();
  EXPECT_EQ(regCache->destroy(), commSuccess);

  testFreeBuf(buf, nbytes, memType);
}

TEST_F(CommRegisterTest, InvalidHybridUsage) {
  NcclCommRAII comm(globalRank, numRanks, localRank);
  EnvRAII env1(NCCL_CTRAN_REGISTER, NCCL_CTRAN_REGISTER::lazy);
  EnvRAII env2(NCCL_LOCAL_REGISTER, (int64_t)1);
  constexpr size_t nbytes = 8192;

  std::vector<TestMemSegment> segments;
  void* buf = testAllocBuf(nbytes, kMemNcclMemAlloc, segments);
  ASSERT_NE(buf, nullptr);

  void* handle = nullptr;
  auto res = ncclCommRegister(comm, buf, nbytes, &handle);
  ASSERT_EQ(res, ncclInvalidUsage);
  ASSERT_EQ(handle, nullptr);

  res = ncclCommDeregister(comm, reinterpret_cast<void*>(0x1000));
  ASSERT_EQ(res, ncclInvalidUsage);

  testFreeBuf(buf, nbytes, kMemNcclMemAlloc);
}

TEST_F(CommRegisterTest, InvalidHandle) {
  NcclCommRAII comm(globalRank, numRanks, localRank);

  auto res = ncclCommDeregister(comm, nullptr);
  ASSERT_EQ(res, ncclInvalidArgument);
}

const std::string testCtranRegisterModeToStr(enum NCCL_CTRAN_REGISTER mode) {
  switch (mode) {
    case NCCL_CTRAN_REGISTER::none:
      return "none";
    case NCCL_CTRAN_REGISTER::lazy:
      return "lazy";
    case NCCL_CTRAN_REGISTER::eager:
      return "eager";
    default:
      return "unknown";
  }
}

INSTANTIATE_TEST_SUITE_P(
    CommRegisterTest,
    CommRegisterTestParam,
    ::testing::Values(
        std::make_tuple(8192, kMemNcclMemAlloc, NCCL_CTRAN_REGISTER::eager),
        std::make_tuple(8192, kMemNcclMemAlloc, NCCL_CTRAN_REGISTER::lazy),
        // cudaMalloc can register
        std::make_tuple(8192, kMemCudaMalloc, NCCL_CTRAN_REGISTER::eager),
        // disjoint segment can register
        std::make_tuple(
            256 * 1024 * 1024,
            kCuMemAllocDisjoint,
            NCCL_CTRAN_REGISTER::eager),
        // small size (expect to fail in eager mode with cudaMalloc)
        std::make_tuple(1, kMemCudaMalloc, NCCL_CTRAN_REGISTER::eager),
        // small size (expect to pass in eager mode with cudaMem because mapper
        // finds whole aligned allocation backing which is >=
        // CTRAN_MIN_REGISTRATION_SIZE)
        std::make_tuple(1, kMemNcclMemAlloc, NCCL_CTRAN_REGISTER::eager),
        std::make_tuple(1, kMemNcclMemAlloc, NCCL_CTRAN_REGISTER::lazy)),
    [&](const testing::TestParamInfo<CommRegisterTestParam::ParamType>& info) {
      return std::to_string(std::get<0>(info.param)) + "bytes_" +
          testMemAllocTypeToStr(std::get<1>(info.param)) + "_" +
          testCtranRegisterModeToStr(std::get<2>(info.param));
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
