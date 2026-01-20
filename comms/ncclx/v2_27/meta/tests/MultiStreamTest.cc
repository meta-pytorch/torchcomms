// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

class MultiStreamTest : public NcclxBaseTest {
 public:
  MultiStreamTest() = default;
  void SetUp() override {
    NcclxBaseTest::SetUp();
    comm = createNcclComm(globalRank, numRanks, localRank);

    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    NcclxBaseTest::TearDown();
  }

 protected:
  ncclComm_t comm;
};

class MultiStreamAllGatherTestParam
    : public MultiStreamTest,
      public ::testing::WithParamInterface<std::tuple<
          enum NCCL_ALLGATHER_ALGO,
          int /* number of streams */
          >> {};

TEST_P(MultiStreamAllGatherTestParam, Test) {
  const auto& [algo, numStreams] = GetParam();

  const size_t bufCount = 16 * 1e6;
  constexpr int numIter = 1000;

  auto envGuard = EnvRAII(NCCL_ALLGATHER_ALGO, algo);

  std::vector<cudaStream_t> streams(numStreams);
  for (auto& stream : streams) {
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  // Each stream carries different amount of data
  std::vector<size_t> counts(numStreams);
  for (int i = 0; i < numStreams; i++) {
    counts[i] = std::min(8192UL + 8192 * i, bufCount);
  }

  std::vector<int*> bufs(numStreams);
  for (int i = 0; i < numStreams; i++) {
    auto& buf = bufs[i];
    NCCLCHECK_TEST(
        ncclMemAlloc((void**)&buf, bufCount * numRanks * sizeof(int)));
    assignChunkValue(buf, bufCount * numRanks, -1);

    // each stream transfers different values for correctness check
    int expectedVal = globalRank + i * bufCount;
    int* sendBuf = buf + counts[i] * this->globalRank;
    assignChunkValue(sendBuf, counts[i], expectedVal);
  }

  std::vector<void*> segHdls(numStreams);
  for (int i = 0; i < numStreams; i++) {
    auto& buf = bufs[i];
    NCCLCHECK_TEST(ncclCommRegister(comm, buf, bufCount, &segHdls[i]));
  }

  // Run communication
  for (int iter = 0; iter < numIter; iter++) {
    for (int i = 0; i < numStreams; i++) {
      auto& buf = bufs[i];
      int* sendBuf = buf + counts[i] * this->globalRank;
      ASSERT_EQ(
          ncclAllGather(sendBuf, buf, counts[i], ncclInt, comm, streams[i]),
          ncclSuccess);
    }
  }

  for (int i = 0; i < numStreams; i++) {
    CUDACHECK_TEST(cudaStreamSynchronize(streams[i]));
  }

  // Check each received chunk
  for (int i = 0; i < numStreams; i++) {
    auto& buf = bufs[i];

    for (int r = 0; r < numRanks; r++) {
      int expectedVal = r + i * bufCount;
      int* recvBuf = buf + counts[i] * r; // recv chunkt from this rank

      int errs = checkChunkValue(recvBuf, counts[i], expectedVal);
      EXPECT_EQ(errs, 0)
          << fmt::format(
                 "Rank {} checked chunk received from peer rank {} in bufs[{}], with errors {}",
                 globalRank,
                 r,
                 i,
                 errs)
          << std::endl;
    }
  }

  for (int i = 0; i < numStreams; i++) {
    NCCLCHECK_TEST(ncclCommDeregister(comm, segHdls[i]));
    NCCLCHECK_TEST(ncclMemFree(bufs[i]));
    CUDACHECK_TEST(cudaStreamDestroy(streams[i]));
  }
}

class MultiStreamAllToAllTestParam
    : public MultiStreamTest,
      public ::testing::WithParamInterface<std::tuple<
          enum NCCL_ALLTOALL_ALGO,
          int /* number of streams */
          >> {};

TEST_P(MultiStreamAllToAllTestParam, Test) {
  const auto& [algo, numStreams] = GetParam();

  const size_t bufCount = 16 * 1e6;
  constexpr int numIter = 1000;

  auto envGuard = EnvRAII(NCCL_ALLTOALL_ALGO, algo);

  std::vector<cudaStream_t> streams(numStreams);
  for (auto& stream : streams) {
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  // Each stream carries different amount of data
  std::vector<size_t> counts(numStreams);
  for (int i = 0; i < numStreams; i++) {
    counts[i] = std::min(8192UL * (i + 1), bufCount);
  }

  std::vector<int*> sbufs(numStreams), rbufs(numStreams);
  for (int i = 0; i < numStreams; i++) {
    auto& sbuf = sbufs[i];
    auto& rbuf = rbufs[i];
    NCCLCHECK_TEST(
        ncclMemAlloc((void**)&sbuf, bufCount * numRanks * sizeof(int)));
    NCCLCHECK_TEST(
        ncclMemAlloc((void**)&rbuf, bufCount * numRanks * sizeof(int)));
    assignChunkValue(rbuf, bufCount * numRanks, -1);

    // each stream transfers different values for correctness check
    int expectedVal = globalRank + i * bufCount;
    assignChunkValue(sbuf, counts[i] * numRanks, expectedVal);
  }

  // Run communication
  for (int x = 0; x < numIter; x++) {
    for (int i = 0; i < numStreams; i++) {
      ASSERT_EQ(
          ncclAllToAll(
              sbufs[i], rbufs[i], counts[i], ncclInt, comm, streams[i]),
          ncclSuccess);
    }
    std::cout << fmt::format(
                     "Rank {} finished {}-th AllToAlls with {} streams",
                     globalRank,
                     x,
                     numStreams)
              << std::endl;
  }

  for (int i = 0; i < numStreams; i++) {
    CUDACHECK_TEST(cudaStreamSynchronize(streams[i]));
  }

  // Check each received chunk
  for (int i = 0; i < numStreams; i++) {
    auto& buf = rbufs[i];

    for (int r = 0; r < numRanks; r++) {
      int expectedVal = r + i * bufCount;
      int* recvBuf = buf + counts[i] * r; // recv chunk from this rank

      int errs = checkChunkValue(recvBuf, counts[i], expectedVal);
      EXPECT_EQ(errs, 0)
          << fmt::format(
                 "Rank {} checked chunk received from peer rank {} in bufs[{}], with errors {}",
                 globalRank,
                 r,
                 i,
                 errs)
          << std::endl;
    }
  }

  for (int i = 0; i < numStreams; i++) {
    NCCLCHECK_TEST(ncclMemFree(sbufs[i]));
    NCCLCHECK_TEST(ncclMemFree(rbufs[i]));
    CUDACHECK_TEST(cudaStreamDestroy(streams[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    MultiStreamAllGatherTestInstance,
    MultiStreamAllGatherTestParam,
    ::testing::Values(
#ifdef TEST_ENABLE_CTRAN
        std::make_tuple(NCCL_ALLGATHER_ALGO::ctdirect, 10)
#else
        std::make_tuple(NCCL_ALLGATHER_ALGO::orig, 10)
#endif
            ),
    [&](const testing::TestParamInfo<MultiStreamAllGatherTestParam::ParamType>&
            info) {
      return fmt::format(
          "{}_{}streams",
          allGatherAlgoName(std::get<0>(info.param)),
          std::to_string(std::get<1>(info.param)));
    });

INSTANTIATE_TEST_SUITE_P(
    MultiStreamAllToAllTestInstance,
    MultiStreamAllToAllTestParam,
    ::testing::Values(
#ifdef TEST_ENABLE_CTRAN
        std::make_tuple(NCCL_ALLTOALL_ALGO::ctran, 10)
#else
        std::make_tuple(NCCL_ALLTOALL_ALGO::orig, 10)
#endif
            ),
    [&](const testing::TestParamInfo<MultiStreamAllToAllTestParam::ParamType>&
            info) {
      return fmt::format(
          "{}_{}streams",
          allToAllAlgoName(std::get<0>(info.param)),
          std::to_string(std::get<1>(info.param)));
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
