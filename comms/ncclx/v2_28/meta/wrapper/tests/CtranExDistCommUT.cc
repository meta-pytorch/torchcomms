// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <stdlib.h>
#include "comms/ctran/CtranEx.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/wrapper/CtranExComm.h"

#include "gmock/gmock.h"
#include "nccl.h"

using namespace ctran;

class CtranExCommTest : public CtranExBaseTest {
 public:
  void SetUp() override {
    CtranExBaseTest::SetUp();
  }

 protected:
  template <typename T>
  void
  assignChunkHostValue(std::vector<T>& buf, size_t count, T valSeed, T valInc) {
    for (int i = 0; i < count; i++) {
      buf.at(i) = valSeed + valInc * i;
    }
  }

  template <typename T>
  size_t
  checkChunkHostValue(std::vector<T>& buf, ssize_t count, T seed, T inc) {
    size_t errs = 0;
    // Use manual print rather than EXPECT_THAT to print first 10 failing
    // location
    for (auto i = 0; i < count; ++i) {
      T expVal = seed + i * inc;
      if (buf[i] != expVal) {
        if (errs < 10) {
          printf(
              "[%d] buf[%d] = %d, expectedVal = %d\n",
              globalRank,
              i,
              buf[i],
              expVal);
        }
        errs++;
      }
    }
    return errs;
  }
};

TEST_F(CtranExCommTest, Initialized) {
  auto ctranExComm = std::make_unique<CtranExComm>(ncclComm_, "ctranExCommUT");
  EXPECT_TRUE(ctranExComm->isInitialized());
}

TEST_F(CtranExCommTest, TestNonBlockingThrow) {
  ncclComm_t nonBlockingComm;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  config.commDesc = "nonBlockingParent";
  ncclUniqueId id;
  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (globalRank == 0) {
    NCCLCHECK_TEST(ncclGetUniqueId(&id));
  }
  MPICHECK_TEST(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  CUDACHECK_TEST(cudaSetDevice(localRank));
  auto nccl_result = ncclCommInitRankConfig(
      &nonBlockingComm, numRanks, id, globalRank, &config);
  do {
    NCCLCHECK_TEST(ncclCommGetAsyncError(nonBlockingComm, &nccl_result));
    // Handle outside events, timeouts, progress, ...
  } while (nccl_result == ncclInProgress);

  try {
    auto ctranExComm =
        CtranExComm(nonBlockingComm, "ctranExCommTestNonBlockingThrow");
  } catch (const std::exception& e) {
    EXPECT_STREQ(e.what(), "CTRAN-EX: parent communicator is non-blocking");
  }
  ncclCommDestroy(nonBlockingComm);
}

TEST_F(CtranExCommTest, RegHostMem) {
  EnvRAII env(NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT, 0);

  auto ctranExComm = std::make_unique<CtranExComm>(ncclComm_, "ctranExCommUT");
  EXPECT_TRUE(ctranExComm->isInitialized());

  const size_t count = 4096;
  std::vector<int> dataH(count);

  void* segHdl = nullptr;
  // Register the buffer as CTran cached segment
  ctranExComm->regMem(dataH.data(), count * sizeof(int), &segHdl);
  EXPECT_NE(segHdl, nullptr);

  // Deregister the segment record
  ctranExComm->deregMem(segHdl);
}

TEST_F(CtranExCommTest, RegHostMemEager) {
  EnvRAII env(NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT, 0);

  auto ctranExComm = std::make_unique<CtranExComm>(ncclComm_, "ctranExCommUT");
  EXPECT_TRUE(ctranExComm->isInitialized());

  const size_t count = 4096;
  std::vector<int> dataH(count);

  void* segHdl = nullptr;
  // Register the buffer as CTran cached segment and also force the backend
  // registration
  ctranExComm->regMem(
      dataH.data(), count * sizeof(int), &segHdl, true /* forceRegister */);
  EXPECT_NE(segHdl, nullptr);

  // Deregister the segment record and also internal backend registration
  ctranExComm->deregMem(segHdl);
}

class CtranExCommBroadcastFixture
    : public CtranExCommTest,
      public ::testing::WithParamInterface<
          std::tuple<size_t, TestInPlaceType, bool>> {};

TEST_P(CtranExCommBroadcastFixture, BcastWithReq) {
  const auto& [count, inplace, concurrentColl] = GetParam();
  const size_t pageSize = getpagesize();
  const int nIter = 10;

  auto ctranExComm = std::make_unique<CtranExComm>(ncclComm_, "ctranExCommUT");

  ASSERT_NE(ctranExComm, nullptr);
  EXPECT_TRUE(ctranExComm->isInitialized());
  EXPECT_TRUE(ctranExComm->supportBroadcast());

  // Always allocate and register with aligned size
  size_t alignedCount =
      (count * sizeof(int) + pageSize - 1) / pageSize * pageSize / sizeof(int);

  std::vector<int> inputBuf(alignedCount, 0);
  std::vector<int> outputBuf(alignedCount, 0);
  void* inputBufHdl = nullptr;

  ASSERT_EQ(
      ctranExComm->regMem(
          inputBuf.data(),
          alignedCount * sizeof(int),
          &inputBufHdl,
          true /* forceRegister */),
      commSuccess);
  ASSERT_NE(inputBufHdl, nullptr);

  std::vector<int>& outputBufRef =
      inplace == kTestOutOfPlace ? outputBuf : inputBuf;
  void* outputBufHdl = nullptr;
  if (inplace == kTestOutOfPlace) {
    ASSERT_EQ(
        ctranExComm->regMem(
            outputBufRef.data(),
            alignedCount * sizeof(int),
            &outputBufHdl,
            true /* forceRegister */),
        commSuccess);
    ASSERT_NE(outputBufHdl, nullptr);
  }

  // Setup buffers for concurrent bcast using original communicator
  int* inputBuf2 = nullptr;
  void* inputBufHdl2 = nullptr;
  cudaStream_t stream = nullptr;
  if (concurrentColl) {
    CUDACHECK_TEST(cudaMalloc(&inputBuf2, alignedCount * sizeof(int)));
    ASSERT_NE(inputBuf2, nullptr);
    NCCLCHECK_TEST(ncclCommRegister(
        ncclComm_, inputBuf2, alignedCount * sizeof(int), &inputBufHdl2));
    ASSERT_NE(inputBufHdl2, nullptr);
    CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  // Run bcast and allreduce in parallel by two threads
  std::thread sideThread(
      [&](const size_t count) {
        for (int x = 0; x < nIter; x++) {
          // Rank 0 assigns the input data
          if (ncclComm_->rank == 0) {
            assignChunkHostValue(inputBuf, count, x, 1);
          }

          std::unique_ptr<CtranExRequest> req;
          CtranExRequest* reqPtr = nullptr;
          ASSERT_EQ(
              ctranExComm->broadcast(
                  inputBuf.data(),
                  outputBufRef.data(),
                  count,
                  ncclInt,
                  0,
                  &reqPtr),
              commSuccess);

          ASSERT_NE(reqPtr, nullptr);
          req.reset(reqPtr);

          // Wait completion of the bcast
          bool complete = false;
          while (!complete) {
            ASSERT_EQ(req->test(complete), ncclSuccess);
          }

          // Check result
          EXPECT_EQ(checkChunkHostValue(outputBufRef, count, x, 1), 0);
        }
      },
      count);

  if (concurrentColl) {
    // mainThread runs a separate Ctran bcast concurrently.
    // Expect both threads to finish without error.
    EnvRAII defaultCommBcast =
        EnvRAII(NCCL_BROADCAST_ALGO, NCCL_BROADCAST_ALGO::ctran);
    EXPECT_TRUE(ctranBroadcastSupport(
        ncclComm_->ctranComm_.get(), NCCL_BROADCAST_ALGO));
    for (int x = 0; x < nIter; x++) {
      const int root = 0;
      if (ncclComm_->rank == root) {
        // Specify different value than the sideThread one
        int val = root + x + 99;
        assignChunkValue(inputBuf2, count, val, 2);
        CUDACHECK_TEST(cudaDeviceSynchronize());
      }

      std::unique_ptr<CtranExRequest> req;
      ASSERT_EQ(
          ncclBcast(inputBuf2, count, ncclInt, root, ncclComm_, stream),
          ncclSuccess);
      CUDACHECK_TEST(cudaStreamSynchronize(stream));
      // Check result
      int expVal = root + x + 99;
      EXPECT_EQ(checkChunkValue(inputBuf2, count, expVal, 2), 0);
    }
  }

  sideThread.join();

  // Ensure all ranks are done before deregistering buffer
  barrier();

  ctranExComm->deregMem(inputBufHdl);
  if (inplace == kTestOutOfPlace) {
    ctranExComm->deregMem(outputBufHdl);
  }

  auto exComm = ctranExComm->unsafeGetNcclComm();
  EXPECT_NE(exComm, nullptr);
  // internal communicator should not allocate any channels
  EXPECT_EQ(exComm->nChannelsReady, 0);

  if (concurrentColl) {
    NCCLCHECK_TEST(ncclCommDeregister(ncclComm_, inputBufHdl2));
    CUDACHECK_TEST(cudaFree(inputBuf2));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
  }
}

// Basic test cases for testing broadcast with request and without stream;
// more edge cases tested by CtranDistBroadcastTests.cc
INSTANTIATE_TEST_SUITE_P(
    CtranExCommBroadcastTest,
    CtranExCommBroadcastFixture,
    ::testing::Combine(
        ::testing::Values(
            1024 * 1024 * 64UL,
            // unaligned size
            1024 * 1024 * 64UL + 5),
        ::testing::Values(kTestInPlace, kTestOutOfPlace),
        // concurrentColl
        ::testing::Values(true)),
    [&](const ::testing::TestParamInfo<CtranExCommBroadcastFixture::ParamType>&
            info) {
      return std::to_string(std::get<0>(info.param)) + "int_" +
          testInPlaceTypeToStr(std::get<1>(info.param)) + "_concurrentColl_" +
          std::to_string(std::get<2>(info.param));
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
