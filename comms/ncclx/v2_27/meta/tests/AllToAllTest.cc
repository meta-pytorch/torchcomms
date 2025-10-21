// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "checks.h"

#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/testinfra/tests_common.cuh"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTrace.h"

static const int kTotalColls = 5;

class AllToAllTest : public NcclxBaseTest {
 public:
  AllToAllTest() = default;
  void SetUp() override {
#ifdef TEST_ENABLE_CTRAN
    setenv("NCCL_COLLTRACE", "trace", 0);
#endif

    NcclxBaseTest::SetUp();

    this->comm = createNcclComm(globalRank, numRanks, localRank);

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T val) {
    std::vector<T> expectedVals(count, val);
    CUDACHECKIGNORE(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print failing location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              globalRank,
              i,
              observedVals[i],
              val);
        }
        errs++;
      }
    }
    return errs;
  }

  void run(bool registFlag = false) {
#ifdef NCCL_ALLTOALL_SUPPORTED

    // create and register buffers
    constexpr int count = 1048576;
    int *sendBuf = nullptr, *recvBuf = nullptr;

    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * numRanks * sizeof(int)));

    for (int r = 0; r < numRanks; r++) {
      int expectedVal = globalRank * 100 + r + 1;
      assignChunkValue(sendBuf + r * count, count, expectedVal);
      assignChunkValue(recvBuf + r * count, count, -1);
    }

#ifdef NCCL_REGISTRATION_SUPPORTED
    if (registFlag) {
      NCCLCHECK_TEST(ncclCommRegister(
          comm, sendBuf, count * numRanks * sizeof(int), &sendHandle));
      NCCLCHECK_TEST(ncclCommRegister(
          comm, recvBuf, count * numRanks * sizeof(int), &recvHandle));
    }
#endif

    // run alltoall
    for (int i = 0; i < kTotalColls; i++) {
      auto res = ncclAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream);
      ASSERT_EQ(res, ncclSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    for (int r = 0; r < numRanks; r++) {
      int expectedVal = r * 100 + globalRank + 1;
      int errs = checkChunkValue(recvBuf + r * count, count, expectedVal);
      EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << r
                         << " at " << recvBuf + r * count << " with " << errs
                         << " errors";
    }

#ifdef NCCL_REGISTRATION_SUPPORTED
    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }
#endif

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));

#ifdef TEST_ENABLE_CTRAN
    // CollTrace is updated by a separate thread, need wait for it to finish to
    // avoid flaky test
    comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();
    auto dump = comm->ctranComm_->collTrace_->dump();
    int totalColls = kTotalColls;
    if (totalColls > NCCL_COLLTRACE_RECORD_MAX) {
      totalColls = NCCL_COLLTRACE_RECORD_MAX;
    }
    if (count == 0) {
      totalColls = 0;
    }
    EXPECT_EQ(dump.pastColls.size(), totalColls);

    for (auto& coll : dump.pastColls) {
      if (NCCL_ALLTOALL_ALGO == NCCL_ALLTOALL_ALGO::ctran) {
        EXPECT_EQ(coll.count, count);
        EXPECT_EQ(coll.dataType, commInt);
        EXPECT_EQ(coll.opName, "AllToAll");
        EXPECT_EQ(coll.codepath, CollTraceColl::Codepath::CTRAN);
      } else {
        EXPECT_EQ(coll.opName, "SendRecv");
        EXPECT_EQ(coll.codepath, CollTraceColl::Codepath::BASELINE);
      }
    }
#endif
#endif
  }

 protected:
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(AllToAllTest, OutOfPlace) {
  run();
}

#ifdef TEST_ENABLE_CTRAN
TEST_F(AllToAllTest, Ctran) {
  auto envGuard = EnvRAII(NCCL_ALLTOALL_ALGO, NCCL_ALLTOALL_ALGO::ctran);
  run();
}
#endif

TEST_F(AllToAllTest, InvalidSendbuf) {
#ifdef NCCL_ALLTOALL_SUPPORTED
  constexpr int count = 1048576;
  int* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, count * numRanks * sizeof(int)));

  // run alltoall
  auto res = ncclAllToAll(nullptr, buf, count, ncclInt, comm, stream);
  ASSERT_EQ(res, ncclInvalidArgument);
  CUDACHECK_TEST(cudaFree(buf));
#endif
}

TEST_F(AllToAllTest, InvalidRecvbuf) {
#ifdef NCCL_ALLTOALL_SUPPORTED
  constexpr int count = 1048576;
  int* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, count * numRanks * sizeof(int)));

  // run alltoall
  auto res = ncclAllToAll(buf, nullptr, count, ncclInt, comm, stream);
  ASSERT_EQ(res, ncclInvalidArgument);
  CUDACHECK_TEST(cudaFree(buf));
#endif
}

TEST_F(AllToAllTest, InvalidInPlace) {
#ifdef NCCL_ALLTOALL_SUPPORTED
  constexpr int count = 1048576;
  int* buf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buf, count * numRanks * sizeof(int)));

  // run alltoall
  auto res = ncclAllToAll(buf, buf, count, ncclInt, comm, stream);
  ASSERT_EQ(res, ncclInvalidArgument);
  CUDACHECK_TEST(cudaFree(buf));
#endif
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
