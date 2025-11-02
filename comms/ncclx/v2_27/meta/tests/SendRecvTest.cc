// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <comm.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "checks.h"
#include "comms/testinfra/AlgoTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/hints/GlobalHints.h"

static bool VERBOSE = true;
using testinfra::AlgoRAII;

class SendRecvTest : public NcclxBaseTest {
 public:
  SendRecvTest() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    NcclxBaseTest::SetUp();
    this->comm = createNcclComm(
        this->globalRank,
        this->numRanks,
        this->localRank,
        false,
        nullptr,
        server.get());

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    NcclxBaseTest::TearDown();
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
              this->globalRank,
              i,
              observedVals[i],
              val);
        }
        errs++;
      }
    }
    return errs;
  }

  void prepareBufs(const size_t count, bool registFlag = false) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));

    int expectedVal = comm->rank * 100 + 1;
    assignChunkValue(sendBuf, count, expectedVal);
    assignChunkValue(recvBuf, count, -1);

    if (registFlag) {
      NCCLCHECK_TEST(
          ncclCommRegister(comm, sendBuf, count * sizeof(int), &sendHandle));
      NCCLCHECK_TEST(
          ncclCommRegister(comm, recvBuf, count * sizeof(int), &recvHandle));
    }
  }

  void checkResults(const int sendRank, const size_t count) {
    int expectedVal = sendRank * 100 + 1;
    int errs = checkChunkValue(recvBuf, count, expectedVal);
    EXPECT_EQ(errs, 0) << "Rank " << this->globalRank
                       << " checked result from rank " << sendRank << " at "
                       << recvBuf << " with " << errs << " errors";
  }

  void releaseBufs(bool registFlag = false) {
    if (registFlag) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
  }

  uint64_t getOpCount(ncclComm_t ncclComm) const {
    // Since we may override algo from hints, let each test case sets whether
    // isCtranAlgo is true
    if (expectCtranAlgo_) {
      // Use Ctran-only opCount to track Ctran sendrecv is called
      return ncclComm->ctranComm_->ctran_->getCtranOpCount();
    } else {
      return ncclComm->opCount;
    }
  }

  void resetOpCount() {
    comm->opCount = 0;
  }

  void runSend(void) {
    // create and register buffers
    constexpr int count = 1048576, commCount = 1024;
    int sendRank, recvRank;
    prepareBufs(count, true);

    // only odd ranks send, and even ranks receive
    if (comm->rank % 2) {
      sendRank = comm->rank;
      recvRank = (comm->rank + 1) % comm->nRanks;
    } else {
      sendRank = (comm->rank + comm->nRanks - 1) % comm->nRanks;
      recvRank = comm->rank;
    }

    if (VERBOSE) {
      printf(
          "Rank %d sendRank %d recvRank %d\n", comm->rank, sendRank, recvRank);
    }

    for (int x = 0; x < 5; x++) {
      // Expect opCount increase per send/receive call
      const auto opCount = getOpCount(comm);
      if (comm->rank == sendRank) {
        NCCLCHECK_TEST(
            ncclSend(sendBuf, commCount, ncclInt, recvRank, comm, stream));
        EXPECT_EQ(opCount, x);
      } else if (comm->rank == recvRank) {
        NCCLCHECK_TEST(
            ncclRecv(recvBuf, commCount, ncclInt, sendRank, comm, stream));
        EXPECT_EQ(opCount, x);
      }
    }

    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    if (comm->rank == recvRank) {
      checkResults(sendRank, commCount);
    }
    releaseBufs(true);
  }

  void runGroupedSend(void) {
    // create and register buffers
    constexpr int count = 1048576, commCount = 1024;
    int sendRank, recvRank;
    prepareBufs(count, true);

    // only odd ranks send, and even ranks receive
    if (comm->rank % 2) {
      sendRank = comm->rank;
      recvRank = (comm->rank + 1) % comm->nRanks;
    } else {
      sendRank = (comm->rank + comm->nRanks - 1) % comm->nRanks;
      recvRank = comm->rank;
    }

    if (VERBOSE) {
      printf(
          "Rank %d sendRank %d recvRank %d\n", comm->rank, sendRank, recvRank);
    }

    constexpr int numGroups = 2;
    constexpr int numOpsInGroup = 5;

    for (int g = 0; g < numGroups; g++) {
      if (comm->rank == sendRank) {
        // Expect opCount increase per group of send/receive calls
        const auto opCount = getOpCount(comm);
        ncclGroupStart();
        for (int x = 0; x < numOpsInGroup; x++) {
          NCCLCHECK_TEST(
              ncclSend(sendBuf, commCount, ncclInt, recvRank, comm, stream));
        }
        ncclGroupEnd();
        EXPECT_EQ(opCount, g);
      } else if (comm->rank == recvRank) {
        const auto opCount = getOpCount(comm);
        ncclGroupStart();
        for (int x = 0; x < numOpsInGroup; x++) {
          NCCLCHECK_TEST(
              ncclRecv(recvBuf, commCount, ncclInt, sendRank, comm, stream));
        }
        ncclGroupEnd();
        EXPECT_EQ(opCount, g);
      }
    }

    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    if (comm->rank == recvRank) {
      checkResults(sendRank, commCount);
    }
    releaseBufs(true);
  }

  void runGroupedSendRecv(void) {
    // create and register buffers
    constexpr int count = 1048576, commCount = 1024;
    int sendRank, recvRank;
    prepareBufs(count, true);

    // every rank sends to the next and receives from previous
    sendRank = (comm->rank + 1) % comm->nRanks;
    recvRank = (comm->rank + comm->nRanks - 1) % comm->nRanks;

    if (VERBOSE) {
      printf(
          "Rank %d sendRank %d recvRank %d\n", comm->rank, sendRank, recvRank);
    }

    for (int x = 0; x < 5; x++) {
      // Expect opCount increase per group of send/receive calls
      const auto opCount = getOpCount(comm);
      ncclGroupStart();
      NCCLCHECK_TEST(
          ncclSend(sendBuf, commCount, ncclInt, recvRank, comm, stream));
      NCCLCHECK_TEST(
          ncclRecv(recvBuf, commCount, ncclInt, sendRank, comm, stream));
      ncclGroupEnd();
      EXPECT_EQ(opCount, x);
    }

    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    checkResults(sendRank, commCount);
    releaseBufs(true);
  }

  void runSendRecvSelf(void) {
    // create and register buffers
    constexpr int count = 1048576, commCount = 1024;
    prepareBufs(count, true);

    for (int x = 0; x < 5; x++) {
      // groupEnd() will increase opCount; query here for the current round
      // NOTE: we have fixed opCount update in D83294734 to ensure it can be
      // exactly increased per kernel launch
      const auto opCount = getOpCount(comm);
      ncclGroupStart();
      NCCLCHECK_TEST(
          ncclSend(sendBuf, commCount, ncclInt, comm->rank, comm, stream));
      NCCLCHECK_TEST(
          ncclRecv(recvBuf, commCount, ncclInt, comm->rank, comm, stream));
      ncclGroupEnd();
      EXPECT_EQ(opCount, x);
    }

    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    checkResults(comm->rank, commCount);
    releaseBufs(true);
  }

 protected:
  ncclComm_t comm;
  cudaStream_t stream;
  bool expectCtranAlgo_{false};

  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  void* sendHandle{nullptr};
  void* recvHandle{nullptr};
};

class SendRecvTestParam : public SendRecvTest,
                          public ::testing::WithParamInterface<
                              std::tuple<enum NCCL_SENDRECV_ALGO>> {};

TEST_P(SendRecvTestParam, SingleSend) {
  auto& [algo] = GetParam();
  AlgoRAII algoEnv(NCCL_SENDRECV_ALGO, algo);

  expectCtranAlgo_ = algo == NCCL_SENDRECV_ALGO::ctran;

  runSend();
}

TEST_P(SendRecvTestParam, GroupdSend) {
  auto& [algo] = GetParam();
  AlgoRAII algoEnv(NCCL_SENDRECV_ALGO, algo);
  expectCtranAlgo_ = algo == NCCL_SENDRECV_ALGO::ctran;

  runGroupedSend();
}

TEST_P(SendRecvTestParam, GroupedSendRecv) {
  auto& [algo] = GetParam();
  AlgoRAII algoEnv(NCCL_SENDRECV_ALGO, algo);
  expectCtranAlgo_ = algo == NCCL_SENDRECV_ALGO::ctran;

  runGroupedSendRecv();
}

TEST_P(SendRecvTestParam, SendRecvSelf) {
  auto& [algo] = GetParam();
  AlgoRAII algoEnv(NCCL_SENDRECV_ALGO, algo);
  expectCtranAlgo_ = algo == NCCL_SENDRECV_ALGO::ctran;

  runSendRecvSelf();
}

TEST_F(SendRecvTestParam, GroupedSendRecvWithHintOverride) {
  AlgoRAII algoEnv(NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::orig);

  ASSERT_TRUE(ncclx::setGlobalHint("algo_sendrecv", "ctran"));
  expectCtranAlgo_ = true;
  runGroupedSendRecv();

  // Ctran algo would also update the opCount shared with baseline; reset for
  // baseline test
  resetOpCount();

  ASSERT_TRUE(ncclx::resetGlobalHint("algo_sendrecv"));
  expectCtranAlgo_ = false;
  runGroupedSendRecv();
}

INSTANTIATE_TEST_SUITE_P(
    SendRecvTest,
    SendRecvTestParam,
    ::testing::Values(NCCL_SENDRECV_ALGO::orig, NCCL_SENDRECV_ALGO::ctran),
    [&](const testing::TestParamInfo<SendRecvTestParam::ParamType>& info) {
      return std::get<0>(info.param) == NCCL_SENDRECV_ALGO::orig ? "Baseline"
                                                                 : "Ctran";
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
