// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <unistd.h>

#include "comm.h" // @manual

#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"

#define CAPTURE_STDOUT_WITH_FAIL_SAFE()                                    \
  testing::internal::CaptureStdout();                                      \
  SCOPE_FAIL {                                                             \
    std::string output = testing::internal::GetCapturedStdout();           \
    std::cout << "Test failed with stdout being: " << output << std::endl; \
  };

class CollTraceTest : public RcclxBaseTest {
 public:
  CollTraceTest() = default;
  void SetUp() override {
    // Set up dummy values for environment variables for Scuba test
    setenv("WORLD_SIZE", "2", 0);
    setenv("HPC_JOB_NAME", "CollTraceUT", 0);
    setenv("HPC_JOB_VERSION", "1", 0);
    setenv("HPC_JOB_ATTEMPT_INDEX", "2", 0);
    setenv(
        "NCCL_HPC_JOB_IDS",
        "HPC_JOB_NAME,HPC_JOB_VERSION,HPC_JOB_ATTEMPT_INDEX",
        0);
    setenv("NCCL_CTRAN_ENABLE", "0", 0);

    RcclxBaseTest::SetUp();
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    // cudaFree in case test case doesn't free
    if (sendBuf) {
      CUDACHECK_TEST(cudaFree(sendBuf));
    }
    if (recvBuf) {
      CUDACHECK_TEST(cudaFree(recvBuf));
    }
  }

  void prepareAllreduce(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

  void prepareSendRecv(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  void* sendHandle{nullptr};
  void* recvHandle{nullptr};
  cudaStream_t stream;
};

TEST_F(CollTraceTest, DumpSendRecv) {
  if (this->numRanks % 2) {
    GTEST_SKIP() << "This test requires even number of ranks";
  }

  setenv("RCCL_LATENCY_PROFILER", "1", 1);
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  prepareSendRecv(count);
  for (int i = 0; i < nColl; i++) {
    // even rank sends to odd rank (e.g, 0->1, 2->3)
    if (this->globalRank % 2 == 0) {
      int peer = this->globalRank + 1;
      NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, peer, comm, stream));
    } else {
      int peer = this->globalRank - 1;
      NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, peer, comm, stream));
    }
  }
  EXPECT_TRUE(comm->ctrace != nullptr);
  comm->ctrace->waitForWorkerFinishQueue();

  auto dump = comm->ctrace->dump();
  EXPECT_EQ(dump.pastColls.size(), nColl);
  EXPECT_EQ(dump.currentColl, nullptr);

  for (auto& coll : dump.pastColls) {
    EXPECT_EQ(coll.opName, this->globalRank % 2 == 0 ? "Send" : "Recv");
    EXPECT_EQ(coll.dataType, "ncclInt8");
    EXPECT_EQ(coll.count, count * ncclTypeSize(ncclInt));
    EXPECT_GT(coll.startTs.time_since_epoch().count(), 0);
    EXPECT_GT(coll.latencyMs, 0);
  }
}

TEST_F(CollTraceTest, DumpAllFinished) {
  setenv("RCCL_LATENCY_PROFILER", "1", 1);
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->ctrace != nullptr);
  comm->ctrace->waitForWorkerFinishQueue();
  auto dump = comm->ctrace->dump();
  EXPECT_EQ(dump.pastColls.size(), nColl);
  EXPECT_EQ(dump.currentColl, nullptr);
}

TEST_F(CollTraceTest, DumpWithUnfinished) {
  setenv("RCCL_LATENCY_PROFILER", "1", 1);
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->ctrace != nullptr);
  comm->ctrace->waitForWorkerFinishQueue();

  // schedule more after the first 10 coll are finished
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  auto dump = comm->ctrace->dump();

  EXPECT_GE(dump.pastColls.size(), nColl);
  EXPECT_LE(dump.pendingColls.size(), nColl);

  auto totalSize = dump.pastColls.size() + dump.pendingColls.size();
  // nColl * 2 - 1 <= Total Size <= nColl * 2
  // We might have 1 coll that has been popped out of the queue
  // but not yet set as current collective.
  EXPECT_GE(totalSize, nColl * 2 - 1);
  EXPECT_LE(totalSize, nColl * 2);
  if (totalSize == nColl * 2) {
    // If we don't have any ongoing coll, we should have currentColl as null
    EXPECT_EQ(dump.currentColl, nullptr);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
