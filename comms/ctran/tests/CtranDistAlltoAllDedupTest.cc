// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <new>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllDedupImpl.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/colltrace/CollTrace.h"

class ctranAllToAllDedupTest : public CtranDistBaseTest {
 public:
  ctranAllToAllDedupTest() = default;

  void generateUnevenRailDisps() {
    size_t sendTotalCount = 0;
    size_t recvTotalCount = 0;
    int nNodes = comm->ctranComm_->statex_->nNodes();
    int myNode = comm->ctranComm_->statex_->rank() /
        comm->ctranComm_->statex_->nLocalRanks();
    if (myNode == 0) {
      // send count to self and count/2 to everyone else
      // get count from everyone
      sendCounts[0] = count;
      recvCounts[0] = count;
      for (int i = 1; i < nNodes; i++) {
        sendCounts[i] = count / 2;
        recvCounts[i] = count;
      }
    } else {
      // send count to node 0 and count/2 to everyone else
      // get count/2 from everyone
      sendCounts[0] = count;
      recvCounts[0] = count / 2;
      for (int i = 1; i < nNodes; i++) {
        sendCounts[i] = count / 2;
        recvCounts[i] = count / 2;
      }
    }

    for (int i = 0; i < nNodes; ++i) {
      sendDisps[i] = sendTotalCount;
      recvDisps[i] = recvTotalCount;
      sendTotalCount += sendCounts[i];
      recvTotalCount += recvCounts[i];
    }
  }

  void generateRailDisps() {
    size_t sendTotalCount = 0;
    size_t recvTotalCount = 0;
    int nNodes = comm->ctranComm_->statex_->nNodes();
    for (int i = 0; i < nNodes; ++i) {
      sendCounts[i] = count;
      recvCounts[i] = count;
      sendDisps[i] = sendTotalCount;
      recvDisps[i] = recvTotalCount;
      sendTotalCount += sendCounts[i];
      recvTotalCount += recvCounts[i];
    }
  }

  // Split up parallel alltoall into nLocalRanks chunks where each local rank
  // is responsible for a specific split
  void computeParallelSplitOffsets() {
    int nNodes = comm->ctranComm_->statex_->nNodes();
    int nLocalRanks = comm->ctranComm_->statex_->nLocalRanks();
    for (int i = 0; i < nNodes; i++) {
      size_t send_split_size = sendCounts[i] / nLocalRanks;
      size_t recv_split_size = recvCounts[i] / nLocalRanks;
      splitSendCounts[i] = send_split_size;
      splitRecvCounts[i] = recv_split_size;
      splitSendDisps[i] = sendDisps[i] + send_split_size * localRank;
      splitRecvDisps[i] = recvDisps[i] + recv_split_size * localRank;
    }
  }

  void* createDataBuf(size_t nbytes, void** handle) {
    void* buf = nullptr;
    // Allocate data buffer, and assign different value for each send chunk
    NCCLCHECK_TEST(ncclMemAlloc(&buf, nbytes));
    if (buf && handle) {
      NCCLCHECK_TEST(ncclCommRegister(comm, buf, nbytes, handle));
    }
    return buf;
  }

  void releaseDataBuf(void* buf, void* handle) {
    if (handle) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, handle));
    }
    NCCLCHECK_TEST(ncclMemFreeWithRefCheck(buf));
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t chunkCount, T val) {
    std::vector<T> expectedVals(chunkCount, val);
    CUDACHECK_TEST(cudaMemcpy(
        buf,
        expectedVals.data(),
        chunkCount * sizeof(T),
        cudaMemcpyHostToDevice));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t chunkCount, T val) {
    std::vector<T> observedVals(chunkCount, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, chunkCount * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print first 10 failing
    // location
    for (auto i = 0; i < chunkCount; ++i) {
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

  void SetUp() override {
    CtranDistBaseTest::SetUp();
    comm = commWorld;
    if (!ctranAllToAllDedupSupport(comm->ctranComm_.get())) {
      GTEST_SKIP() << "Skip the test because ctranAllToAllv is not supported";
    }

    // Allocate enough space for arguments, value assignment set in each test
    sendBuf = nullptr;
    sendHdl = nullptr;
    recvHdl = nullptr;
    int nNodes = comm->ctranComm_->statex_->nNodes();
    sendCounts.resize(nNodes, 0);
    recvCounts.resize(nNodes, 0);
    sendDisps.resize(nNodes, 0);
    recvDisps.resize(nNodes, 0);
    splitSendCounts.resize(nNodes, 0);
    splitRecvCounts.resize(nNodes, 0);
    splitSendDisps.resize(nNodes, 0);
    splitRecvDisps.resize(nNodes, 0);
  }

  void TearDown() override {
    CtranDistBaseTest::TearDown();
  }

  void run() {
    // Assign different value for each send chunk
    int nLocalRanks = comm->ctranComm_->statex_->nLocalRanks();
    int numSplits = nLocalRanks;
    int nNodes = comm->ctranComm_->statex_->nNodes();

    comm->ctranComm_->collTrace_->resetPastColls();

    CtranPersistentRequest* request = nullptr;
    void* recvBuf = nullptr;
    COMMCHECK_TEST(ctranAllToAllDedupInit(
        sendBuf,
        splitSendCounts.data(),
        splitSendDisps.data(),
        maxSendCount,
        recvBuf,
        splitRecvCounts.data(),
        splitRecvDisps.data(),
        maxRecvCount,
        commInt,
        comm->ctranComm_.get(),
        stream,
        request));

    for (int x = 0; x < numTimesRunExec; x++) {
      if (execs[x] == "even") {
        generateRailDisps();
      } else if (execs[x] == "uneven") {
        generateUnevenRailDisps();
      }
      computeParallelSplitOffsets();
      for (int i = 0; i < nNodes; ++i) {
        // get send buff to ith node
        int* curSendBuf = sendBuf + sendDisps[i];
        for (int j = 0; j < numSplits; j++) {
          size_t split_size = sendCounts[i] / numSplits;
          assignChunkValue<int>(
              curSendBuf + j * split_size,
              split_size,
              expectedVal + x * 1000 + (globalRank / nLocalRanks) * 100 +
                  i * 10 + j + 1);
        }
      }

      auto res = ctranAllToAllDedupExec(request);
      ASSERT_EQ(res, commSuccess);
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      // Check each received chunk
      for (int i = 0; i < nNodes; ++i) {
        int* curRecvBuf = (int*)recvBuf + recvDisps[i];
        for (int j = 0; j < numSplits; j++) {
          size_t split_size = recvCounts[i] / numSplits;
          int errs = checkChunkValue<int>(
              curRecvBuf + j * split_size,
              split_size,
              expectedVal + x * 1000 + (globalRank / nLocalRanks) * 10 +
                  i * 100 + j + 1);
          EXPECT_EQ(errs, 0)
              << "rank " << globalRank << " checked chunk " << i << " at "
              << (int*)recvBuf + recvDisps[i] << " with " << errs << " errors";
        }
      }
    }

    auto destroyRes = ctranAllToAllDedupDestroy(request);
    ASSERT_EQ(destroyRes, commSuccess);

    // CollTrace is updated by a separate thread, need wait for it to finish to
    // avoid flaky test
    comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

    auto dump = comm->ctranComm_->collTrace_->dump();
    // make sure we are collecting enough records
    EXPECT_GE(NCCL_COLLTRACE_RECORD_MAX, numTimesRunExec);
    EXPECT_EQ(dump.pastColls.size(), numTimesRunExec);

    for (auto& coll : dump.pastColls) {
      EXPECT_EQ(coll.opName, "AllToAllDedup");
      // Count should be nullOpt for AllToAllV at the moment
      EXPECT_THAT(coll.count, ::testing::Eq(std::nullopt));
      EXPECT_EQ(coll.dataType, commInt);
      EXPECT_EQ(coll.algoName, "ctranAllToAllDedup");
    }

    size_t sendCount =
        std::accumulate(sendCounts.begin(), sendCounts.end(), 0UL);
    if (sendCount > 0) {
      // Alltoall uses kernel staged copy not NVL iput
      std::vector<CtranMapperBackend> excludedBackends = {
          CtranMapperBackend::NVL};
      // If single node, uses only kernel staged copy
      if (comm->ctranComm_->statex_->nNodes() == 1) {
        excludedBackends.push_back(CtranMapperBackend::IB);
      }
      verifyBackendsUsed(
          comm->ctranComm_->ctran_.get(),
          comm->ctranComm_->statex_.get(),
          kMemCudaMalloc,
          excludedBackends);
    }
  }

 protected:
  ncclComm_t comm{nullptr};
  int* sendBuf{nullptr};
  size_t countSet{0};
  std::vector<size_t> sendCounts;
  std::vector<size_t> recvCounts;
  std::vector<size_t> sendDisps;
  std::vector<size_t> recvDisps;
  std::vector<size_t> splitSendCounts;
  std::vector<size_t> splitRecvCounts;
  std::vector<size_t> splitSendDisps;
  std::vector<size_t> splitRecvDisps;
  size_t count{0};
  size_t maxSendCount{0};
  size_t maxRecvCount{0};
  void* sendHdl{nullptr};
  void* recvHdl{nullptr};
  int expectedVal{0};
  int numTimesRunExec{3};
  std::vector<std::string> execs;
};

// Set up buffers so each local rank gets the same information
// Vary the values so splitting will yield different values
// Input buffers
//           Local rank 0          Local rank 1
//           001  002  003  004    001  002  003  004    ...
// Node0     011  012  013  014    011  012  013  014    ...
//           021  022  023  024    021  022  023  024    ...
//
//           Local rank 0          Local rank 1
//           101  102  103  104    101  102  103  104    ...
// Node1     111  112  113  104    111  112  113  114    ...
//           121  122  123  104    121  122  123  124    ...
//
//           Local rank 0          Local rank 1
//           201  202  203  204    201  202  203  204    ...
// Node2     211  212  213  214    211  212  213  214    ...
//           221  222  223  224    221  222  223  224    ...
//
// Expected Output buffers
//           Local rank 0          Local rank 1
//           001  002  003  004    001  002  003  004    ...
// Node0     101  102  103  104    101  102  103  104    ...
//           201  202  203  204    201  202  203  204    ...

//           011  012  013  014    011  012  013  014    ...
// Node1     111  112  113  104    111  112  113  114    ...
//           211  212  213  214    211  212  213  214    ...

//           021  022  023  024    021  022  023  024    ...
// Node2     121  122  123  104    121  122  123  124    ...
//           221  222  223  224    221  222  223  224    ...
TEST_F(ctranAllToAllDedupTest, even) {
  if (comm->ctranComm_->statex_->nNodes() < 2) {
    GTEST_SKIP() << "Skip the test because single node";
  }

  numTimesRunExec = 100;
  auto collRecordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, numTimesRunExec);

  count = 1024UL;
  for (int i = 0; i < numTimesRunExec; i++) {
    execs.emplace_back("even");
  }
  maxSendCount = count * comm->ctranComm_->statex_->nNodes();
  maxRecvCount = count * comm->ctranComm_->statex_->nNodes();

  sendBuf = (int*)createDataBuf(maxSendCount * sizeof(int), &sendHdl);

  run();

  releaseDataBuf(sendBuf, sendHdl);
}

TEST_F(ctranAllToAllDedupTest, uneven) {
  if (comm->ctranComm_->statex_->nNodes() < 2) {
    GTEST_SKIP() << "Skip the test because single node";
  }

  numTimesRunExec = 100;
  auto collRecordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, numTimesRunExec);

  count = 1024UL;
  for (int i = 0; i < numTimesRunExec; i++) {
    execs.emplace_back("uneven");
  }
  maxSendCount = count * comm->ctranComm_->statex_->nNodes();
  maxRecvCount = count * comm->ctranComm_->statex_->nNodes();

  sendBuf = (int*)createDataBuf(maxSendCount * sizeof(int), &sendHdl);

  run();

  releaseDataBuf(sendBuf, sendHdl);
}

TEST_F(ctranAllToAllDedupTest, evenAndUneven) {
  if (comm->ctranComm_->statex_->nNodes() < 2) {
    GTEST_SKIP() << "Skip the test because single node";
  }
  numTimesRunExec = 3;
  auto collRecordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, numTimesRunExec);

  count = 1024UL;
  execs.emplace_back("even");
  execs.emplace_back("uneven");
  execs.emplace_back("even");

  maxSendCount = count * comm->ctranComm_->statex_->nNodes();
  maxRecvCount = count * comm->ctranComm_->statex_->nNodes();

  sendBuf = (int*)createDataBuf(maxSendCount * sizeof(int), &sendHdl);

  run();

  releaseDataBuf(sendBuf, sendHdl);
}

TEST_F(ctranAllToAllDedupTest, invalidSendBuff) {
  if (comm->ctranComm_->statex_->nNodes() < 2) {
    GTEST_SKIP() << "Skip the test because single node";
  }

  void* recvBuf = nullptr;
  sendBuf = nullptr;
  CtranPersistentRequest* request = nullptr;

  auto res = ctranAllToAllDedupInit(
      sendBuf,
      splitSendCounts.data(),
      splitSendDisps.data(),
      maxSendCount,
      recvBuf,
      splitRecvCounts.data(),
      splitRecvDisps.data(),
      maxRecvCount,
      commInt,
      comm->ctranComm_.get(),
      stream,
      request);
  ASSERT_EQ(res, commInvalidArgument);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
