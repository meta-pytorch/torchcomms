// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <new>
#include <thread>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllDedupImpl.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"

class ctranAllToAllDedupTest : public ctran::CtranDistTestFixture,
                               public CtranBaseTest {
 public:
  ctranAllToAllDedupTest() = default;

  void generateUnevenRailDisps() {
    size_t sendTotalCount = 0;
    size_t recvTotalCount = 0;
    int nNodes = ctranComm->statex_->nNodes();
    int myNode = ctranComm->statex_->rank() / ctranComm->statex_->nLocalRanks();
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
    int nNodes = ctranComm->statex_->nNodes();
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
    int nNodes = ctranComm->statex_->nNodes();
    int nLocalRanks = ctranComm->statex_->nLocalRanks();
    for (int i = 0; i < nNodes; i++) {
      size_t send_split_size = sendCounts[i] / nLocalRanks;
      size_t recv_split_size = recvCounts[i] / nLocalRanks;
      splitSendCounts[i] = send_split_size;
      splitRecvCounts[i] = recv_split_size;
      splitSendDisps[i] = sendDisps[i] + send_split_size * localRank;
      splitRecvDisps[i] = recvDisps[i] + recv_split_size * localRank;
    }
  }

  void* createDataBuf(size_t nbytes) {
    void* buf = nullptr;
    NCCLCHECK_TEST(ncclMemAlloc(&buf, nbytes));
    if (buf) {
      COMMCHECK_TEST(ctran::globalRegisterWithPtr(buf, nbytes));
    }
    return buf;
  }

  void releaseDataBuf(void* buf, size_t nbytes) {
    if (buf) {
      COMMCHECK_TEST(ctran::globalDeregisterWithPtr(buf, nbytes));
    }
    NCCLCHECK_TEST(ncclMemFreeWithRefCheck(buf));
  }

  void SetUp() override {
    ctran::CtranDistTestFixture::SetUp();
    ctranComm = makeCtranComm();
    if (!ctranAllToAllDedupSupport(ctranComm.get())) {
      GTEST_SKIP() << "Skip the test because ctranAllToAllv is not supported";
    }

    // Allocate enough space for arguments, value assignment set in each test
    sendBuf = nullptr;
    int nNodes = ctranComm->statex_->nNodes();
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
    ctran::CtranDistTestFixture::TearDown();
  }

  void run() {
    // Assign different value for each send chunk
    int nLocalRanks = ctranComm->statex_->nLocalRanks();
    int numSplits = nLocalRanks;
    int nNodes = ctranComm->statex_->nNodes();

    ASSERT_TRUE(
        meta::comms::colltrace::testOnlyClearCollTraceRecords(ctranComm.get()));

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
        ctranComm.get(),
        testStream,
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
      CUDACHECK_TEST(cudaStreamSynchronize(testStream));

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

    CUDACHECK_TEST(cudaDeviceSynchronize());
    // Sleep for a while to make sure all the colls are finished
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_NE(ctranComm->colltraceNew_, nullptr);
    auto dumpMap = ctran::dumpCollTrace(ctranComm.get());

    EXPECT_NE(dumpMap["CT_pastColls"], "[]");
    EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
    EXPECT_EQ(dumpMap["CT_currentColl"], "null");

    auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
    // make sure we are collecting enough records
    EXPECT_GE(NCCL_COLLTRACE_RECORD_MAX, numTimesRunExec);
    EXPECT_EQ(pastCollsJson.size(), numTimesRunExec);

    for (const auto& coll : pastCollsJson) {
      EXPECT_EQ(coll["opName"].asString(), "AllToAll_Dedup");
      EXPECT_THAT(
          coll["algoName"].asString(),
          testing::HasSubstr("ctranAllToAllDedup"));
    }

    size_t sendCount =
        std::accumulate(sendCounts.begin(), sendCounts.end(), 0UL);
    if (sendCount > 0) {
      // Alltoall uses kernel staged copy not NVL iput
      std::vector<CtranMapperBackend> excludedBackends = {
          CtranMapperBackend::NVL};
      // If single node, uses only kernel staged copy
      if (ctranComm->statex_->nNodes() == 1) {
        excludedBackends.push_back(CtranMapperBackend::IB);
      }
      verifyBackendsUsed(
          ctranComm->ctran_.get(),
          ctranComm->statex_.get(),
          kMemCudaMalloc,
          excludedBackends);
    }
  }

 protected:
  cudaStream_t testStream{0};
  std::unique_ptr<CtranComm> ctranComm{nullptr};
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
  if (ctranComm->statex_->nNodes() < 2) {
    GTEST_SKIP() << "Skip the test because single node";
  }

  numTimesRunExec = 100;
  auto collRecordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, numTimesRunExec);

  count = 1024UL;
  for (int i = 0; i < numTimesRunExec; i++) {
    execs.emplace_back("even");
  }
  maxSendCount = count * ctranComm->statex_->nNodes();
  maxRecvCount = count * ctranComm->statex_->nNodes();

  size_t sendBufSize = maxSendCount * sizeof(int);
  sendBuf = (int*)createDataBuf(sendBufSize);

  run();

  releaseDataBuf(sendBuf, sendBufSize);
}

TEST_F(ctranAllToAllDedupTest, uneven) {
  if (ctranComm->statex_->nNodes() < 2) {
    GTEST_SKIP() << "Skip the test because single node";
  }

  numTimesRunExec = 100;
  auto collRecordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, numTimesRunExec);

  count = 1024UL;
  for (int i = 0; i < numTimesRunExec; i++) {
    execs.emplace_back("uneven");
  }
  maxSendCount = count * ctranComm->statex_->nNodes();
  maxRecvCount = count * ctranComm->statex_->nNodes();

  size_t sendBufSize = maxSendCount * sizeof(int);
  sendBuf = (int*)createDataBuf(sendBufSize);

  run();

  releaseDataBuf(sendBuf, sendBufSize);
}

TEST_F(ctranAllToAllDedupTest, evenAndUneven) {
  if (ctranComm->statex_->nNodes() < 2) {
    GTEST_SKIP() << "Skip the test because single node";
  }
  numTimesRunExec = 3;
  auto collRecordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, numTimesRunExec);

  count = 1024UL;
  execs.emplace_back("even");
  execs.emplace_back("uneven");
  execs.emplace_back("even");

  maxSendCount = count * ctranComm->statex_->nNodes();
  maxRecvCount = count * ctranComm->statex_->nNodes();

  size_t sendBufSize = maxSendCount * sizeof(int);
  sendBuf = (int*)createDataBuf(sendBufSize);

  run();

  releaseDataBuf(sendBuf, sendBufSize);
}

TEST_F(ctranAllToAllDedupTest, invalidSendBuff) {
  if (ctranComm->statex_->nNodes() < 2) {
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
      ctranComm.get(),
      testStream,
      request);
  ASSERT_EQ(res, commInvalidArgument);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
