// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <unordered_map>

#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTrace.h"
#include "meta/colltrace/ProxyMock.h"
#include "meta/colltrace/ProxyTrace.h"

#include "comm.h"

static bool VERBOSE = true;

class ProxyTraceTest : public NcclxBaseTest {
 public:
  ProxyTraceTest() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0); // enable ctran
    // Initialize CVAR so that we can overwrite global variable in each test
    initEnv();
    NcclxBaseTest::SetUp();
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
    NcclxBaseTest::TearDown();
  }

  void runAllReduce(const int count, const int nColl, ncclComm_t comm) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));

    for (int i = 0; i < nColl; i++) {
      NCCLCHECK_TEST(ncclAllReduce(
          sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
    }
  }

  void runAllToAll(const int count, const int nColl, ncclComm_t comm) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * comm->nRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * comm->nRanks * sizeof(int)));
    for (int i = 0; i < nColl; i++) {
      NCCLCHECK_TEST(
          ncclAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream));
    }
  }

  void runSendRecv(const int count, const int nColl, ncclComm_t comm) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
    for (int i = 0; i < nColl; i++) {
      // localRank on node0 sends to the same localRank on node1
      // TestCase ensures it runs with 2 nodes
      if (comm->node == 0) {
        int peer = comm->localRank + comm->localRanks;
        NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, peer, comm, stream));
      } else if (comm->node == 1) {
        int peer = comm->localRank;
        NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, peer, comm, stream));
      }
    }
  }

  void verbosePrintPastColl(ProxyTraceColl& past, ncclComm_t comm) {
    if (comm->rank == 0 && VERBOSE) {
      printf("Rank %d past coll: %s\n", comm->rank, past.serialize().c_str());
    }
  }

  void checkPastCollsUnderLimit(std::deque<ProxyTraceColl> pastColls) {
    std::unordered_map<uint64_t, uint64_t> commToCount;
    for (const auto& coll : pastColls) {
      commToCount[coll.collInfo.commHash]++;
    }
    for (const auto& [commHash, count] : commToCount) {
      EXPECT_LE(count, NCCL_PROXYTRACE_RECORD_MAX);
    }
  }

  // Common check for dumpping after finished collectives
  void checkCompletedDump(ProxyTrace::Dump& dump, int nCompletedColls) {
    EXPECT_EQ(dump.activeOps.size(), 0);
    EXPECT_EQ(dump.pastOps.size(), 0);
    if (dump.pastColls.size() < NCCL_PROXYTRACE_RECORD_MAX) {
      EXPECT_EQ(dump.pastColls.size(), nCompletedColls);
    }
  }

  struct SendFailureConfig {
    int opCount;
    int rank;
    int remoteRank;
    int step;
    int numMatch;
    int delaySec;
  };

  struct MinOpCountStep {
    uint64_t opCount{UINT64_MAX};
    int step{INT_MAX};
    uint64_t ts{UINT64_MAX};

    void update(uint64_t opCount, int step, uint64_t ts) {
      if (opCount < this->opCount ||
          (opCount == this->opCount && step < this->step) ||
          (opCount == this->opCount && step == this->step && ts < this->ts)) {
        this->opCount = opCount;
        this->step = step;
        this->ts = ts;
      }
    }

    void update(ProxyTraceOp& entry) {
      auto& stepRecord = entry.opType == ProxyTraceOp::OpType::SEND
          ? entry.stepRecords[ProxyOpStepStatus::TRANSMITTED]
          : entry.stepRecords[ProxyOpStepStatus::RECEIVED];
      update(
          entry.collInfo.opCount,
          stepRecord.step,
          stepRecord.ts.time_since_epoch().count());
    }

    bool match(ProxyTraceOp& entry) {
      auto& stepRecord = entry.opType == ProxyTraceOp::OpType::SEND
          ? entry.stepRecords[ProxyOpStepStatus::TRANSMITTED]
          : entry.stepRecords[ProxyOpStepStatus::RECEIVED];

      return entry.collInfo.opCount == opCount && stepRecord.step == step &&
          stepRecord.ts.time_since_epoch().count() == ts;
    }

    std::string toString() {
      return "OpCount:" + std::to_string(opCount) +
          ", Step:" + std::to_string(step) + ", Ts:" + std::to_string(ts);
    }
  };

  void setMockConfig(SendFailureConfig& config) {
    NCCL_PROXYMOCK_NET_SEND_FAILURE.clear();

    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(config.opCount));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(config.rank));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(
        std::to_string(config.remoteRank));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(config.step));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(config.numMatch));
    NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(config.delaySec));

    // Manually re-initialze state of the mock instance
    auto& instance = ProxyMockNetSendFailure::getInstance();
    instance.initialize();
  }

  void findMinStep(
      ProxyTrace::Dump& dump,
      ProxyTraceOp::OpType opType,
      MinOpCountStep& minStep) {
    for (auto& entry : dump.activeOps) {
      if (entry.opType == opType) {
        minStep.update(entry);
      }
    }
  }

  bool findMatchingActiveOp(
      ProxyTrace::Dump& dump,
      MinOpCountStep& minStep,
      ProxyTraceOp& foundOp) {
    for (auto& entry : dump.activeOps) {
      if (minStep.match(entry)) {
        foundOp = entry;
        return true;
      }
    }
    return false;
  }

  void checkHangMatch(
      ProxyTrace::Dump& dump,
      SendFailureConfig& failureConfig) {
    // no hang found
    if (dump.activeOps.size() == 0) {
      return;
    }

    // Find global min hanging opCount:step:ts as root cause report
    // - First find local min hanging opCount:step
    MinOpCountStep minSend, minRecv;
    findMinStep(dump, ProxyTraceOp::OpType::SEND, minSend);
    findMinStep(dump, ProxyTraceOp::OpType::RECV, minRecv);

    // Note that comm->rank may perform proxy ops for other ranks due to NCCL
    // topology detection, so we cannot assume rank 0 will see hanging at proxy
    // send from rank 0. Thus, we collect min steps/rank from all ranks and find
    // the global min
    std::vector<MinOpCountStep> allMinSends(numRanks);
    std::vector<MinOpCountStep> allMinRecvs(numRanks);
    allMinSends[globalRank] = minSend;
    allMinRecvs[globalRank] = minRecv;
    oobAllGather(allMinSends);
    oobAllGather(allMinRecvs);
    for (auto& send : allMinSends) {
      minSend.update(send.opCount, send.step, send.ts);
    }
    for (auto& recv : allMinRecvs) {
      minRecv.update(recv.opCount, recv.step, recv.ts);
    }

    // Expect global min hanging point matches mock config
    // - Same opCount
    EXPECT_EQ(minSend.opCount, failureConfig.opCount);
    EXPECT_EQ(minRecv.opCount, failureConfig.opCount);
    // We might not get the exact step due to slicing
    EXPECT_GE(minSend.step, failureConfig.step);
    // - Both send and recv hanging at the same step
    EXPECT_EQ(minSend.step, minRecv.step);

    // Check if my local send/recv hanging is the root cause
    ProxyTraceOp foundOp;
    bool foundMinSend = findMatchingActiveOp(dump, minSend, foundOp);
    if (foundMinSend) {
      if (VERBOSE) {
        printf(
            "Rank %d found root cause bewteen ranks %d:%d event %s\n",
            globalRank,
            foundOp.rank,
            foundOp.remoteRank,
            foundOp.serialize().c_str());
      }
      // hanging send rank matches specified rank
      EXPECT_EQ(foundOp.rank, failureConfig.rank);
    }

    bool foundMinRecv = findMatchingActiveOp(dump, minRecv, foundOp);
    if (foundMinRecv) {
      if (VERBOSE) {
        printf(
            "Rank %d found root cause bewteen ranks %d:%d event %s\n",
            globalRank,
            foundOp.rank,
            foundOp.remoteRank,
            foundOp.serialize().c_str());
      }
      // hanging recv remoteRank matches specified rank
      EXPECT_EQ(foundOp.remoteRank, failureConfig.rank);
    }
  };

  bool checkTestRequirement(ncclComm_t comm) {
    // FIXME: this check seems flaky on RE when using with socket transport
    // Disable it for now to unblock other DIFF landing. More investigation is
    // needed.
    //
    // We don't have a good way to detect whether baseline IB transport is
    // turned on. Thus, use ctran IB backend to valid for now.
    if (comm->nNodes < 2 || !ctranInitialized(comm->ctranComm_.get()) ||
        !comm->ctranComm_->ctran_->mapper->hasBackend()) {
      std::cout
          << "This test requires 2+ nodes and valid IB transport, but nNodes="
          << comm->nNodes << ", ctranInitialized(comm)="
          << ctranInitialized(comm->ctranComm_.get())
          << ", hasBackend()=" << comm->ctranComm_->ctran_->mapper->hasBackend()
          << ". Skip test "
          << ::testing::UnitTest::GetInstance()->current_test_info()->name()
          << std::endl;
      return false;
    }
    return true;
  }

 protected:
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  cudaStream_t stream;
};

static void
checkPastColl(ProxyTraceColl& past, uint64_t opCount, ncclComm* comm) {
  EXPECT_EQ(past.collInfo.commHash, comm->commHash);
  EXPECT_EQ(past.collInfo.opCount, opCount);
  EXPECT_GT(past.collInfo.nChannels, 0);
}

TEST_F(ProxyTraceTest, PastCollNoDropUnderLimit) {
  auto traceGuard = EnvRAII(NCCL_PROXYTRACE, {"trace"});
  auto recordGuard = EnvRAII(
      NCCL_PROXYTRACE_RECORD_MAX, std::max(NCCL_PROXYTRACE_RECORD_MAX, 100));

  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};
  if (!checkTestRequirement(comm)) {
    GTEST_SKIP();
  }

  EXPECT_THAT(comm->proxyState->trace, ::testing::NotNull());

  const int count = 1048500;
  const int nColl = NCCL_PROXYTRACE_RECORD_MAX - 10;

  runAllReduce(count, nColl, comm);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. How to check it properly?
  sleep(3);

  auto dump = comm->proxyState->trace->dump(comm->commHash);
  checkCompletedDump(dump, nColl);
  checkPastCollsUnderLimit(dump.pastColls);
}

TEST_F(ProxyTraceTest, TestRecordNoDropByEnv) {
  auto traceGuard = EnvRAII(NCCL_PROXYTRACE, {"trace"});
  auto recordGuard = EnvRAII(NCCL_PROXYTRACE_RECORD_MAX, -1);

  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};
  if (!checkTestRequirement(comm)) {
    GTEST_SKIP();
  }

  EXPECT_THAT(comm->proxyState->trace, ::testing::NotNull());

  const int count = 1048500;
  const int nColl =
      std::max(NCCL_PROXYTRACE_RECORD_MAX_DEFAULTCVARVALUE, 100) * 5;

  runAllReduce(count, nColl, comm);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. How to check it properly?
  sleep(3);

  auto dump = comm->proxyState->trace->dump(comm->commHash);
  checkCompletedDump(dump, nColl);
  EXPECT_EQ(dump.pastColls.size(), nColl);
}

TEST_F(ProxyTraceTest, TestRecordDropExceedLimit) {
  auto traceGuard = EnvRAII(NCCL_PROXYTRACE, {"trace"});
  auto recordGuard = EnvRAII(
      NCCL_PROXYTRACE_RECORD_MAX,
      std::max(NCCL_PROXYTRACE_RECORD_MAX_DEFAULTCVARVALUE, 100));

  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};
  if (!checkTestRequirement(comm)) {
    GTEST_SKIP();
  }

  EXPECT_THAT(comm->proxyState->trace, ::testing::NotNull());

  const int count = 1048500;
  const int nColl = NCCL_PROXYTRACE_RECORD_MAX * 5;

  runAllReduce(count, nColl, comm);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. How to check it properly?
  sleep(3);

  auto dump = comm->proxyState->trace->dump(comm->commHash);
  checkCompletedDump(dump, nColl);
  checkPastCollsUnderLimit(dump.pastColls);
}

TEST_F(ProxyTraceTest, QueryFinishedAllReduce) {
  auto traceGuard = EnvRAII(NCCL_PROXYTRACE, {"trace"});
  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};
  if (!checkTestRequirement(comm)) {
    GTEST_SKIP();
  }

  EXPECT_NE(comm->proxyState->trace, nullptr);

  const int count = 1048500;
  const int nColl = 10;

  uint64_t opCountStart = comm->opCount;

  runAllReduce(count, nColl, comm);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. How to check it properly?
  sleep(3);

  auto dump = comm->proxyState->trace->dump(comm->commHash);
  checkCompletedDump(dump, nColl);

  // Check past collective details
  for (int i = 0; i < nColl; i++) {
    checkPastColl(dump.pastColls[i], opCountStart + i, comm);
    EXPECT_EQ(dump.pastColls[i].collInfo.coll, ncclFuncAllReduce);
    // Skip check for nProxyOps as we don't know allreduce internal

    verbosePrintPastColl(dump.pastColls[i], comm);
  }
}

TEST_F(ProxyTraceTest, QueryFinishedAllToAll) {
  auto traceGuard = EnvRAII(NCCL_PROXYTRACE, {"trace"});
  // disable PXN so that each proxy thread can have deterministic behavior:
  // send and recv for the local rank with PPN remote ranks on the other node
  NCCL_PXN_DISABLE = 1;
  // ensure we use default proxy path
  NCCL_ALLTOALL_ALGO = NCCL_ALLTOALL_ALGO::orig;

  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};
  if (!checkTestRequirement(comm)) {
    GTEST_SKIP();
  }

  EXPECT_NE(comm->proxyState->trace, nullptr);

  // use size cannot be evenly divided by stepSize to test trace size
  // correctness
  const int count = 1048500;
  const int nColl = 10;
  uint64_t opCountStart = comm->opCount;

  runAllToAll(count, nColl, comm);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. How to check it properly?
  sleep(3);

  auto dump = comm->proxyState->trace->dump(comm->commHash);
  checkCompletedDump(dump, nColl);

  // Check past collective details
  for (int i = 0; i < nColl; i++) {
    checkPastColl(dump.pastColls[i], opCountStart + i, comm);
    EXPECT_EQ(dump.pastColls[i].collInfo.coll, ncclFuncSendRecv);

    // Expect nChannels number of send and recv to each remote rank
    size_t nChannels = dump.pastColls[i].channelIds.size();
    int numRemoteRanks = comm->localRanks * (comm->nNodes - 1);
    EXPECT_EQ(dump.pastColls[i].nProxyOps, numRemoteRanks * 2 * nChannels);

    // Expect total send size to be count * sizeof(int) * numRemoteRanks
    EXPECT_EQ(
        dump.pastColls[i].totalSendSize, count * sizeof(int) * numRemoteRanks);
    // DO NOT check totalRecvSize which can be inaccurate (see ProxyTraceColl
    // description).

    verbosePrintPastColl(dump.pastColls[i], comm);
  }
}

TEST_F(ProxyTraceTest, QueryFinishedSendRecv) {
  auto traceGuard = EnvRAII(NCCL_PROXYTRACE, {"trace"});
  // disable PXN so that each proxy thread can have deterministic behavior:
  // send and recv for the local rank with PPN remote ranks on the other node
  NCCL_PXN_DISABLE = 1;
  // ensure we use default proxy path
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::orig;

  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};
  if (!checkTestRequirement(comm)) {
    GTEST_SKIP();
  }
  EXPECT_NE(comm->proxyState->trace, nullptr);

  // use size cannot be evenly divided by stepSize to test trace size
  // correctness
  const int count = 1048500;
  const int nColl = 2;
  uint64_t opCountStart = comm->opCount;

  runSendRecv(count, nColl, comm);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. How to check it properly?
  sleep(3);

  auto dump = comm->proxyState->trace->dump(comm->commHash);
  checkCompletedDump(dump, nColl);

  // Check past collective details
  EXPECT_EQ(dump.pastColls.size(), nColl);
  for (int i = 0; i < nColl; i++) {
    checkPastColl(dump.pastColls[i], opCountStart + i, comm);
    // localRank on node0 sends to the same localRank on node1 (see
    // runSendRecv). skipSingleNodeRun check ensures it runs with 2+nodes
    if (comm->node == 0) {
      EXPECT_EQ(dump.pastColls[i].collInfo.coll, ncclFuncSend);
      EXPECT_EQ(dump.pastColls[i].totalSendSize, count * sizeof(int));
      EXPECT_EQ(dump.pastColls[i].totalRecvSize, 0);
    } else if (comm->node == 1) {
      EXPECT_EQ(dump.pastColls[i].collInfo.coll, ncclFuncRecv);
      // DO NOT check totalRecvSize which can be inaccurate (see ProxyTraceColl
      // description).
      EXPECT_EQ(dump.pastColls[i].totalSendSize, 0);
    }
    // 1 send@sender and 1 recv@receiver are expected.
    // Each send/recv may be divided into nChannels.
    size_t nChannels = dump.pastColls[i].channelIds.size();
    EXPECT_EQ(dump.pastColls[i].nProxyOps, nChannels);

    verbosePrintPastColl(dump.pastColls[i], comm);
  }
}

TEST_F(ProxyTraceTest, QueryHangAllReduce) {
  auto traceGuard = EnvRAII(NCCL_PROXYTRACE, {"trace"});

  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};
  if (!checkTestRequirement(comm)) {
    GTEST_SKIP();
  }

  // Configure mock failure relative to current opCount to account for
  // init-phase operations (e.g., fast-init) that consume opCounts.
  SendFailureConfig failureConfig = {
      static_cast<int>(comm->opCount) + 8 /*opCount*/,
      0 /*rank*/,
      -1 /*remoteRank*/,
      1 /*step*/,
      1 /*num of matches*/,
      30 /*delay*/
  };
  setMockConfig(failureConfig);

  EXPECT_NE(comm->proxyState->trace, nullptr);

  const int count = 1048500;
  const int nColl = 10;

  auto begin = std::chrono::high_resolution_clock::now();
  runAllReduce(count, nColl, comm);

  // sleep 10 seconds to reach the hanging point
  sleep(10);

  auto dump = comm->proxyState->trace->dump(comm->commHash);

  // Expect hanging happens and active ops are not empty
  EXPECT_GT(dump.activeOps.size(), 0);

  for (auto& entry : dump.activeOps) {
    // Check basic info of each active op
    EXPECT_EQ(entry.collInfo.coll, ncclFuncAllReduce);
    EXPECT_EQ(entry.collInfo.commHash, comm->commHash);
    EXPECT_GE(entry.channelId, 0);
    EXPECT_GT(
        entry.startTs.time_since_epoch().count(),
        begin.time_since_epoch().count());
    EXPECT_FALSE(entry.done);
  }

  // Find hanging root cause across all ranks
  // and check if matchs failure config
  checkHangMatch(dump, failureConfig);

  // Now let's wait for all communication to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
}

TEST_F(ProxyTraceTest, QueryHangSendRecv) {
  auto traceGuard = EnvRAII(NCCL_PROXYTRACE, {"trace"});
  // disable PXN so that each proxy thread is guaranteed to send and recv with
  // PPN remote ranks
  NCCL_PXN_DISABLE = 1;
  // ensure we use default proxy path
  NCCL_SENDRECV_ALGO = NCCL_SENDRECV_ALGO::orig;

  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};
  if (!checkTestRequirement(comm)) {
    GTEST_SKIP();
  }

  // Configure mock failure relative to current opCount to account for
  // init-phase operations (e.g., fast-init) that consume opCounts.
  SendFailureConfig failureConfig = {
      static_cast<int>(comm->opCount) + 8 /*opCount*/,
      0 /*rank*/,
      comm->localRanks /*remoteRank*/,
      1 /*step*/,
      1 /*num of matches*/,
      30 /*delay*/
  };
  // assumed no network communication now, so safe to reset mock instance
  setMockConfig(failureConfig);

  EXPECT_NE(comm->proxyState->trace, nullptr);

  const int count = 1048500;
  const int nColl = 10;

  runSendRecv(count, nColl, comm);

  // sleep 10 seconds to reach the hanging point
  sleep(10);

  auto dump = comm->proxyState->trace->dump(comm->commHash);

  // Expect active ops are not empty only on the hanging pairs
  // Also find the min hanging step on each rank
  MinOpCountStep minStep;
  if (comm->rank == failureConfig.rank) {
    EXPECT_GT(dump.activeOps.size(), 0);
    findMinStep(dump, ProxyTraceOp::OpType::SEND, minStep);
    EXPECT_EQ(minStep.opCount, failureConfig.opCount);
    EXPECT_EQ(minStep.step, failureConfig.step);

    ProxyTraceOp foundOp;
    EXPECT_TRUE(findMatchingActiveOp(dump, minStep, foundOp));
    EXPECT_EQ(foundOp.rank, failureConfig.rank);
  } else if (comm->rank == failureConfig.remoteRank) {
    EXPECT_GT(dump.activeOps.size(), 0);
    findMinStep(dump, ProxyTraceOp::OpType::RECV, minStep);
    EXPECT_EQ(minStep.opCount, failureConfig.opCount);
    EXPECT_EQ(minStep.step, failureConfig.step);

    ProxyTraceOp foundOp;
    EXPECT_TRUE(findMatchingActiveOp(dump, minStep, foundOp));
    EXPECT_EQ(foundOp.rank, failureConfig.remoteRank);
  } else {
    checkCompletedDump(dump, nColl);
  }

  // Now let's wait for all communication to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
}

// Verify that CollTrace (CT) and ProxyTrace (PT) record the same opCount
// for the same collective. A bug introduced by D83294734 caused opCount to be
// incremented in doLaunches (after kernel launch) instead of in
// ncclLaunchPrepare (before proxy ops are created), causing PT to capture a
// stale value when multiple plans exist in a single group.
TEST_F(ProxyTraceTest, CTAndPTOpCountsMatch) {
  auto traceGuard = EnvRAII(NCCL_PROXYTRACE, {"trace"});
  auto recordGuard = EnvRAII(
      NCCL_PROXYTRACE_RECORD_MAX, std::max(NCCL_PROXYTRACE_RECORD_MAX, 100));

  NcclCommRAII comm{globalRank, numRanks, localRank, bootstrap_.get()};
  if (!checkTestRequirement(comm)) {
    GTEST_SKIP();
  }

  EXPECT_THAT(comm->proxyState->trace, ::testing::NotNull());
  EXPECT_THAT(comm->ctranComm_->collTrace_, ::testing::NotNull());

  const int count = 1048500;
  const int nColl = 20;

  runAllReduce(count, nColl, comm);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Wait for proxy ops to finish
  sleep(3);

  // Dump both CT and PT
  auto ptDump = comm->proxyState->trace->dump(comm->commHash);
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();
  auto ctDump = comm->ctranComm_->collTrace_->dump();

  // Build sets of opCounts from CT and PT pastColls
  std::set<uint64_t> ctOpCounts;
  for (const auto& coll : ctDump.pastColls) {
    ctOpCounts.insert(coll.opCount);
  }

  std::set<uint64_t> ptOpCounts;
  for (const auto& coll : ptDump.pastColls) {
    ptOpCounts.insert(coll.collInfo.opCount);
  }

  // The set of opCounts in CT and PT should be identical — both should have
  // recorded the same operations under the same opCount values.
  // A mismatch here means PT captured stale/wrong opCount values.
  if (comm->rank == 0 && VERBOSE) {
    printf(
        "Rank %d: CT pastColls=%zu (opCounts %lu-%lu), "
        "PT pastColls=%zu (opCounts %lu-%lu)\n",
        comm->rank,
        ctDump.pastColls.size(),
        ctOpCounts.empty() ? 0UL : *ctOpCounts.begin(),
        ctOpCounts.empty() ? 0UL : *ctOpCounts.rbegin(),
        ptDump.pastColls.size(),
        ptOpCounts.empty() ? 0UL : *ptOpCounts.begin(),
        ptOpCounts.empty() ? 0UL : *ptOpCounts.rbegin());
  }

  // Verify the opCount sets match
  EXPECT_EQ(ctOpCounts, ptOpCounts)
      << "CT and PT pastColls have different opCount sets. "
      << "CT range: [" << (ctOpCounts.empty() ? 0UL : *ctOpCounts.begin())
      << ", " << (ctOpCounts.empty() ? 0UL : *ctOpCounts.rbegin()) << "], "
      << "PT range: [" << (ptOpCounts.empty() ? 0UL : *ptOpCounts.begin())
      << ", " << (ptOpCounts.empty() ? 0UL : *ptOpCounts.rbegin()) << "]";
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
