// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <iostream>
#include <string>
#include <unordered_map>

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "nccl.h"
#include "rccl.h" // For ncclCommDump function declaration

// Forward declaration for ncclCommDump function (C++ function, not C)
ncclResult_t ncclCommDump(
    ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map);

// #include "comms/utils/cvars/nccl_cvars.h"
#include "CollTrace.h"
#include "comms/utils/StrUtils.h"
// #include "meta/colltrace/ProxyMock.h"

// #include "meta/wrapper/CtranExComm.h"

static bool VERBOSE = true;
enum class sourceToDump { comm, telemetryData };

class CommDumpTest : public ::testing::TestWithParam<enum sourceToDump> {
 public:
  CommDumpTest() = default;

  void SetUp() override {
    setenv("RCCL_LATENCY_PROFILER", "1", 1);
    setenv("NCCL_ENABLE_PROXY_TRACE", "1", 1);

    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    // Prepare data for sanity check after commSplit
    CUDACHECK_TEST(cudaMalloc(&this->dataBuf, sizeof(int) * this->dataCount));
  }

  void initData(int myRank) {
    std::vector<int> initVals(this->dataCount);
    for (int i = 0; i < this->dataCount; i++) {
      initVals[i] = i * myRank;
    }
    CUDACHECK_TEST(cudaMemcpy(
        this->dataBuf,
        initVals.data(),
        sizeof(int) * this->dataCount,
        cudaMemcpyHostToDevice));
  }

  void TearDown() override {
    if (sendHandle != nullptr) {
      ncclCommDeregister(comm, sendHandle);
    }
    if (recvHandle != nullptr) {
      ncclCommDeregister(comm, recvHandle);
    }
    CUDACHECK_TEST(cudaFree(this->dataBuf));
    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));

    if (cpuRecvBuf != nullptr) {
      free(cpuRecvBuf);
    }
    if (cpuSendBuf != nullptr) {
      free(cpuSendBuf);
    }
  }

  void prepareSendRecv(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
    NCCLCHECK_TEST(
        ncclCommRegister(comm, sendBuf, count * sizeof(int), &sendHandle));
    NCCLCHECK_TEST(
        ncclCommRegister(comm, recvBuf, count * sizeof(int), &recvHandle));
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  int* cpuSendBuf{nullptr};
  int* cpuRecvBuf{nullptr};
  void* sendHandle{nullptr};
  void* recvHandle{nullptr};

  int* dataBuf{nullptr};
  const int dataCount{65536};
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(CommDumpTest, SingleComm) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;

  res = ncclCommDump(this->comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  // Check if all the values can be parsed as json entries
  for (const auto& [key, val] : dump) {
    if (key != "KernelTrace")
      EXPECT_NO_THROW(folly::parseJson(val));
  }

  // commHash is intentially stored as hex string for readability
  std::stringstream commHashSs;
  commHashSs << "\"" << std::hex << comm->commHash << "\"";
  std::string commHashStr = commHashSs.str();

  EXPECT_EQ(dump.count("commHash"), 1);
  EXPECT_EQ(dump["commHash"], commHashStr);
  EXPECT_EQ(dump.count("rank"), 1);
  EXPECT_EQ(dump["rank"], std::to_string(this->comm->rank));
  EXPECT_EQ(dump.count("localRank"), 1);
  EXPECT_EQ(dump["localRank"], std::to_string(this->comm->localRank));
  EXPECT_EQ(dump.count("node"), 1);
  EXPECT_EQ(dump["node"], std::to_string(this->comm->node));

  EXPECT_EQ(dump.count("nRanks"), 1);
  EXPECT_EQ(dump["nRanks"], std::to_string(this->comm->nRanks));
  EXPECT_EQ(dump.count("localRanks"), 1);
  EXPECT_EQ(dump["localRanks"], std::to_string(this->comm->localRanks));
  EXPECT_EQ(dump.count("nNodes"), 1);
  EXPECT_EQ(dump["nNodes"], std::to_string(this->comm->nNodes));

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);

  EXPECT_EQ(dump.count("PT_pastOps"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  if (comm->rank == 1 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }
}

TEST_F(CommDumpTest, DumpAfterSendRecv) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  const int count = 1048576;
  const int nColl = 10;

  if (comm->nRanks < 3) {
    GTEST_SKIP()
        << "Skip test because this comm does not have enough ranks to properly test send recv.";
  }

  // commHash is intentially stored as hex string for readability
  std::stringstream commHashSs;
  commHashSs << std::hex << comm->commHash;
  std::string commHashStr = commHashSs.str();

  this->prepareSendRecv(count);
  int sendPeer = (this->globalRank + 1) % this->numRanks;
  int recvPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclGroupStart());
    NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, sendPeer, comm, stream));
    NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, recvPeer, comm, stream));
    NCCLCHECK_TEST(ncclGroupEnd());
  }
  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. We need add a hook to check for proxy
  // completion
  sleep(3);

  res = ncclCommDump(this->comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  // Check if all the values can be parsed as json entries
  for (const auto& [key, val] : dump) {
    if (key != "KernelTrace")
      EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastOps"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    folly::dynamic ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(ctPastCollsObjs.size(), nColl);
    for (int i = 0; i < nColl; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
      EXPECT_TRUE(ctPastCollsObjs[i].count("ranksInGroupedP2P"));
      EXPECT_TRUE(ctPastCollsObjs[i]["ranksInGroupedP2P"].isArray());
      if (ctPastCollsObjs[i].count("ranksInGroupedP2P") &&
          ctPastCollsObjs[i]["ranksInGroupedP2P"].isArray()) {
        std::vector<int> ranksVec{};
        for (const auto& rank : ctPastCollsObjs[i]["ranksInGroupedP2P"]) {
          if (!rank.isInt()) {
            ADD_FAILURE() << "ranksInGroupedP2P contains wrong type";
            break;
          }
          ranksVec.push_back(rank.asInt());
        }
        EXPECT_EQ(ranksVec.size(), 3);
        EXPECT_THAT(ranksVec, ::testing::Contains(this->globalRank));
        EXPECT_THAT(ranksVec, ::testing::Contains(sendPeer));
        EXPECT_THAT(ranksVec, ::testing::Contains(recvPeer));
      }
    }
  }

  // Proxy trace might only exist for selected ranks. Skip checking it.

  // Check no pending/current entries
  if (dump.count("CT_pendingColls")) {
    auto ctPendingCollsObjs = folly::parseJson(dump["CT_pendingColls"]);
    EXPECT_EQ(ctPendingCollsObjs.size(), 0);
  }

  if (dump.count("CT_currentColl")) {
    EXPECT_EQ(dump["CT_currentColl"], "null");
  }

  if (dump.count("PT_activeOps")) {
    auto ptActiveOpsObjs = folly::parseJson(dump["PT_activeOps"]);
    EXPECT_EQ(ptActiveOpsObjs.size(), 0);
  }

  if (comm->rank == 0 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }
}

TEST_F(CommDumpTest, DumpAfterColl) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  constexpr int numColls = 10;

  // commHash is intentially stored as hex string for readability
  std::stringstream commHashSs;
  commHashSs << std::hex << comm->commHash;
  std::string commHashStr = commHashSs.str();

  this->initData(this->globalRank);
  for (int i = 0; i < numColls; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        this->dataBuf,
        this->dataBuf,
        this->dataCount,
        ncclInt,
        ncclSum,
        this->comm,
        this->stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. We need add a hook to check for proxy
  // completion
  sleep(3);

  res = ncclCommDump(this->comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  // Check if all the values can be parsed as json entries
  for (const auto& [key, val] : dump) {
    if (key != "KernelTrace")
      EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastOps"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    auto ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(ctPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
    }
  }

  // Proxy trace would be empty if nNodes == 1
  if (dump.count("PT_pastOps") && comm->nNodes > 1) {
    auto ptPastCollsObjs = folly::parseJson(dump["PT_pastOps"]);
    EXPECT_EQ(ptPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ptPastCollsObjs[i]["commHash"].asString(), commHashStr);
      EXPECT_EQ(ptPastCollsObjs[i]["opCount"].asInt(), i);
    }
  }

  // Check no pending/current entries
  if (dump.count("CT_pendingColls")) {
    auto ctPendingCollsObjs = folly::parseJson(dump["CT_pendingColls"]);
    EXPECT_EQ(ctPendingCollsObjs.size(), 0);
  }

  if (dump.count("CT_currentColl")) {
    EXPECT_EQ(dump["CT_currentColl"], "null");
  }

  if (dump.count("PT_activeOps")) {
    auto ptActiveOpsObjs = folly::parseJson(dump["PT_activeOps"]);
    EXPECT_EQ(ptActiveOpsObjs.size(), 0);
  }

  if (comm->rank == 0 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
