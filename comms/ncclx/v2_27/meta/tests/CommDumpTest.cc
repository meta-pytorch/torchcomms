// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <iostream>
#include <string>
#include <unordered_map>

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

#include "comms/utils/StrUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTrace.h"
#include "meta/colltrace/ProxyMock.h"

#include "meta/wrapper/CtranExComm.h"

static bool VERBOSE = true;
enum class sourceToDump { comm, telemetryData };

class CommDumpTest : public ::testing::TestWithParam<enum sourceToDump> {
 public:
  CommDumpTest() = default;

  void SetUp() override {
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_PROXYTRACE", "trace", 0);

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

  void prepareCtranExBcast(
      std::unique_ptr<::ctran::CtranExComm>& ctranExComm,
      const int count) {
    cpuSendBuf = reinterpret_cast<int*>(malloc(count * sizeof(int)));
    cpuRecvBuf = reinterpret_cast<int*>(malloc(count * sizeof(int)));
    NCCLCHECK_TEST(ctranExComm->regMem(
        cpuSendBuf, count * sizeof(int), &sendHandle, true));
    NCCLCHECK_TEST(ctranExComm->regMem(
        cpuRecvBuf, count * sizeof(int), &recvHandle, true));
  }

  void releaseCtranExBcast(std::unique_ptr<::ctran::CtranExComm>& ctranExComm) {
    NCCLCHECK_TEST(ctranExComm->deregMem(sendHandle));
    NCCLCHECK_TEST(ctranExComm->deregMem(recvHandle));
    free(cpuSendBuf);
    free(cpuRecvBuf);

    // teardown won't free them again
    sendHandle = nullptr;
    recvHandle = nullptr;
    cpuSendBuf = nullptr;
    cpuRecvBuf = nullptr;
  }

  void prepareAllReduce(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

  void prepareSendRecv(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
    NCCLCHECK_TEST(
        ncclCommRegister(comm, sendBuf, count * sizeof(int), &sendHandle));
    NCCLCHECK_TEST(
        ncclCommRegister(comm, recvBuf, count * sizeof(int), &recvHandle));
  }

  void prepareCtranAllGather(ncclComm* comm, const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));
    NCCLCHECK_TEST(
        ncclCommRegister(comm, sendBuf, count * sizeof(int), &sendHandle));
    NCCLCHECK_TEST(ncclCommRegister(
        comm, recvBuf, count * this->numRanks * sizeof(int), &recvHandle));
  }

  void prepareCtranAllToAll(ncclComm* comm, const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * this->numRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));
    NCCLCHECK_TEST(ncclCommRegister(
        comm, sendBuf, count * this->numRanks * sizeof(int), &sendHandle));
    NCCLCHECK_TEST(ncclCommRegister(
        comm, recvBuf, count * this->numRanks * sizeof(int), &recvHandle));
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
  for (const auto& [_, val] : dump) {
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
  EXPECT_EQ(dump.count("commDesc"), 1);
  EXPECT_EQ(dump["commDesc"], toQuotedString(this->comm->config.commDesc));

  EXPECT_EQ(dump.count("nRanks"), 1);
  EXPECT_EQ(dump["nRanks"], std::to_string(this->comm->nRanks));
  EXPECT_EQ(dump.count("localRanks"), 1);
  EXPECT_EQ(dump["localRanks"], std::to_string(this->comm->localRanks));
  EXPECT_EQ(dump.count("nNodes"), 1);
  EXPECT_EQ(dump["nNodes"], std::to_string(this->comm->nNodes));

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);

  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  if (comm->rank == 1 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }
}

TEST_F(CommDumpTest, DumpAfterSendRecv) {
  auto baselineGuard = EnvRAII{NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::orig};
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto collRecordGuard =
      EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1); // -1 for no max records

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
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    folly::dynamic ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(ctPastCollsObjs.size(), nColl);
    for (int i = 0; i < nColl; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["codepath"].asString(), "baseline");
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

TEST_F(CommDumpTest, DumpAfterCtranSendRecv) {
  auto ctranGuard = EnvRAII{NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctran};
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto checksumSampleRateGuard =
      EnvRAII(NCCL_CTRAN_SENDRECV_CHECKSUM_SAMPLE_RATE, 1);
  auto collRecordGuard =
      EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1); // -1 for no max records

  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  const int count = 1048576;
  const int nColl = 10;

  if (comm->nRanks < 3) {
    GTEST_SKIP()
        << "Skip test because this comm does not have enough ranks to properly test send recv.";
  }

  int sendPeer = (this->globalRank + 1) % this->numRanks;
  int recvPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;

  if (!ctranSendRecvSupport(sendPeer, comm->ctranComm_.get()) &&
      !ctranSendRecvSupport(recvPeer, comm->ctranComm_.get())) {
    GTEST_SKIP() << "Skip test because no ctran support.";
  }

  this->prepareSendRecv(count);
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
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    auto ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(ctPastCollsObjs.size(), nColl);
    for (int i = 0; i < nColl; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["codepath"].asString(), "ctran");
      // Check if checksum is dumped
      EXPECT_TRUE(ctPastCollsObjs[i].count("checksum"));
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
  auto reduceGuard = EnvRAII{NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig};
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto collRecordGuard =
      EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1); // -1 for no max records

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
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    auto ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(ctPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["codepath"].asString(), "baseline");
    }
  }

  // Proxy trace would be empty if nNodes == 1
  if (dump.count("PT_pastColls") && comm->nNodes > 1) {
    auto ptPastCollsObjs = folly::parseJson(dump["PT_pastColls"]);
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

TEST_F(CommDumpTest, DumpAfterCtranColl) {
  auto ctranGuard = EnvRAII(NCCL_ALLTOALL_ALGO, NCCL_ALLTOALL_ALGO::ctran);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto collRecordGuard =
      EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1); // -1 for no max records

  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  constexpr int numColls = 10;

  if (!ctranAllToAllvSupport(this->comm->ctranComm_.get())) {
    GTEST_SKIP()
        << "Skip test because this comm does not have Ctran All to All support.";
  }

  const int count = 1048576;
  const int nColl = 10;

  prepareCtranAllToAll(this->comm, count);

  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclAllToAll(
        this->sendBuf,
        this->recvBuf,
        count,
        ncclInt,
        this->comm,
        this->stream));
  }

  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  res = ncclCommDump(this->comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  // Check if all the values can be parsed as json entries
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    auto ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(ctPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["codepath"].asString(), "ctran");
    }
  }

  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  EXPECT_EQ(dump.count("MT_recvNotifiedByPeer"), 1);
  EXPECT_EQ(dump.count("MT_unfinishedRequests"), 1);
  EXPECT_EQ(dump.count("MT_putFinishedByPeer"), 1);
  EXPECT_EQ(dump.count("MT_currentColl"), 1);

  if (comm->rank == 0 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }
}

TEST_F(CommDumpTest, DumpAfterCtranAllGather) {
  auto ctranGuard = EnvRAII(NCCL_ALLGATHER_ALGO, NCCL_ALLGATHER_ALGO::ctran);
  auto checksumSampleRateGuard =
      EnvRAII(NCCL_CTRAN_ALLGATHER_CHECKSUM_SAMPLE_RATE, 1);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto collRecordGuard =
      EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1); // -1 for no max records

  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;

  if (!ctranAllGatherSupport(
          this->comm->ctranComm_.get(), NCCL_ALLGATHER_ALGO)) {
    GTEST_SKIP()
        << "Skip test because this comm does not have Ctran AllGather support.";
  }

  const int count = 1048576;

  prepareCtranAllGather(this->comm, count);

  NCCLCHECK_TEST(ncclAllGather(
      this->sendBuf, this->recvBuf, count, ncclInt, this->comm, this->stream));

  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  res = ncclCommDump(this->comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  // Check if all the values can be parsed as json entries
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  if (dump.count("CT_pastColls")) {
    auto ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(ctPastCollsObjs.size(), 1);
    // Check if checksum is dumped
    EXPECT_TRUE(ctPastCollsObjs[0].count("checksum"));
  }

  if (comm->rank == 0 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }
}

TEST_F(CommDumpTest, DumpDuringColl) {
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::string> dump;
  constexpr int numColls = 10;

  if (comm->nNodes < 2) {
    GTEST_SKIP() << "Skipping test since nNodes < 2";
  }

  // commHash is intentially stored as hex string for readability
  std::stringstream commHashSs;
  commHashSs << std::hex << comm->commHash;
  std::string commHashStr = commHashSs.str();

  // Manually set the hanging point at opCount 5
  constexpr int hangOpCount = 5;
  constexpr int hangRank = 0;
  NCCL_PROXYMOCK_NET_SEND_FAILURE.clear();
  NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(hangOpCount));
  NCCL_PROXYMOCK_NET_SEND_FAILURE.push_back(std::to_string(hangRank));
  NCCL_PROXYMOCK_NET_SEND_FAILURE.emplace_back("-1");
  NCCL_PROXYMOCK_NET_SEND_FAILURE.emplace_back("-1");
  NCCL_PROXYMOCK_NET_SEND_FAILURE.emplace_back("1"); // match only once
  NCCL_PROXYMOCK_NET_SEND_FAILURE.emplace_back("30"); // delay 30 seconds

  // Manually re-initialze state of the mock instance
  auto& instance = ProxyMockNetSendFailure::getInstance();
  instance.initialize();

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

  // Wait till the hanging point is reached
  sleep(10);

  res = ncclCommDump(this->comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  // Check if all the values can be parsed as json entries
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);
  EXPECT_EQ(dump.count("PT_activeColls"), 1);

  // Check records are dumped correctly and simply check if can be
  // parsed as json entries.

  // PastColl: Except some ranks may stuck at the hanging opCount but some
  // others may have finished and stuck at the next.
  if (dump.count("CT_pastColls")) {
    auto ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    size_t numPasts = ctPastCollsObjs.size();
    // For CollTrace, we know rank 0 must be hanging at hangOpCount
    if (comm->rank == hangRank) {
      EXPECT_EQ(numPasts, hangOpCount);
    } else {
      EXPECT_TRUE(numPasts == hangOpCount || numPasts == hangOpCount + 1);
    }
  }

  if (dump.count("PT_pastColls") && comm->nNodes > 1) {
    auto ptPastCollsObjs = folly::parseJson(dump["PT_pastColls"]);
    size_t numPasts = ptPastCollsObjs.size();
    // For ProxyTrace, since rank A's proxy thread may serve rank B's network
    // op, we cannot assume a exact hang point based on rank
    EXPECT_TRUE(numPasts == hangOpCount || numPasts == hangOpCount + 1);
  }

  // Pending collectives
  if (dump.count("CT_pendingColls")) {
    auto ctPendingCollsObjs = folly::parseJson(dump["CT_pendingColls"]);
    size_t numPending = ctPendingCollsObjs.size();
    if (comm->rank == hangRank) {
      // should hang exactly at hangOpCount, and 1 current
      EXPECT_EQ(numPending, numColls - hangOpCount - 1);
    } else {
      // may hang at hangOpCount or next
      EXPECT_TRUE(
          numPending == numColls - hangOpCount - 1 ||
          numPending == numColls - hangOpCount - 2);
    }
  }

  if (dump.count("CT_currentColl")) {
    EXPECT_NE(dump["CT_currentColl"], "null");
    auto ctCurrentCollsObj = folly::parseJson(dump["CT_currentColl"]);
    if (comm->rank == hangRank) {
      EXPECT_EQ(ctCurrentCollsObj["collId"].asInt(), hangOpCount);
      EXPECT_EQ(ctCurrentCollsObj["opCount"].asInt(), hangOpCount);
      EXPECT_EQ(ctCurrentCollsObj["opName"], "AllReduce");
    }
  }

  if (dump.count("PT_activeOps")) {
    auto ptActiveOpsObjs = folly::parseJson(dump["PT_activeOps"]);
    EXPECT_GT(ptActiveOpsObjs.size(), 0);

    for (auto& op : ptActiveOpsObjs) {
      EXPECT_TRUE(op["rank"].asInt() >= 0 && op["rank"].asInt() < comm->nRanks);
      EXPECT_TRUE(
          op["remoteRank"].asInt() >= 0 &&
          op["remoteRank"].asInt() < comm->nRanks);
      EXPECT_TRUE(op["opCount"].asInt() >= 0);
      EXPECT_TRUE(op["coll"].asString() == "AllReduce");
      EXPECT_TRUE(
          op["opType"].asString() == "SEND" ||
          op["opType"].asString() == "RECV");
      EXPECT_EQ(op["commHash"].asString(), commHashStr);

      // Each rank may hang at hangOpCount and/or hangOpCount + 1 and may see
      // active ops in both opCounts
      EXPECT_TRUE(
          op["opCount"].asInt() == hangOpCount ||
          op["opCount"].asInt() == hangOpCount + 1);
    }

    // Skip cross-rank check as already covered in ProxyTraceDistTest
  }

  if (dump.count("PT_activeColls")) {
    auto ptActiveCollsObjs = folly::parseJson(dump["PT_activeColls"]);
    EXPECT_GT(ptActiveCollsObjs.size(), 0);

    for (auto& coll : ptActiveCollsObjs) {
      EXPECT_EQ(coll["commHash"].asString(), commHashStr);

      // Each rank may hang at hangOpCount and/or hangOpCount + 1 and may see
      // active ops in both opCounts
      EXPECT_TRUE(
          coll["opCount"].asInt() == hangOpCount ||
          coll["opCount"].asInt() == hangOpCount + 1);
      EXPECT_EQ(coll["coll"].asString(), "AllReduce");
      EXPECT_GT(coll["channelIds"].size(), 0);
    }
  }

  // Now let's wait for all communication to finish
  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

  if (comm->rank == 0 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }
}

TEST_F(CommDumpTest, TestDumpAllWithTwoComms) {
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});

  constexpr int count = 1048576;
  constexpr int nColl = 10;

  // Could not use this->comm as it is created before CommsMonitor is enabled
  NcclCommRAII origComm{this->globalRank, this->numRanks, this->localRank};

  auto ctranExComm =
      std::make_unique<::ctran::CtranExComm>(origComm, "collTraceCpuBcastUt");
  ASSERT_NE(ctranExComm, nullptr);

  if (!ctranExComm->isInitialized() || !ctranExComm->supportBroadcast()) {
    GTEST_SKIP() << fmt::format(
        "Skip test because this comm does not have CtranEx support {} or no broadcast support {}.",
        ctranExComm->isInitialized(),
        ctranExComm->supportBroadcast());
  }

  prepareCtranExBcast(ctranExComm, count);

  std::vector<std::unique_ptr<::ctran::CtranExRequest>> reqs;

  for (int i = 0; i < nColl; i++) {
    ::ctran::CtranExRequest* reqPtr = nullptr;
    NCCLCHECK_TEST(ctranExComm->broadcast(
        cpuSendBuf, cpuRecvBuf, count, ncclInt, 0, &reqPtr));

    ASSERT_NE(reqPtr, nullptr);
    reqs.push_back(std::unique_ptr<::ctran::CtranExRequest>(reqPtr));
  }

  // Wait completion of all bcast
  for (auto& req : reqs) {
    ASSERT_EQ(req->wait(), ncclSuccess);
  }

  releaseCtranExBcast(ctranExComm);

  prepareAllReduce(count);

  // Run GPU collectives on the original communicator
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclAllReduce(
        this->sendBuf,
        this->recvBuf,
        count,
        ncclInt,
        ncclSum,
        origComm,
        this->stream));
  }

  origComm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  // // Dump the communicator
  auto res = ncclSuccess;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      dumpAll;
  res = ncclCommDumpAll(dumpAll);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(dumpAll.size(), 2);
  auto exNcclComm = ctranExComm->unsafeGetNcclComm();

  for (const auto& [commHash, commDump] : dumpAll) {
    EXPECT_EQ(commDump.count("CT_pastColls"), 1);
    if (!commDump.count("CT_pastColls")) {
      ADD_FAILURE() << fmt::format(
          "comm hash {} in dumpAll does not contain CT_pastColls", commHash);
      continue;
    }

    auto ctPastCollsObjs = folly::parseJson(commDump.at("CT_pastColls"));
    EXPECT_EQ(ctPastCollsObjs.size(), nColl);

    if (commHash == hashToHexStr(exNcclComm->commHash)) {
      for (int i = 0; i < nColl; i++) {
        EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
        EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
        EXPECT_EQ(ctPastCollsObjs[i]["codepath"].asString(), "ctran_cpu");
      }
    } else if (commHash == hashToHexStr(origComm->commHash)) {
      for (int i = 0; i < nColl; i++) {
        EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
        EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
        EXPECT_EQ(ctPastCollsObjs[i]["codepath"].asString(), "baseline");
      }
    } else {
      FAIL() << fmt::format("Unexpected comm hash {} in dumpAll", commHash);
    }
  }
}

TEST_F(CommDumpTest, DumpAfterCollNewCollTrace) {
  auto reduceGuard = EnvRAII{NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig};
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto newColltraceGuard = EnvRAII(NCCL_COLLTRACE_USE_NEW_COLLTRACE, true);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

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
        comm,
        this->stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. We need add a hook to check for proxy
  // completion
  sleep(3);

  ASSERT_TRUE(comm->newCollTrace != nullptr);

  res = ncclCommDump(comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  if (comm->rank == 0 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }

  // Check if all the values can be parsed as json entries
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    XLOG(DBG1) << "Entered CT_pastColls if statement";
    auto ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(ctPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
      // For new colltrace, we no longer uses codepath but Metadata type to
      // signal the type of the coll
      // EXPECT_EQ(ctPastCollsObjs[i]["codepath"].asString(), "baseline");
    }
  }

  // Proxy trace would be empty if nNodes == 1
  if (dump.count("PT_pastColls") && comm->nNodes > 1) {
    XLOG(DBG1) << "Entered PT_pastColls if statement";
    auto ptPastCollsObjs = folly::parseJson(dump["PT_pastColls"]);
    EXPECT_EQ(ptPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ptPastCollsObjs[i]["commHash"].asString(), commHashStr);
      EXPECT_EQ(ptPastCollsObjs[i]["opCount"].asInt(), i);
    }
  }

  // Check no pending/current entries
  if (dump.count("CT_pendingColls")) {
    XLOG(DBG1) << "Entered CT_pendingColls if statement";
    auto ctPendingCollsObjs = folly::parseJson(dump["CT_pendingColls"]);
    EXPECT_EQ(ctPendingCollsObjs.size(), 0);
  }

  if (dump.count("CT_currentColl")) {
    XLOG(DBG1) << "Entered CT_currentColl if statement";
    EXPECT_EQ(dump["CT_currentColl"], "null");
  }

  if (dump.count("PT_activeOps")) {
    XLOG(DBG1) << "Entered PT_activeOps if statement";
    auto ptActiveOpsObjs = folly::parseJson(dump["PT_activeOps"]);
    EXPECT_EQ(ptActiveOpsObjs.size(), 0);
  }
}

TEST_F(CommDumpTest, DumpAfterCollNewCollTraceWithCommsMonitor) {
  auto reduceGuard = EnvRAII{NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig};
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto newColltraceGuard = EnvRAII(NCCL_COLLTRACE_USE_NEW_COLLTRACE, true);
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

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
        comm,
        this->stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

  // FIXME: last a few tail sends may not be finished when kernel is done;
  // Sleep 3 sec to wait as workaround. We need add a hook to check for proxy
  // completion
  sleep(3);

  ASSERT_TRUE(comm->newCollTrace != nullptr);

  res = ncclCommDump(comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  if (comm->rank == 0 && VERBOSE) {
    for (auto& it : dump) {
      printf("%s: %s\n", it.first.c_str(), it.second.c_str());
    }
  }

  // Check if all the values can be parsed as json entries
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColl"), 1);
  EXPECT_EQ(dump.count("PT_pastColls"), 1);
  EXPECT_EQ(dump.count("PT_activeOps"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    XLOG(DBG1) << "Entered CT_pastColls if statement";
    auto ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(ctPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
      // For new colltrace, we no longer uses codepath but Metadata type to
      // signal the type of the coll
      // EXPECT_EQ(ctPastCollsObjs[i]["codepath"].asString(), "baseline");
    }
  }

  // Proxy trace would be empty if nNodes == 1
  if (dump.count("PT_pastColls") && comm->nNodes > 1) {
    XLOG(DBG1) << "Entered PT_pastColls if statement";
    auto ptPastCollsObjs = folly::parseJson(dump["PT_pastColls"]);
    EXPECT_EQ(ptPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ptPastCollsObjs[i]["commHash"].asString(), commHashStr);
      EXPECT_EQ(ptPastCollsObjs[i]["opCount"].asInt(), i);
    }
  }

  // Check no pending/current entries
  if (dump.count("CT_pendingColls")) {
    XLOG(DBG1) << "Entered CT_pendingColls if statement";
    auto ctPendingCollsObjs = folly::parseJson(dump["CT_pendingColls"]);
    EXPECT_EQ(ctPendingCollsObjs.size(), 0);
  }

  if (dump.count("CT_currentColl")) {
    XLOG(DBG1) << "Entered CT_currentColl if statement";
    EXPECT_EQ(dump["CT_currentColl"], "null");
  }

  if (dump.count("PT_activeOps")) {
    XLOG(DBG1) << "Entered PT_activeOps if statement";
    auto ptActiveOpsObjs = folly::parseJson(dump["PT_activeOps"]);
    EXPECT_EQ(ptActiveOpsObjs.size(), 0);
  }
}

TEST_F(CommDumpTest, DumpWhileCommsInDestruct) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto proxyGuard = EnvRAII{NCCL_PROXYTRACE, {"trace"}};
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);

  for (int i = 0; i < 100; i++) {
    auto comm_ptr = std::make_unique<NcclCommRAII>(
        this->globalRank, this->numRanks, this->localRank);
    std::thread t(
        [](ncclComm_t comm_t_ptr) {
          for (int j = 0; j < 100; j++) {
            std::unordered_map<std::string, std::string> dump;
            ncclCommDump(comm_t_ptr, dump);
          }
        },
        comm_ptr->get());
    comm_ptr.reset();
    t.join();
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
