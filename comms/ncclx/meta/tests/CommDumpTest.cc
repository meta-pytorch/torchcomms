// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <iostream>
#include <string>
#include <unordered_map>

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comm.h"
#include "nccl.h"

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include "meta/NcclxConfig.h"
#include "meta/commDump.h"
#include "meta/comms-monitor/CommsMonitor.h"
static bool VERBOSE = true;
enum class sourceToDump { comm, telemetryData };

using meta::comms::ncclx::waitForCollTraceDrain;

class CommDumpTest : public NcclxBaseTestFixture,
                     public ::testing::WithParamInterface<enum sourceToDump> {
 public:
  void SetUp() override {
    setenv("NCCL_COLLTRACE", "trace", 0);

    NcclxBaseTestFixture::SetUp();
    this->comm = ncclx::test::createNcclComm(
        globalRank, numRanks, localRank, bootstrap_.get());
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

    NcclxBaseTestFixture::TearDown();
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

  void prepareCtranAllGather(ncclComm* commPtr, const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));
    NCCLCHECK_TEST(
        ncclCommRegister(commPtr, sendBuf, count * sizeof(int), &sendHandle));
    NCCLCHECK_TEST(ncclCommRegister(
        commPtr, recvBuf, count * this->numRanks * sizeof(int), &recvHandle));
  }

  void prepareCtranAllToAll(ncclComm* commPtr, const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * this->numRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));
    NCCLCHECK_TEST(ncclCommRegister(
        commPtr, sendBuf, count * this->numRanks * sizeof(int), &sendHandle));
    NCCLCHECK_TEST(ncclCommRegister(
        commPtr, recvBuf, count * this->numRanks * sizeof(int), &recvHandle));
  }

  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
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
  EXPECT_EQ(
      dump["commDesc"],
      "\"" + NCCLX_CONFIG_FIELD(this->comm->config, commDesc) + "\"");

  EXPECT_EQ(dump.count("nRanks"), 1);
  EXPECT_EQ(dump["nRanks"], std::to_string(this->comm->nRanks));
  EXPECT_EQ(dump.count("localRanks"), 1);
  EXPECT_EQ(dump["localRanks"], std::to_string(this->comm->localRanks));
  EXPECT_EQ(dump.count("nNodes"), 1);
  EXPECT_EQ(dump["nNodes"], std::to_string(this->comm->nNodes));
  EXPECT_EQ(dump.count("cliqueSize"), 1);
  EXPECT_EQ(dump["cliqueSize"], std::to_string(this->comm->clique.size));

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColls"), 1);

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
  ASSERT_NE(this->comm->newCollTrace, nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*this->comm->newCollTrace));

  res = ncclCommDump(this->comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  // Check if all the values can be parsed as json entries
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColls"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    folly::dynamic ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    ASSERT_EQ(ctPastCollsObjs.size(), nColl);
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

  // Check no pending/current entries
  if (dump.count("CT_pendingColls")) {
    auto ctPendingCollsObjs = folly::parseJson(dump["CT_pendingColls"]);
    EXPECT_EQ(ctPendingCollsObjs.size(), 0);
  }

  if (dump.count("CT_currentColls")) {
    EXPECT_EQ(dump["CT_currentColls"], "[]");
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
  ASSERT_NE(this->comm->newCollTrace, nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*this->comm->newCollTrace));

  res = ncclCommDump(this->comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  // Check if all the values can be parsed as json entries
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColls"), 1);

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

  // Check no pending/current entries
  if (dump.count("CT_pendingColls")) {
    auto ctPendingCollsObjs = folly::parseJson(dump["CT_pendingColls"]);
    EXPECT_EQ(ctPendingCollsObjs.size(), 0);
  }

  if (dump.count("CT_currentColls")) {
    EXPECT_EQ(dump["CT_currentColls"], "[]");
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
  // TODO: Currently CommsMonitor has an issue of communicators with same addr
  // will have issue. temporarily disable it, will turn it back after the fix.
  auto monitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, false);

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
  ASSERT_NE(this->comm->newCollTrace, nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*this->comm->newCollTrace));

  res = ncclCommDump(this->comm, dump);

  ASSERT_EQ(res, ncclSuccess);

  // Check if all the values can be parsed as json entries
  for (const auto& [_, val] : dump) {
    EXPECT_NO_THROW(folly::parseJson(val));
  }

  EXPECT_EQ(dump.count("CT_pastColls"), 1);
  EXPECT_EQ(dump.count("CT_pendingColls"), 1);
  EXPECT_EQ(dump.count("CT_currentColls"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    auto ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    ASSERT_EQ(ctPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
    }
  }

  // Check no pending/current entries
  if (dump.count("CT_pendingColls")) {
    auto ctPendingCollsObjs = folly::parseJson(dump["CT_pendingColls"]);
    EXPECT_EQ(ctPendingCollsObjs.size(), 0);
  }

  if (dump.count("CT_currentColls")) {
    EXPECT_EQ(dump["CT_currentColls"], "[]");
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

  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));
  ASSERT_NE(this->comm->newCollTrace, nullptr);
  EXPECT_EQ(waitForCollTraceDrain(*this->comm->newCollTrace), true);

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
  EXPECT_EQ(dump.count("CT_currentColls"), 1);

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

  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));
  ASSERT_NE(this->comm->newCollTrace, nullptr);
  EXPECT_EQ(waitForCollTraceDrain(*this->comm->newCollTrace), true);

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

TEST_F(CommDumpTest, DumpAfterCollNewCollTrace) {
  auto reduceGuard = EnvRAII{NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig};
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

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
  ASSERT_TRUE(comm->newCollTrace != nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*comm->newCollTrace));

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
  EXPECT_EQ(dump.count("CT_currentColls"), 1);

  // Check past collectives are dumped correctly and simply check if can be
  // parsed as json entries.
  if (dump.count("CT_pastColls")) {
    XLOG(DBG1) << "Entered CT_pastColls if statement";
    auto ctPastCollsObjs = folly::parseJson(dump["CT_pastColls"]);
    EXPECT_EQ(ctPastCollsObjs.size(), numColls);
    for (int i = 0; i < numColls; i++) {
      EXPECT_EQ(ctPastCollsObjs[i]["collId"].asInt(), i);
      EXPECT_EQ(ctPastCollsObjs[i]["opCount"].asInt(), i);
    }
  }

  // Check no pending/current entries
  if (dump.count("CT_pendingColls")) {
    XLOG(DBG1) << "Entered CT_pendingColls if statement";
    auto ctPendingCollsObjs = folly::parseJson(dump["CT_pendingColls"]);
    EXPECT_EQ(ctPendingCollsObjs.size(), 0);
  }

  if (dump.count("CT_currentColls")) {
    XLOG(DBG1) << "Entered CT_currentColls if statement";
    EXPECT_EQ(dump["CT_currentColls"], "[]");
  }
}

TEST_F(CommDumpTest, DumpAfterCollNewCollTraceWithCommsMonitor) {
  auto reduceGuard = EnvRAII{NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig};
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

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
  ASSERT_TRUE(comm->newCollTrace != nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*comm->newCollTrace));

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
  EXPECT_EQ(dump.count("CT_currentColls"), 1);

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

  // Check no pending/current entries
  if (dump.count("CT_pendingColls")) {
    XLOG(DBG1) << "Entered CT_pendingColls if statement";
    auto ctPendingCollsObjs = folly::parseJson(dump["CT_pendingColls"]);
    EXPECT_EQ(ctPendingCollsObjs.size(), 0);
  }

  if (dump.count("CT_currentColls")) {
    XLOG(DBG1) << "Entered CT_currentColls if statement";
    EXPECT_EQ(dump["CT_currentColls"], "[]");
  }
}

TEST_F(CommDumpTest, AlgoStatInCommDump) {
  ncclx::comms_monitor::CommsMonitor::testOnlyClearComms();
  auto reduceGuard = EnvRAII{NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig};
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace", "algostat"});
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  constexpr int numColls = 5;
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

  std::unordered_map<std::string, std::string> dump;
  auto res = ncclCommDump(comm, dump);
  ASSERT_EQ(res, ncclSuccess);

  ASSERT_EQ(dump.count("algoStat"), 1);
  auto algoStatObj = folly::parseJson(dump["algoStat"]);
  ASSERT_TRUE(algoStatObj.count("AllReduce"));

  int totalCalls = 0;
  for (const auto& [algoName, sizeMap] : algoStatObj["AllReduce"].items()) {
    ASSERT_TRUE(sizeMap.isObject());
    for (const auto& [sizeStr, count] : sizeMap.items()) {
      EXPECT_GT(count.asInt(), 0);
      EXPECT_GT(folly::to<size_t>(sizeStr.asString()), 0);
      totalCalls += count.asInt();
    }
  }
  EXPECT_EQ(totalCalls, numColls);
}

TEST_F(CommDumpTest, DumpAllWithRequestFieldsCommInfoOnly) {
  ncclx::comms_monitor::CommsMonitor::testOnlyClearComms();
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  constexpr int numColls = 3;
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
  ASSERT_NE(comm->newCollTrace, nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*comm->newCollTrace));

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      dumpAll;
  auto res = ncclCommDumpAll(
      dumpAll, {{"comm_dump::requestFields", "commHash;rank;nRanks"}});
  ASSERT_EQ(res, ncclSuccess);
  EXPECT_GE(dumpAll.size(), 1);

  auto commHash = hashToHexStr(comm->commHash);
  ASSERT_TRUE(dumpAll.count(commHash));
  const auto& commDump = dumpAll.at(commHash);

  EXPECT_EQ(commDump.count("commHash"), 1);
  EXPECT_EQ(commDump.count("rank"), 1);
  EXPECT_EQ(commDump.count("nRanks"), 1);

  // Keys not requested should be absent
  EXPECT_EQ(commDump.count("localRank"), 0);
  EXPECT_EQ(commDump.count("node"), 0);
  EXPECT_EQ(commDump.count("CT_pastColls"), 0);
  EXPECT_EQ(commDump.count("CT_pendingColls"), 0);
  EXPECT_EQ(commDump.count("CT_currentColls"), 0);
  EXPECT_EQ(commDump.count("memory"), 0);
}

TEST_F(CommDumpTest, DumpAllWithRequestFieldsTotalCommDurPerIteration) {
  ncclx::comms_monitor::CommsMonitor::testOnlyClearComms();
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  constexpr int numColls = 3;
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
  ASSERT_NE(comm->newCollTrace, nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*comm->newCollTrace));

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      dumpAll;
  auto res = ncclCommDumpAll(
      dumpAll,
      {{"comm_dump::requestFields", "GlobalInfo::totalCommDurPerIterationUs"}});
  ASSERT_EQ(res, ncclSuccess);

  // Per-communicator maps should be empty (no fields requested)
  auto commHash = hashToHexStr(comm->commHash);
  if (dumpAll.count(commHash)) {
    const auto& commDump = dumpAll.at(commHash);
    EXPECT_EQ(commDump.count("CT_pastColls"), 0);
    EXPECT_EQ(commDump.count("commHash"), 0);
  }

  // Aggregated result should be in GlobalInfo
  ASSERT_TRUE(dumpAll.count("GlobalInfo"));
  const auto& globalInfo = dumpAll.at("GlobalInfo");
  EXPECT_EQ(globalInfo.count("totalCommDurPerIterationUs"), 1);
}

TEST_F(CommDumpTest, DumpAllWithEmptyRequestFieldsDumpsEverything) {
  ncclx::comms_monitor::CommsMonitor::testOnlyClearComms();
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  constexpr int numColls = 3;
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
  ASSERT_NE(comm->newCollTrace, nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*comm->newCollTrace));

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      dumpAll;
  auto res = ncclCommDumpAll(dumpAll);
  ASSERT_EQ(res, ncclSuccess);
  EXPECT_GE(dumpAll.size(), 1);

  auto commHash = hashToHexStr(comm->commHash);
  ASSERT_TRUE(dumpAll.count(commHash));
  const auto& commDump = dumpAll.at(commHash);

  EXPECT_EQ(commDump.count("commHash"), 1);
  EXPECT_EQ(commDump.count("rank"), 1);
  EXPECT_EQ(commDump.count("CT_pastColls"), 1);
  EXPECT_EQ(commDump.count("CT_pendingColls"), 1);
  EXPECT_EQ(commDump.count("CT_currentColls"), 1);
}

TEST_F(CommDumpTest, DumpAllGlobalInfoOnlySkipsPerCommDump) {
  ncclx::comms_monitor::CommsMonitor::testOnlyClearComms();
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  constexpr int numColls = 3;
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
  ASSERT_NE(comm->newCollTrace, nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*comm->newCollTrace));

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      dumpAll;
  auto res = ncclCommDumpAll(
      dumpAll,
      {{"comm_dump::requestFields", "GlobalInfo::totalCommDurPerIterationUs"}});
  ASSERT_EQ(res, ncclSuccess);

  // Per-communicator entries should NOT exist at all (shortcut skipped the
  // loop)
  auto commHash = hashToHexStr(comm->commHash);
  EXPECT_EQ(dumpAll.count(commHash), 0)
      << "Per-communicator dump should be skipped when only GlobalInfo keys requested";

  // GlobalInfo should be present
  ASSERT_TRUE(dumpAll.count("GlobalInfo"));
  EXPECT_EQ(dumpAll.at("GlobalInfo").count("totalCommDurPerIterationUs"), 1);
}

TEST_F(CommDumpTest, DumpAllSingleCollTraceKey) {
  ncclx::comms_monitor::CommsMonitor::testOnlyClearComms();
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto collRecordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1);

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  constexpr int numColls = 3;
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
  ASSERT_NE(comm->newCollTrace, nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*comm->newCollTrace));

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      dumpAll;
  auto res =
      ncclCommDumpAll(dumpAll, {{"comm_dump::requestFields", "CT_pastColls"}});
  ASSERT_EQ(res, ncclSuccess);

  auto commHash = hashToHexStr(comm->commHash);
  ASSERT_TRUE(dumpAll.count(commHash));
  const auto& commDump = dumpAll.at(commHash);

  // Only CT_pastColls should be present
  EXPECT_EQ(commDump.count("CT_pastColls"), 1);
  EXPECT_EQ(commDump.count("CT_pendingColls"), 0);
  EXPECT_EQ(commDump.count("CT_currentColls"), 0);
  EXPECT_EQ(commDump.count("CT_currentIteration"), 0);
  EXPECT_EQ(commDump.count("commHash"), 0);
  EXPECT_EQ(commDump.count("rank"), 0);
  EXPECT_EQ(commDump.count("memory"), 0);
}

TEST_F(CommDumpTest, DumpAllMixedPerCommAndGlobalInfoKeys) {
  ncclx::comms_monitor::CommsMonitor::testOnlyClearComms();
  auto commsMonitorGuard = EnvRAII(NCCL_COMMSMONITOR_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};

  constexpr int numColls = 3;
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
  ASSERT_NE(comm->newCollTrace, nullptr);
  EXPECT_TRUE(waitForCollTraceDrain(*comm->newCollTrace));

  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      dumpAll;
  auto res = ncclCommDumpAll(
      dumpAll,
      {{"comm_dump::requestFields",
        "rank;nRanks;GlobalInfo::totalCommDurPerIterationUs"}});
  ASSERT_EQ(res, ncclSuccess);

  // Per-communicator should have only rank and nRanks
  auto commHash = hashToHexStr(comm->commHash);
  ASSERT_TRUE(dumpAll.count(commHash));
  const auto& commDump = dumpAll.at(commHash);
  EXPECT_EQ(commDump.count("rank"), 1);
  EXPECT_EQ(commDump.count("nRanks"), 1);
  EXPECT_EQ(commDump.count("commHash"), 0);
  EXPECT_EQ(commDump.count("CT_pastColls"), 0);

  // GlobalInfo should also be present
  ASSERT_TRUE(dumpAll.count("GlobalInfo"));
  EXPECT_EQ(dumpAll.at("GlobalInfo").count("totalCommDurPerIterationUs"), 1);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
