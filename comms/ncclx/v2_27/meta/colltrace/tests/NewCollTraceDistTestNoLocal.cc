// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>
#include <unistd.h>
#include <cstddef>
#include <exception>
#include <filesystem>

#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rfe/scubadata/ScubaData.h>

#include "comm.h" // @manual
#include "nccl.h" // @manual

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranEx.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/colltrace/CollTrace.h"
#include "comms/utils/colltrace/tests/nvidia-only/CPUControlledKernel.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/commDump.h"
#include "meta/wrapper/CtranExComm.h"

using ::meta::comms::colltrace::CollTraceConfig;

class CollTraceTest : public NcclxBaseTest {
 public:
  CollTraceTest() = default;
  void SetUp() override {
    // Set up dummy values for environment variables for Scuba test
    setenv("WORLD_SIZE", "4", 0);
    setenv("HPC_JOB_NAME", "CollTraceUT", 0);
    setenv("HPC_JOB_VERSION", "1", 0);
    setenv("HPC_JOB_ATTEMPT_INDEX", "2", 0);
    setenv(
        "NCCL_HPC_JOB_IDS",
        "HPC_JOB_NAME,HPC_JOB_VERSION,HPC_JOB_ATTEMPT_INDEX",
        0);
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);

    NcclxBaseTest::SetUp();
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

  void prepareCtranAllGather(ncclComm* comm, const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));
  }

  void prepareCtranAllToAll(ncclComm* comm, const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * this->numRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));
    NCCLCHECK_TEST(ncclCommRegister(
        comm, sendBuf, count * this->numRanks * sizeof(int), &sendHandle));
    NCCLCHECK_TEST(ncclCommRegister(
        comm, recvBuf, count * this->numRanks * sizeof(int), &recvHandle));
  }

  void prepareCtranExBcast(
      std::unique_ptr<::ctran::CtranExComm>& ctranExComm,
      const int count) {
    sendBuf = reinterpret_cast<int*>(malloc(count * sizeof(int)));
    recvBuf = reinterpret_cast<int*>(malloc(count * sizeof(int)));
    NCCLCHECK_TEST(
        ctranExComm->regMem(sendBuf, count * sizeof(int), &sendHandle, true));
    NCCLCHECK_TEST(
        ctranExComm->regMem(recvBuf, count * sizeof(int), &recvHandle, true));
  }

  void releaseCtranExBcast(std::unique_ptr<::ctran::CtranExComm>& ctranExComm) {
    NCCLCHECK_TEST(ctranExComm->deregMem(sendHandle));
    NCCLCHECK_TEST(ctranExComm->deregMem(recvHandle));
    free(sendBuf);
    free(recvBuf);

    // teardown won't free them again
    sendBuf = nullptr;
    recvBuf = nullptr;
  }

  void prepareAllToAll(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * this->numRanks * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * this->numRanks * sizeof(int)));
  }

  void prepareSendRecv(const int count) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(int)));
  }

  bool prepareDumpDir(const std::string& dir) {
    try {
      // always re-create a fresh dir to ensure output files are up-to-date
      if (std::filesystem::exists(dir)) {
        std::filesystem::remove_all(dir);
      }
      std::filesystem::create_directories(dir);
    } catch (const std::exception& e) {
      printf(
          "Rank %d failed to create directory %s: %s\n",
          this->globalRank,
          dir.c_str(),
          e.what());
      return false;
    }
    return true;
  }
  void startVerboseLogging() {
    NcclLogger::close();
    prevDebug = NCCL_DEBUG.empty() ? "WARN" : NCCL_DEBUG;
    prevDebugSubsys =
        NCCL_DEBUG_SUBSYS.empty() ? "INIT,BOOTSTRAP,ENV" : NCCL_DEBUG_SUBSYS;
    NCCL_DEBUG = "INFO";
    NCCL_DEBUG_SUBSYS = "INIT,COLL";
    initNcclLogger();
  }

  void endVerboseLogging() {
    sleep(1);
    NcclLogger::close();
    NCCL_DEBUG = prevDebug;
    NCCL_DEBUG_SUBSYS = prevDebugSubsys;
    initNcclLogger();
  }

  void barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
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
  std::string prevDebug;
  std::string prevDebugSubsys;
};

TEST_F(CollTraceTest, NewCollTraceAllReduce) {
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  // Sleep for a while to make sure all the colls are finished
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  ASSERT_TRUE(comm->newCollTrace != nullptr);
  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

  EXPECT_NE(dumpMap["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");

  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  EXPECT_EQ(pastCollsJson.size(), nColl);
}

TEST_F(CollTraceTest, MixedCtranBaseline) {
  auto ctranAlgoGuard =
      EnvRAII(NCCL_ALLGATHER_ALGO, NCCL_ALLGATHER_ALGO::ctring);
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  // CTran has temporarily disabled NVL backend support. Set to nolocal to
  // enable test
  auto nolocalGuard =
      EnvRAII(NCCL_COMM_STATE_DEBUG_TOPO, NCCL_COMM_STATE_DEBUG_TOPO::nolocal);
  auto checksumSampleRateGuard =
      EnvRAII(NCCL_CTRAN_ALLGATHER_CHECKSUM_SAMPLE_RATE, 1);
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);

  constexpr int count = 1048576;
  constexpr int nColl = 10;

  if (!ctranAllGatherSupport(comm->ctranComm_.get())) {
    GTEST_SKIP()
        << "Skip test because this comm does not have Ctran AllGather support.";
  }

  prepareCtranAllGather(comm, count);

  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllGather(sendBuf, recvBuf, count, ncclInt, comm, stream));
    NCCLCHECK_TEST(ncclReduceScatter(
        recvBuf, sendBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  // Sleep for a while to make sure all the colls are finished
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  ASSERT_TRUE(comm->newCollTrace != nullptr);
  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

  EXPECT_NE(dumpMap["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");

  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  EXPECT_EQ(pastCollsJson.size(), nColl * 2);
  int curOpCount = 0;
  for (const auto& coll : pastCollsJson) {
    EXPECT_EQ(coll["collId"].asInt(), curOpCount);
    EXPECT_EQ(coll["opCount"].asInt(), curOpCount);
    if (coll["opName"].asString() == "AllGather") {
      EXPECT_THAT(coll["algoName"].asString(), testing::HasSubstr("Ctran"));
    }
    EXPECT_GT(coll["startTs"].asInt(), 0);
    curOpCount++;
  }
}

TEST_F(CollTraceTest, TestBcastCtranEx) {
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);

  constexpr int count = 1048576;
  constexpr int nColl = 10;

  auto ctranExComm =
      std::make_unique<::ctran::CtranExComm>(comm, "collTraceCpuBcastUt");
  ASSERT_NE(ctranExComm, nullptr);

  if (!ctranExComm->isInitialized() || !ctranExComm->supportBroadcast()) {
    GTEST_SKIP() << fmt::format(
        "Skip test because this comm does not have CtranEx support {} or no broadcast support {}.",
        ctranExComm->isInitialized(),
        ctranExComm->supportBroadcast());
  }

  prepareCtranExBcast(ctranExComm, count);

  auto actualComm = ctranExComm->unsafeGetNcclComm();

  std::vector<std::unique_ptr<::ctran::CtranExRequest>> reqs;

  for (int i = 0; i < nColl; i++) {
    ::ctran::CtranExRequest* reqPtr = nullptr;
    NCCLCHECK_TEST(
        ctranExComm->broadcast(sendBuf, recvBuf, count, ncclInt, 0, &reqPtr));

    ASSERT_NE(reqPtr, nullptr);
    reqs.push_back(std::unique_ptr<::ctran::CtranExRequest>(reqPtr));
  }

  // Wait completion of all bcast
  for (auto& req : reqs) {
    ASSERT_EQ(req->wait(), ncclSuccess);
  }

  // Sleep for a while to make sure all the colls are finished
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto startOpCount = 0;
  auto dumpMap =
      meta::comms::ncclx::dumpNewCollTrace(*actualComm->newCollTrace);

  EXPECT_EQ(folly::parseJson(dumpMap["CT_pastColls"]).size(), nColl);
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");
  EXPECT_EQ(folly::parseJson(dumpMap["CT_pendingColls"]).size(), 0);
  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  for (const auto& coll : pastCollsJson) {
    EXPECT_EQ(coll["collId"].asInt(), startOpCount);
    EXPECT_EQ(coll["opCount"].asInt(), startOpCount);
    EXPECT_EQ(
        coll["dataType"].asString(),
        commDataTypeToString(ncclToMetaComm(ncclInt)));
    EXPECT_EQ(coll["count"].asInt(), count);
    // sendbuff and recvbuff are pointers, compare as string or skip if not
    // available EXPECT_EQ(coll["sendbuff"].asInt(),
    // reinterpret_cast<intptr_t>(sendBuf)); EXPECT_EQ(coll["recvbuff"].asInt(),
    // reinterpret_cast<intptr_t>(recvBuf));
    EXPECT_GT(coll["startTs"].asInt(), 0);
    startOpCount++;
  }

  releaseCtranExBcast(ctranExComm);
}

TEST_F(CollTraceTest, GroupedSendRecv) {
  auto ctranGuard = EnvRAII{NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::orig};

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);

  const int count = 1048576;
  const int nColl = 10;

  if (comm->nRanks < 3) {
    GTEST_SKIP()
        << "Skip test because this comm does not have enough ranks to properly test send recv.";
  }

  prepareSendRecv(count);
  int sendPeer = (this->globalRank + 1) % this->numRanks;
  int recvPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclGroupStart());
    NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, sendPeer, comm, stream));
    NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, recvPeer, comm, stream));
    NCCLCHECK_TEST(ncclGroupEnd());
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);
  EXPECT_EQ(folly::parseJson(dumpMap["CT_pendingColls"]).size(), 0);
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");
  EXPECT_EQ(folly::parseJson(dumpMap["CT_pastColls"]).size(), nColl);
  auto curOpCount = 0;
  for (const auto& coll : folly::parseJson(dumpMap["CT_pastColls"])) {
    EXPECT_EQ(coll["collId"].asInt(), curOpCount);
    EXPECT_EQ(coll["opCount"].asInt(), curOpCount);
    EXPECT_TRUE(coll["ranksInGroupedP2P"] != nullptr);
    if (!coll["ranksInGroupedP2P"].isNull()) {
      ASSERT_TRUE(coll["ranksInGroupedP2P"].isArray());
      const auto& ranksVec =
          folly::convertTo<std::vector<int>>(coll["ranksInGroupedP2P"]);
      EXPECT_EQ(ranksVec.size(), 3);
      EXPECT_THAT(ranksVec, ::testing::Contains(this->globalRank));
      EXPECT_THAT(ranksVec, ::testing::Contains(sendPeer));
      EXPECT_THAT(ranksVec, ::testing::Contains(recvPeer));
    }
    curOpCount++;
  }
}

TEST_F(CollTraceTest, GroupedSendRecvCtran) {
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  auto ctranSRGuard = EnvRAII{NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctran};

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);

  const int count = 1048576;
  const int nColl = 10;

  if (comm->nRanks < 3) {
    GTEST_SKIP()
        << "Skip test because this comm does not have enough ranks to properly test send recv.";
  }

  int sendPeer = (this->globalRank + 1) % this->numRanks;
  int recvPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;

  if (!ctranSendRecvSupport(sendPeer, comm->ctranComm_.get()) ||
      !ctranSendRecvSupport(recvPeer, comm->ctranComm_.get())) {
    GTEST_SKIP()
        << "Skip test because this comm does not support ctran sendrecv.";
  }

  prepareSendRecv(count);

  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclGroupStart());
    NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, sendPeer, comm, stream));
    NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, recvPeer, comm, stream));
    NCCLCHECK_TEST(ncclGroupEnd());
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);
  EXPECT_EQ(folly::parseJson(dumpMap["CT_pendingColls"]).size(), 0);
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");
  EXPECT_EQ(folly::parseJson(dumpMap["CT_pastColls"]).size(), nColl);
  auto curOpCount = 0;
  for (const auto& coll : folly::parseJson(dumpMap["CT_pastColls"])) {
    EXPECT_EQ(coll["collId"].asInt(), curOpCount);
    EXPECT_EQ(coll["opCount"].asInt(), curOpCount);
    EXPECT_TRUE(coll["ranksInGroupedP2P"] != nullptr);
    if (!coll["ranksInGroupedP2P"].isNull()) {
      ASSERT_TRUE(coll["ranksInGroupedP2P"].isArray());
      const auto& ranksVec =
          folly::convertTo<std::vector<int>>(coll["ranksInGroupedP2P"]);
      EXPECT_EQ(ranksVec.size(), 3);
      EXPECT_THAT(ranksVec, ::testing::Contains(this->globalRank));
      EXPECT_THAT(ranksVec, ::testing::Contains(sendPeer));
      EXPECT_THAT(ranksVec, ::testing::Contains(recvPeer));
    }
    curOpCount++;
  }
}

TEST_F(CollTraceTest, SimulatePPSendRecv) {
  auto ctranGuard = EnvRAII{NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::orig};

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);

  const int count = 1048576;
  const int nColl = 10;

  if (comm->nRanks < 3) {
    GTEST_SKIP()
        << "Skip test because this comm does not have enough ranks to properly test send recv.";
  }

  prepareSendRecv(count);
  int sendPeer = (this->globalRank + 1) % this->numRanks;
  int recvPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;
  for (int i = 0; i < nColl; i++) {
    // Not the last rank
    if (this->globalRank != comm->nRanks - 1) {
      NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, sendPeer, comm, stream));
    }
    if (this->globalRank != 0) {
      NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, recvPeer, comm, stream));
    }
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);
  EXPECT_EQ(folly::parseJson(dumpMap["CT_pendingColls"]).size(), 0);
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");
  if (this->globalRank == 0 || this->globalRank == comm->nRanks - 1) {
    EXPECT_EQ(folly::parseJson(dumpMap["CT_pastColls"]).size(), nColl);
  } else {
    EXPECT_EQ(folly::parseJson(dumpMap["CT_pastColls"]).size(), 2 * nColl);
  }
  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  auto curOpCount = 0;
  for (const auto& coll : pastCollsJson) {
    EXPECT_EQ(coll["collId"].asInt(), curOpCount);
    EXPECT_EQ(coll["opCount"].asInt(), curOpCount);
    EXPECT_TRUE(coll.contains("ranksInGroupedP2P"));
    if (coll.contains("ranksInGroupedP2P")) {
      ASSERT_TRUE(coll["ranksInGroupedP2P"].isArray());
      const auto& ranksVec =
          folly::convertTo<std::vector<int>>(coll["ranksInGroupedP2P"]);
      EXPECT_EQ(ranksVec.size(), 2);
      EXPECT_THAT(ranksVec, ::testing::Contains(this->globalRank));
      if (coll["opName"].asString() == "Send") {
        EXPECT_THAT(ranksVec, ::testing::Contains(sendPeer));
      } else if (coll["opName"].asString() == "Recv") {
        EXPECT_THAT(ranksVec, ::testing::Contains(recvPeer));
      } else {
        ADD_FAILURE() << "Unexpected opName: " << coll["opName"].asString();
      }
    }
    curOpCount++;
  }
}

TEST_F(CollTraceTest, SimulateCtranPPSendRecv) {
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  auto ctranSRGuard = EnvRAII{NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctran};

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);

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

  prepareSendRecv(count);
  for (int i = 0; i < nColl; i++) {
    // Not the last rank
    if (this->globalRank != comm->nRanks - 1) {
      NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, sendPeer, comm, stream));
    }
    if (this->globalRank != 0) {
      NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, recvPeer, comm, stream));
    }
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);
  EXPECT_EQ(folly::parseJson(dumpMap["CT_pendingColls"]).size(), 0);
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");
  if (this->globalRank == 0 || this->globalRank == comm->nRanks - 1) {
    EXPECT_EQ(folly::parseJson(dumpMap["CT_pastColls"]).size(), nColl);
  } else {
    EXPECT_EQ(folly::parseJson(dumpMap["CT_pastColls"]).size(), 2 * nColl);
  }
  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  auto curOpCount = 0;
  for (const auto& coll : pastCollsJson) {
    EXPECT_EQ(coll["collId"].asInt(), curOpCount);
    EXPECT_EQ(coll["opCount"].asInt(), curOpCount);
    EXPECT_TRUE(coll.contains("ranksInGroupedP2P"));
    if (coll.contains("ranksInGroupedP2P")) {
      ASSERT_TRUE(coll["ranksInGroupedP2P"].isArray());
      const auto& ranksVec =
          folly::convertTo<std::vector<int>>(coll["ranksInGroupedP2P"]);
      EXPECT_EQ(ranksVec.size(), 2);
      EXPECT_THAT(ranksVec, ::testing::Contains(this->globalRank));
      if (coll["opName"].asString() == "Send") {
        EXPECT_THAT(ranksVec, ::testing::Contains(sendPeer));
      } else if (coll["opName"].asString() == "Recv") {
        EXPECT_THAT(ranksVec, ::testing::Contains(recvPeer));
      } else {
        ADD_FAILURE() << "Unexpected opName: " << coll["opName"].asString();
      }
    }
    curOpCount++;
  }
}

template <typename T>
void assignChunkValue(T* buf, size_t count, T seed, T inc) {
  std::vector<T> expectedVals(count, 0);
  for (size_t i = 0; i < count; ++i) {
    expectedVals[i] = seed + i * inc;
  }
  CUDACHECK_TEST(cudaMemcpy(
      buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
}

TEST_F(CollTraceTest, winPutWait) {
  constexpr auto kNumElements = 1 << 20;
  constexpr auto kNumIters = 10;

  auto envGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::ctdirect);
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  auto recordGuard = EnvRAII{NCCL_COLLTRACE_RECORD_MAX, 1000};

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t put_stream, wait_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  ncclWin_t win = nullptr;
  void* winBase = nullptr;
  auto res = ncclWinAllocate(sizeBytes, comm, &winBase, &win);
  ASSERT_EQ(res, ncclSuccess);
  ASSERT_NE(winBase, nullptr);
  int* localbuf = reinterpret_cast<int*>(winBase);

  EXPECT_THAT(win, testing::NotNull());

  for (int peer = 0; peer < this->numRanks; peer++) {
    void* remoteAddr = nullptr;
    res = ncclWinSharedQuery(peer, comm, win, &remoteAddr);
    EXPECT_EQ(res, ncclSuccess);
    if (statex->node() == statex->node(peer)) {
      EXPECT_THAT(remoteAddr, testing::NotNull());
    } else {
      EXPECT_THAT(remoteAddr, testing::IsNull());
    }
    if (peer == this->globalRank) {
      EXPECT_EQ(remoteAddr, winBase);
    }
  }

  assignChunkValue(
      localbuf, kNumElements * statex->nRanks(), statex->rank(), 1);
  // Barrier to ensure all peers have finished creation
  this->barrier();

  int nextPeer = (this->globalRank + 1) % this->numRanks;
  int prevPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;

  for (auto iter = 0; iter < kNumIters; iter++) {
    NCCLCHECK_TEST(ncclPutSignal(
        localbuf + kNumElements * statex->rank(),
        kNumElements,
        ncclInt32,
        nextPeer,
        kNumElements * statex->rank(),
        win,
        put_stream));
    NCCLCHECK_TEST(ncclWaitSignal(prevPeer, win, wait_stream));
  }

  int errs = 0;
  // A couple of ctran all-reduce after RMA tests
  for (auto iter = 0; iter < kNumIters; iter++) {
    NCCLCHECK_TEST(ncclAllReduce(
        localbuf + kNumElements * prevPeer,
        localbuf + kNumElements * prevPeer,
        kNumElements,
        ncclInt32,
        ncclSum,
        comm,
        wait_stream));
  }

  CUDACHECK_TEST(cudaStreamSynchronize(put_stream));
  CUDACHECK_TEST(cudaStreamSynchronize(wait_stream));
  // Barrier to ensure all peers have finished put
  this->barrier();

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

  // Parse the dumpMap JSON to check values, similar to previous tests
  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  auto pendingCollsJson = folly::parseJson(dumpMap["CT_pendingColls"]);
  auto currentCollJson = dumpMap["CT_currentColl"];

  EXPECT_EQ(pendingCollsJson.size(), 0);
  EXPECT_EQ(currentCollJson, "null");
  // 1 put + 1 wait + 1 allreduce per iteration
  EXPECT_EQ(pastCollsJson.size(), 3 * kNumIters);

  for (int i = 0; i < pastCollsJson.size(); i++) {
    XLOGF(DBG, "coll {}: {}", i, pastCollsJson[i]["opName"].asString());
  }

  for (int i = 0; i < kNumIters; i++) {
    EXPECT_EQ(pastCollsJson[2 * i]["opName"].asString(), "PutNotify");
    EXPECT_EQ(pastCollsJson[2 * i + 1]["opName"].asString(), "WaitNotify");
  }

  for (int i = 0; i < kNumIters; i++) {
    const auto& coll = pastCollsJson[2 * kNumIters + i];
    EXPECT_EQ(coll["opName"].asString(), "AllReduce");
  }

  res = ncclWinFree(comm, win);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));

  EXPECT_EQ(errs, 0);
}

TEST_F(CollTraceTest, DumpWithUnfinished) {
  auto wakeUpGuard = EnvRAII(NCCL_COLLTRACE_WAKEUP_INTERVAL_MS, 10L);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Create a kernel that will block the stream, so that all the following
  // scheduled collective will be pending
  ::meta::comms::colltrace::CPUControlledKernel blockingKernel(stream);
  blockingKernel.launch();

  // schedule more after the first 10 coll are finished
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  // Give CollTrace some time to start tracking next coll
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  auto pendingCollsJson = folly::parseJson(dumpMap["CT_pendingColls"]);
  auto currentCollJson = dumpMap["CT_currentColl"];

  EXPECT_EQ(pastCollsJson.size(), nColl);
  // We will have 1 coll as current coll
  EXPECT_EQ(pendingCollsJson.size(), nColl - 1);
  EXPECT_NE(currentCollJson, "null");

  for (const auto& coll : pastCollsJson) {
    EXPECT_GT(coll["startTs"].asInt(), 0);
  }
  for (const auto& coll : pendingCollsJson) {
    EXPECT_EQ(coll["startTs"].asInt(), 0);
  }

  blockingKernel.endKernel();
}

TEST_F(CollTraceTest, DumpWithUnfinishedCtran) {
  auto wakeUpGuard = EnvRAII(NCCL_COLLTRACE_WAKEUP_INTERVAL_MS, 10L);
  auto envGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::ctdirect);
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Create a kernel that will block the stream, so that all the following
  // scheduled collective will be pending
  ::meta::comms::colltrace::CPUControlledKernel blockingKernel(stream);
  blockingKernel.launch();

  // schedule more after the first 10 coll are finished
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  // Give CollTrace some time to start tracking next coll and exited wait once
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  auto pendingCollsJson = folly::parseJson(dumpMap["CT_pendingColls"]);
  auto currentCollJson = dumpMap["CT_currentColl"];

  EXPECT_EQ(pastCollsJson.size(), nColl);
  // We will have 1 coll as current coll
  EXPECT_EQ(pendingCollsJson.size(), nColl - 1);
  EXPECT_NE(currentCollJson, "null");

  for (const auto& coll : pastCollsJson) {
    EXPECT_GT(coll["startTs"].asInt(), 0);
  }
  for (const auto& coll : pendingCollsJson) {
    EXPECT_EQ(coll["startTs"].asInt(), 0);
  }

  blockingKernel.endKernel();
}

TEST_F(CollTraceTest, GroupedAllReduce) {
  auto envGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig);
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);

  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclGroupStart());
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
    NCCLCHECK_TEST(ncclGroupEnd());
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  // Sleep for a while to make sure all the colls are finished
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  ASSERT_TRUE(comm->newCollTrace != nullptr);
  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

  EXPECT_NE(dumpMap["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");

  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  EXPECT_EQ(pastCollsJson.size(), nColl);
}

TEST_F(CollTraceTest, GroupedSendRecvAllReduce) {
  auto ctranGuard = EnvRAII{NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::orig};
  auto envGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);

  const int count = 1048576;
  const int nColl = 10;

  if (comm->nRanks < 3) {
    GTEST_SKIP()
        << "Skip test because this comm does not have enough ranks to properly test send recv.";
  }

  prepareSendRecv(count);
  int sendPeer = (this->globalRank + 1) % this->numRanks;
  int recvPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(ncclGroupStart());
    NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, sendPeer, comm, stream));
    NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, recvPeer, comm, stream));
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
    NCCLCHECK_TEST(ncclGroupEnd());
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  ASSERT_TRUE(comm->newCollTrace != nullptr);
  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

  EXPECT_NE(dumpMap["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");

  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  EXPECT_EQ(pastCollsJson.size(), nColl);
}

TEST_F(CollTraceTest, CollTraceQueryInCapture) {
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);
  const int count = 1048576;
  const int nColl = 20;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  CUDACHECK_TEST(cudaStreamBeginCapture(
      stream, cudaStreamCaptureMode::cudaStreamCaptureModeGlobal));
  // Sleep for a while to make sure all the colls are finished
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  cudaGraph_t graph;
  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));

  ASSERT_TRUE(comm->newCollTrace != nullptr);
  auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

  EXPECT_NE(dumpMap["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");

  auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
  EXPECT_EQ(pastCollsJson.size(), nColl);
}

// Previously we found that if we enqueue more collectives than what the pending
// queue can hold, we will get segfault due to colltrace handle pointing to
// freed memory. Add a test to ensure we don't encounter this issue again.
TEST_F(CollTraceTest, CollTraceTestEnqueueMoreThanPendingQueue) {
  auto wakeUpGuard = EnvRAII(NCCL_COLLTRACE_WAKEUP_INTERVAL_MS, 10L);
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->collTrace, nullptr);
  ASSERT_EQ(comm->ctranComm_->collTrace_, nullptr);

  const auto kNumElements = 8388608;
  const auto kNumIters = CollTraceConfig::kDefaultMaxPendingQueueSize;
  const auto cpuWin = true;
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t put_stream, wait_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  meta::comms::Hints hints;
  if (cpuWin) {
    hints.set("window_buffer_location", "cpu");
  }
  ctran::CtranWin* win = nullptr;
  void* winBase = nullptr;
  auto res = ctranWinAllocate(
      sizeBytes, comm->ctranComm_.get(), &winBase, &win, hints);
  ASSERT_EQ(res, ncclSuccess);
  ASSERT_NE(winBase, nullptr);

  // Always allocate localBuf from GPU mem so it can be used in ctranAllReduce
  int* localBuf = nullptr;
  void* localHdl = nullptr;
  ASSERT_EQ(
      ncclMemAlloc((void**)&localBuf, kNumElements * sizeof(int)), ncclSuccess);
  ASSERT_EQ(
      ncclCommRegister(comm, localBuf, kNumElements * sizeof(int), &localHdl),
      ncclSuccess);

  EXPECT_THAT(win, testing::NotNull());

  for (int peer = 0; peer < this->numRanks; peer++) {
    void* remoteAddr = nullptr;
    res = ctranWinSharedQuery(peer, win, &remoteAddr);
    EXPECT_EQ(res, ncclSuccess);
    if (peer == statex->rank()) {
      EXPECT_EQ(remoteAddr, winBase);
    } else if (!win->nvlEnabled(peer)) {
      EXPECT_THAT(remoteAddr, testing::IsNull());
    } else {
      EXPECT_THAT(remoteAddr, testing::NotNull());
    }
  }

  int nextPeer = (this->globalRank + 1) % this->numRanks;
  int prevPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;

  for (auto iter = 0; iter < kNumIters; iter++) {
    COMMCHECK_TEST(ctranPutSignal(
        localBuf,
        kNumElements,
        commInt32,
        nextPeer,
        kNumElements * statex->rank(),
        win,
        win->comm,
        put_stream,
        true));
    COMMCHECK_TEST(ctranWaitSignal(prevPeer, win, win->comm, wait_stream));
  }

  CUDACHECK_TEST(cudaDeviceSynchronize());

  res = ctranWinFree(win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);

  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
