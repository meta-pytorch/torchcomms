// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "comm.h" // @manual

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranEx.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/trainer/TrainerContext.h"
#include "meta/colltrace/CollTrace.h"

#include "meta/wrapper/CtranExComm.h"
#include "meta/wrapper/DataTypeStrUtils.h"

#define CAPTURE_STDOUT_WITH_FAIL_SAFE()                                    \
  testing::internal::CaptureStdout();                                      \
  SCOPE_FAIL {                                                             \
    std::string output = testing::internal::GetCapturedStdout();           \
    std::cout << "Test failed with stdout being: " << output << std::endl; \
  };

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

TEST_F(CollTraceTest, TraceFeatureEnableCollTrace) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  startVerboseLogging();
  CAPTURE_STDOUT_WITH_FAIL_SAFE()
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  endVerboseLogging();
  std::string output = testing::internal::GetCapturedStdout();
  //
  EXPECT_THAT(
      output, testing::HasSubstr("enabled features: trace - Init COMPLETE"));
  EXPECT_THAT(
      output,
      testing::Not(testing::HasSubstr("COLLTRACE initialization failed")));
}

TEST_F(CollTraceTest, VerboseAllReduce) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"verbose"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  startVerboseLogging();
  CAPTURE_STDOUT_WITH_FAIL_SAFE()

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  endVerboseLogging();
  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    std::stringstream ss;
    ss << "COLLTRACE: opCount " << std::hex << i << " opName AllReduce";
    std::string traceLog = ss.str();
    EXPECT_THAT(
        output,
        testing::HasSubstr(
            fmt::format(
                "COLLTRACE: collId {} opCount {} opName AllReduce", i, i)));
  }
}

TEST_F(CollTraceTest, VerboseAllToAll) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"verbose"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  startVerboseLogging();
  CAPTURE_STDOUT_WITH_FAIL_SAFE()

  prepareAllToAll(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  endVerboseLogging();
  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    EXPECT_THAT(
        output,
        testing::HasSubstr(
            fmt::format(
                "COLLTRACE: collId {} opCount {} opName SendRecv", i, i)));
  }
}

TEST_F(CollTraceTest, VerboseSendRecv) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"verbose"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  startVerboseLogging();
  CAPTURE_STDOUT_WITH_FAIL_SAFE()

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
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  endVerboseLogging();
  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    EXPECT_THAT(
        output,
        testing::HasSubstr(
            fmt::format(
                "COLLTRACE: collId {} opCount {} opName SendRecv", i, i)));
  }
}

TEST_F(CollTraceTest, VerboseSendOrRecv) {
  if (this->numRanks % 2) {
    GTEST_SKIP() << "This test requires even number of ranks";
  }

  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"verbose"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  startVerboseLogging();
  CAPTURE_STDOUT_WITH_FAIL_SAFE()

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
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  endVerboseLogging();
  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    EXPECT_THAT(
        output,
        testing::HasSubstr(
            fmt::format(
                "COLLTRACE: collId {} opCount {} opName {}",
                i,
                i,
                this->globalRank % 2 == 0 ? "Send" : "Recv")));
  }
}

TEST_F(CollTraceTest, DumpSendRecv) {
  if (this->numRanks % 2) {
    GTEST_SKIP() << "This test requires even number of ranks";
  }

  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"verbose"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
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
  EXPECT_TRUE(comm->ctranComm_->collTrace_ != nullptr);
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pastColls.size(), nColl);
  EXPECT_EQ(dump.currentColl, nullptr);

  for (auto& coll : dump.pastColls) {
    EXPECT_EQ(coll.opName, this->globalRank % 2 == 0 ? "Send" : "Recv");
    EXPECT_EQ(coll.dataType, ncclInt8);
    EXPECT_EQ(coll.count, count * ncclTypeSize(ncclInt));
    EXPECT_EQ(coll.stream, stream);
    EXPECT_GT(coll.startTs.time_since_epoch().count(), 0);
    EXPECT_GT(coll.latency, 0);
  }
}

TEST_F(CollTraceTest, DumpAllFinished) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->ctranComm_->collTrace_ != nullptr);
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();
  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pastColls.size(), nColl);
  EXPECT_EQ(dump.currentColl, nullptr);
}

TEST_F(CollTraceTest, DumpWithUnfinished) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->ctranComm_->collTrace_ != nullptr);
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  // schedule more after the first 10 coll are finished
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  auto dump = comm->ctranComm_->collTrace_->dump();

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
  for (auto& coll : dump.pastColls) {
    EXPECT_GT(coll.startTs.time_since_epoch().count(), 0);
  }
  for (auto& coll : dump.pendingColls) {
    EXPECT_EQ(coll.startTs.time_since_epoch().count(), 0);
  }
}

TEST_F(CollTraceTest, TestSerializedDump) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->ctranComm_->collTrace_ != nullptr);
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  // schedule more after the first 10 coll are finished
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  auto dump = comm->ctranComm_->collTrace_->dump();

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
  constexpr std::string_view startTsStr = "\"startTs\": ";
  for (auto& coll : dump.pastColls) {
    auto serialized = coll.serialize(true);
    EXPECT_THAT(serialized, testing::HasSubstr(startTsStr));
    EXPECT_GT(coll.startTs.time_since_epoch().count(), 0);
  }
  for (auto& coll : dump.pendingColls) {
    auto serialized = coll.serialize(true);
    EXPECT_THAT(serialized, testing::HasSubstr(startTsStr));
    EXPECT_EQ(coll.startTs.time_since_epoch().count(), 0);
  }
}

TEST_F(CollTraceTest, TestScubaEntry) {
  // overwrite CollTrace features before creating comm
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl = 10;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_TRUE(comm->ctranComm_->collTrace_ != nullptr);
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();

  for (auto& coll : dump.pastColls) {
    auto scubaEntry = coll.toScubaEntry();
    auto& normalMap = scubaEntry.getNormalMap();
    auto& intMap = scubaEntry.getIntMap();
    EXPECT_EQ(normalMap["codepath"], "baseline");
    EXPECT_EQ(intMap["rank"], this->globalRank);
    EXPECT_EQ(intMap["collId"], static_cast<int64_t>(coll.collId));
    EXPECT_EQ(intMap["opCount"], static_cast<int64_t>(coll.opCount));
    EXPECT_EQ(intMap["stream"], reinterpret_cast<int64_t>(stream));
    EXPECT_EQ(intMap["iteration"], coll.iteration);
    EXPECT_EQ(normalMap["opName"], "AllReduce");
    if (coll.sendbuff.has_value()) {
      EXPECT_EQ(intMap["sendbuff"], reinterpret_cast<int64_t>(sendBuf));
    }
    if (coll.recvbuff.has_value()) {
      EXPECT_EQ(intMap["recvbuff"], reinterpret_cast<int64_t>(recvBuf));
    }
    if (coll.count.has_value()) {
      EXPECT_EQ(intMap["count"], static_cast<int64_t>(count));
    }
    EXPECT_EQ(intMap["nThreads"], coll.nThreads);
    EXPECT_EQ(normalMap["dataType"], getDatatypeStr(coll.dataType));
    EXPECT_TRUE(coll.baselineAttr.has_value());
    if (coll.baselineAttr.has_value()) {
      auto& baselineAttr = coll.baselineAttr.value();
      EXPECT_EQ(normalMap["redOp"], getRedOpStr(baselineAttr.op));
      EXPECT_EQ(intMap["root"], baselineAttr.root);
      EXPECT_EQ(normalMap["algorithm"], ncclAlgoStr[baselineAttr.algorithm]);
      EXPECT_EQ(normalMap["protocol"], ncclProtoStr[baselineAttr.protocol]);
      EXPECT_EQ(intMap["nChannels"], baselineAttr.nChannels);
    }
    EXPECT_EQ(
        intMap["startTs"],
        std::chrono::duration_cast<std::chrono::microseconds>(
            coll.startTs.time_since_epoch())
            .count());
    EXPECT_EQ(
        intMap["enqueueTs"],
        std::chrono::duration_cast<std::chrono::microseconds>(
            coll.enqueueTs.time_since_epoch())
            .count());
    EXPECT_EQ(intMap["InterCollTimeUs"], coll.interCollTime.count());
    EXPECT_NEAR(intMap["ExecutionTimeUs"], 1000 * coll.latency, 1);
    EXPECT_EQ(
        intMap["QueueingTimeUs"],
        std::chrono::duration_cast<std::chrono::microseconds>(
            coll.startTs - coll.enqueueTs)
            .count());
  }
}

TEST_F(CollTraceTest, TestRecordNoDropBelowLimit) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto recordGuard = EnvRAII(
      NCCL_COLLTRACE_RECORD_MAX, NCCL_COLLTRACE_RECORD_MAX_DEFAULTCVARVALUE);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  if (NCCL_COLLTRACE_RECORD_MAX_DEFAULTCVARVALUE <= 1) {
    GTEST_SKIP()
        << "NCCL_COLLTRACE_RECORD_MAX_DEFAULTCVARVALUE is too small. Skipping test.";
  }
  const int nColl = std::max(NCCL_COLLTRACE_RECORD_MAX - 5, 1);

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  EXPECT_THAT(comm->ctranComm_->collTrace_, testing::NotNull());

  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto traceDump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(traceDump.pastColls.size(), nColl);
}

TEST_F(CollTraceTest, TestRecordNoDropByEnv) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto recordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl =
      std::min(NCCL_COLLTRACE_RECORD_MAX_DEFAULTCVARVALUE, 100) * 5;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_THAT(comm->ctranComm_->collTrace_, testing::NotNull());

  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto traceDump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(traceDump.pastColls.size(), nColl);
}

TEST_F(CollTraceTest, TestRecordDropOverLIMIT) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto recordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, 100);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1048576;
  const int nColl = NCCL_COLLTRACE_RECORD_MAX * 5;

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }

  EXPECT_THAT(comm->ctranComm_->collTrace_, testing::NotNull());

  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto traceDump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(traceDump.pastColls.size(), NCCL_COLLTRACE_RECORD_MAX);
}

TEST_F(CollTraceTest, TestCtranScubaEntry) {
  // overwrite CollTrace features before creating comm
  auto ctranAlgoGuard = EnvRAII(NCCL_ALLTOALL_ALGO, NCCL_ALLTOALL_ALGO::ctran);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);

  constexpr int count = 1048576;
  constexpr int nColl = 10;

  if (!ctranAllToAllSupport(
          count, commInt, comm->ctranComm_.get(), NCCL_ALLTOALL_ALGO)) {
    GTEST_SKIP()
        << "Skip test because this comm does not have Ctran All to All support.";
  }

  prepareCtranAllToAll(comm, count);
  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream));
  }

  EXPECT_TRUE(comm->ctranComm_->collTrace_ != nullptr);
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();

  for (auto& coll : dump.pastColls) {
    auto scubaEntry = coll.toScubaEntry();
    auto& normalMap = scubaEntry.getNormalMap();
    auto& intMap = scubaEntry.getIntMap();
    EXPECT_EQ(normalMap["codepath"], "ctran");
    EXPECT_EQ(intMap["rank"], this->globalRank);
    EXPECT_EQ(intMap["collId"], static_cast<int64_t>(coll.collId));
    EXPECT_EQ(intMap["opCount"], static_cast<int64_t>(coll.opCount));
    EXPECT_EQ(intMap["stream"], reinterpret_cast<int64_t>(stream));
    EXPECT_EQ(intMap["iteration"], coll.iteration);
    EXPECT_EQ(normalMap["opName"], "AllToAll");
    if (coll.sendbuff.has_value()) {
      EXPECT_EQ(intMap["sendbuff"], reinterpret_cast<int64_t>(sendBuf));
    }
    if (coll.recvbuff.has_value()) {
      EXPECT_EQ(intMap["recvbuff"], reinterpret_cast<int64_t>(recvBuf));
    }
    if (coll.count.has_value()) {
      EXPECT_EQ(intMap["count"], static_cast<int64_t>(count));
    }
    EXPECT_EQ(normalMap["dataType"], getDatatypeStr(ncclInt));
    EXPECT_EQ(
        intMap["startTs"],
        std::chrono::duration_cast<std::chrono::microseconds>(
            coll.startTs.time_since_epoch())
            .count());
    EXPECT_EQ(
        intMap["enqueueTs"],
        std::chrono::duration_cast<std::chrono::microseconds>(
            coll.enqueueTs.time_since_epoch())
            .count());
    EXPECT_EQ(intMap["InterCollTimeUs"], coll.interCollTime.count());
    EXPECT_NEAR(intMap["ExecutionTimeUs"], 1000 * coll.latency, 1);
    EXPECT_EQ(
        intMap["QueueingTimeUs"],
        std::chrono::duration_cast<std::chrono::microseconds>(
            coll.startTs - coll.enqueueTs)
            .count());
  }
  if (sendHandle != nullptr) {
    ncclCommDeregister(comm, sendHandle);
  }
  if (recvHandle != nullptr) {
    ncclCommDeregister(comm, recvHandle);
  }
}

TEST_F(CollTraceTest, VerboseAllToAllCtran) {
  auto ctranAlgoGuard = EnvRAII(NCCL_ALLTOALL_ALGO, NCCL_ALLTOALL_ALGO::ctran);
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"verbose"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);

  constexpr int count = 1048576;
  constexpr int nColl = 10;

  if (!ctranAllToAllSupport(
          count, commInt, comm->ctranComm_.get(), NCCL_ALLTOALL_ALGO)) {
    GTEST_SKIP()
        << "Skip test because this comm does not have Ctran All to All support.";
  }

  prepareCtranAllToAll(comm, count);

  startVerboseLogging();
  CAPTURE_STDOUT_WITH_FAIL_SAFE()

  for (int i = 0; i < nColl; i++) {
    NCCLCHECK_TEST(
        ncclAllToAll(sendBuf, recvBuf, count, ncclInt, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  endVerboseLogging();
  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < nColl; i++) {
    EXPECT_THAT(
        output,
        testing::HasSubstr(
            fmt::format(
                "COLLTRACE: collId {} opCount {} opName AllToAll", i, i)));
  }
  if (sendHandle != nullptr) {
    ncclCommDeregister(comm, sendHandle);
  }
  if (recvHandle != nullptr) {
    ncclCommDeregister(comm, recvHandle);
  }
}

TEST_F(CollTraceTest, TestBcastCtranEx) {
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);

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

  actualComm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto startOpCount = 0;
  auto collDump = actualComm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(collDump.pastColls.size(), nColl);
  EXPECT_EQ(collDump.currentColl, nullptr);
  EXPECT_EQ(collDump.pendingColls.size(), 0);
  for (const auto& coll : collDump.pastColls) {
    EXPECT_EQ(coll.collId, startOpCount);
    EXPECT_EQ(coll.opCount, startOpCount);
    EXPECT_EQ(coll.dataType, ncclInt);
    EXPECT_EQ(coll.count, count);
    EXPECT_EQ(coll.sendbuff, sendBuf);
    EXPECT_EQ(coll.recvbuff, recvBuf);
    EXPECT_EQ(coll.codepath, CollTraceColl::Codepath::CTRAN_CPU);
    EXPECT_GT(coll.startTs.time_since_epoch().count(), 0);
    EXPECT_GT(coll.latency, 0);
    startOpCount++;
  }

  releaseCtranExBcast(ctranExComm);
}

TEST_F(CollTraceTest, MixedCtranBaseline) {
  auto ctranAlgoGuard =
      EnvRAII(NCCL_ALLGATHER_ALGO, NCCL_ALLGATHER_ALGO::ctring);
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  // CTran has temporarily disabled NVL backend support. Set to nolocal to
  // enable test
  auto nolocalGuard =
      EnvRAII(NCCL_COMM_STATE_DEBUG_TOPO, NCCL_COMM_STATE_DEBUG_TOPO::nolocal);
  auto checksumSampleRateGuard =
      EnvRAII(NCCL_CTRAN_ALLGATHER_CHECKSUM_SAMPLE_RATE, 1);
  auto gx = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1);
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);

  constexpr int count = 1048576;
  constexpr int nColl = 10;

  if (!ctranAllGatherSupport(comm->ctranComm_.get(), NCCL_ALLGATHER_ALGO)) {
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
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pendingColls.size(), 0);
  EXPECT_EQ(dump.currentColl, nullptr);
  auto curOpCount = 0;
  for (const auto& coll : dump.pastColls) {
    EXPECT_EQ(coll.collId, curOpCount);
    EXPECT_EQ(coll.opCount, curOpCount);
    if (coll.opName == "AllGather") {
      EXPECT_TRUE(coll.ctranAttr.has_value());
      EXPECT_TRUE(coll.ctranAttr.value().checksum.has_value());
    }
    EXPECT_GT(coll.startTs.time_since_epoch().count(), 0);
    EXPECT_GT(coll.latency, 0);
    curOpCount++;
  }
}

TEST_F(CollTraceTest, GroupedSendRecv) {
  auto ctranGuard = EnvRAII{NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::orig};
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto gx = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);

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
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pendingColls.size(), 0);
  EXPECT_EQ(dump.currentColl, nullptr);
  EXPECT_EQ(dump.pastColls.size(), nColl);
  auto curOpCount = 0;
  for (const auto& coll : dump.pastColls) {
    EXPECT_EQ(coll.collId, curOpCount);
    EXPECT_EQ(coll.opCount, curOpCount);
    EXPECT_TRUE(coll.ranksInGroupedP2P != std::nullopt);
    if (coll.ranksInGroupedP2P.has_value()) {
      const auto& ranksVec = coll.ranksInGroupedP2P.value();
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
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto gx = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);

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
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pendingColls.size(), 0);
  EXPECT_EQ(dump.currentColl, nullptr);
  if (this->globalRank == 0 || this->globalRank == comm->nRanks - 1) {
    EXPECT_EQ(dump.pastColls.size(), nColl);
  } else {
    EXPECT_EQ(dump.pastColls.size(), 2 * nColl);
  }
  auto curOpCount = 0;
  for (const auto& coll : dump.pastColls) {
    EXPECT_EQ(coll.collId, curOpCount);
    EXPECT_EQ(coll.opCount, curOpCount);
    EXPECT_TRUE(coll.ranksInGroupedP2P != std::nullopt);
    if (coll.ranksInGroupedP2P.has_value()) {
      const auto& ranksVec = coll.ranksInGroupedP2P.value();
      EXPECT_EQ(ranksVec.size(), 2);
      EXPECT_THAT(ranksVec, ::testing::Contains(this->globalRank));
      if (coll.opName == "Send") {
        EXPECT_THAT(ranksVec, ::testing::Contains(sendPeer));
      } else if (coll.opName == "Recv") {
        EXPECT_THAT(ranksVec, ::testing::Contains(recvPeer));
      } else {
        ADD_FAILURE() << "Unexpected opName: " << coll.opName;
      }
    }
    curOpCount++;
  }
}

TEST_F(CollTraceTest, GroupedSendRecvCtran) {
  auto ctranAlgoGuard = EnvRAII{NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctran};
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto gx = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1);
  // CTran has temporarily disabled NVL backend support. Set to nolocal to
  // enable test
  auto nolocalGuard =
      EnvRAII(NCCL_COMM_STATE_DEBUG_TOPO, NCCL_COMM_STATE_DEBUG_TOPO::nolocal);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);

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
    NCCLCHECK_TEST(ncclGroupStart());
    NCCLCHECK_TEST(ncclSend(sendBuf, count, ncclInt, sendPeer, comm, stream));
    NCCLCHECK_TEST(ncclRecv(recvBuf, count, ncclInt, recvPeer, comm, stream));
    NCCLCHECK_TEST(ncclGroupEnd());
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pendingColls.size(), 0);
  EXPECT_EQ(dump.currentColl, nullptr);
  EXPECT_EQ(dump.pastColls.size(), nColl);
  auto curOpCount = 0;
  for (const auto& coll : dump.pastColls) {
    EXPECT_EQ(coll.collId, curOpCount);
    EXPECT_EQ(coll.opCount, curOpCount);
    EXPECT_TRUE(coll.ranksInGroupedP2P != std::nullopt);
    if (coll.ranksInGroupedP2P.has_value()) {
      const auto& ranksVec = coll.ranksInGroupedP2P.value();
      EXPECT_EQ(ranksVec.size(), 3);
      EXPECT_THAT(ranksVec, ::testing::Contains(this->globalRank));
      EXPECT_THAT(ranksVec, ::testing::Contains(sendPeer));
      EXPECT_THAT(ranksVec, ::testing::Contains(recvPeer));
    }
    curOpCount++;
  }
}

TEST_F(CollTraceTest, SimulatePPSendRecvCtran) {
  auto ctranAlgoGuard = EnvRAII{NCCL_SENDRECV_ALGO, NCCL_SENDRECV_ALGO::ctran};
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  // CTran has temporarily disabled NVL backend support. Set to nolocal to
  // enable test
  auto nolocalGuard =
      EnvRAII(NCCL_COMM_STATE_DEBUG_TOPO, NCCL_COMM_STATE_DEBUG_TOPO::nolocal);
  auto checksumSampleRateGuard =
      EnvRAII(NCCL_CTRAN_SENDRECV_CHECKSUM_SAMPLE_RATE, 1);
  auto gx = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, -1);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

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
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pendingColls.size(), 0);
  EXPECT_EQ(dump.currentColl, nullptr);
  if (this->globalRank == 0 || this->globalRank == comm->nRanks - 1) {
    EXPECT_EQ(dump.pastColls.size(), nColl);
  } else {
    EXPECT_EQ(dump.pastColls.size(), 2 * nColl);
  }
  auto curOpCount = 0;
  for (const auto& coll : dump.pastColls) {
    EXPECT_EQ(coll.collId, curOpCount);
    EXPECT_EQ(coll.opCount, curOpCount);
    EXPECT_TRUE(coll.ranksInGroupedP2P != std::nullopt);
    if (coll.ranksInGroupedP2P.has_value()) {
      const auto& ranksVec = coll.ranksInGroupedP2P.value();
      EXPECT_EQ(ranksVec.size(), 2);
      EXPECT_THAT(ranksVec, ::testing::Contains(this->globalRank));
      if (coll.opName == "Send") {
        EXPECT_THAT(ranksVec, ::testing::Contains(sendPeer));
      } else if (coll.opName == "Recv") {
        EXPECT_THAT(ranksVec, ::testing::Contains(recvPeer));
      } else {
        ADD_FAILURE() << "Unexpected opName: " << coll.opName;
      }
    }
    EXPECT_TRUE(coll.ctranAttr.has_value());
    EXPECT_TRUE(coll.ctranAttr.value().checksum.has_value());

    curOpCount++;
  }
}

TEST_F(CollTraceTest, TestIterLimit) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto iterGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX_ITERATIONS, 4);
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);

  constexpr int count = 1048576;
  constexpr int nColl = 10;

  prepareAllreduce(count);

  for (int i = 0; i < nColl; i++) {
    ncclxSetIteration(i);
    NCCLCHECK_TEST(
        ncclAllReduce(sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pendingColls.size(), 0);
  EXPECT_EQ(dump.currentColl, nullptr);
  EXPECT_EQ(dump.pastColls.size(), 4);
}

TEST_F(CollTraceTest, winPutWait) {
  constexpr auto kNumElements = 1 << 20;
  constexpr auto kNumIters = 10;

  auto envGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::ctdirect);
  auto ctranGuard = EnvRAII(NCCL_CTRAN_ENABLE, true);
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto cpuGuard = EnvRAII(NCCL_COLLTRACE_CTRAN_USE_CPU_RECORD, true);
  auto recordGuard = EnvRAII(NCCL_COLLTRACE_RECORD_MAX, kNumIters * 3);

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t put_stream, wait_stream;
  cudaEvent_t start_event, end_event;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaEventCreate(&start_event));
  CUDACHECK_TEST(cudaEventCreate(&end_event));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  ncclWindow_t win = nullptr;
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
    if (iter == 0) {
      CUDACHECK_TEST(cudaEventRecord(start_event, put_stream));
    }
  }
  CUDACHECK_TEST(cudaEventRecord(end_event, put_stream));

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

  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();
  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pendingColls.size(), 0);
  EXPECT_EQ(dump.currentColl, nullptr);
  // 1 put + 1 wait + 1 allreduce per iteration
  EXPECT_EQ(dump.pastColls.size(), 3 * kNumIters);
  for (int i = 0; i < kNumIters; i++) {
    EXPECT_EQ(dump.pastColls[2 * i].opName, "PutSignal");
    EXPECT_EQ(dump.pastColls[2 * i + 1].opName, "WaitSignal");
  }

  for (int i = 0; i < kNumIters; i++) {
    const auto& coll = dump.pastColls[2 * kNumIters + i];
    EXPECT_EQ(coll.opName, "AllReduce");
  }

  res = ncclWinFree(comm, win);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaEventDestroy(start_event));
  CUDACHECK_TEST(cudaEventDestroy(end_event));
  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));

  EXPECT_EQ(errs, 0);
}

TEST_F(CollTraceTest, TestCudaGraphAllReduce) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto algoGuard = EnvRAII{NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig};
  auto graphGuard = EnvRAII{NCCL_COLLTRACE_TRACE_CUDA_GRAPH, true};
  // Set a big enough number to get all the colls
  auto recordGuard = EnvRAII{NCCL_COLLTRACE_RECORD_MAX, 1000};
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1024;
  const int nColl = 3;

  bool graphCreated = false;
  cudaGraph_t graph;
  cudaGraphExec_t instance;

  int checkBuf[count];

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    for (int j = 0; j < nColl; j++) {
      checkBuf[j] = 1;
    }
    CUDACHECK_TEST(
        cudaMemcpy(sendBuf, checkBuf, count, cudaMemcpyHostToDevice));
    if (!graphCreated) {
      CUDACHECK_TEST(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      for (int j = 0; j < nColl; j++) {
        NCCLCHECK_TEST(ncclAllReduce(
            sendBuf + j, recvBuf + j, 1, ncclInt, ncclSum, comm, stream));
      }
      CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
      CUDACHECK_TEST(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
      graphCreated = true;
    }
    CUDACHECK_TEST(cudaGraphLaunch(instance, stream));
    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    CUDACHECK_TEST(
        cudaMemcpy(checkBuf, recvBuf, count, cudaMemcpyDeviceToHost));
    for (int j = 0; j < nColl; j++) {
      EXPECT_EQ(checkBuf[j], this->numRanks);
    }
  }
  ASSERT_TRUE(comm->ctranComm_->collTrace_ != nullptr);
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();
  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pendingColls.size(), 0);
  EXPECT_EQ(dump.currentColl, nullptr);
  EXPECT_EQ(dump.pastColls.size(), nColl * nColl);
  CUDACHECK_TEST(cudaGraphExecDestroy(instance));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
}

TEST_F(CollTraceTest, TestCudaGraphDisabledByEnvVar) {
  auto traceGuard = EnvRAII(NCCL_COLLTRACE, {"trace"});
  auto algoGuard = EnvRAII{NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig};
  auto graphGuard = EnvRAII{NCCL_COLLTRACE_TRACE_CUDA_GRAPH, false};
  // Set a big enough number to get all the colls
  auto recordGuard = EnvRAII{NCCL_COLLTRACE_RECORD_MAX, 1000};
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  ASSERT_EQ(comm->newCollTrace, nullptr);
  const int count = 1024;
  const int nColl = 3;

  bool graphCreated = false;
  cudaGraph_t graph;
  cudaGraphExec_t instance;

  int checkBuf[count];

  prepareAllreduce(count);
  for (int i = 0; i < nColl; i++) {
    for (int j = 0; j < nColl; j++) {
      checkBuf[j] = 1;
    }
    CUDACHECK_TEST(
        cudaMemcpy(sendBuf, checkBuf, count, cudaMemcpyHostToDevice));
    if (!graphCreated) {
      CUDACHECK_TEST(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      for (int j = 0; j < nColl; j++) {
        NCCLCHECK_TEST(ncclAllReduce(
            sendBuf + j, recvBuf + j, 1, ncclInt, ncclSum, comm, stream));
      }
      CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
      CUDACHECK_TEST(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
      graphCreated = true;
    }
    CUDACHECK_TEST(cudaGraphLaunch(instance, stream));
    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    CUDACHECK_TEST(
        cudaMemcpy(checkBuf, recvBuf, count, cudaMemcpyDeviceToHost));
    for (int j = 0; j < nColl; j++) {
      EXPECT_EQ(checkBuf[j], this->numRanks);
    }
  }
  ASSERT_TRUE(comm->ctranComm_->collTrace_ != nullptr);
  comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();
  auto dump = comm->ctranComm_->collTrace_->dump();
  EXPECT_EQ(dump.pendingColls.size(), 0);
  EXPECT_EQ(dump.currentColl, nullptr);
  EXPECT_EQ(dump.pastColls.size(), 0);
  CUDACHECK_TEST(cudaGraphExecDestroy(instance));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
