// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <new>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/commDump.h"

#include <folly/json/json.h>

class ctranAllToAllPTest : public CtranDistBaseTest {
 public:
  ctranAllToAllPTest() = default;

  void generateDistRandomExpValue() {
    if (globalRank == 0) {
      expectedVal = rand();
    }
    MPI_Bcast(&expectedVal, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  void generateDistRandomCount(bool small_msg = false) {
    if (globalRank == 0) {
      if (small_msg) {
        count = std::min(8192, (int)maxRecvCount / numRanks);
      } else {
        count = rand() % (maxRecvCount / numRanks) + 1;
      }
    }
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
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
  void assignChunkValue(T* buf, size_t count, T val) {
    std::vector<T> expectedVals(count, val);
    CUDACHECK_TEST(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print first 10 failing
    // location
    for (auto i = 0; i < count; ++i) {
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
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);
    // -1 for not limiting the number of colls to trace
    setenv("NCCL_COLLTRACE_RECORD_MAX", "-1", 0);
    CtranDistBaseTest::SetUp();
    comm = commWorld;
    if (!ctran::AllToAllPSupport(comm->ctranComm_.get())) {
      GTEST_SKIP() << "Skip the test because ctran::AllToAllP is not supported";
    }

    // Allocate enough space for arguments, value assignment set in each test
    sendBuf = nullptr;
    sendHdl = nullptr;
  }

  void TearDown() override {
    CtranDistBaseTest::TearDown();
  }

  void setupHints(bool skip_ctrl_msg) {
    if (skip_ctrl_msg) {
      ASSERT_EQ(
          hints.set("ncclx_alltoallp_skip_ctrl_msg_exchange", "true"),
          ncclSuccess);
    } else {
      ASSERT_EQ(
          hints.set("ncclx_alltoallp_skip_ctrl_msg_exchange", "false"),
          ncclSuccess);
    }
  }

  void run() {
    ASSERT_TRUE(
        meta::comms::colltrace::testOnlyClearCollTraceRecords(
            comm->ctranComm_.get()));

    // Initialize double persistent requests using double recv buffer allocated.
    std::array<CtranPersistentRequest*, 2> doublePRequests;
    for (int idx = 0; idx < 2; idx++) {
      COMMCHECK_TEST(
          ctran::AllToAllPInit(
              doubleRecvbuffs[idx],
              maxRecvCount,
              hints,
              commInt,
              comm->ctranComm_.get(),
              stream,
              doublePRequests[idx]));
    }

    std::vector<size_t> counts(numTimesRunExec);
    for (int x = 0; x < numTimesRunExec; x++) {
      generateDistRandomExpValue();
      // Make sure there is at least one small message to cover fast put tests.
      generateDistRandomCount(/*small_msg*/ x == 0);
      counts[x] = count;

      // Assign different value for each send chunk
      for (int i = 0; i < numRanks; ++i) {
        assignChunkValue<int>(
            sendBuf + count * i, count, expectedVal + globalRank * 100 + i + 1);
      }
      const int idx = x % 2;
      auto res = ctran::AllToAllPExec(sendBuf, count, doublePRequests[idx]);
      ASSERT_EQ(res, commSuccess);
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      // Check each received chunk
      int* recvbuff = (int*)doubleRecvbuffs[idx];
      for (int i = 0; i < numRanks; ++i) {
        int errs = checkChunkValue<int>(
            recvbuff + count * i,
            count,
            expectedVal + i * 100 + globalRank + 1);
        EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << i
                           << " at " << recvbuff + count * i << " with " << errs
                           << " errors";
      }
    }

    for (int idx = 0; idx < 2; idx++) {
      auto destroyRes = ctran::AllToAllPDestroy(doublePRequests[idx]);
      ASSERT_EQ(destroyRes, commSuccess);
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
    // Sleep for a while to make sure all the colls are finished
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(comm->newCollTrace != nullptr);
    auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

    EXPECT_NE(dumpMap["CT_pastColls"], "[]");
    EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
    EXPECT_EQ(dumpMap["CT_currentColl"], "null");

    auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
    auto statex = comm->ctranComm_->statex_.get();
    // If there are remote peers, AllToAllPInit submits gpe op and was recorded
    // by colltrace.
    int numTimesRunInit = statex->nNodes() == 1 ? 0 : 2;
    EXPECT_EQ(pastCollsJson.size(), numTimesRunInit + numTimesRunExec);

    // Skip the check for the AllToAllPInit (first 2) colls.
    for (int i = numTimesRunInit; i < pastCollsJson.size(); i++) {
      const auto& coll = pastCollsJson[i];
      if (statex->nNodes() == 1) {
        // If only cuda kernel is launched (no IB put), AlltoAllP is essentially
        // alltoall because it shares cuda kernel logic with AlltoAll.
        EXPECT_EQ(coll["opName"].asString(), "AllToAll");
      } else {
        EXPECT_EQ(coll["opName"].asString(), "AllToAllP");
      }
      EXPECT_EQ(coll["count"].asInt(), counts[i - numTimesRunInit]);
      EXPECT_EQ(
          coll["algoName"].asString(),
          ctran::alltoallp::AlgoImpl::algoName(NCCL_ALLTOALL_ALGO::ctran));
    }

    // Alltoall uses kernel staged copy not NVL iput
    std::vector<CtranMapperBackend> excludedBackends = {
        CtranMapperBackend::NVL};
    // If all ranks are local, uses only kernel staged copy
    if (comm->ctranComm_->statex_->nLocalRanks() ==
        comm->ctranComm_->statex_->nRanks()) {
      excludedBackends.push_back(CtranMapperBackend::IB);
    }
    verifyBackendsUsed(
        comm->ctranComm_->ctran_.get(),
        comm->ctranComm_->statex_.get(),
        kMemCudaMalloc,
        excludedBackends);
  }

 protected:
  ncclComm_t comm{nullptr};
  meta::comms::Hints hints;
  int* sendBuf{nullptr};
  std::array<void*, 2> doubleRecvbuffs;
  size_t maxRecvCount{1024 * 1024};
  size_t count{0};
  void* sendHdl{nullptr};
  std::array<void*, 2> recvHdls;
  int expectedVal{0};
  int numTimesRunExec{7};
};

class ctranAllToAllPTestParam
    : public ctranAllToAllPTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool, bool>> {};

TEST_P(ctranAllToAllPTestParam, normalRun) {
  const auto& [enable_lowlatency_config, skip_ctrl_msg, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  setupHints(skip_ctrl_msg);

  sendBuf = (int*)createDataBuf(maxRecvCount * sizeof(int), &sendHdl);
  for (int idx = 0; idx < 2; idx++) {
    doubleRecvbuffs[idx] =
        (int*)createDataBuf(maxRecvCount * sizeof(int), &recvHdls[idx]);
  }
  run();

  releaseDataBuf(sendBuf, sendHdl);
  for (int idx = 0; idx < 2; idx++) {
    releaseDataBuf(doubleRecvbuffs[idx], recvHdls[idx]);
  }
}

TEST_P(ctranAllToAllPTestParam, countExceedsPreregBufferSize) {
  const auto& [enable_lowlatency_config, skip_ctrl_msg, enable_put_fast_path_for_small_msgs] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(
      NCCL_CTRAN_ENABLE_PUT_FAST_PATH_FOR_SMALL_MSGS,
      enable_put_fast_path_for_small_msgs);
  setupHints(skip_ctrl_msg);

  sendBuf = (int*)createDataBuf(maxRecvCount * sizeof(int), &sendHdl);
  for (int idx = 0; idx < 2; idx++) {
    doubleRecvbuffs[idx] =
        (int*)createDataBuf(maxRecvCount * sizeof(int), &recvHdls[idx]);
  }

  CtranPersistentRequest* pRequest;
  COMMCHECK_TEST(
      ctran::AllToAllPInit(
          doubleRecvbuffs[0],
          maxRecvCount,
          hints,
          commInt,
          comm->ctranComm_.get(),
          stream,
          pRequest));

  auto res = ctran::AllToAllPExec(
      sendBuf, /* count */ maxRecvCount / numRanks + 1, pRequest);
  ASSERT_EQ(res, commInvalidArgument);

  releaseDataBuf(sendBuf, sendHdl);
  for (int idx = 0; idx < 2; idx++) {
    releaseDataBuf(doubleRecvbuffs[idx], recvHdls[idx]);
  }
}

TEST_F(ctranAllToAllPTest, InvalidPreq) {
  auto request = std::make_unique<CtranPersistentRequest>(
      CtranPersistentRequest::Type::ALLGATHER_P,
      comm->ctranComm_.get(),
      stream);
  ASSERT_EQ(
      ctran::AllToAllPExec(nullptr, 0, request.get()), commInvalidArgument);
}

// Tests for PerfConfig and hints
inline std::string getTestName(
    const testing::TestParamInfo<ctranAllToAllPTestParam::ParamType>& info) {
  return "lowlatencyconfig_" + std::to_string(std::get<0>(info.param)) +
      "_skipctrlmsg" + std::to_string(std::get<1>(info.param)) +
      "_enablefastput_" + std::to_string(std::get<2>(info.param));
}

INSTANTIATE_TEST_SUITE_P(
    ctranAllToAllPTest,
    ctranAllToAllPTestParam,
    ::testing::Combine(
        testing::Values(true, false),
        testing::Values(true, false),
        testing::Values(true, false)),
    getTestName);

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
