// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <thread>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

class ctranAllToAllPTest : public ctran::CtranDistTestFixture,
                           public CtranBaseTest {
 public:
  ctranAllToAllPTest() = default;

  void generateDistRandomExpValue() {
    if (globalRank == 0) {
      expectedVal = rand();
    }
    oobBroadcast(&expectedVal, 1, 0);
  }

  void generateDistRandomCount(bool small_msg = false) {
    if (globalRank == 0) {
      if (small_msg) {
        count = std::min(8192, (int)maxRecvCount / numRanks);
      } else {
        count = rand() % (maxRecvCount / numRanks) + 1;
      }
    }
    oobBroadcast(&count, 1, 0);
  }

  void* createDataBuf(size_t nbytes, bool doRegister) {
    void* buf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&buf, nbytes));
    if (buf && doRegister) {
      COMMCHECK_TEST(ctran::globalRegisterWithPtr(buf, nbytes));
    }
    return buf;
  }

  void releaseDataBuf(void* buf, size_t nbytes, bool doDeregister) {
    if (doDeregister) {
      COMMCHECK_TEST(ctran::globalDeregisterWithPtr(buf, nbytes));
    }
    CUDACHECK_TEST(cudaFree(buf));
  }

  // Recv buffers must be CCA-cached (via the regcache globalRegister path) so
  // that AllToAllPInit's scoped registration can acquire them; the force-reg
  // (globalRegisterWithPtr) path used for sendbufs does not satisfy that.
  void* createDataBufCached(size_t nbytes) {
    void* buf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&buf, nbytes));
    if (buf) {
      COMMCHECK_TEST(
          ctran::RegCache::getInstance()->globalRegister(buf, nbytes));
    }
    return buf;
  }

  void releaseDataBufCached(void* buf, size_t nbytes) {
    COMMCHECK_TEST(
        ctran::RegCache::getInstance()->globalDeregister(buf, nbytes));
    CUDACHECK_TEST(cudaFree(buf));
  }

  void SetUp() override {
    ctran::CtranDistTestFixture::SetUp();
    ctranComm = makeCtranComm();
    if (!ctran::AllToAllPSupport(ctranComm.get())) {
      GTEST_SKIP() << "Skip the test because ctran::AllToAllP is not supported";
    }

    sendBuf = nullptr;
  }

  void TearDown() override {
    ctran::CtranDistTestFixture::TearDown();
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
        meta::comms::colltrace::testOnlyClearCollTraceRecords(ctranComm.get()));

    // Initialize double persistent requests using double recv buffer allocated.
    std::array<CtranPersistentRequest*, 2> doublePRequests;
    for (int idx = 0; idx < 2; idx++) {
      COMMCHECK_TEST(
          ctran::AllToAllPInit(
              doubleRecvbuffs[idx],
              maxRecvCount,
              hints,
              commInt,
              ctranComm.get(),
              testStream,
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
      CUDACHECK_TEST(cudaStreamSynchronize(testStream));

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

    // Verify colltrace records the AllToAllP operations
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ASSERT_NE(ctranComm->colltraceNew_, nullptr);
    auto dumpMap = ctran::dumpCollTrace(ctranComm.get());
    EXPECT_NE(dumpMap["CT_pastColls"], "[]");
    EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
    EXPECT_EQ(dumpMap["CT_currentColls"], "[]");

    auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
    auto statex = ctranComm->statex_.get();
    // AllToAllPInit now always exchanges the intra-node NVL IPC handles via a
    // submitHost op (mirroring AGP), so init is colltrace-recorded on every
    // topology, including single-node. Two persistent requests are created, so
    // two init ops are recorded regardless of nNodes.
    int numTimesRunInit = 2;
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
    if (ctranComm->statex_->nLocalRanks() == ctranComm->statex_->nRanks()) {
      excludedBackends.push_back(CtranMapperBackend::IB);
    }
    verifyBackendsUsed(
        ctranComm->ctran_.get(),
        ctranComm->statex_.get(),
        kMemCudaMalloc,
        excludedBackends);
  }

 protected:
  cudaStream_t testStream{0};
  std::unique_ptr<CtranComm> ctranComm{nullptr};
  meta::comms::Hints hints;
  int* sendBuf{nullptr};
  std::array<void*, 2> doubleRecvbuffs;
  size_t maxRecvCount{1024 * 1024};
  size_t count{0};
  int expectedVal{0};
  int numTimesRunExec{7};
  size_t bufNbytes{0};
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

  bufNbytes = maxRecvCount * sizeof(int);
  sendBuf = (int*)createDataBuf(bufNbytes, true);
  for (int idx = 0; idx < 2; idx++) {
    doubleRecvbuffs[idx] = createDataBufCached(bufNbytes);
  }
  run();

  releaseDataBuf(sendBuf, bufNbytes, true);
  for (int idx = 0; idx < 2; idx++) {
    releaseDataBufCached(doubleRecvbuffs[idx], bufNbytes);
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

  bufNbytes = maxRecvCount * sizeof(int);
  sendBuf = (int*)createDataBuf(bufNbytes, true);
  for (int idx = 0; idx < 2; idx++) {
    doubleRecvbuffs[idx] = createDataBufCached(bufNbytes);
  }

  CtranPersistentRequest* pRequest;
  COMMCHECK_TEST(
      ctran::AllToAllPInit(
          doubleRecvbuffs[0],
          maxRecvCount,
          hints,
          commInt,
          ctranComm.get(),
          testStream,
          pRequest));

  auto res = ctran::AllToAllPExec(
      sendBuf, /* count */ maxRecvCount / numRanks + 1, pRequest);
  ASSERT_EQ(res, commInvalidArgument);

  ASSERT_EQ(ctran::AllToAllPDestroy(pRequest), commSuccess);
  delete pRequest;

  releaseDataBuf(sendBuf, bufNbytes, true);
  for (int idx = 0; idx < 2; idx++) {
    releaseDataBufCached(doubleRecvbuffs[idx], bufNbytes);
  }
}

TEST_F(ctranAllToAllPTest, InvalidPreq) {
  auto request = std::make_unique<CtranPersistentRequest>(
      CtranPersistentRequest::Type::ALLGATHER_P, ctranComm.get(), testStream);
  ASSERT_EQ(
      ctran::AllToAllPExec(nullptr, 0, request.get()), commInvalidArgument);
}

// Back-to-back / overlapping eager AllToAllP execs with NO
// cudaStreamSynchronize between them. This stresses the sync-only phase-lock
// protecting the comm-shared IB notify counter: the first exec on each
// persistent request does the one-shot IB rkey exchange, while subsequent execs
// perform only the SYNC-only handshake. Queuing execs back-to-back (letting
// them overlap on the stream) mirrors the graph-replay path but for the eager
// entry point, which normalRun never exercises because it syncs after every
// exec. A broken phase-lock would either hang the test or corrupt the final
// result via a late-landing earlier op. skip_ctrl_msg=false only:
// skip_ctrl_msg=true would require the user to double-buffer / avoid
// back-to-back writes to the same recvbuf, which is out of contract here.
TEST_F(ctranAllToAllPTest, BackToBackExecNoSync) {
  setupHints(/*skip_ctrl_msg=*/false);

  bufNbytes = maxRecvCount * sizeof(int);
  sendBuf = (int*)createDataBuf(bufNbytes, true);
  for (int idx = 0; idx < 2; idx++) {
    doubleRecvbuffs[idx] = createDataBufCached(bufNbytes);
  }

  std::array<CtranPersistentRequest*, 2> reqs;
  for (int idx = 0; idx < 2; idx++) {
    COMMCHECK_TEST(
        ctran::AllToAllPInit(
            doubleRecvbuffs[idx],
            maxRecvCount,
            hints,
            commInt,
            ctranComm.get(),
            testStream,
            reqs[idx]));
  }

  // Fix the message size once so verification against each recvbuf's last
  // writer is simple.
  generateDistRandomCount(/*small_msg=*/false);

  // Allocate a distinct send buffer per exec so the host fill of iteration x
  // cannot race a prior exec's device read of a shared sendBuf (there is no
  // inter-exec sync).
  std::vector<int*> sendBufs;
  sendBufs.reserve(numTimesRunExec);
  std::vector<int> execExpectedVal(numTimesRunExec);
  for (int x = 0; x < numTimesRunExec; x++) {
    generateDistRandomExpValue();
    execExpectedVal[x] = expectedVal;
    int* buf = (int*)createDataBuf(bufNbytes, true);
    for (int i = 0; i < numRanks; ++i) {
      assignChunkValue<int>(
          buf + count * i, count, expectedVal + globalRank * 100 + i + 1);
    }
    sendBufs.push_back(buf);
  }

  // Track the expected value of the last exec that targeted each recvbuf; only
  // the last writer is observable since back-to-back execs overwrite the same
  // buffer.
  std::array<int, 2> lastExpectedVal;
  for (int x = 0; x < numTimesRunExec; x++) {
    const int idx = x % 2;
    lastExpectedVal[idx] = execExpectedVal[x];
    ASSERT_EQ(ctran::AllToAllPExec(sendBufs[x], count, reqs[idx]), commSuccess);
  }

  CUDACHECK_TEST(cudaStreamSynchronize(testStream));

  for (int idx = 0; idx < 2; idx++) {
    int* recvbuff = (int*)doubleRecvbuffs[idx];
    for (int i = 0; i < numRanks; ++i) {
      int errs = checkChunkValue<int>(
          recvbuff + count * i,
          count,
          lastExpectedVal[idx] + i * 100 + globalRank + 1);
      EXPECT_EQ(errs, 0) << "rank " << globalRank << " recvbuf " << idx
                         << " checked chunk " << i << " at "
                         << recvbuff + count * i << " with " << errs
                         << " errors";
    }
  }

  for (int idx = 0; idx < 2; idx++) {
    ASSERT_EQ(ctran::AllToAllPDestroy(reqs[idx]), commSuccess);
  }

  CUDACHECK_TEST(cudaDeviceSynchronize());

  for (int x = 0; x < numTimesRunExec; x++) {
    releaseDataBuf(sendBufs[x], bufNbytes, true);
  }
  releaseDataBuf(sendBuf, bufNbytes, true);
  for (int idx = 0; idx < 2; idx++) {
    releaseDataBufCached(doubleRecvbuffs[idx], bufNbytes);
  }
}

TEST_F(ctranAllToAllPTest, UncachedRecvbufFailsCleanly) {
  // Eager A2AP registers recvbuf via the scoped regcache API, which requires
  // the buffer's segment to already be CCA-cached. An uncached recvbuf must be
  // rejected cleanly with commInvalidUsage. Allocate via raw cudaMalloc (NOT
  // the cached helper) so the buffer stays uncached.
  const size_t uncachedMaxRecvCount = 8192 * numRanks;
  const size_t recvRegBytes = uncachedMaxRecvCount * commTypeSize(commInt);
  void* uncachedRecvbuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&uncachedRecvbuf, recvRegBytes));

  CtranPersistentRequest* request = nullptr;
  ASSERT_EQ(
      ctran::AllToAllPInit(
          uncachedRecvbuf,
          uncachedMaxRecvCount,
          hints,
          commInt,
          ctranComm.get(),
          testStream,
          request),
      commInvalidUsage);
  // createPersistentRequest sets *out = nullptr up front, so request stays null
  // when it fails early (here, on the uncached recvbuf) before any allocation.
  ASSERT_EQ(request, nullptr);

  CUDACHECK_TEST(cudaFree(uncachedRecvbuf));
}

// Validates that a persistent request's pooled resources are released safely
// regardless of whether the CtranComm or the persistent request is destroyed
// first. The "comm before preq" sub-case must not hang.
TEST_F(ctranAllToAllPTest, CommDestroyBeforePreqDestroy) {
  const size_t testMaxRecvCount = 8192 * numRanks;
  const size_t testCount = 8192;
  const size_t nbytes = testMaxRecvCount * sizeof(int);

  // Creates a comm + persistent request over a freshly allocated recvbuf, runs
  // exactly one exec, and returns the pieces so the caller controls teardown
  // ordering. Buffers are freed by the caller after the request is destroyed.
  auto makeReqAndExec = [&](std::unique_ptr<CtranComm>& comm,
                            CtranPersistentRequest*& request,
                            int*& sendBufLocal,
                            void*& recvBufLocal) {
    comm = makeCtranComm();
    sendBufLocal = (int*)createDataBuf(nbytes, true);
    recvBufLocal = createDataBufCached(nbytes);

    COMMCHECK_TEST(
        ctran::AllToAllPInit(
            recvBufLocal,
            testMaxRecvCount,
            hints,
            commInt,
            comm.get(),
            testStream,
            request));
    ASSERT_EQ(
        ctran::AllToAllPExec(sendBufLocal, testCount, request), commSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(testStream));
  };

  // Sub-case 1: normal ordering -- destroy the preq (comm alive), then the
  // comm. The comm drain then finds the token already spent (no-op).
  {
    std::unique_ptr<CtranComm> comm;
    CtranPersistentRequest* request = nullptr;
    int* sendBufLocal = nullptr;
    void* recvBufLocal = nullptr;
    makeReqAndExec(comm, request, sendBufLocal, recvBufLocal);

    ASSERT_EQ(ctran::AllToAllPDestroy(request), commSuccess);
    delete request;
    comm.reset();
    releaseDataBuf(sendBufLocal, nbytes, true);
    releaseDataBufCached(recvBufLocal, nbytes);
  }

  // Sub-case 2: destroy the COMM before the preq. The comm drain must release
  // the pooled resources so this returns instead of hanging. The user then
  // only frees the request object (the comm is gone; AllToAllPDestroy is not
  // valid post-comm-destroy).
  {
    std::unique_ptr<CtranComm> comm;
    CtranPersistentRequest* request = nullptr;
    int* sendBufLocal = nullptr;
    void* recvBufLocal = nullptr;
    makeReqAndExec(comm, request, sendBufLocal, recvBufLocal);

    comm.reset();
    delete request;
    releaseDataBuf(sendBufLocal, nbytes, true);
    releaseDataBufCached(recvBufLocal, nbytes);
  }
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
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
