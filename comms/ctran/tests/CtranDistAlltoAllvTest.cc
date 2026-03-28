// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <folly/logging/xlog.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <thread>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

class ctranAllToAllvTest : public ctran::CtranDistTestFixture,
                           public CtranBaseTest {
 public:
  ctranAllToAllvTest() = default;

  void generateDistRandomExpValue() {
    if (globalRank == 0) {
      expectedVal = rand();
    }
    oobBroadcast(&expectedVal, 1, 0);
  }

  void generateFixedCountsDisps(size_t count) {
    // each send/recv are with the same count and displacement
    int stride = count * 2;
    sendTotalCount = stride * numRanks;
    recvTotalCount = stride * numRanks;
    for (int i = 0; i < numRanks; ++i) {
      sendCounts[i] = count;
      sendDisps[i] = stride * i;
      recvCounts[i] = count;
      recvDisps[i] = stride * i;
    }
  }

  void generateDistRandomCountsDisps() {
    // assign random send count for each peer
    srand(time(NULL) + globalRank);

    sendTotalCount = 0;
    for (int i = 0; i < numRanks; ++i) {
      sendCounts[i] = (rand() % 10) * getpagesize(); // always page aligned size
      sendDisps[i] = sendTotalCount;
      sendTotalCount += sendCounts[i];
    }

    // Exchange send counts to derive recv counts using oobAllGather.
    // For each destination rank, gather what each source rank sends to it.
    recvTotalCount = 0;
    for (int dest = 0; dest < numRanks; ++dest) {
      std::vector<size_t> countsForDest(numRanks, 0);
      countsForDest[globalRank] = sendCounts[dest];
      oobAllGather(countsForDest);
      if (dest == globalRank) {
        for (int src = 0; src < numRanks; ++src) {
          recvCounts[src] = countsForDest[src];
          recvDisps[src] = recvTotalCount;
          recvTotalCount += recvCounts[src];
        }
      }
    }
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

  void SetUp() override {
    ctran::CtranDistTestFixture::SetUp();
    ctranComm = makeCtranComm();
    if (!ctranAllToAllvSupport(ctranComm.get())) {
      GTEST_SKIP() << "Skip the test because ctranAllToAllv is not supported";
    }

    sendBuf = nullptr;
    recvBuf = nullptr;
    sendCounts.resize(numRanks, 0);
    recvCounts.resize(numRanks, 0);
    sendDisps.resize(numRanks, 0);
    recvDisps.resize(numRanks, 0);
  }

  void TearDown() override {
    ctran::CtranDistTestFixture::TearDown();
  }

  void run() {
    // Assign different value for each send chunk
    for (int i = 0; i < numRanks; ++i) {
      assignChunkValue<int>(
          sendBuf + sendDisps[i],
          sendCounts[i],
          expectedVal + globalRank * 100 + i + 1);
    }

    ASSERT_TRUE(
        meta::comms::colltrace::testOnlyClearCollTraceRecords(ctranComm.get()));

    // Run communication
    for (int x = 0; x < 1; x++) {
      auto res = ctranAllToAllv(
          sendBuf,
          sendCounts.data(),
          sendDisps.data(),
          recvBuf,
          recvCounts.data(),
          recvDisps.data(),
          commInt,
          ctranComm.get(),
          testStream);
      ASSERT_EQ(res, commSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(testStream));

    // Check each received chunk
    for (int i = 0; i < numRanks; ++i) {
      int errs = checkChunkValue<int>(
          recvBuf + recvDisps[i],
          recvCounts[i],
          expectedVal + i * 100 + globalRank + 1);
      EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << i
                         << " at " << recvBuf + recvDisps[i] << " with " << errs
                         << " errors";
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Verify colltrace records the AllToAllv operation
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_NE(ctranComm->colltraceNew_, nullptr);
    auto dumpMap = ctran::dumpCollTrace(ctranComm.get());
    EXPECT_NE(dumpMap["CT_pastColls"], "[]");
    EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
    EXPECT_EQ(dumpMap["CT_currentColl"], "null");
    auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
    ASSERT_GE(pastCollsJson.size(), 1);
    for (const auto& coll : pastCollsJson) {
      EXPECT_EQ(coll["opName"].asString(), "AllToAllv");
      EXPECT_THAT(
          coll["algoName"].asString(),
          testing::HasSubstr(allToAllvAlgoName(NCCL_ALLTOALLV_ALGO::ctran)));
    }

    size_t sendCount = std::accumulate(sendCounts.begin(), sendCounts.end(), 0);
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
  int* recvBuf{nullptr};
  std::vector<size_t> sendCounts;
  std::vector<size_t> recvCounts;
  std::vector<size_t> sendDisps;
  std::vector<size_t> recvDisps;
  size_t sendTotalCount{0};
  size_t recvTotalCount{0};
  int expectedVal{0};
};

class ctranAllToAllvTestParam : public ctranAllToAllvTest,
                                public ::testing::WithParamInterface<bool> {};

TEST_P(ctranAllToAllvTestParam, AllToAllv) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), true);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), true);

  run();

  releaseDataBuf(sendBuf, sendTotalCount * sizeof(int), true);
  releaseDataBuf(recvBuf, recvTotalCount * sizeof(int), true);
}

TEST_P(ctranAllToAllvTestParam, AllToAll) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);

  generateFixedCountsDisps(1024 * 1024UL);
  generateDistRandomExpValue();

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), true);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), true);

  run();

  releaseDataBuf(sendBuf, sendTotalCount * sizeof(int), true);
  releaseDataBuf(recvBuf, recvTotalCount * sizeof(int), true);
}

TEST_P(ctranAllToAllvTestParam, ZeroByteAllToAllv) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);

  generateFixedCountsDisps(0);

  // reassign non-zero total buffer sizes
  sendTotalCount = 1048576;
  recvTotalCount = 1048576;

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), true);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), true);

  // Reset buffers' value
  assignChunkValue(sendBuf, sendTotalCount, globalRank);
  assignChunkValue(recvBuf, recvTotalCount, -1);

  run();

  // Check receive buffer is not updated
  int errs = checkChunkValue<int>(recvBuf, recvTotalCount, -1);
  EXPECT_EQ(errs, 0) << "rank " << globalRank
                     << " checked receive buffer (expect no update) with "
                     << errs << " errors";

  releaseDataBuf(sendBuf, sendTotalCount * sizeof(int), true);
  releaseDataBuf(recvBuf, recvTotalCount * sizeof(int), true);
}

// Do not support all to all v with different memory allocs
// AllToAllv only used in alltoall_single which has contigous tensor
// meaning memory is not across cudaMallocs.
TEST_P(ctranAllToAllvTestParam, DISABLED_AllToAllvMultiBufs) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);

  std::vector<int*> sendBufs(numRanks, nullptr), recvBufs(numRanks, nullptr);
  std::vector<size_t> sendBufSizes(numRanks, 0), recvBufSizes(numRanks, 0);

  ASSERT_NE(nullptr, ctranComm.get());
  ASSERT_NE(nullptr, ctranComm->ctran_);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  // Allocate different buffer for each send/recv chunk, and re-generate
  // displacement as offset to first buffer
  for (int i = 0; i < numRanks; ++i) {
    sendBufSizes[i] = sendCounts[i] * sizeof(int);
    sendBufs[i] = (int*)createDataBuf(sendBufSizes[i], true);
    sendDisps[i] = (i == 0) ? 0 : sendBufs[i] - sendBufs[0];
    recvBufSizes[i] = recvCounts[i] * sizeof(int);
    recvBufs[i] = (int*)createDataBuf(recvBufSizes[i], true);
    recvDisps[i] = (i == 0) ? 0 : recvBufs[i] - recvBufs[0];
  }

  sendBuf = sendBufs[0];
  recvBuf = recvBufs[0];
  run();

  for (int i = 0; i < numRanks; ++i) {
    releaseDataBuf(sendBufs[i], sendBufSizes[i], true);
    releaseDataBuf(recvBufs[i], recvBufSizes[i], true);
  }
}

TEST_P(ctranAllToAllvTestParam, AllToAllvDynamicRegister) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);

  ASSERT_NE(nullptr, ctranComm.get());
  ASSERT_NE(nullptr, ctranComm->ctran_);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  // Skip registration as for dynamic registration test
  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), false);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), false);

  run();

  releaseDataBuf(sendBuf, sendTotalCount * sizeof(int), false);
  releaseDataBuf(recvBuf, recvTotalCount * sizeof(int), false);
}

// Tests for PerfConfig
INSTANTIATE_TEST_SUITE_P(
    ctranAllToAllvTest,
    ctranAllToAllvTestParam,
    ::testing::Values(true, false),
    [&](const testing::TestParamInfo<ctranAllToAllvTestParam::ParamType>&
            info) {
      if (info.param) {
        return "low_latency_perfconfig";
      } else {
        return "default_perfconfig";
      }
    });

class ctranAllToAllvIbTest : public ctranAllToAllvTest {
 public:
  ctranAllToAllvIbTest() = default;

  void SetUp() override {
    setenv("NCCL_CTRAN_IB_QP_CONFIG_ALGO", "alltoall:131072,1,dqplb,8,192", 1);
    ncclCvarInit();
    ctranAllToAllvTest::SetUp();
  }
};

TEST_F(ctranAllToAllvIbTest, AllToAllvIbconfig) {
  generateDistRandomCountsDisps();
  generateDistRandomExpValue();
  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), true);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), true);
  run();

  if (this->globalRank == 0) {
    if (ctranComm->ctran_->algo == nullptr) {
      XLOGF(INFO, "No ctran algo found, skip test");
    } else {
      CtranIbConfig* ctranIbConfigPtr =
          ctranComm->ctran_->algo->getCollToVcConfig(CollType::ALLTOALL);
      if (ctranIbConfigPtr != nullptr) {
        EXPECT_EQ(ctranIbConfigPtr->qpScalingTh, 131072);
        EXPECT_EQ(ctranIbConfigPtr->numQps, 1);
        EXPECT_EQ(ctranIbConfigPtr->vcMode, NCCL_CTRAN_IB_VC_MODE::dqplb);
        EXPECT_EQ(ctranIbConfigPtr->qpMsgs, 8);

      } else {
        GTEST_SKIP() << "Alltoall IB config override not found, skip test";
      }
    }
  }
  releaseDataBuf(sendBuf, sendTotalCount * sizeof(int), true);
  releaseDataBuf(recvBuf, recvTotalCount * sizeof(int), true);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
