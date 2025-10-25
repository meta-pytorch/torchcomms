// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/tracing/CollTraceWrapper.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/commDump.h"

class ctranAllToAllvTest : public CtranDistBaseTest {
 public:
  ctranAllToAllvTest() = default;

  void generateDistRandomExpValue() {
    if (globalRank == 0) {
      expectedVal = rand();
    }
    MPI_Bcast(&expectedVal, 1, MPI_INT, 0, MPI_COMM_WORLD);
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
    std::vector<MPI_Request> reqs(numRanks * 2, MPI_REQUEST_NULL);

    // assign random send count for each peer
    srand(time(NULL) + globalRank);

    sendTotalCount = 0;
    for (int i = 0; i < numRanks; ++i) {
      sendCounts[i] = (rand() % 10) * getpagesize(); // always page aligned size
      sendDisps[i] = sendTotalCount;
      sendTotalCount += sendCounts[i];
      // exchange send count to receiver side
      MPI_Isend(&sendCounts[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &reqs[i]);
      MPI_Irecv(
          &recvCounts[i],
          1,
          MPI_INT,
          i,
          0,
          MPI_COMM_WORLD,
          &reqs[numRanks + i]);
    }
    MPI_Waitall(numRanks * 2, reqs.data(), MPI_STATUSES_IGNORE);

    // updates recvDisp based on received counts from sender
    recvTotalCount = 0;
    for (int i = 0; i < numRanks; ++i) {
      recvDisps[i] = recvTotalCount;
      recvTotalCount += recvCounts[i];
    }
  }

  void* createDataBuf(size_t nbytes, void** handle) {
    void* buf = nullptr;
    // Allocate data buffer, and assign different value for each send chunk
    CUDACHECK_TEST(cudaMalloc(&buf, nbytes));
    if (buf && handle) {
      NCCLCHECK_TEST(ncclCommRegister(comm, buf, nbytes, handle));
    }
    return buf;
  }

  void releaseDataBuf(void* buf, void* handle) {
    if (handle) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, handle));
    }
    CUDACHECK_TEST(cudaFree(buf));
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
    if (!ctranAllToAllvSupport(comm->ctranComm_.get())) {
      GTEST_SKIP() << "Skip the test because ctranAllToAllv is not supported";
    }

    // Allocate enough space for arguments, value assignment set in each test
    sendBuf = nullptr;
    recvBuf = nullptr;
    sendHdl = nullptr;
    recvHdl = nullptr;
    sendCounts.resize(numRanks, 0);
    recvCounts.resize(numRanks, 0);
    sendDisps.resize(numRanks, 0);
    recvDisps.resize(numRanks, 0);
  }

  void TearDown() override {
    CtranDistBaseTest::TearDown();
  }

  void run() {
    // Assign different value for each send chunk
    for (int i = 0; i < numRanks; ++i) {
      assignChunkValue<int>(
          sendBuf + sendDisps[i],
          sendCounts[i],
          expectedVal + globalRank * 100 + i + 1);
    }

    ASSERT_TRUE(meta::comms::colltrace::testOnlyClearCollTraceRecords(
        comm->ctranComm_.get()));

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
          comm->ctranComm_.get(),
          stream);
      ASSERT_EQ(res, commSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

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
    // Sleep for a while to make sure all the colls are finished
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(comm->newCollTrace != nullptr);
    auto dumpMap = meta::comms::ncclx::dumpNewCollTrace(*comm->newCollTrace);

    EXPECT_NE(dumpMap["CT_pastColls"], "[]");
    EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
    EXPECT_EQ(dumpMap["CT_currentColl"], "null");

    auto pastCollsJson = folly::parseJson(dumpMap["CT_pastColls"]);
    EXPECT_EQ(pastCollsJson.size(), 1);

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
      if (comm->ctranComm_->statex_->nNodes() == 1) {
        excludedBackends.push_back(CtranMapperBackend::IB);
      }
      verifyBackendsUsed(
          comm->ctranComm_->ctran_.get(),
          comm->ctranComm_->statex_.get(),
          kMemCudaMalloc,
          excludedBackends);
    }
  }

 protected:
  ncclComm_t comm{nullptr};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  std::vector<size_t> sendCounts;
  std::vector<size_t> recvCounts;
  std::vector<size_t> sendDisps;
  std::vector<size_t> recvDisps;
  size_t sendTotalCount{0};
  size_t recvTotalCount{0};
  void* sendHdl{nullptr};
  void* recvHdl{nullptr};
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

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  run();

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_P(ctranAllToAllvTestParam, AllToAll) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);

  generateFixedCountsDisps(1024 * 1024UL);
  generateDistRandomExpValue();

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  run();

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_P(ctranAllToAllvTestParam, ZeroByteAllToAllv) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);

  generateFixedCountsDisps(0);

  // reassign non-zero total buffer sizes
  sendTotalCount = 1048576;
  recvTotalCount = 1048576;

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  // Reset buffers' value
  assignChunkValue(sendBuf, sendTotalCount, globalRank);
  assignChunkValue(recvBuf, recvTotalCount, -1);

  run();

  // Check receive buffer is not updated
  int errs = checkChunkValue<int>(recvBuf, recvTotalCount, -1);
  EXPECT_EQ(errs, 0) << "rank " << globalRank
                     << " checked receive buffer (expect no update) with "
                     << errs << " errors";

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

// Do not support all to all v with different memory allocs
// AllToAllv only used in alltoall_single which has contigous tensor
// meaning memory is not across cudaMallocs.
TEST_P(ctranAllToAllvTestParam, DISABLED_AllToAllvMultiBufs) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);

  std::vector<int*> sendBufs(numRanks, nullptr), recvBufs(numRanks, nullptr);
  std::vector<void*> sendHdls(numRanks, nullptr), recvHdls(numRanks, nullptr);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  // Allocate different buffer for each send/recv chunk, and re-generate
  // displacement as offset to first buffer
  for (int i = 0; i < numRanks; ++i) {
    sendBufs[i] =
        (int*)createDataBuf(sendCounts[i] * sizeof(int), &sendHdls[i]);
    sendDisps[i] = (i == 0) ? 0 : sendBufs[i] - sendBufs[0];
    recvBufs[i] =
        (int*)createDataBuf(recvCounts[i] * sizeof(int), &recvHdls[i]);
    recvDisps[i] = (i == 0) ? 0 : recvBufs[i] - recvBufs[0];
  }

  sendBuf = sendBufs[0];
  recvBuf = recvBufs[0];
  run();

  for (int i = 0; i < numRanks; ++i) {
    releaseDataBuf(sendBufs[i], sendHdls[i]);
    releaseDataBuf(recvBufs[i], recvHdls[i]);
  }
}

TEST_P(ctranAllToAllvTestParam, AllToAllvDynamicRegister) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  // Skip registration as for dynamic registration test
  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), nullptr);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), nullptr);

  run();

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
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
  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);
  run();

  if (this->globalRank == 0) {
    if (comm->ctranComm_->ctran_->algo == nullptr) {
      XLOGF(INFO, "No ctran algo found, skip test");
    } else {
      CtranIbConfig* ctranIbConfigPtr =
          comm->ctranComm_->ctran_->algo->getCollToVcConfig(CollType::ALLTOALL);
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
  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
