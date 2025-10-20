// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/colltrace/CollTrace.h"

#ifdef ENABLE_META_COMPRESSION

#include "comms/ctran/algos/AllToAll/compressed/CompressedAllToAllv.h"

class ctranCompAllToAllvTest : public CtranDistBaseTest {
 public:
  ctranCompAllToAllvTest() = default;

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
    CtranDistBaseTest::SetUp();
    comm = commWorld;
    if (!ctranCompressedAllToAllvSupport(comm->ctranComm_.get())) {
      GTEST_SKIP()
          << "Skip the test because ctranCompAllToAllv is not supported";
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

  void run(IbImplType ibImplType) {
    // Assign different value for each send chunk
    for (int i = 0; i < numRanks; ++i) {
      assignChunkValue<int>(
          sendBuf + sendDisps[i],
          sendCounts[i],
          expectedVal + globalRank * 100 + i + 1);
    }

    comm->ctranComm_->collTrace_->resetPastColls();

    // Run communication
    for (int x = 0; x < 1; x++) {
      commResult_t res;
      if (ibImplType == IbImplType::IbExchange) {
        res = ctranCompressedAllToAllv(
            sendBuf,
            sendCounts.data(),
            sendDisps.data(),
            recvBuf,
            recvCounts.data(),
            recvDisps.data(),
            commInt,
            comm->ctranComm_.get(),
            stream);
      } else if (ibImplType == IbImplType::Bootstrap) {
        res = ctranBootstrapCompressedAllToAllv(
            sendBuf,
            sendCounts.data(),
            sendDisps.data(),
            recvBuf,
            recvCounts.data(),
            recvDisps.data(),
            commInt,
            comm->ctranComm_.get(),
            stream);
      }
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

    // CollTrace is updated by a separate thread, need wait for it to finish to
    // avoid flaky test
    comm->ctranComm_->collTrace_->waitForWorkerFinishQueue();

    auto dump = comm->ctranComm_->collTrace_->dump();
    EXPECT_EQ(dump.pastColls.size(), 1);

    for (auto& coll : dump.pastColls) {
      EXPECT_EQ(coll.opName, "AllToAllV");
      // Count should be nullOpt for AllToAllV at the moment
      EXPECT_THAT(coll.count, ::testing::Eq(std::nullopt));
      EXPECT_EQ(
          coll.dataType, commUint8); // for compressed op, we always use bytes

      if (ibImplType == IbImplType::IbExchange) {
        EXPECT_EQ(
            coll.algoName, allToAllvAlgoName(NCCL_ALLTOALLV_ALGO::compCtran));
      } else if (ibImplType == IbImplType::Bootstrap) {
        EXPECT_EQ(
            coll.algoName, allToAllvAlgoName(NCCL_ALLTOALLV_ALGO::bsCompCtran));
      }
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

class ctranCompAllToAllvTestParam : public ctranCompAllToAllvTest,
                                    public ::testing::WithParamInterface<bool> {
};

TEST_P(ctranCompAllToAllvTestParam, CompAllToAllv) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::compCtran);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  run(IbImplType::IbExchange);

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_P(ctranCompAllToAllvTestParam, BsCompAllToAllv) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::bsCompCtran);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  run(IbImplType::Bootstrap);

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_P(ctranCompAllToAllvTestParam, CompAllToAll) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::compCtran);

  generateFixedCountsDisps(1024 * 1024UL);
  generateDistRandomExpValue();

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  run(IbImplType::IbExchange);

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_P(ctranCompAllToAllvTestParam, BsCompAllToAll) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::bsCompCtran);

  generateFixedCountsDisps(1024 * 1024UL);
  generateDistRandomExpValue();

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  run(IbImplType::Bootstrap);

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_P(ctranCompAllToAllvTestParam, ZeroByteCompAllToAllv) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::compCtran);

  generateFixedCountsDisps(0);

  // reassign non-zero total buffer sizes
  sendTotalCount = 1048576;
  recvTotalCount = 1048576;

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  // Reset buffers' value
  assignChunkValue(sendBuf, sendTotalCount, globalRank);
  assignChunkValue(recvBuf, recvTotalCount, -1);

  run(IbImplType::IbExchange);

  // Check receive buffer is not updated
  int errs = checkChunkValue<int>(recvBuf, recvTotalCount, -1);
  EXPECT_EQ(errs, 0) << "rank " << globalRank
                     << " checked receive buffer (expect no update) with "
                     << errs << " errors";

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_P(ctranCompAllToAllvTestParam, ZeroByteBsCompAllToAllv) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::bsCompCtran);

  generateFixedCountsDisps(0);

  // reassign non-zero total buffer sizes
  sendTotalCount = 1048576;
  recvTotalCount = 1048576;

  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), &sendHdl);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), &recvHdl);

  // Reset buffers' value
  assignChunkValue(sendBuf, sendTotalCount, globalRank);
  assignChunkValue(recvBuf, recvTotalCount, -1);

  run(IbImplType::Bootstrap);

  // Check receive buffer is not updated
  int errs = checkChunkValue<int>(recvBuf, recvTotalCount, -1);
  EXPECT_EQ(errs, 0) << "rank " << globalRank
                     << " checked receive buffer (expect no update) with "
                     << errs << " errors";

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_P(ctranCompAllToAllvTestParam, CompAllToAllvDynamicRegister) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::compCtran);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  // Skip registration as for dynamic registration test
  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), nullptr);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), nullptr);

  run(IbImplType::IbExchange);

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

TEST_P(ctranCompAllToAllvTestParam, BsCompAllToAllvDynamicRegister) {
  const bool enable_lowlatency_config = GetParam();
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, enable_lowlatency_config);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, enable_lowlatency_config);
  EnvRAII env3(NCCL_ALLTOALLV_ALGO, NCCL_ALLTOALLV_ALGO::bsCompCtran);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  generateDistRandomCountsDisps();
  generateDistRandomExpValue();

  // Skip registration as for dynamic registration test
  sendBuf = (int*)createDataBuf(sendTotalCount * sizeof(int), nullptr);
  recvBuf = (int*)createDataBuf(recvTotalCount * sizeof(int), nullptr);

  run(IbImplType::Bootstrap);

  releaseDataBuf(sendBuf, sendHdl);
  releaseDataBuf(recvBuf, recvHdl);
}

// Tests for PerfConfig
INSTANTIATE_TEST_SUITE_P(
    ctranCompAllToAllvTest,
    ctranCompAllToAllvTestParam,
    ::testing::Values(true, false),
    [&](const testing::TestParamInfo<ctranCompAllToAllvTestParam::ParamType>&
            info) {
      if (info.param) {
        return "low_latency_perfconfig";
      } else {
        return "default_perfconfig";
      }
    });
#endif // ENABLE_META_COMPRESSION

// TODO Tests only enabled when -c nccl.enable_compression=True compilation
// flag is passed
int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
