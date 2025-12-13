// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <nccl.h>
#include <stdlib.h>
#include <cstdio>

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "CtranUtUtils.h"
#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAllvDedup/tests/AllToAllvDedupTestBase.h"
#include "comms/testinfra/TestsDistUtils.h"

// #define VERBOSE 0

// Global flag to enable deterministic bucket selection using iteration as seed;
// use it for debugging only
constexpr bool kUseDeterministicBucketSeed = false;

class CtranAllToAllvDedupTest : public CtranDistBaseTest,
                                public AllToAllvDedupTestBase {
 public:
  CtranAllToAllvDedupTest() = default;
  void SetUp() override {
    CtranDistBaseTest::SetUp();

    comm_ = commWorld->ctranComm_.get();
    CUDACHECK_ASSERT(cudaEventCreate(&execStart_));
    CUDACHECK_ASSERT(cudaEventCreate(&execStop_));

    myRank_ = comm_->statex_->rank();
    nRanks_ = comm_->statex_->nRanks();
    setStatex(comm_->statex_.get());

    CUDACHECK_ASSERT(cudaMalloc(&barrierFlag_, sizeof(int)));
  }

  void TearDown() override {
    CUDACHECK_ASSERT(cudaFree(barrierFlag_));
    CUDACHECK_ASSERT(cudaEventDestroy(execStart_));
    CUDACHECK_ASSERT(cudaEventDestroy(execStop_));

    CtranDistBaseTest::TearDown();
  }

  template <typename T>
  void assignSendBuff(std::vector<T>& sendBuffH) {
    const int totalNumSendBlocks = pArgs_.totalNumSendBlocks;
    const int blockCount = pArgs_.blockCount;

    for (int i = 0; i < totalNumSendBlocks; ++i) {
      T val = T(expectedVal_ + globalRank * 10000 + i);
      for (int j = 0; j < blockCount; j++) {
        sendBuffH[i * blockCount + j] = val;
      }
    }
  }

  template <typename T>
  void checkExecResult(
      T* recvBuff,
      const int* recvBlockIds,
      const std::vector<int>& recvIdxH,
      const int iter);

  template <commDataType_t DataType = commInt>
  void run(int numIter, std::vector<int>& allowBuckets, bool skipCheck = false);

  void barrier(cudaStream_t stream) {
    // simple Allreduce as barrier before get data from other ranks
    COMMCHECK_ASSERT(ctranAllReduce(
        barrierFlag_, barrierFlag_, 1, commInt, commSum, comm_, stream));
  }

  std::vector<std::vector<int>> genAllRankIndices(
      const std::vector<int>& allowBuckets,
      int iter = 0);

 protected:
  cudaStream_t stream_{0};
  CtranComm* comm_{nullptr};
  int expectedVal_{0};
  int myRank_;
  int nRanks_;
  int* barrierFlag_ = nullptr;

  cudaEvent_t execStart_, execStop_;
  const int defaultNumIters_ = 20;
};

template <typename T>
void CtranAllToAllvDedupTest::checkExecResult(
    T* recvBuff,
    const int* recvBlockIds,
    const std::vector<int>& recvIdxH,
    const int iter) {
  // Check received value
  size_t sRankOffset = 0;
  const auto totalNumSendBlocks = pArgs_.totalNumSendBlocks;
  for (int e = 0; e < pArgs_.numRecvBuckets; e++) {
    for (int rank = 0; rank < nRanks_; rank++) {
      const auto slotId = e * nRanks_ + rank;
      std::vector<size_t> expRecvBlockIds = getRecvBlockIds(
          &recvIdxH[slotId * totalNumSendBlocks], totalNumSendBlocks);

      if (expRecvBlockIds.size()) {
        std::vector<std::string> obsValStrs;
        std::vector<T> expVals;
        obsValStrs.reserve(expRecvBlockIds.size());
        expVals.reserve(expRecvBlockIds.size());

        std::vector<int> obsRecvBlkIds(expRecvBlockIds.size(), -2);
        CUDACHECK_ASSERT(cudaMemcpy(
            obsRecvBlkIds.data(),
            recvBlockIds + sRankOffset,
            sizeof(int) * expRecvBlockIds.size(),
            cudaMemcpyDefault));

        for (auto b = 0; b < expRecvBlockIds.size(); b++) {
          auto expectedVal =
              T(expectedVal_ + rank * 10000 + expRecvBlockIds[b]);
          expVals.push_back(expectedVal);

          const auto blkOffset = sRankOffset + b;
          const auto rDataOffset = blkOffset * pArgs_.blockCount;

          EXPECT_EQ(obsRecvBlkIds[b], expRecvBlockIds[b]) << fmt::format(
              "rank {} iter {} checked recvBlockIds[bkt {}][sendRank {}][{}/{}] expected blockId {} observed {}",
              myRank_,
              iter,
              e,
              rank,
              b,
              expRecvBlockIds.size(),
              expRecvBlockIds[b],
              obsRecvBlkIds[b]);

          std::vector<std::string> errs;
          checkDevArg<T>(
              recvBuff + rDataOffset, pArgs_.blockCount, expectedVal, errs);
          EXPECT_EQ(errs.size(), 0) << fmt::format(
              "rank {} iter {} checked recvBuff[bkt {}][sendRank {}][{}] blockIdx {} offset {}/{} with {} errors: \n{}",
              myRank_,
              iter,
              e,
              rank,
              b,
              expRecvBlockIds[b],
              blkOffset,
              rDataOffset,
              errs.size(),
              folly::join("\n", errs));
        }

#if defined(VERBOSE) and VERBOSE > 1
        std::cout
            << fmt::format(
                   "TEST rank {} iter {} recvBuff[{}][{}]: {} (expected {}, expRecvBlockIds: {})",
                   myRank_,
                   iter,
                   e,
                   rank,
                   folly::join(", ", obsValStrs),
                   folly::join(",", expVals),
                   folly::join(",", expRecvBlockIds))
            << std::endl;
#endif
      }
      sRankOffset += expRecvBlockIds.size();
    }
  }
}

std::vector<std::vector<int>> CtranAllToAllvDedupTest::genAllRankIndices(
    const std::vector<int>& allowBuckets,
    int iter) {
  const auto numBlocksPerRank =
      pArgs_.totalNumSendBlocks * pArgs_.blockNumRecvBuckets;
  const int myRank = statex_->rank();

  std::vector<int> allRankTmpContig(nRanks_ * numBlocksPerRank);

  // Optionally use iteration as seed for deterministic bucket selection
  std::optional<unsigned int> seed = kUseDeterministicBucketSeed
      ? std::make_optional(static_cast<unsigned int>(iter + myRank))
      : std::nullopt;

  // generate block buckets for my rank
  genRankBlockRecvBuckets(
      allowBuckets, allRankTmpContig.data() + myRank * numBlocksPerRank, seed);

  // allgather block buckets from all ranks
  allGather(allRankTmpContig.data(), numBlocksPerRank * sizeof(int));

  // copy to 2D allRankBlkBkts_ for easier access
  std::vector<std::vector<int>> allRankBlkBkts;
  allRankBlkBkts.resize(nRanks_);
  for (int sendRank = 0; sendRank < nRanks_; sendRank++) {
    auto sendRankTmpContig =
        allRankTmpContig.data() + sendRank * numBlocksPerRank;
    allRankBlkBkts[sendRank] = std::vector<int>(
        sendRankTmpContig, sendRankTmpContig + numBlocksPerRank);
  }
  return allRankBlkBkts;
}

template <commDataType_t DataType>
void CtranAllToAllvDedupTest::run(
    int numIter,
    std::vector<int>& allowBuckets,
    bool skipCheck) {
  using DT = typename CommTypeTraits<DataType>::T;
  size_t dataTypeSize = sizeof(DT);
  meta::comms::Hints hints; // unused

  if (!::ctran::allToAllvDedupSupport(comm_, hints)) {
    GTEST_SKIP() << "Skip test because allToAllvDedupSupport returns false";
  }

  expectedVal_ = 0;
  std::vector<double> execTimes;
  execTimes.reserve(numIter);

  CtranPersistentRequest* request = nullptr;
  COMMCHECK_ASSERT(
      ::ctran::allToAllvDedupInit(
          pArgs_.totalNumSendBlocks,
          pArgs_.blockCount,
          pArgs_.blockNumRecvBuckets,
          pArgs_.numRecvBuckets,
          hints,
          DataType,
          comm_,
          stream_,
          request));
  ASSERT_NE(request, nullptr);

  for (int x = 0; x < numIter; x++) {
    // generate index distribution and allgather from all ranks
    auto allRankBlkBkts = genAllRankIndices(allowBuckets, x);
    setAllRankBlkBkts(allRankBlkBkts);

    // generate index maps and totalNumRecvBlocks_ based on
    // allRankBlkBkts_
    std::vector<int> sendIdxH, fwdIdxH, recvIdxH;
    setupIndices(sendIdxH, fwdIdxH, recvIdxH);

    int *sendIdx = nullptr, *fwdIdx = nullptr, *recvIdx = nullptr;
    allocDevArg<int>(sendIdxH, sendIdx);
    ASSERT_NE(sendIdx, nullptr);

    allocDevArg<int>(fwdIdxH, fwdIdx);
    ASSERT_NE(fwdIdx, nullptr);

    allocDevArg<int>(recvIdxH, recvIdx);
    ASSERT_NE(recvIdx, nullptr);
    CUDACHECK_ASSERT(cudaDeviceSynchronize());

    const auto totalNumRecvBlocks = getTotalNumRecvBlocks();
    // Update expectedVal to use different value per iteration
    expectedVal_ += 1000000;
#ifdef VERBOSE
    logIndices(sendIdxH, fwdIdxH, recvIdxH, x);
    std::cout << "TEST prepare iteration " << x << ": rank " << myRank_
              << " totalNumRecvBlocks: " << totalNumRecvBlocks << std::endl;
    std::cout << "TEST prepare iteration " << x << ": rank " << myRank_
              << " expectedVal base: " << expectedVal_ << std::endl;
#endif

    // allocate and assign values for sendBuff
    DT* sendBuff = nullptr;
    DT* recvBuff = nullptr;
    int* recvBlockIds = nullptr;

    const auto recvCount = totalNumRecvBlocks * pArgs_.blockCount;
    const auto sendCount = pArgs_.totalNumSendBlocks * pArgs_.blockCount;

    std::vector<DT> sendBuffH(sendCount);
    assignSendBuff<DT>(sendBuffH);

    allocDevArg(sendBuffH, sendBuff);
    allocDevArg(std::vector<DT>(recvCount, -1), recvBuff);
    allocDevArg<int>(std::vector<int>(totalNumRecvBlocks, -1), recvBlockIds);

    // execute
    std::cout << "exec on rank " << myRank_ << " iteration " << x << std::endl;

    CUDACHECK_ASSERT(cudaEventRecord(execStart_, stream_));
    COMMCHECK_ASSERT(
        ::ctran::allToAllvDedupExec(
            sendBuff,
            sendIdx,
            fwdIdx,
            recvIdx,
            recvBuff,
            recvBlockIds,
            request));
    CUDACHECK_ASSERT(cudaEventRecord(execStop_, stream_));
    CUDACHECK_ASSERT(cudaEventSynchronize(execStop_));
    CUDACHECK_ASSERT(cudaDeviceSynchronize());

    std::cout << "exec done on rank " << myRank_ << " iteration " << x
              << std::endl;

    float execMs;
    CUDACHECK_ASSERT(cudaEventElapsedTime(&execMs, execStart_, execStop_));
    execTimes.push_back(execMs);

    if (!skipCheck) {
      checkExecResult<DT>(recvBuff, recvBlockIds, recvIdxH, x);
      std::cout << "exec finished check on rank " << myRank_ << " iteration "
                << x << std::endl;
    }

    releaseDevArgs();
  }

  // skip first time to account for warmup
  double execAvg =
      std::accumulate(execTimes.begin() + 1, execTimes.end(), 0.0) /
      execTimes.size();
  double size = dataTypeSize * pArgs_.blockCount * pArgs_.totalNumSendBlocks;
  double sizeGB = size / (1 << 30);
  double execBW = sizeGB / (execAvg / 1e3);
  std::cout
      << fmt::format(
             "rank {}, {} avg exec time: {:.2f} ms ({:.2f} GB/s) : [{:.2f}]",
             myRank_,
             pArgs_.toString(),
             execAvg,
             execBW,
             fmt::join(execTimes, ","))
      << std::endl;

  COMMCHECK_ASSERT(::ctran::allToAllvDedupDestroy(request));
  delete request;
}

TEST_F(CtranAllToAllvDedupTest, InitDestroy) {
  meta::comms::Hints hints; // unused

  const TestPArgsParam pArgs = {16, 8192, 4, 2};
  setPersistArgs(pArgs);

  std::cout << "calling support check with comm_ " << comm_ << std::endl;
  if (!::ctran::allToAllvDedupSupport(comm_, hints)) {
    GTEST_SKIP() << "Skip test because allToAllvDedupSupport returns false";
  }

  auto usedBytesBase =
      ncclx::memory::memCacheAllocator::getInstance()->getUsedMem();

  // Multiple times init / destory to ensure no memory leak nor race condition
  const int myRank = comm_->statex_->rank();
  const auto numIters = defaultNumIters_;
  for (int x = 0; x < numIters; x++) {
    if (myRank == 0) {
      std::cout << " InitDestroy starts iter " << x << std::endl;
    }
    CtranPersistentRequest* request = nullptr;
    auto numUsedSegsBeforeInit =
        ncclx::memory::memCacheAllocator::getInstance()->getNumUsedReg();

    COMMCHECK_ASSERT(
        ::ctran::allToAllvDedupInit(
            pArgs_.totalNumSendBlocks,
            pArgs_.blockCount,
            pArgs_.blockNumRecvBuckets,
            pArgs_.numRecvBuckets,
            hints,
            commInt,
            comm_,
            stream_,
            request));

    auto numUsedSegsAfterInit =
        ncclx::memory::memCacheAllocator::getInstance()->getNumUsedReg();
    ASSERT_NE(request, nullptr);

    // memory pool may not release the memory after dedup destroy, thus get
    // delta based on usage before first dedup init
    auto usedBytes =
        ncclx::memory::memCacheAllocator::getInstance()->getUsedMem() -
        usedBytesBase;

    COMMCHECK_ASSERT(::ctran::allToAllvDedupDestroy(request));
    delete request;
    auto numUsedSegsAfterDestroy =
        ncclx::memory::memCacheAllocator::getInstance()->getNumUsedReg();

    // Track memory usage from memory pool
    // - After init, expect increased used segments
    EXPECT_LT(numUsedSegsBeforeInit, numUsedSegsAfterInit);
    // - After destory, expect used segments are released
    EXPECT_EQ(numUsedSegsBeforeInit, numUsedSegsAfterDestroy);

    if (myRank == 0) {
      std::cout << "InitDestroy finished iter " << x << ", used segments "
                << numUsedSegsAfterInit - numUsedSegsBeforeInit
                << " total bytes " << usedBytes << std::endl;
    }
  }
}

class CtranTestAllToAllvDedupParamFixture
    : public CtranAllToAllvDedupTest,
      public ::testing::WithParamInterface<
          std::
              tuple<TestPArgsParam, TestTmpChunkParam, TestExecWgParam, bool>> {
};

TEST_P(CtranTestAllToAllvDedupParamFixture, Basic) {
  auto& [pArgsParam, tmpChkParam, execWgParam, skipBucket] = GetParam();

  SET_TMPCHK_ENVRAII(tmpChkParam);
  SET_EXEC_WG_ENVRAII(execWgParam);

  const auto nRanks = comm_->statex_->nRanks();
  std::vector<int> allowBuckets =
      genAllowBuckets(nRanks * pArgsParam.numRecvBuckets, {});

  // Exclude the last one if skipBucket is true
  if (skipBucket) {
    allowBuckets.pop_back();
  }

  const int numAllowedBuckets = static_cast<int>(allowBuckets.size());
  setPersistArgs(pArgsParam, numAllowedBuckets);

  run<commInt32>(defaultNumIters_, allowBuckets);
}

TEST_F(CtranAllToAllvDedupTest, TracingExec) {
  const TestPArgsParam pArgsParam{8192, 8192, 2, 2};
  const TestTmpChunkParam tmpChkParam{4, 8};
  const TestExecWgParam execWgParam{1, 4, 16, 1, 8, 8, 8};

  EnvRAII<bool> envTraceLogger(NCCL_CTRAN_ENABLE_TRACE_LOGGER, true);
  SET_TMPCHK_ENVRAII(tmpChkParam);
  SET_EXEC_WG_ENVRAII(execWgParam);

  const auto nRanks = comm_->statex_->nRanks();
  std::vector<int> allowBuckets =
      genAllowBuckets(nRanks * pArgsParam.numRecvBuckets, {});

  setPersistArgs(pArgsParam);

  run<commInt32>(defaultNumIters_, allowBuckets);
}

TEST_F(CtranAllToAllvDedupTest, TracingExecSmall) {
  const TestPArgsParam pArgsParam{2, 8192, 2, 2};
  const TestTmpChunkParam tmpChkParam{4, 8};
  const TestExecWgParam execWgParam{1, 4, 4, 1, 4, 1, 1};

  EnvRAII<bool> envTraceLogger(NCCL_CTRAN_ENABLE_TRACE_LOGGER, true);
  SET_TMPCHK_ENVRAII(tmpChkParam);
  SET_EXEC_WG_ENVRAII(execWgParam);

  const auto nRanks = comm_->statex_->nRanks();
  if (nRanks != 4) {
    GTEST_SKIP() << "Skip test because special 4 rank test";
  }

  // exclude buckets 2 and 4
  std::vector<int> allowBuckets =
      genAllowBuckets(nRanks * pArgsParam.numRecvBuckets, {2, 4});

  setPersistArgs(pArgsParam);

  run<commInt32>(defaultNumIters_, allowBuckets);
}

TEST_F(CtranAllToAllvDedupTest, SmallChunkSize) {
  const TestPArgsParam pArgsParam{32, 8192, 2, 1};
  const TestTmpChunkParam tmpChkParam{1, 64 /* KB */};
  const TestExecWgParam execWgParam{1, 8, 1, 8, 1, 8, 8};

  SET_TMPCHK_ENVRAII(tmpChkParam);
  SET_EXEC_WG_ENVRAII(execWgParam);

  const auto nLocalRanks = comm_->statex_->nLocalRanks();
  const auto myNode = comm_->statex_->node();
  const auto nRanks = comm_->statex_->nRanks();

  const auto numBucketsPerNode = nLocalRanks * pArgsParam.numRecvBuckets;

  // Build allowBuckets to include only inter-node buckets (exclude local node
  // buckets)
  std::unordered_set<int> intraNodeBuckets;
  for (int b = 0; b < numBucketsPerNode; b++) {
    intraNodeBuckets.insert(numBucketsPerNode * myNode + b);
  }

  std::vector<int> allowBuckets =
      genAllowBuckets(nRanks * pArgsParam.numRecvBuckets, intraNodeBuckets);

  setPersistArgs(pArgsParam);
  run<commInt32>(1, allowBuckets);
}

TEST_F(CtranAllToAllvDedupTest, InvalidExecBuffs) {
  meta::comms::Hints hints; // unused

  pArgs_.totalNumSendBlocks = 16;
  pArgs_.blockCount = 8192;
  pArgs_.blockNumRecvBuckets = 4;
  pArgs_.numRecvBuckets = 2;

  std::cout << "calling support check with comm_ " << comm_ << std::endl;
  if (!::ctran::allToAllvDedupSupport(comm_, hints)) {
    GTEST_SKIP() << "Skip test because allToAllvDedupSupport returns false";
  }

  CtranPersistentRequest* request = nullptr;
  COMMCHECK_ASSERT(
      ::ctran::allToAllvDedupInit(
          pArgs_.totalNumSendBlocks,
          pArgs_.blockCount,
          pArgs_.blockNumRecvBuckets,
          pArgs_.numRecvBuckets,
          hints,
          commInt,
          comm_,
          stream_,
          request));
  ASSERT_NE(request, nullptr);

  // generate global index distribution
  const auto nRanks = comm_->statex_->nRanks();
  std::vector<int> allowBuckets =
      genAllowBuckets(nRanks * pArgs_.numRecvBuckets, {});
  setAllRankBlkBkts(genAllRankIndices(allowBuckets));

  // generate index maps and totalNumRecvBlocks_ based on
  // allRankBlkBkts_
  std::vector<int> sendIdxH, fwdIdxH, recvIdxH;
  setupIndices(sendIdxH, fwdIdxH, recvIdxH);

  int *sendIdx = nullptr, *fwdIdx = nullptr, *recvIdx = nullptr;
  allocDevArg(sendIdxH, sendIdx);
  ASSERT_NE(sendIdx, nullptr);

  allocDevArg(fwdIdxH, fwdIdx);
  ASSERT_NE(fwdIdx, nullptr);

  allocDevArg(recvIdxH, recvIdx);
  ASSERT_NE(recvIdx, nullptr);
  CUDACHECK_ASSERT(cudaDeviceSynchronize());

  int* dataBuff = nullptr;
  int* recvBlockIds = nullptr;
  const auto totalNumRecvBlocks = getTotalNumRecvBlocks();

  const auto sendCount = pArgs_.totalNumSendBlocks * pArgs_.blockCount;
  allocDevArg(std::vector<int>(sendCount, -1), dataBuff);
  allocDevArg(std::vector<int>(totalNumRecvBlocks, -1), recvBlockIds);

  // Invalid sendBuff
  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          nullptr, sendIdx, fwdIdx, recvIdx, dataBuff, recvBlockIds, request),
      commInvalidArgument);

  //  other invalid argumetns
  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          dataBuff, nullptr, fwdIdx, recvIdx, dataBuff, recvBlockIds, request),
      commInvalidArgument);

  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          dataBuff, sendIdx, nullptr, recvIdx, dataBuff, recvBlockIds, request),
      commInvalidArgument);

  ASSERT_EQ(
      ::ctran::allToAllvDedupExec(
          dataBuff, sendIdx, fwdIdx, nullptr, dataBuff, recvBlockIds, request),
      commInvalidArgument);

  COMMCHECK_ASSERT(::ctran::allToAllvDedupDestroy(request));
  delete request;
  releaseDevArgs();
}

TEST_F(CtranAllToAllvDedupTest, InvalidEnvConfig) {
  meta::comms::Hints hints; // unused

  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, 16);
  EnvRAII<int> envFwdW(NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS, 16);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, 16);

  pArgs_.totalNumSendBlocks = 16;
  pArgs_.blockCount = 16;
  pArgs_.blockNumRecvBuckets = 4;
  pArgs_.numRecvBuckets = 2;

  CtranPersistentRequest* request = nullptr;
  ASSERT_EQ(
      ::ctran::allToAllvDedupInit(
          pArgs_.totalNumSendBlocks,
          pArgs_.blockCount,
          pArgs_.blockNumRecvBuckets,
          pArgs_.numRecvBuckets,
          hints,
          commInt,
          comm_,
          stream_,
          request),
      commInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranTestAllToAllvDedupParamFixture,
    ::testing::Values(
        std::make_tuple(
            TestPArgsParam{4000, 8192, 4, 1},
            TestTmpChunkParam{4, 4},
            TestExecWgParam{1, 1, 1, 1, 1, 1, 1},
            false),
        std::make_tuple(
            TestPArgsParam{8, 2048, 2, 1},
            TestTmpChunkParam{4, 4},
            TestExecWgParam{1, 1, 1, 1, 1, 1, 1},
            false),
        std::make_tuple(
            TestPArgsParam{4000, 8192, 4, 1},
            TestTmpChunkParam{1, 8},
            TestExecWgParam{2, 1, 4, 2, 1, 4, 4},
            false),
        std::make_tuple(
            TestPArgsParam{4000, 8192, 4, 1},
            TestTmpChunkParam{2, 8},
            TestExecWgParam{2, 2, 4, 2, 1, 4, 4},
            false),
        std::make_tuple(
            TestPArgsParam{8192, 8192, 4, 1},
            TestTmpChunkParam{4, 4},
            TestExecWgParam{2, 2, 4, 2, 2, 4, 8},
            false),
        std::make_tuple(
            TestPArgsParam{8, 2048, 2, 2},
            TestTmpChunkParam{4, 4},
            TestExecWgParam{1, 1, 1, 1, 1, 1, 1},
            false),
        std::make_tuple(
            TestPArgsParam{4000, 8192, 4, 2},
            TestTmpChunkParam{4, 4},
            TestExecWgParam{2, 1, 4, 2, 1, 4, 4},
            false),
        std::make_tuple(
            TestPArgsParam{8, 2048, 2, 4},
            TestTmpChunkParam{4, 4},
            TestExecWgParam{1, 1, 1, 1, 1, 1, 1},
            false),
        // skip last bucket
        std::make_tuple(
            TestPArgsParam{8192, 8192, 4, 1},
            TestTmpChunkParam{4, 4},
            TestExecWgParam{1, 8, 8, 1, 8, 8, 8},
            true),
        std::make_tuple(
            TestPArgsParam{8192, 8192, 4, 1},
            TestTmpChunkParam{4, 4},
            TestExecWgParam{2, 8, 8, 1, 8, 8, 8},
            true),

        // ModelComparisonTracingExec
        std::make_tuple(
            TestPArgsParam{4096, 2048, 8, 16},
            TestTmpChunkParam{4, 4},
            TestExecWgParam{1, 4, 8, 4, 8, 8, 16},
            false)),
    [&](const testing::TestParamInfo<
        CtranTestAllToAllvDedupParamFixture::ParamType>& info) {
      const auto& pArgsParam = std::get<0>(info.param);
      const auto& tmpChkParam = std::get<1>(info.param);
      const auto& execWgParam = std::get<2>(info.param);
      const auto skipBucket = std::get<3>(info.param);

      return pArgsParam.toTestName() + "_" + tmpChkParam.toTestName() + "_" +
          execWgParam.toTestName() + (skipBucket ? "_skipBucket" : "");
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
