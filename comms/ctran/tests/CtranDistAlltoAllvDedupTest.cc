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
  void run(int numIter, bool overrideBuckets = false, bool skipCheck = false);

  void barrier(cudaStream_t stream) {
    // simple Allreduce as barrier before get data from other ranks
    COMMCHECK_ASSERT(ctranAllReduce(
        barrierFlag_, barrierFlag_, 1, commInt, commSum, comm_, stream));
  }

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

template <commDataType_t DataType>
void CtranAllToAllvDedupTest::run(
    int numIter,
    bool overrideBuckets,
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
    // generate global index distribution
    if (!overrideBuckets) {
      genAllRankIndices(x, {});
    }

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

  setPersistArgs(16, 8192, 4, 2);

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
          std::tuple<int, int, int, int, int, int, int, int, int, int, bool>> {
};

TEST_P(CtranTestAllToAllvDedupParamFixture, Basic) {
  const auto& [totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets, chunkSizeMb, sendG, sendW, fwdW, recvG, recvW, skipBucket] =
      GetParam();
  EnvRAII<int> env1(
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, chunkSizeMb * 1 << 20);
  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, sendG);
  EnvRAII<int> envSendW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP, sendW);
  EnvRAII<int> envFwdW(NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS, fwdW);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, recvG);
  EnvRAII<int> envRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP, recvW);
  EnvRAII<int> envIntraFwdW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_INTRA_FWD_NUM_THREAD_BLOCKS, fwdW);
  EnvRAII<int> envIntraRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_INTRA_RECV_NUM_THREAD_BLOCKS, recvW);

  const auto nRanks = comm_->statex_->nRanks();

  bool overrideBuckets = false;
  std::unordered_set<int> skippedBuckets;
  if (skipBucket) {
    skippedBuckets.insert(nRanks - 1);
  }

  const int actualBlockNumRecvBuckets =
      std::min(nRanks - (int)skippedBuckets.size(), blockNumRecvBuckets);

  setPersistArgs(
      totalNumSendBlocks,
      blockCount,
      actualBlockNumRecvBuckets,
      numRecvBuckets);

  // optionally override bucket assignement to skip one bucket; generate indices
  // after setPersistArgs
  if (skipBucket) {
    genAllRankIndices(0, skippedBuckets);
    overrideBuckets = true;
  }

  run<commInt32>(defaultNumIters_, overrideBuckets);
}

TEST_F(CtranAllToAllvDedupTest, TracingExec) {
  const auto& [totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets, chunkSizeMb, sendG, sendW, fwdW, recvG, recvW] =
      std::make_tuple(8192, 8192, 2, 2, 8 /* MB */, 1, 4, 16, 1, 8);
  EnvRAII<bool> envTraceLogger(NCCL_CTRAN_ENABLE_TRACE_LOGGER, true);
  EnvRAII<int> env1(
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, chunkSizeMb * 1 << 20);
  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, sendG);
  EnvRAII<int> envSendW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP, sendW);
  EnvRAII<int> envFwdW(NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS, fwdW);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, recvG);
  EnvRAII<int> envRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP, recvW);
  // EnvRAII<int> envNumChunks(NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS, 4);

  const auto nRanks = comm_->statex_->nRanks();

  int actualBlockNumRecvBuckets = std::min(nRanks, blockNumRecvBuckets);
  setPersistArgs(
      totalNumSendBlocks,
      blockCount,
      actualBlockNumRecvBuckets,
      numRecvBuckets);

  run<commInt32>(defaultNumIters_);
}

TEST_F(CtranAllToAllvDedupTest, TracingExecSmall) {
  const auto& [totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets, chunkSizeMb, sendG, sendW, fwdW, recvG, recvW, skipBucket] =
      std::make_tuple(2, 8192, 2, 2, 4 /* MB */, 1, 4, 4, 1, 4, false);
  EnvRAII<bool> envTraceLogger(NCCL_CTRAN_ENABLE_TRACE_LOGGER, true);
  EnvRAII<int> env1(
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, chunkSizeMb * 1 << 20);
  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, sendG);
  EnvRAII<int> envSendW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP, sendW);
  EnvRAII<int> envFwdW(NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS, fwdW);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, recvG);
  EnvRAII<int> envRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP, recvW);

  const auto nRanks = comm_->statex_->nRanks();

  if (nRanks != 4) {
    GTEST_SKIP() << "Skip test because special 4 rank test";
  }

  int actualBlockNumRecvBuckets = std::min(nRanks, blockNumRecvBuckets);
  const bool kOverrideBuckets = true;
  allRankBlkBkts_.resize(nRanks);
  allRankBlkBkts_[0] = {1, 5, 5, 6};
  allRankBlkBkts_[1] = {0, 1, 3, 5};
  allRankBlkBkts_[2] = {3, 5, 0, 7};
  allRankBlkBkts_[3] = {5, 6, 5, 6};

  setPersistArgs(
      totalNumSendBlocks,
      blockCount,
      actualBlockNumRecvBuckets,
      numRecvBuckets);
  run<commInt32>(defaultNumIters_, kOverrideBuckets);
}

TEST_F(CtranAllToAllvDedupTest, SmallChunkSize) {
  const auto& [totalNumSendBlocks, blockCount, blockNumRecvBuckets, numRecvBuckets, chunkSizeKb, sendG, sendW, fwdW, recvG, recvW, numChunks] =
      std::make_tuple(32, 8192, 2, 1, 64 /* KB */, 1, 1, 8, 1, 8, 1);
  EnvRAII<int> env1(
      NCCL_CTRAN_ALLTOALLV_DEDUP_CHUNK_SIZE, chunkSizeKb * 1 << 10);
  EnvRAII<int> envSendG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCK_GROUPS, sendG);
  EnvRAII<int> envSendW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_SEND_NUM_THREAD_BLOCKS_PER_GROUP, sendW);
  EnvRAII<int> envFwdW(NCCL_CTRAN_ALLTOALLV_DEDUP_FWD_NUM_THREAD_BLOCKS, fwdW);
  EnvRAII<int> envRecvG(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCK_GROUPS, recvG);
  EnvRAII<int> envRecvW(
      NCCL_CTRAN_ALLTOALLV_DEDUP_RECV_NUM_THREAD_BLOCKS_PER_GROUP, recvW);
  EnvRAII<int> envNumChunks(NCCL_CTRAN_ALLTOALLV_DEDUP_NUM_CHUNKS, numChunks);

  const auto nRanks = comm_->statex_->nRanks();
  const auto nLocalRanks = comm_->statex_->nLocalRanks();
  const auto nNodes = comm_->statex_->nNodes();

  int actualBlockNumRecvBuckets = std::min(nRanks, blockNumRecvBuckets);
  const bool kOverrideBuckets = true;
  allRankBlkBkts_.resize(nRanks);
  const auto numBucketsPerNode = nLocalRanks * numRecvBuckets;

  for (int i = 0; i < nRanks; i++) {
    const auto node = comm_->statex_->node(i);
    std::vector<int> candidates;
    for (int n = 0; n < nNodes; n++) {
      if (n == node) {
        continue;
      }
      for (int j = 0; j < numBucketsPerNode; j++) {
        candidates.push_back(j + numBucketsPerNode * n);
      }
    }

    for (int j = 0; j < totalNumSendBlocks; j++) {
      for (int k = 0; k < blockNumRecvBuckets; k++) {
        allRankBlkBkts_[i].push_back(candidates[(j + k) % candidates.size()]);
      }
    }
  }

  setPersistArgs(
      totalNumSendBlocks,
      blockCount,
      actualBlockNumRecvBuckets,
      numRecvBuckets);
  run<commInt32>(1, kOverrideBuckets);
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
  genAllRankIndices(0, {});

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
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 1, 1, 1, 1, 1, false),
#if 1
        std::make_tuple(8, 2048, 2, 1, 4 /* MB */, 1, 1, 1, 1, 1, false),
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 2, 1, 4, 2, 1, false),
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 2, 2, 4, 2, 1, false),
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 2, 2, 4, 2, 2, false),
        std::make_tuple(8, 2048, 2, 2, 4 /* MB */, 1, 1, 1, 1, 1, false),
        std::make_tuple(8192, 8192, 2, 2, 4 /* MB */, 1, 1, 1, 1, 1, false),
        std::make_tuple(8192, 8192, 4, 2, 4 /* MB */, 2, 1, 4, 2, 1, false),
        std::make_tuple(8, 2048, 2, 4, 4 /* MB */, 1, 1, 1, 1, 1, false),
        // skip last bucket
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 1, 8, 8, 1, 8, true),
        std::make_tuple(8192, 8192, 4, 1, 4 /* MB */, 2, 2, 8, 1, 8, true),

        // ModelComparisonTracingExec
        std::make_tuple(4096, 2048, 8, 16, 4 /* MB */, 1, 4, 8, 4, 8, false)
#endif
            ),
    [&](const testing::TestParamInfo<
        CtranTestAllToAllvDedupParamFixture::ParamType>& info) {
      const bool skipBucket = std::get<10>(info.param);
      return std::to_string(std::get<0>(info.param)) + "numblocks_" +
          std::to_string(std::get<1>(info.param)) + "count_" +
          std::to_string(std::get<2>(info.param)) + "bBuckets_" +
          std::to_string(std::get<3>(info.param)) + "buckets_" +
          std::to_string(std::get<4>(info.param)) + "MBchunkSz_" +
          std::to_string(std::get<5>(info.param)) + "sendG_" +
          std::to_string(std::get<6>(info.param)) + "sendW_" +
          std::to_string(std::get<7>(info.param)) + "fwdW_" +
          std::to_string(std::get<8>(info.param)) + "recvG" +
          std::to_string(std::get<9>(info.param)) + "recvW" +
          (skipBucket ? "_skipBucket" : "");
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
