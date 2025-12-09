// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <cstddef>

#if !defined(USE_ROCM)
// needed because we use ncclMemAlloc to test kMemNcclMemAlloc mem type.
// cuMem API is not supported on AMD so we don't use test it on AMD.
#include <nccl.h>
#include "comms/testinfra/TestUtils.h"
#endif

#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/tests_common.cuh"

#define dceil(x, y) ((x / y) + !!(x % y))

// The main send/recv buffers cannot use more than this amount of
// memory.  Does not include smaller buffers such as count buffers.
constexpr size_t MAX_MEM_USAGE = 36 * 1024 * 1024 * 1024ULL;

static size_t rand64() {
  size_t ret = 0;

  for (int i = 0; i < sizeof(size_t) * 8; i++) {
    ret |= (((size_t)(rand() % 2)) << i);
  }

  return ret;
}

__global__ void initializeDataBuffersKernel(
    size_t maxCount,
    int** sendbuffs,
    int** recvbuffs,
    size_t* sendcountsDev,
    int maxNumExperts,
    int numRanks);
__global__ void initializeBufferPtrKernel(
    size_t maxCount,
    int* sendbuff,
    int** sendbuffs,
    size_t* inputChunkSizesDev);
__global__ void checkDataBuffersKernel(
    size_t maxCount,
    size_t* counts,
    int globalRank,
    int** recvbuffs);
__global__ void checkDataBuffersNonContigKernel(
    size_t maxCount,
    int maxNumExperts,
    size_t* recvSplits,
    size_t* recvIndices,
    size_t* recvIndicesBlockLengths,
    size_t inputChunkSizesCount,
    int** recvbuffs,
    int globalRank);
__global__ void equalCountsKernel(size_t* sendCounts, size_t count);
__global__ void checkEqualCountsKernel(size_t* recvCounts, size_t count);
__global__ void randomCountsKernel(
    size_t* sendCounts,
    size_t* randomCountsMatrixDev,
    int globalRank,
    int numRanks);
__global__ void checkRandomCountsKernel(
    size_t* recvCounts,
    size_t* randomCountsMatrixDev,
    int globalRank,
    int numRanks);
__global__ void checkRandomCountsNonContigKernel(
    size_t* recvSplits,
    size_t* randomCountsMatrix,
    size_t inputChunkSizesCount,
    int numRanks,
    int maxNumExperts);
__global__ void initRecvIndicesKernel(
    size_t* recvIndices,
    size_t* recvIndicesBlockLengths,
    size_t* sendIndices,
    size_t curSendIndicesPos);
__global__ void initRecvIndicesBlockLengthKernel(
    size_t* recvIndicesBlockLengths,
    size_t myIndicesBlockLengths);

class AllToAllvDynamicSplitNonContigTestCommon : public CtranDistTest {
 public:
  AllToAllvDynamicSplitNonContigTestCommon() = default;
  void SetUp() override {
    CtranDistTest::SetUp();
    comm = commRAII->ctranComm.get();

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    maxAllowedCount = MAX_MEM_USAGE / (2 * numRanks * sizeof(int));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    CtranDistTest::TearDown();
  }

  void InitTestSetup(size_t maxCount_, int numExperts_, bool dupExpertFlag) {
    numExperts = numExperts_;

    if (dupExpertFlag) {
      maxNumExperts = numExperts + defaultNumDupExpert;
    } else {
      maxNumExperts = numExperts;
    }
    maxTotalExperts = numRanks * maxNumExperts;

    maxCount = fmin(maxCount_, maxAllowedCount / (maxNumExperts * numRanks));

    maxCountBuff = maxCount;
    if (maxCountBuff * sizeof(int) < CTRAN_MIN_REGISTRATION_SIZE) {
      maxCountBuff = 2 * CTRAN_MIN_REGISTRATION_SIZE / sizeof(int);
    }

    maxSendcount = maxCountBuff * maxNumExperts * numRanks;
    maxRecvcount = maxCountBuff * maxNumExperts * numRanks;

    setHints();
  }

  void AllocateBuffers(MemAllocType memType, bool registFlag) {
    // Create metadata buffers
    CUDACHECK_TEST(cudaMalloc(
        &inputChunkSizesDev, numRanks * maxNumExperts * sizeof(size_t)));
    CUDACHECK_TEST(cudaMalloc(
        &inputChunkIndicesDev, numRanks * maxNumExperts * sizeof(size_t)));
    CUDACHECK_TEST(
        cudaMalloc(&recvIndicesDev, numRanks * maxNumExperts * sizeof(size_t)));
    CUDACHECK_TEST(cudaHostAlloc(
        &inputChunkIndicesHost,
        numRanks * maxNumExperts * sizeof(size_t),
        cudaHostAllocDefault));
    CUDACHECK_TEST(cudaHostAlloc(
        &recvIndicesHost,
        numRanks * maxNumExperts * sizeof(size_t),
        cudaHostAllocDefault));
    CUDACHECK_TEST(
        cudaMalloc(&inputChunkCountPerRankDev, numRanks * sizeof(size_t)));
    CUDACHECK_TEST(
        cudaMalloc(&recvIndicesBlockLengthsDev, numRanks * sizeof(size_t)));
    CUDACHECK_TEST(cudaHostAlloc(
        &inputChunkCountPerRankHost,
        numRanks * sizeof(size_t),
        cudaHostAllocDefault));
    CUDACHECK_TEST(cudaHostAlloc(
        &recvIndicesBlockLengthsHost,
        numRanks * sizeof(size_t),
        cudaHostAllocDefault));
    CUDACHECK_TEST(cudaMalloc(
        &recvSplitsDev, numRanks * maxNumExperts * numRanks * sizeof(size_t)));

    // Create and register data buffers.
    CUDACHECK_TEST(cudaHostAlloc(
        &recvbuffsHost, numRanks * sizeof(void*), cudaHostAllocDefault));
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaMalloc(
          &sendbuffDev, maxCountBuff * maxNumExperts * numRanks * sizeof(int)));
      for (int i = 0; i < numRanks; i++) {
        CUDACHECK_TEST(cudaMalloc(
            &recvbuffsHost[i],
            maxCountBuff * maxNumExperts * numRanks * sizeof(int)));
      }
    } else {
#if !defined(USE_ROCM)
      NCCLCHECK_TEST(ncclMemAlloc(
          &sendbuffDev, maxCountBuff * maxNumExperts * numRanks * sizeof(int)));
      for (int i = 0; i < numRanks; i++) {
        NCCLCHECK_TEST(ncclMemAlloc(
            &recvbuffsHost[i],
            maxCountBuff * maxNumExperts * numRanks * sizeof(int)));
      }
#else
      CHECK(false) << "cuMem API is not supported on AMD";
#endif
    }
    CUDACHECK_TEST(
        cudaMalloc(&sendbuffsDev, numRanks * maxNumExperts * sizeof(void*)));
    CUDACHECK_TEST(cudaMalloc(&recvbuffsDev, numRanks * sizeof(void*)));
    CUDACHECK_TEST(cudaMemcpy(
        recvbuffsDev,
        recvbuffsHost,
        numRanks * sizeof(void*),
        cudaMemcpyDefault));

    if (registFlag) {
      void* hdl = nullptr;
      for (int i = 0; i < numRanks; i++) {
        COMMCHECK_TEST(comm->ctran_->commRegister(
            recvbuffsHost[i],
            maxCountBuff * maxNumExperts * numRanks * sizeof(int),
            &hdl));
        recvhdls.push_back(hdl);
      }

      COMMCHECK_TEST(comm->ctran_->commRegister(
          sendbuffDev,
          maxCountBuff * maxNumExperts * numRanks * sizeof(int),
          &hdl));
      sendhdls.push_back(hdl);
    }
  }

  void DeallocateBuffers(MemAllocType memType, bool registFlag) {
    // Deregister and free data buffers
    if (registFlag) {
      for (auto h : recvhdls) {
        COMMCHECK_TEST(comm->ctran_->commDeregister(h));
      }
      for (auto h : sendhdls) {
        COMMCHECK_TEST(comm->ctran_->commDeregister(h));
      }
    }

    if (memType == kMemCudaMalloc) {
      for (int i = 0; i < numRanks; i++) {
        CUDACHECK_TEST(cudaFree(recvbuffsHost[i]));
      }
      CUDACHECK_TEST(cudaFree(sendbuffDev));
    } else {
#if !defined(USE_ROCM)
      for (int i = 0; i < numRanks; i++) {
        NCCLCHECK_TEST(ncclMemFree(recvbuffsHost[i]));
      }
      NCCLCHECK_TEST(ncclMemFree(sendbuffDev));
#else
      CHECK(false) << "cuMem API is not supported on AMD";
#endif
    }
    CUDACHECK_TEST(cudaFreeHost(recvbuffsHost));
    CUDACHECK_TEST(cudaFree(sendbuffsDev));
    CUDACHECK_TEST(cudaFree(recvbuffsDev));

    // Free metadata buffers
    CUDACHECK_TEST(cudaFree(inputChunkSizesDev));
    CUDACHECK_TEST(cudaFree(inputChunkIndicesDev));
    CUDACHECK_TEST(cudaFree(recvIndicesDev));
    CUDACHECK_TEST(cudaFreeHost(inputChunkIndicesHost));
    CUDACHECK_TEST(cudaFreeHost(recvIndicesHost));
    CUDACHECK_TEST(cudaFree(inputChunkCountPerRankDev));
    CUDACHECK_TEST(cudaFree(recvIndicesBlockLengthsDev));
    CUDACHECK_TEST(cudaFreeHost(inputChunkCountPerRankHost));
    CUDACHECK_TEST(cudaFreeHost(recvIndicesBlockLengthsHost));
    CUDACHECK_TEST(cudaFree(recvSplitsDev));
  }

  void EnqueueDataBuffersInitialization() {
    std::vector<void*> kernelArgs;
    kernelArgs.push_back((void*)&maxCount);
    kernelArgs.push_back((void*)&sendbuffsDev);
    kernelArgs.push_back((void*)&recvbuffsDev);
    kernelArgs.push_back((void*)&inputChunkSizesDev);
    kernelArgs.push_back((void*)&maxTotalExperts);
    kernelArgs.push_back((void*)&numRanks);
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)initializeDataBuffersKernel,
        inputChunkSizesCount,
        1024,
        kernelArgs.data(),
        0,
        stream));
  }

  void EnqueueInitializeBufferPtrKernel() {
    std::vector<void*> kernelArgs;
    kernelArgs.push_back((void*)&maxCount);
    kernelArgs.push_back((void*)&sendbuffDev);
    kernelArgs.push_back((void*)&sendbuffsDev);
    kernelArgs.push_back((void*)&inputChunkSizesDev);
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)initializeBufferPtrKernel,
        inputChunkSizesCount,
        1,
        kernelArgs.data(),
        0,
        stream));
  }

  void EnqueueDataBuffersCheck() {
    std::vector<void*> kernelArgs;
    kernelArgs.push_back((void*)&maxCount);
    kernelArgs.push_back((void*)&maxNumExperts);
    kernelArgs.push_back((void*)&recvSplitsDev);
    kernelArgs.push_back((void*)&recvIndicesDev);
    kernelArgs.push_back((void*)&recvIndicesBlockLengthsDev);
    kernelArgs.push_back((void*)&inputChunkSizesCount);
    kernelArgs.push_back((void*)&recvbuffsDev);
    kernelArgs.push_back((void*)&globalRank);
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)checkDataBuffersNonContigKernel,
        numRanks,
        1,
        kernelArgs.data(),
        0,
        stream));
  }

  // TODO: will need to add unit test for combine. Now only have dispatch.
  void EnqueueAllToAllvDynamicSplitNonContig() {
    auto res = ctranAlltoallvDynamicSplitNonContig(
        sendbuffDev,
        inputChunkSizesDev,
        inputChunkSizesCount,
        inputChunkIndicesDev,
        inputChunkCountPerRankDev,
        recvbuffsHost,
        nullptr,
        maxSendcount,
        maxRecvcount,
        hints,
        commInt32,
        comm,
        stream,
        false,
        recvSplitsDev);
    ASSERT_EQ(res, commSuccess);
  }

  enum class CountType {
    EQUAL,
    RANDOM,
  };

  void InitSendIndices(bool expertType) {
    // Initilze inputChunkCountPerRankDev
    inputChunkSizesCount = 0;
    for (int r = 0; r < numRanks; r++) {
      inputChunkCountPerRankHost[r] = numExperts;
      if (expertType) {
        inputChunkCountPerRankHost[r] += r % (maxNumExperts - numExperts + 1);
      }
      inputChunkSizesCount += inputChunkCountPerRankHost[r];
    }
    CUDACHECK_TEST(cudaMemcpy(
        inputChunkCountPerRankDev,
        inputChunkCountPerRankHost,
        numRanks * sizeof(size_t),
        cudaMemcpyDefault));

    // Initialize inputChunkIndicesDev
    auto sendIndicesPos = 0;
    if (contigSendIndices) {
      auto lastIndices = 0;
      for (int r = 0; r < numRanks; r++) {
        for (int i = 0; i < inputChunkCountPerRankHost[r]; i++) {
          inputChunkIndicesHost[sendIndicesPos + i] = lastIndices;
          lastIndices++;
        }
        sendIndicesPos += inputChunkCountPerRankHost[r];
      }
    } else {
      auto lastIndices = numRanks * numExperts;
      for (int r = 0; r < numRanks; r++) {
        // the extra indices will start from lastIndices, and will be coninuous
        // for each rank
        for (int i = 0; i < inputChunkCountPerRankHost[r]; i++) {
          if (i < numExperts) {
            inputChunkIndicesHost[sendIndicesPos + i] = r + numRanks * i;
          } else {
            inputChunkIndicesHost[sendIndicesPos + i] = lastIndices;
            lastIndices++;
          }
        }
        sendIndicesPos += inputChunkCountPerRankHost[r];
      }
    }
    CUDACHECK_TEST(cudaMemcpy(
        inputChunkIndicesDev,
        inputChunkIndicesHost,
        inputChunkSizesCount * sizeof(size_t),
        cudaMemcpyDefault));
  }

  void EnqueueSplitsInitialization(
      size_t count,
      AllToAllvDynamicSplitNonContigTestCommon::CountType type,
      int matrixId) {
    std::vector<void*> kernelArgs;
    // Initialize sendSplitLengthsDev
    if (type == CountType::EQUAL) {
      kernelArgs.push_back((void*)&inputChunkSizesDev);
      kernelArgs.push_back((void*)&count);
      CUDACHECK_TEST(cudaLaunchKernel(
          (void*)equalCountsKernel,
          1,
          numRanks * maxNumExperts,
          kernelArgs.data(),
          0,
          stream));
    } else {
      kernelArgs.push_back((void*)&inputChunkSizesDev);
      kernelArgs.push_back((void*)&randomCountsMatricesDev[matrixId]);
      kernelArgs.push_back((void*)&globalRank);
      kernelArgs.push_back((void*)&numRanks);
      CUDACHECK_TEST(cudaLaunchKernel(
          (void*)randomCountsKernel,
          1,
          numRanks * maxNumExperts,
          kernelArgs.data(),
          0,
          stream));
    }
  }

  void EnqueueCountsCheck(
      size_t count,
      AllToAllvDynamicSplitNonContigTestCommon::CountType type,
      int matrixId) {
    std::vector<void*> kernelArgs;
    if (type == CountType::EQUAL) {
      kernelArgs.clear();
      kernelArgs.push_back((void*)&recvSplitsDev);
      kernelArgs.push_back((void*)&count);
      CUDACHECK_TEST(cudaLaunchKernel(
          (void*)checkEqualCountsKernel,
          1,
          inputChunkSizesCount * numRanks,
          kernelArgs.data(),
          0,
          stream));
    } else {
      kernelArgs.push_back((void*)&recvSplitsDev);
      kernelArgs.push_back((void*)&randomCountsMatricesDev[matrixId]);
      kernelArgs.push_back((void*)&inputChunkSizesCount);
      kernelArgs.push_back((void*)&numRanks);
      kernelArgs.push_back((void*)&maxNumExperts);
      CUDACHECK_TEST(cudaLaunchKernel(
          (void*)checkRandomCountsNonContigKernel,
          numRanks,
          inputChunkSizesCount,
          kernelArgs.data(),
          0,
          stream));
    }
  }

  void EnqueueRecvIndicesInitialization() {
    // Initialize recvIndicesDev
    curSendIndicesPos = 0;
    for (int r = 0; r < globalRank; r++) {
      curSendIndicesPos += inputChunkCountPerRankHost[r];
    }

    std::vector<void*> kernelArgs;
    kernelArgs.push_back((void*)&recvIndicesBlockLengthsDev);
    kernelArgs.push_back((void*)&inputChunkCountPerRankHost[globalRank]);
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)initRecvIndicesBlockLengthKernel,
        numRanks,
        1,
        kernelArgs.data(),
        0,
        stream));

    kernelArgs.clear();
    kernelArgs.push_back((void*)&recvIndicesDev);
    kernelArgs.push_back((void*)&recvIndicesBlockLengthsDev);
    kernelArgs.push_back((void*)&inputChunkIndicesDev);
    kernelArgs.push_back((void*)&curSendIndicesPos);
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)initRecvIndicesKernel,
        numRanks,
        inputChunkCountPerRankHost[globalRank],
        kernelArgs.data(),
        0,
        stream));
  }

  void InitializeRandomMatrices(int numMatrices) {
    // Don't use srand(0), it resets to srand(1)
    // (https://sourceware.org/git/?p=glibc.git;a=blob;f=stdlib/random_r.c;h=b49f03f5becd5063b445c1ebdf683730bf3dea07;hb=HEAD#l193)
    std::srand(1);

    size_t* matrixHost;
    CUDACHECK_TEST(cudaHostAlloc(
        &matrixHost,
        numRanks * numRanks * maxNumExperts * sizeof(size_t),
        cudaHostAllocDefault));

    for (int m = 0; m < numMatrices; m++) {
      size_t* matrixDev;
      CUDACHECK_TEST(cudaMalloc(
          &matrixDev, numRanks * numRanks * maxNumExperts * sizeof(size_t)));
      randomCountsMatricesDev.push_back(matrixDev);

      for (int i = 0; i < numRanks; i++) {
        for (int j = 0; j < numRanks * maxNumExperts; j++) {
          if (maxCount) {
            matrixHost[i * numRanks * maxNumExperts + j] = rand64() % maxCount;
          } else {
            matrixHost[i * numRanks * maxNumExperts + j] = 0;
          }
        }
      }
      CUDACHECK_TEST(cudaMemcpy(
          matrixDev,
          matrixHost,
          numRanks * numRanks * maxNumExperts * sizeof(size_t),
          cudaMemcpyDefault));
    }

    CUDACHECK_TEST(cudaFreeHost(matrixHost));
  }

  void DeallocateRandomCountsMatrices() {
    for (auto x : randomCountsMatricesDev) {
      CUDACHECK_TEST(cudaFree(x));
    }
  }

  void setHints() {
    hints.set("ncclx_alltoallv_dynamic_sendbuffs_location", "cpu");
    hints.set("ncclx_alltoallv_dynamic_recvbuffs_location", "cpu");
    hints.set("ncclx_alltoallv_dynamic_sendcounts_location", "gpu");
    hints.set("ncclx_alltoallv_dynamic_max_sendcounts_location", "cpu");
    hints.set("ncclx_alltoallv_dynamic_max_recvcounts_location", "cpu");
    hints.set("ncclx_alltoallv_dynamic_actual_recvcounts_location", "gpu");
  }

 protected:
  CtranComm* comm{nullptr};
  cudaStream_t stream{};
  int numExperts{0};
  int maxNumExperts{0};
  int maxTotalExperts{0};
  int defaultNumDupExpert{2};

  void* sendbuffDev{nullptr};
  void** sendbuffsDev{nullptr};

  void** recvbuffsHost{nullptr};
  void** recvbuffsDev{nullptr};
  size_t* recvSplitsDev{nullptr};

  std::vector<void*> sendhdls;
  std::vector<void*> recvhdls;

  size_t* inputChunkSizesDev{nullptr};
  size_t inputChunkSizesCount{0};

  size_t* inputChunkIndicesDev{nullptr};
  size_t* inputChunkIndicesHost{nullptr};
  size_t* inputChunkCountPerRankDev{nullptr};
  size_t* inputChunkCountPerRankHost{nullptr};
  size_t curSendIndicesPos{0};

  size_t* recvIndicesDev{nullptr};
  size_t* recvIndicesHost{nullptr};
  size_t* recvIndicesBlockLengthsDev{nullptr};
  size_t* recvIndicesBlockLengthsHost{nullptr};

  // The maxCount per expert
  size_t maxCount{0};
  size_t maxCountBuff{0};
  size_t maxSendcount{0};
  size_t maxRecvcount{0};

  std::vector<size_t*> randomCountsMatricesDev;

  meta::comms::Hints hints;
  size_t maxAllowedCount{0};

  size_t* actualRcountsDev{nullptr};

  // TODO: add UT for second AllToAll case.
  bool contigSendIndices{true};
};

class AllToAllvDynamicSplitNonContigTestSuite
    : public AllToAllvDynamicSplitNonContigTestCommon,
      public ::testing::WithParamInterface<
          std::tuple<MemAllocType, size_t, int, bool, bool, bool>> {};

/********************** UnchangedEqualCounts **************************/
/* set the scounts to the max values at issue time, and do not change
 * them after issuing the collective. */
/**********************************************************************/
TEST_P(AllToAllvDynamicSplitNonContigTestSuite, UnchangedEqualCounts) {
  const auto& [memType, maxCount_, numExperts_, registFlag, dupExpertFlag, lowLatencyFlag] =
      GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }
  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, lowLatencyFlag);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, lowLatencyFlag);

  InitTestSetup(maxCount_, numExperts_, dupExpertFlag);

  AllocateBuffers(memType, registFlag);

  InitSendIndices(dupExpertFlag);

  // Enqueue count initialization
  EnqueueSplitsInitialization(maxCount, CountType::EQUAL, -1);

  // Enqueue recvIndices / recvIndicesBlocklength initialization
  EnqueueRecvIndicesInitialization();

  // Wait for count update to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  EnqueueInitializeBufferPtrKernel();

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization();

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplitNonContig();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount, CountType::EQUAL, -1);

  // Enqueue data check
  EnqueueDataBuffersCheck();

  // Wait for everything to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  DeallocateBuffers(memType, registFlag);
}

/********************** ChangedEqualCounts ****************************/
/* set the scounts to the max values at issue time, but have a kernel
 * halve them after issuing the collective, but before the execution
 * of the collective. */
/**********************************************************************/
TEST_P(AllToAllvDynamicSplitNonContigTestSuite, ChangedEqualCounts) {
  const auto& [memType, maxCount_, numExperts_, registFlag, dupExpertFlag, lowLatencyFlag] =
      GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, lowLatencyFlag);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, lowLatencyFlag);

  InitTestSetup(maxCount_, numExperts_, dupExpertFlag);

  AllocateBuffers(memType, registFlag);

  InitSendIndices(dupExpertFlag);

  // Enqueue count initialization
  EnqueueSplitsInitialization(maxCount / 2, CountType::EQUAL, -1);

  // Enqueue recvIndices / recvIndicesBlocklength initialization
  EnqueueRecvIndicesInitialization();

  // Enqueue buffer ptr initialization
  EnqueueInitializeBufferPtrKernel();

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization();

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplitNonContig();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount / 2, CountType::EQUAL, -1);

  // Enqueue data check
  EnqueueDataBuffersCheck();

  // Wait for everything to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  DeallocateBuffers(memType, registFlag);
}

/********************** UnchangedRandomCounts *************************/
/* set the scounts to the random values at issue time, and do not
 * change them after issuing the collective. */
/**********************************************************************/
TEST_P(AllToAllvDynamicSplitNonContigTestSuite, UnchangedRandomCounts) {
  const auto& [memType, maxCount_, numExperts_, registFlag, dupExpertFlag, lowLatencyFlag] =
      GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, lowLatencyFlag);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, lowLatencyFlag);

  InitTestSetup(maxCount_, numExperts_, dupExpertFlag);

  // Overwrite the registFlag to true. Change it back after fix the small buffer
  // reigstration issue.
  AllocateBuffers(memType, true);

  InitSendIndices(dupExpertFlag);

  InitializeRandomMatrices(1);

  // Enqueue count initialization
  EnqueueSplitsInitialization(maxCount, CountType::RANDOM, 0);

  // Enqueue recvIndices / recvIndicesBlocklength initialization
  EnqueueRecvIndicesInitialization();

  // Wait for count update to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Enqueue buffer ptr initialization
  EnqueueInitializeBufferPtrKernel();

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization();

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplitNonContig();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount, CountType::RANDOM, 0);

  // Enqueue data check
  EnqueueDataBuffersCheck();

  // Wait for everything to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Overwrite the registFlag to true. Change it back after fix the small buffer
  // reigstration issue.
  DeallocateBuffers(memType, true);
}

/********************** ChangedRandomCounts ***************************/
/* set the scounts to the random values at issue time, but have a kernel
 * halve them after issuing the collective, but before the execution
 * of the collective. */
/**********************************************************************/
TEST_P(AllToAllvDynamicSplitNonContigTestSuite, ChangedRandomCounts) {
  const auto& [memType, maxCount_, numExperts_, registFlag, dupExpertFlag, lowLatencyFlag] =
      GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, lowLatencyFlag);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, lowLatencyFlag);

  InitTestSetup(maxCount_, numExperts_, dupExpertFlag);

  // Overwrite the registFlag to true. Change it back after fix the small buffer
  // reigstration issue.
  AllocateBuffers(memType, true);

  InitSendIndices(maxCount);

  InitializeRandomMatrices(1);

  // Enqueue count initialization
  EnqueueSplitsInitialization(maxCount, CountType::RANDOM, 0);

  // Enqueue recvIndices / recvIndicesBlocklength initialization
  EnqueueRecvIndicesInitialization();

  // Enqueue buffer ptr initialization
  EnqueueInitializeBufferPtrKernel();

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization();

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplitNonContig();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount, CountType::RANDOM, 0);

  // Enqueue data check
  EnqueueDataBuffersCheck();

  // Wait for everything to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Overwrite the registFlag to true. Change it back after fix the small buffer
  // reigstration issue.
  DeallocateBuffers(memType, true);
}

/********************** MultipleRandomCounts ***************************/
/* set the scounts to the random values at issue time, but have a kernel
 * halve them after issuing the collective, but before the execution
 * of the collective. */
/**********************************************************************/
TEST_P(AllToAllvDynamicSplitNonContigTestSuite, MultipleRandomCounts) {
  const auto& [memType, maxCount_, numExperts_, registFlag, dupExpertFlag, lowLatencyFlag] =
      GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, lowLatencyFlag);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, lowLatencyFlag);

  InitTestSetup(maxCount_, numExperts_, dupExpertFlag);

  // Overwrite the registFlag to true. Change it back after fix the small buffer
  // reigstration issue.
  AllocateBuffers(memType, true);

  InitSendIndices(dupExpertFlag);

  constexpr int numIters = 10;

  InitializeRandomMatrices(numIters);

  for (int i = 0; i < numIters; i++) {
    // Enqueue count initialization
    EnqueueSplitsInitialization(maxCount, CountType::RANDOM, i);

    // Enqueue recvIndices / recvIndicesBlocklength initialization
    EnqueueRecvIndicesInitialization();

    // Enqueue buffer ptr initialization
    EnqueueInitializeBufferPtrKernel();

    // Enqueue buffer initialization
    EnqueueDataBuffersInitialization();

    // Enqueue communication kernel
    EnqueueAllToAllvDynamicSplitNonContig();

    // Enqueue counts check
    EnqueueCountsCheck(maxCount, CountType::RANDOM, i);

    // Enqueue data check
    EnqueueDataBuffersCheck();
  }

  // Wait for everything to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Overwrite the registFlag to true. Change it back after fix the small buffer
  // reigstration issue.
  DeallocateBuffers(memType, true);
}

#ifdef TEST_CUDA_GRAPH_MODE
/********************** ChangedEqualCountsGraph ********************/
TEST_P(AllToAllvDynamicSplitNonContigTestSuite, UnchangedEqualCountsGraph) {
  const auto& [memType, maxCount_, numExperts_, registFlag, dupExpertFlag, lowLatencyFlag] =
      GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  EnvRAII env1(NCCL_CTRAN_NO_ERROR_CHECK, lowLatencyFlag);
  EnvRAII env2(NCCL_CTRAN_ENABLE_PRECONNECT, lowLatencyFlag);

  InitTestSetup(maxCount_, numExperts_, dupExpertFlag);

  AllocateBuffers(memType, registFlag);

  InitSendIndices(dupExpertFlag);

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  // Enqueue count initialization
  EnqueueSplitsInitialization(maxCount, CountType::EQUAL, -1);

  // Enqueue recvIndices / recvIndicesBlocklength initialization
  EnqueueRecvIndicesInitialization();

  // Enqueue buffer ptr initialization
  EnqueueInitializeBufferPtrKernel();

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization();

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplitNonContig();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount, CountType::EQUAL, -1);

  // Enqueue data check
  EnqueueDataBuffersCheck();

  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
  CUDACHECK_TEST(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

  constexpr int numIters = 10;
  for (int i = 0; i < numIters; i++) {
    CUDACHECK_TEST(cudaGraphLaunch(instance, stream));
    auto nelems = comm->ctran_->gpe->numInUseKernelElems();
    EXPECT_NE(nelems, 0);
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  CUDACHECK_TEST(cudaGraphExecDestroy(instance));
  CUDACHECK_TEST(cudaGraphDestroy(graph));

  DeallocateBuffers(memType, registFlag);
}

TEST_P(
    AllToAllvDynamicSplitNonContigTestSuite,
    UnchangedEqualCountsGraphWithGraphAware) {
  const auto& [memType, maxCount_, numExperts_, registFlag, dupExpertFlag, lowLatencyFlag] =
      GetParam();
  EnvRAII env1(NCCL_CTRAN_ALLTOALL_CUDAGRAPH_AWARE_ENABLE, true);
  EnvRAII env2(NCCL_CTRAN_NO_ERROR_CHECK, lowLatencyFlag);
  EnvRAII env3(NCCL_CTRAN_ENABLE_PRECONNECT, lowLatencyFlag);

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }
  if (!registFlag) {
    GTEST_SKIP() << "Cuda graph aware only support when pre-registered";
  }

  InitTestSetup(maxCount_, numExperts_, dupExpertFlag);

  AllocateBuffers(memType, registFlag);

  InitSendIndices(dupExpertFlag);

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  // Enqueue count initialization
  EnqueueSplitsInitialization(maxCount, CountType::EQUAL, -1);

  // Enqueue recvIndices / recvIndicesBlocklength initialization
  EnqueueRecvIndicesInitialization();

  // Enqueue buffer ptr initialization
  EnqueueInitializeBufferPtrKernel();

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization();

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplitNonContig();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount, CountType::EQUAL, -1);

  // Enqueue data check
  EnqueueDataBuffersCheck();

  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
  CUDACHECK_TEST(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

  // When using cudagraph aware, need double-buffer to avoid data race on one
  // buffer because of skipping sync. For simplicity, we use 1 iteration here.
  constexpr int numIters = 1;
  for (int i = 0; i < numIters; i++) {
    CUDACHECK_TEST(cudaGraphLaunch(instance, stream));
    auto nelems = comm->ctran_->gpe->numInUseKernelElems();
    EXPECT_NE(nelems, 0);
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  CUDACHECK_TEST(cudaGraphExecDestroy(instance));
  CUDACHECK_TEST(cudaGraphDestroy(graph));

  DeallocateBuffers(memType, registFlag);
}
#endif

INSTANTIATE_TEST_SUITE_P(
    AllToAllvDynamicSplitNonContigTestInstance,
    AllToAllvDynamicSplitNonContigTestSuite,
    ::testing::Values(
        // memType, maxCount, number of experts, registFlag, dupExpertFlag,
        // lowLatencyFlag maxCount: max number of elements to send per expert.
        // numExperts: number of experts to test. Currently only support
        // numExperts * numRanks <= 256.
        std::make_tuple(kMemCudaMalloc, 1048576, 1, true, false, true),
        std::make_tuple(kMemCudaMalloc, 1048576, 1, true, false, false),
        std::make_tuple(kMemCudaMalloc, 1048576, 1, false, false, true),
        // std::make_tuple(kMemCudaMalloc, 1048576, 2, true, false, true),
        // std::make_tuple(kMemCudaMalloc, 1048576, 2, true, false, false),
        // std::make_tuple(kMemCudaMalloc, 1048576, 2, false, false, true),
        std::make_tuple(kMemCudaMalloc, 1048576, 2, true, true, true),
        std::make_tuple(kMemCudaMalloc, 1048576, 2, false, true, true),
        std::make_tuple(kMemCudaMalloc, 262144, 4, true, false, true),
        std::make_tuple(kMemCudaMalloc, 262144, 4, true, false, false),
        std::make_tuple(kMemCudaMalloc, 262144, 4, false, false, true),
        std::make_tuple(kMemNcclMemAlloc, 262144, 4, true, false, true),
        std::make_tuple(kMemNcclMemAlloc, 262144, 4, false, false, true),
        // std::make_tuple(
        //     kMemCudaMalloc,
        //     dceil(CTRAN_MIN_REGISTRATION_SIZE, commTypeSize(commInt)),
        //     4,
        //     false,
        //     false),

        // Comment out the following tests to save time for CI.
        // std::make_tuple(
        //     kMemCudaMalloc,
        //     std::numeric_limits<size_t>::max(),
        //     4,
        //     true,
        //     false,
        //     true)));
        std::make_tuple(
            kMemCudaMalloc,
            dceil(CTRAN_MIN_REGISTRATION_SIZE, commTypeSize(commInt)),
            4,
            true,
            false,
            true)));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranDistTestEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
