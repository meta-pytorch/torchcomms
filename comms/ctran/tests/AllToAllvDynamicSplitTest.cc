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
    size_t* sendSplitLengthsDev);
__global__ void checkDataBuffersKernel(
    size_t maxCount,
    size_t* counts,
    int globalRank,
    int** recvbuffs);
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

class AllToAllvDynamicSplitTestCommon : public CtranDistTest {
 public:
  AllToAllvDynamicSplitTestCommon() = default;
  void SetUp() override {
    CtranDistTest::SetUp();
    comm = commRAII->ctranComm;

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    maxAllowedCount = MAX_MEM_USAGE / (2 * numRanks * sizeof(int));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    CtranDistTest::TearDown();
  }

  void AllocateBuffers(MemAllocType memType, size_t maxCount, bool registFlag) {
    maxCountBuff = maxCount;
    if (maxCountBuff * sizeof(int) < CTRAN_MIN_REGISTRATION_SIZE) {
      maxCountBuff = 2 * CTRAN_MIN_REGISTRATION_SIZE / sizeof(int);
    }

    // Create metadata buffers
    CUDACHECK_TEST(cudaMalloc(&sendSplitLengthsDev, numRanks * sizeof(size_t)));
    CUDACHECK_TEST(cudaMalloc(&actualRcountsDev, numRanks * sizeof(size_t)));
    CUDACHECK_TEST(cudaHostAlloc(
        &recvbuffsHost, numRanks * sizeof(void*), cudaHostAllocDefault));

    // Create and register data buffers.
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(
          cudaMalloc(&sendbuffDev, maxCountBuff * numRanks * sizeof(int)));
      for (int i = 0; i < numRanks; i++) {
        CUDACHECK_TEST(
            cudaMalloc(&recvbuffsHost[i], maxCountBuff * sizeof(int)));
      }
    } else {
#if !defined(USE_ROCM)
      NCCLCHECK_TEST(
          ncclMemAlloc(&sendbuffDev, maxCountBuff * numRanks * sizeof(int)));
      for (int i = 0; i < numRanks; i++) {
        NCCLCHECK_TEST(
            ncclMemAlloc(&recvbuffsHost[i], maxCountBuff * sizeof(int)));
      }
#else
      CHECK(false) << "cuMem API is not supported on AMD";
#endif
    }
    CUDACHECK_TEST(cudaMalloc(&sendbuffsDev, numRanks * sizeof(void*)));
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
            recvbuffsHost[i], maxCountBuff * sizeof(int), &hdl));
        recvhdls.push_back(hdl);
      }

      COMMCHECK_TEST(comm->ctran_->commRegister(
          sendbuffDev, maxCountBuff * numRanks * sizeof(int), &hdl));
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
      CUDACHECK_TEST(cudaFree(sendbuffDev));
      for (int i = 0; i < numRanks; i++) {
        CUDACHECK_TEST(cudaFree(recvbuffsHost[i]));
      }
    } else {
#if !defined(USE_ROCM)
      NCCLCHECK_TEST(ncclMemFree(sendbuffDev));
      for (int i = 0; i < numRanks; i++) {
        NCCLCHECK_TEST(ncclMemFree(recvbuffsHost[i]));
      }
#else
      CHECK(false) << "cuMem API is not supported on AMD";
#endif
    }

    CUDACHECK_TEST(cudaFree(sendbuffsDev));
    CUDACHECK_TEST(cudaFreeHost(recvbuffsHost));
    CUDACHECK_TEST(cudaFree(recvbuffsDev));

    // Free metadata buffers
    CUDACHECK_TEST(cudaFree(sendSplitLengthsDev));
    CUDACHECK_TEST(cudaFree(actualRcountsDev));
  }

  void EnqueueDataBuffersInitialization(size_t maxCount) {
    std::vector<void*> kernelArgs;
    kernelArgs.push_back((void*)&maxCount);
    kernelArgs.push_back((void*)&sendbuffsDev);
    kernelArgs.push_back((void*)&recvbuffsDev);
    kernelArgs.push_back((void*)&sendSplitLengthsDev);
    kernelArgs.push_back((void*)&numExperts);
    kernelArgs.push_back((void*)&numRanks);
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)initializeDataBuffersKernel,
        numRanks,
        1024,
        kernelArgs.data(),
        0,
        stream));
  }

  void EnqueueInitializeBufferPtrKernel(size_t maxCount) {
    std::vector<void*> kernelArgs;
    kernelArgs.push_back((void*)&maxCount);
    kernelArgs.push_back((void*)&sendbuffDev);
    kernelArgs.push_back((void*)&sendbuffsDev);
    kernelArgs.push_back((void*)&sendSplitLengthsDev);
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)initializeBufferPtrKernel,
        numRanks,
        1,
        kernelArgs.data(),
        0,
        stream));
  }

  void EnqueueDataBuffersCheck(size_t maxCount) {
    std::vector<void*> kernelArgs;
    kernelArgs.push_back((void*)&maxCount);
    kernelArgs.push_back((void*)&actualRcountsDev);
    kernelArgs.push_back((void*)&globalRank);
    kernelArgs.push_back((void*)&recvbuffsDev);
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)checkDataBuffersKernel,
        numRanks,
        1024,
        kernelArgs.data(),
        0,
        stream));
  }

  void EnqueueAllToAllvDynamicSplit() {
    auto res = ctranAlltoallvDynamicSplit(
        sendbuffDev,
        sendSplitLengthsDev,
        recvbuffsHost,
        maxSendcount,
        maxRecvcount,
        actualRcountsDev,
        hints,
        commInt32,
        comm,
        stream);
    ASSERT_EQ(res, commSuccess);
  }

  enum class CountType {
    EQUAL,
    RANDOM,
  };

  void EnqueueSplitInitialization(
      size_t count,
      AllToAllvDynamicSplitTestCommon::CountType type,
      int matrixId) {
    std::vector<void*> kernelArgs;
    if (type == CountType::EQUAL) {
      kernelArgs.push_back((void*)&sendSplitLengthsDev);
      kernelArgs.push_back((void*)&count);
      CUDACHECK_TEST(cudaLaunchKernel(
          (void*)equalCountsKernel, 1, numRanks, kernelArgs.data(), 0, stream));
    } else {
      kernelArgs.push_back((void*)&sendSplitLengthsDev);
      kernelArgs.push_back((void*)&randomCountsMatricesDev[matrixId]);
      kernelArgs.push_back((void*)&globalRank);
      kernelArgs.push_back((void*)&numRanks);
      CUDACHECK_TEST(cudaLaunchKernel(
          (void*)randomCountsKernel,
          1,
          numRanks,
          kernelArgs.data(),
          0,
          stream));
    }
  }

  void EnqueueCountsCheck(
      size_t count,
      AllToAllvDynamicSplitTestCommon::CountType type,
      int matrixId) {
    std::vector<void*> kernelArgs;
    if (type == CountType::EQUAL) {
      kernelArgs.push_back((void*)&actualRcountsDev);
      kernelArgs.push_back((void*)&count);
      CUDACHECK_TEST(cudaLaunchKernel(
          (void*)checkEqualCountsKernel,
          1,
          numRanks,
          kernelArgs.data(),
          0,
          stream));
    } else {
      kernelArgs.push_back((void*)&actualRcountsDev);
      kernelArgs.push_back((void*)&randomCountsMatricesDev[matrixId]);
      kernelArgs.push_back((void*)&globalRank);
      kernelArgs.push_back((void*)&numRanks);
      CUDACHECK_TEST(cudaLaunchKernel(
          (void*)checkRandomCountsKernel,
          1,
          numRanks,
          kernelArgs.data(),
          0,
          stream));
    }
  }

  void InitializeRandomMatrices(size_t maxCount, int numMatrices) {
    // Don't use srand(0), it resets to srand(1)
    // (https://sourceware.org/git/?p=glibc.git;a=blob;f=stdlib/random_r.c;h=b49f03f5becd5063b445c1ebdf683730bf3dea07;hb=HEAD#l193)
    std::srand(1);

    size_t* matrixHost;
    CUDACHECK_TEST(cudaHostAlloc(
        &matrixHost,
        numRanks * numRanks * sizeof(size_t),
        cudaHostAllocDefault));

    for (int m = 0; m < numMatrices; m++) {
      size_t* matrixDev;
      CUDACHECK_TEST(
          cudaMalloc(&matrixDev, numRanks * numRanks * sizeof(size_t)));
      randomCountsMatricesDev.push_back(matrixDev);

      for (int i = 0; i < numRanks; i++) {
        for (int j = 0; j < numRanks; j++) {
          if (maxCount) {
            matrixHost[i * numRanks + j] = rand64() % maxCount;
          } else {
            matrixHost[i * numRanks + j] = 0;
          }
        }
      }
      CUDACHECK_TEST(cudaMemcpy(
          matrixDev,
          matrixHost,
          numRanks * numRanks * sizeof(size_t),
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
  int numExperts{1};

  void* sendbuffDev{nullptr};
  void** sendbuffsDev{nullptr};
  void** recvbuffsHost{nullptr};
  void** recvbuffsDev{nullptr};

  std::vector<void*> sendhdls;
  std::vector<void*> recvhdls;

  size_t* sendSplitLengthsDev{nullptr};
  size_t* actualRcountsDev{nullptr};

  size_t maxCountBuff{0};
  size_t maxSendcount{0};
  size_t maxRecvcount{0};

  std::vector<size_t*> randomCountsMatricesDev;

  meta::comms::Hints hints;
  size_t maxAllowedCount{};
};

class AllToAllvDynamicSplitTestSuite
    : public AllToAllvDynamicSplitTestCommon,
      public ::testing::WithParamInterface<
          std::tuple<MemAllocType, size_t, bool>> {};

/********************** UnchangedEqualCounts **************************/
/* set the scounts to the max values at issue time, and do not change
 * them after issuing the collective. */
/**********************************************************************/
TEST_P(AllToAllvDynamicSplitTestSuite, UnchangedEqualCounts) {
  const auto& [memType, maxCount_, registFlag] = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  std::vector<void*> kernelArgs;

  size_t maxCount = std::min(maxCount_, maxAllowedCount);

  AllocateBuffers(memType, maxCount, registFlag);
  setHints();

  maxSendcount = maxCountBuff;
  maxRecvcount = maxCountBuff;

  // Enqueue count initialization
  EnqueueSplitInitialization(maxCount, CountType::EQUAL, -1);

  // Wait for count update to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  EnqueueInitializeBufferPtrKernel(maxCount);

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization(maxCount);

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplit();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount, CountType::EQUAL, -1);

  // Enqueue data check
  EnqueueDataBuffersCheck(maxCount);

  // Wait for everything to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  DeallocateBuffers(memType, registFlag);
}

/********************** ChangedEqualCounts ****************************/
/* set the scounts to the max values at issue time, but have a kernel
 * halve them after issuing the collective, but before the execution
 * of the collective. */
/**********************************************************************/
TEST_P(AllToAllvDynamicSplitTestSuite, ChangedEqualCounts) {
  const auto& [memType, maxCount_, registFlag] = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  std::vector<void*> kernelArgs;

  size_t maxCount = std::min(maxCount_, maxAllowedCount);

  AllocateBuffers(memType, maxCount, registFlag);
  setHints();

  maxSendcount = maxCountBuff;
  maxRecvcount = maxCountBuff;

  // Enqueue count initialization
  EnqueueSplitInitialization(maxCount / 2, CountType::EQUAL, -1);

  EnqueueInitializeBufferPtrKernel(maxCount);

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization(maxCount);

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplit();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount / 2, CountType::EQUAL, -1);

  // Enqueue data check
  EnqueueDataBuffersCheck(maxCount);

  // Wait for everything to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  DeallocateBuffers(memType, registFlag);
}

/********************** UnchangedRandomCounts *************************/
/* set the scounts to the random values at issue time, and do not
 * change them after issuing the collective. */
/**********************************************************************/
TEST_P(AllToAllvDynamicSplitTestSuite, UnchangedRandomCounts) {
  const auto& [memType, maxCount_, registFlag] = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  std::vector<void*> kernelArgs;

  size_t maxCount = std::min(maxCount_, maxAllowedCount);

  AllocateBuffers(memType, maxCount, registFlag);
  setHints();

  InitializeRandomMatrices(maxCount, 1);
  maxSendcount = maxCountBuff;
  maxRecvcount = maxCountBuff;

  // Enqueue count initialization
  EnqueueSplitInitialization(maxCount, CountType::RANDOM, 0);

  // Wait for count update to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  EnqueueInitializeBufferPtrKernel(maxCount);

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization(maxCount);

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplit();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount, CountType::RANDOM, 0);

  // Enqueue data check
  EnqueueDataBuffersCheck(maxCount);

  // Wait for everything to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  DeallocateBuffers(memType, registFlag);
}

/********************** ChangedRandomCounts ***************************/
/* set the scounts to the random values at issue time, but have a kernel
 * halve them after issuing the collective, but before the execution
 * of the collective. */
/**********************************************************************/
TEST_P(AllToAllvDynamicSplitTestSuite, ChangedRandomCounts) {
  const auto& [memType, maxCount_, registFlag] = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  std::vector<void*> kernelArgs;

  size_t maxCount = std::min(maxCount_, maxAllowedCount);

  AllocateBuffers(memType, maxCount, registFlag);
  setHints();

  InitializeRandomMatrices(maxCount, 1);
  maxSendcount = maxCountBuff;
  maxRecvcount = maxCountBuff;

  // Enqueue count initialization
  EnqueueSplitInitialization(maxCount, CountType::RANDOM, 0);

  EnqueueInitializeBufferPtrKernel(maxCount);

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization(maxCount);

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplit();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount, CountType::RANDOM, 0);

  // Enqueue data check
  EnqueueDataBuffersCheck(maxCount);

  // Wait for everything to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  DeallocateBuffers(memType, registFlag);
}

/********************** MultipleRandomCounts ***************************/
/* set the scounts to the random values at issue time, but have a kernel
 * halve them after issuing the collective, but before the execution
 * of the collective. */
/**********************************************************************/
TEST_P(AllToAllvDynamicSplitTestSuite, MultipleRandomCounts) {
  const auto& [memType, maxCount_, registFlag] = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  std::vector<void*> kernelArgs;

  size_t maxCount = std::min(maxCount_, maxAllowedCount);

  AllocateBuffers(memType, maxCount, registFlag);
  setHints();

  constexpr int numIters = 10;

  InitializeRandomMatrices(maxCount, numIters);
  maxSendcount = maxCountBuff;
  maxRecvcount = maxCountBuff;

  for (int i = 0; i < numIters; i++) {
    // Enqueue count initialization
    EnqueueSplitInitialization(maxCount, CountType::RANDOM, i);

    EnqueueInitializeBufferPtrKernel(maxCount);

    // Enqueue buffer initialization
    EnqueueDataBuffersInitialization(maxCount);

    // Enqueue communication kernel
    EnqueueAllToAllvDynamicSplit();

    // Enqueue counts check
    EnqueueCountsCheck(maxCount, CountType::RANDOM, i);

    // Enqueue data check
    EnqueueDataBuffersCheck(maxCount);
  }

  // Wait for everything to finish
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  DeallocateBuffers(memType, registFlag);
}

#ifdef TEST_CUDA_GRAPH_MODE
/********************** ChangedEqualCountsGraph ********************/
TEST_P(AllToAllvDynamicSplitTestSuite, UnchangedEqualCountsGraph) {
  const auto& [memType, maxCount_, registFlag] = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  std::vector<void*> kernelArgs;

  size_t maxCount = std::min(maxCount_, maxAllowedCount);

  AllocateBuffers(memType, maxCount, registFlag);
  setHints();

  maxSendcount = maxCountBuff;
  maxRecvcount = maxCountBuff;

  cudaGraph_t graph;
  cudaGraphExec_t instance;

  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  // Enqueue count initialization
  EnqueueSplitInitialization(maxCount, CountType::EQUAL, -1);

  EnqueueInitializeBufferPtrKernel(maxCount);

  // Enqueue buffer initialization
  EnqueueDataBuffersInitialization(maxCount);

  // Enqueue communication kernel
  EnqueueAllToAllvDynamicSplit();

  // Enqueue counts check
  EnqueueCountsCheck(maxCount, CountType::EQUAL, -1);

  // Enqueue data check
  EnqueueDataBuffersCheck(maxCount);

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

TEST_P(AllToAllvDynamicSplitTestSuite, MultipleRandomCountsGraph) {
  const auto& [memType, maxCount_, registFlag] = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  std::vector<void*> kernelArgs;

  size_t maxCount = std::min(maxCount_, maxAllowedCount);

  AllocateBuffers(memType, maxCount, registFlag);
  setHints();

  constexpr int numIters = 10;

  InitializeRandomMatrices(maxCount, numIters);
  maxSendcount = maxCountBuff;
  maxRecvcount = maxCountBuff;

  cudaGraph_t graph;
  cudaGraphExec_t instance;
  for (int i = 0; i < numIters; i++) {
    CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Enqueue count initialization
    EnqueueSplitInitialization(maxCount, CountType::EQUAL, -1);

    EnqueueInitializeBufferPtrKernel(maxCount);

    // Enqueue buffer initialization
    EnqueueDataBuffersInitialization(maxCount);

    // Enqueue communication kernel
    EnqueueAllToAllvDynamicSplit();

    // Enqueue counts check
    EnqueueCountsCheck(maxCount, CountType::EQUAL, -1);

    // Enqueue data check
    EnqueueDataBuffersCheck(maxCount);

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
  }

  DeallocateBuffers(memType, registFlag);
}
#endif

INSTANTIATE_TEST_SUITE_P(
    AllToAllvDynamicSplitTestInstance,
    AllToAllvDynamicSplitTestSuite,
    ::testing::Values(
        // memType, maxCount, registFlag
        std::make_tuple(kMemCudaMalloc, 1048576, true),
        std::make_tuple(kMemCudaMalloc, 1048576, false),
        std::make_tuple(kMemNcclMemAlloc, 1048576, true),
        std::make_tuple(kMemNcclMemAlloc, 1048576, false),
        std::make_tuple(
            kMemCudaMalloc,
            dceil(CTRAN_MIN_REGISTRATION_SIZE, commTypeSize(commInt)),
            true),
        std::make_tuple(
            kMemCudaMalloc,
            dceil(CTRAN_MIN_REGISTRATION_SIZE, commTypeSize(commInt)),
            false),
        std::make_tuple(
            kMemCudaMalloc,
            std::numeric_limits<size_t>::max(),
            true)));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranDistTestEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
