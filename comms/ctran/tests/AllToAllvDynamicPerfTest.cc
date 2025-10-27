// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/testinfra/tests_common.cuh"

__global__ void halfCountKernel(size_t* sendCounts, int numRanks);

class AllToAllvDynamicPerfTestCommon : public NcclxBaseTest {
 public:
  AllToAllvDynamicPerfTestCommon() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0); // enable ctran

    NcclxBaseTest::SetUp();
    this->comm = createNcclComm(
        this->globalRank,
        this->numRanks,
        this->localRank,
        false,
        nullptr,
        server.get());

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    CUDACHECK_TEST(cudaMalloc(&scounts, numRanks * sizeof(size_t)));
    CUDACHECK_TEST(cudaMalloc(&actualRcounts, numRanks * sizeof(size_t)));
    CUDACHECK_TEST(cudaHostAlloc(
        &scountsHost, numRanks * sizeof(size_t), cudaHostAllocDefault));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaFree(scounts));
    CUDACHECK_TEST(cudaFree(actualRcounts));
    CUDACHECK_TEST(cudaFreeHost(scountsHost));

    NCCLCHECK_TEST(ncclCommDestroy(comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NcclxBaseTest::TearDown();
  }

  void AllocateBuffers(MemAllocType memType, size_t maxCount, bool registFlag) {
    maxCountBuff = maxCount;
    if (maxCountBuff * sizeof(int) < CTRAN_MIN_REGISTRATION_SIZE) {
      maxCountBuff = 2 * CTRAN_MIN_REGISTRATION_SIZE / sizeof(int);
    }

    // Create and register buffers.
    for (int i = 0; i < numRanks; i++) {
      void* buf;

      if (memType == kMemCudaMalloc) {
        CUDACHECK_TEST(cudaMalloc(&buf, maxCountBuff * sizeof(int)));
      } else {
        NCCLCHECK_TEST(ncclMemAlloc(&buf, maxCountBuff * sizeof(int)));
      }
      sendbuffs.push_back(buf);

      if (memType == kMemCudaMalloc) {
        CUDACHECK_TEST(cudaMalloc(&buf, maxCountBuff * sizeof(int)));
      } else {
        NCCLCHECK_TEST(ncclMemAlloc(&buf, maxCountBuff * sizeof(int)));
      }
      recvbuffs.push_back(buf);
    }

    // Allocate buffer for AllToAllv
    CUDACHECK_TEST(
        cudaMalloc(&sendBufContig, numRanks * maxCountBuff * sizeof(int)));
    CUDACHECK_TEST(
        cudaMalloc(&recvBufContig, numRanks * maxCountBuff * sizeof(int)));

    if (registFlag) {
      void* hdl;

      for (auto buf : sendbuffs) {
        NCCLCHECK_TEST(
            ncclCommRegister(comm, buf, maxCountBuff * sizeof(int), &hdl));
        sendhdls.push_back(hdl);
      }

      for (auto buf : recvbuffs) {
        NCCLCHECK_TEST(
            ncclCommRegister(comm, buf, maxCountBuff * sizeof(int), &hdl));
        recvhdls.push_back(hdl);
      }

      NCCLCHECK_TEST(ncclCommRegister(
          comm, sendBufContig, numRanks * maxCountBuff * sizeof(int), &hdl));
      sendhdls.push_back(hdl);

      NCCLCHECK_TEST(ncclCommRegister(
          comm, recvBufContig, numRanks * maxCountBuff * sizeof(int), &hdl));
      recvhdls.push_back(hdl);
    }
  }

  void DeallocateBuffers(MemAllocType memType, bool registFlag) {
    // Deregister and free buffers
    if (registFlag) {
      for (auto h : sendhdls) {
        NCCLCHECK_TEST(ncclCommDeregister(comm, h));
      }
      for (auto h : recvhdls) {
        NCCLCHECK_TEST(ncclCommDeregister(comm, h));
      }
    }

    for (auto buf : sendbuffs) {
      if (memType == kMemCudaMalloc) {
        CUDACHECK_TEST(cudaFree(buf));
      } else {
        NCCLCHECK_TEST(ncclMemFree(buf));
      }
    }
    for (auto buf : recvbuffs) {
      if (memType == kMemCudaMalloc) {
        CUDACHECK_TEST(cudaFree(buf));
      } else {
        NCCLCHECK_TEST(ncclMemFree(buf));
      }
    }

    CUDACHECK_TEST(cudaFree(sendBufContig));
    CUDACHECK_TEST(cudaFree(recvBufContig));
  }

  void InitializeBuffers(size_t maxCount) {
    int* hostBuf;
    CUDACHECK_TEST(
        cudaHostAlloc(&hostBuf, maxCount * sizeof(int), cudaHostAllocDefault));

    for (int r = 0; r < numRanks; r++) {
      for (int i = 0; i < maxCount; i++) {
        hostBuf[i] = r + i;
      }
      CUDACHECK_TEST(cudaMemcpy(
          sendbuffs[r], hostBuf, maxCount * sizeof(int), cudaMemcpyDefault));
      CUDACHECK_TEST(cudaMemcpy(
          sendBufContig + r * maxCount,
          hostBuf,
          maxCount * sizeof(int),
          cudaMemcpyDefault));
    }

    for (int i = 0; i < maxCount; i++) {
      hostBuf[i] = -1;
    }
    for (int r = 0; r < numRanks; r++) {
      CUDACHECK_TEST(cudaMemcpy(
          recvbuffs[r], hostBuf, maxCount * sizeof(int), cudaMemcpyDefault));
      CUDACHECK_TEST(cudaMemcpy(
          recvBufContig + r * maxCount,
          hostBuf,
          maxCount * sizeof(int),
          cudaMemcpyDefault));
    }

    CUDACHECK_TEST(cudaFreeHost(hostBuf));
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
  ncclComm_t comm;
  cudaStream_t stream;
  std::vector<void*> sendbuffs;
  std::vector<void*> recvbuffs;
  int* sendBufContig{nullptr};
  int* recvBufContig{nullptr};
  std::vector<void*> sendhdls, recvhdls;
  std::vector<size_t> maxRcounts;
  size_t* scounts{nullptr};
  size_t* scountsHost{nullptr};
  size_t* actualRcounts{nullptr};
  int warmupTime = 5, totalTimes = 1000;
  ncclx::Hints hints;
  size_t maxCountBuff{0};
  size_t maxSendcount{0};
  size_t maxRecvcount{0};
};

class AllToAllvDynamicPerfTestSuite
    : public AllToAllvDynamicPerfTestCommon,
      public ::testing::WithParamInterface<
          std::tuple<MemAllocType, size_t, bool>> {};

/********************** AllToAllvDynamicPerfTestSuite
 * ****************************/
/* Collect the performance result to compare w/ and w/o the dynamic change */
/**********************************************************************/

TEST_P(AllToAllvDynamicPerfTestSuite, AllToAllvBaseline) {
  const auto& [memType, maxCount, registFlag] = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  AllocateBuffers(memType, maxCount, registFlag);
  InitializeBuffers(maxCount);

  std::vector<size_t> sendDispls(numRanks);
  std::vector<size_t> recvDispls(numRanks);
  sendDispls[0] = 0;
  recvDispls[0] = 0;
  for (int i = 1; i < numRanks; i++) {
    sendDispls[i] = sendDispls[i - 1] + maxCount;
    recvDispls[i] = recvDispls[i - 1] + maxCount;
  }

  float elapsedTime = 0;
  cudaEvent_t eventStart, eventEnd;
  CUDACHECK_TEST(cudaEventCreate(&eventStart));
  CUDACHECK_TEST(cudaEventCreate(&eventEnd));

  if (globalRank == 0) {
    std::cout << "Avg elapsed time for AllToAllv: " << std::endl;
  }
  for (size_t count = 1; count <= maxCount; count *= 2) {
    for (int i = 0; i < numRanks; i++) {
      scountsHost[i] = count;
    }
    int totalTimes_ = totalTimes;
    if (count < maxCount / 1024) {
      totalTimes_ = totalTimes_ * 10;
    }
    for (int i = 0; i < totalTimes_ + warmupTime; i++) {
      if (i == warmupTime) {
        CUDACHECK_TEST(cudaEventRecord(eventStart, stream));
      }
      COMMCHECK_TEST(ctranAllToAllv(
          sendBufContig,
          scountsHost,
          sendDispls.data(),
          recvBufContig,
          scountsHost,
          recvDispls.data(),
          commInt,
          comm->ctranComm_.get(),
          stream));
    }
    CUDACHECK_TEST(cudaEventRecord(eventEnd, stream));
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    CUDACHECK_TEST(cudaEventElapsedTime(&elapsedTime, eventStart, eventEnd));

    if (globalRank == 0) {
      std::cout << count << " " << elapsedTime / (totalTimes_) << std::endl;
    }
  }

  DeallocateBuffers(memType, registFlag);
  CUDACHECK_TEST(cudaEventDestroy(eventStart));
  CUDACHECK_TEST(cudaEventDestroy(eventEnd));
}

TEST_P(AllToAllvDynamicPerfTestSuite, AllToAllvDynamicUnchangedEqualCounts) {
  const auto& [memType, maxCount, registFlag] = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  AllocateBuffers(memType, maxCount, registFlag);
  InitializeBuffers(maxCount);
  setHints();

  float elapsedTime = 0;
  cudaEvent_t eventStart, eventEnd;
  CUDACHECK_TEST(cudaEventCreate(&eventStart));
  CUDACHECK_TEST(cudaEventCreate(&eventEnd));

  // Run communication
  if (globalRank == 0) {
    std::cout << "Avg elapsed time for AllToAllv Dynamic (Unchanged): "
              << std::endl;
  }
  for (size_t count = 1; count <= maxCount; count *= 2) {
    for (int i = 0; i < numRanks; i++) {
      scountsHost[i] = count;
    }

    CUDACHECK_TEST(cudaMemcpyAsync(
        scounts,
        scountsHost,
        numRanks * sizeof(size_t),
        cudaMemcpyDefault,
        stream));

    maxSendcount = maxCountBuff;
    maxRecvcount = maxCountBuff;

    int totalTimes_ = totalTimes;
    if (count < maxCount / 1024) {
      totalTimes_ = totalTimes_ * 10;
    }
    for (int i = 0; i < totalTimes_ + warmupTime; i++) {
      if (i == warmupTime) {
        CUDACHECK_TEST(cudaEventRecord(eventStart, stream));
      }
      NCCLCHECK_TEST(
          ncclx::alltoallvDynamic(
              sendbuffs.data(),
              scounts,
              recvbuffs.data(),
              maxSendcount,
              maxRecvcount,
              actualRcounts,
              hints,
              ncclInt,
              comm,
              stream));
    }
    CUDACHECK_TEST(cudaEventRecord(eventEnd, stream));
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    CUDACHECK_TEST(cudaEventElapsedTime(&elapsedTime, eventStart, eventEnd));

    if (globalRank == 0) {
      std::cout << count << " " << elapsedTime / (totalTimes_) << std::endl;
    }
  }

  DeallocateBuffers(memType, registFlag);
  CUDACHECK_TEST(cudaEventDestroy(eventStart));
  CUDACHECK_TEST(cudaEventDestroy(eventEnd));
}

TEST_P(AllToAllvDynamicPerfTestSuite, AllToAllvDynamicEqualChangedCounts) {
  const auto& [memType, maxCount, registFlag] = GetParam();

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  AllocateBuffers(memType, maxCount, registFlag);
  InitializeBuffers(maxCount);
  setHints();

  // std::vector<void*> kernelArgs;
  // void* tmp1 = reinterpret_cast<void *>(scounts);
  // kernelArgs.push_back((void*)&tmp1);
  // kernelArgs.push_back((void*)&numRanks);
  // CUDACHECK_TEST(cudaLaunchKernel((void *)halfCountKernel, 1, 1,
  // kernelArgs.data(), 0, stream));

  float elapsedTime = 0;
  cudaEvent_t eventStart, eventEnd;
  CUDACHECK_TEST(cudaEventCreate(&eventStart));
  CUDACHECK_TEST(cudaEventCreate(&eventEnd));

  // Run communication
  std::vector<int> changeFactors = {2, 4, 8};
  for (auto factor : changeFactors) {
    if (globalRank == 0) {
      std::cout << "Avg elapsed time for AllToAllv Dynamic (Changed 1/"
                << factor << "): " << std::endl;
    }
    for (size_t count = 1; count <= maxCount; count *= 2) {
      for (int i = 0; i < numRanks; i++) {
        scountsHost[i] = std::ceil(count / factor);
      }

      CUDACHECK_TEST(cudaMemcpyAsync(
          scounts,
          scountsHost,
          numRanks * sizeof(size_t),
          cudaMemcpyDefault,
          stream));

      maxSendcount = maxCountBuff;
      maxRecvcount = maxCountBuff;

      int totalTimes_ = totalTimes;
      if (count < maxCount / 1024) {
        totalTimes_ = totalTimes_ * 10;
      }

      for (int i = 0; i < totalTimes_ + warmupTime; i++) {
        if (i == warmupTime) {
          CUDACHECK_TEST(cudaEventRecord(eventStart, stream));
        }
        NCCLCHECK_TEST(
            ncclx::alltoallvDynamic(
                sendbuffs.data(),
                scounts,
                recvbuffs.data(),
                maxSendcount,
                maxRecvcount,
                actualRcounts,
                hints,
                ncclInt,
                comm,
                stream));
      }
      CUDACHECK_TEST(cudaEventRecord(eventEnd, stream));
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      CUDACHECK_TEST(cudaEventElapsedTime(&elapsedTime, eventStart, eventEnd));

      if (globalRank == 0) {
        std::cout << count << "  " << elapsedTime / (totalTimes_) << std::endl;
      }
    }
  }

  DeallocateBuffers(memType, registFlag);
  CUDACHECK_TEST(cudaEventDestroy(eventStart));
  CUDACHECK_TEST(cudaEventDestroy(eventEnd));
}

INSTANTIATE_TEST_SUITE_P(
    AllToAllvDynamicTestInstance,
    AllToAllvDynamicPerfTestSuite,
    ::testing::Values(
        // memType, maxCount, registFlag
        std::make_tuple(kMemCudaMalloc, 64 * 1048576, true)));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
