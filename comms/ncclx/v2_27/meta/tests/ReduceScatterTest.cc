// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <fmt/core.h>
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
#include "comms/utils/cvars/nccl_cvars.h"

#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"

class ReduceScatterTest : public NcclxBaseTest {
 public:
  ReduceScatterTest() = default;
  void SetUp() override {
    NcclxBaseTest::SetUp();
    comm = createNcclComm(globalRank, numRanks, localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NcclxBaseTest::TearDown();
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T val) {
    std::vector<T> expectedVals(count, val);
    CUDACHECK_TEST(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
  }

  template <typename T>
  int checkChunkValue(T* buf, size_t count, T val) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print failing location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != val) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d/%ld] = %d, expectedVal = %d, at buf %p (offset 0x%lx)\n",
              globalRank,
              i,
              count,
              observedVals[i],
              val,
              buf,
              sizeof(T) * i);
        }
        errs++;
      }
    }
    return errs;
  }

 protected:
  ncclComm_t comm;
  cudaStream_t stream;
};

class ReduceScatterTestParam : public ReduceScatterTest,
                               public ::testing::WithParamInterface<std::tuple<
                                   enum NCCL_REDUCESCATTER_ALGO,
                                   bool,
                                   bool,
                                   MemAllocType,
                                   size_t,
                                   bool>> {};

TEST_P(ReduceScatterTestParam, Test) {
  const auto& [algo, inplace, registFlag, memType, count, reportPerf] =
      GetParam();
  auto envGuard = EnvRAII(NCCL_REDUCESCATTER_ALGO, algo);

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

#if !defined TEST_ENABLE_CTRAN
  if (algo != NCCL_REDUCESCATTER_ALGO::orig) {
    GTEST_SKIP() << "Ctran is disabled, skip test";
  }
#endif

  if (algo != NCCL_REDUCESCATTER_ALGO::orig &&
      !ctranReduceScatterSupport(comm->ctranComm_.get())) {
    GTEST_SKIP() << "Ctran algorithm is not supported, skip test";
  }

  if (memType == kMemCudaMalloc && algo != NCCL_REDUCESCATTER_ALGO::orig &&
      comm->ctranComm_->statex_->nLocalRanks() > 1) {
    GTEST_SKIP()
        << "Ctran does not support cudaMalloc-ed buffer with nLocalRanks > 1, skip test";
  }

  // Create and register buffers. If inplace, we use the same buffer for send
  // and recv.
  int *sendBuf = nullptr, *recvBuf = nullptr;
  void *sendHandle = nullptr, *recvHandle = nullptr;
  size_t allocSize = (count * numRanks * sizeof(int));
  allocSize = allocSize < 8192 ? 8192 : allocSize;

  if (memType == kMemCudaMalloc) {
    CUDACHECK_TEST(cudaMalloc(&sendBuf, allocSize));
  } else {
    void* buf = nullptr;
    NCCLCHECK_TEST(ncclMemAlloc(&buf, allocSize));
    sendBuf = reinterpret_cast<int*>(buf);
  }

  if (inplace) {
    recvBuf = sendBuf + count * globalRank;
  } else {
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaMalloc(&recvBuf, allocSize));
    } else {
      void* buf = nullptr;
      NCCLCHECK_TEST(ncclMemAlloc(&buf, allocSize));
      recvBuf = reinterpret_cast<int*>(buf);
    }
  }

  assignChunkValue(recvBuf, count, -1);
  for (int r = 0; r < numRanks; r++) {
    int val = globalRank * numRanks + r;
    assignChunkValue(sendBuf + r * count, count, val);
  }

  if (registFlag) {
    NCCLCHECK_TEST(ncclCommRegister(comm, sendBuf, allocSize, &sendHandle));
    if (!inplace) {
      NCCLCHECK_TEST(ncclCommRegister(comm, recvBuf, allocSize, &recvHandle));
    }
  }

  // Run communication
  auto res = ncclReduceScatter(
      sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  // CUDACHECK_TEST(cudaStreamSynchronize(stream));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Check received chunk
  int expectedVal = 0;
  for (int r = 0; r < numRanks; r++) {
    expectedVal += (r * numRanks + globalRank);
  }
  int errs = checkChunkValue(recvBuf, count, expectedVal);
  EXPECT_EQ(errs, 0) << "Rank " << globalRank << " checked chunk at " << recvBuf
                     << " with " << errs << " errors with inplace " << inplace;

  if (reportPerf) {
    constexpr int warm = 10, iters = 100;
    for (int i = 0; i < warm; i++) {
      NCCLCHECK_TEST(ncclReduceScatter(
          sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
      CUDACHECK_TEST(cudaDeviceSynchronize());
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
      NCCLCHECK_TEST(ncclReduceScatter(
          sendBuf, recvBuf, count, ncclInt, ncclSum, comm, stream));
      CUDACHECK_TEST(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (globalRank == 0) {
      printf(
          "Rank %d, nRanks %d, algo %d, count %ld, nbytes %ld, time %ld us\n",
          globalRank,
          numRanks,
          static_cast<int>(algo),
          count,
          count * sizeof(int),
          duration.count() / iters);
    }
  }

  // Deregister and free buffers
  if (registFlag) {
    NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
    if (!inplace) {
      NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
    }
  }

  if (memType == kMemCudaMalloc) {
    CUDACHECK_TEST(cudaFree(sendBuf));
  } else {
    NCCLCHECK_TEST(ncclMemFree(sendBuf));
  }
  if (!inplace) {
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaFree(recvBuf));
    } else {
      NCCLCHECK_TEST(ncclMemFree(recvBuf));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterTestInstance,
    ReduceScatterTestParam,
    ::testing::Values(
        // algo, inplace, registFlag, memType, count
        // inplace cudaMalloc with each of algorithms
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::orig,
            true,
            true,
            kMemCudaMalloc,
            8192,
            false),
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::ctran,
            true,
            true,
            kMemCudaMalloc,
            8192,
            false),
        // inplace cumem
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::orig,
            true,
            true,
            kMemNcclMemAlloc,
            8192,
            false),
        // out-place
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::ctran,
            false,
            true,
            kMemNcclMemAlloc,
            33554432,
            false),
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::ctran,
            false,
            true,
            kMemNcclMemAlloc,
            8192,
            true),
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::ctran,
            false,
            true,
            kMemNcclMemAlloc,
            65536,
            true),
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::ctran,
            false,
            true,
            kMemNcclMemAlloc,
            8388608,
            true),
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::ctran,
            false,
            true,
            kMemNcclMemAlloc,
            268435456,
            true),
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::ctrhd,
            false,
            true,
            kMemNcclMemAlloc,
            8192,
            true),
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::ctrhd,
            false,
            true,
            kMemNcclMemAlloc,
            1,
            true),
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::ctring,
            false,
            true,
            kMemNcclMemAlloc,
            8192,
            true),
        std::make_tuple(
            NCCL_REDUCESCATTER_ALGO::ctring,
            false,
            true,
            kMemNcclMemAlloc,
            1,
            true)),
    [&](const testing::TestParamInfo<ReduceScatterTestParam::ParamType>& info) {
      return fmt::format(
          "{}_{}_{}_{}_{}count_{}",
          reduceScatterAlgoName(std::get<0>(info.param)),
          (std::get<1>(info.param)) ? "Inplace" : "OutOfPlace",
          (std::get<2>(info.param)) ? "Regist" : "NoRegist",
          testMemAllocTypeToStr(std::get<3>(info.param)),
          std::to_string(std::get<4>(info.param)),
          (std::get<5>(info.param)) ? "Perf" : "NoPerf");
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
