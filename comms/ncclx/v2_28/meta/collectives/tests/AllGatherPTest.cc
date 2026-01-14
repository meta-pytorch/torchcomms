// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <cstddef>

#include <comm.h>
#include "comms/utils/cvars/nccl_cvars.h"
#include "nccl.h"

#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class AllGatherPTest : public ::testing::Test {
 public:
  AllGatherPTest() = default;
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
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

  void* prepareBuf(size_t bufSize, MemAllocType memType) {
    void* buf = nullptr;
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaMalloc(&buf, bufSize));
    } else {
      NCCLCHECK_TEST(ncclMemAlloc(&buf, bufSize));
    }
    return buf;
  }

  void releaseBuf(void* buf, MemAllocType memType) {
    if (memType == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaFree(buf));
    } else {
      ncclMemFree(buf);
    }
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;
};

class AllgatherPTestParam : public AllGatherPTest,
                            public ::testing::WithParamInterface<std::tuple<
                                enum NCCL_ALLGATHER_P_ALGO,
                                size_t,
                                TestInPlaceType,
                                MemAllocType>> {};

TEST_P(AllgatherPTestParam, Test) {
  const auto& [algo, count, inplace, memType] = GetParam();
  auto envGuard = EnvRAII(NCCL_ALLGATHER_P_ALGO, algo);

  if (memType == kMemNcclMemAlloc && ncclIsCuMemSupported() == false) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  // Create and register buffers. If inplace, we use the same buffer for send
  // and recv.
  int *sendBuf = nullptr, *recvBuf = nullptr;
  void *sendHandle = nullptr, *recvHandle = nullptr;
  size_t allocCount = std::max(count, (size_t)CTRAN_MIN_REGISTRATION_SIZE);

  recvBuf = reinterpret_cast<int*>(
      prepareBuf(allocCount * numRanks * sizeof(int), memType));
  assignChunkValue(recvBuf, count * numRanks, -1);

  NCCLCHECK_TEST(ncclCommRegister(
      comm, recvBuf, allocCount * numRanks * sizeof(int), &recvHandle));

  if (inplace) {
    sendBuf = recvBuf + count * globalRank;
  } else {
    sendBuf =
        reinterpret_cast<int*>(prepareBuf(allocCount * sizeof(int), memType));
    NCCLCHECK_TEST(
        ncclCommRegister(comm, sendBuf, allocCount * sizeof(int), &sendHandle));
  }

  void* request;
  ncclx::Hints hints;
  const auto initMaxRecvCount = count * numRanks * sizeof(int);
  auto res = ncclx::allGatherInit(
      recvBuf, initMaxRecvCount, hints, ncclInt8, comm, stream, &request);
  ASSERT_EQ(res, ncclSuccess);

  // Run communication
  for (int i = 0; i < 5; i++) {
    int sendVal = globalRank + i * 10;
    assignChunkValue(sendBuf, count, sendVal);

    res = ncclx::allGatherExec(sendBuf, count, ncclInt, request);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    // Check each received chunk
    for (int r = 0; r < numRanks; r++) {
      int expectedVal = r + i * 10;
      int errs = checkChunkValue(recvBuf + r * count, count, expectedVal);
      EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << r
                         << " at " << recvBuf + r * count << " with " << errs
                         << " errors";
    }
  }

  ncclx::pFree(request);

  if (!inplace) {
    NCCLCHECK_TEST(ncclCommDeregister(comm, sendHandle));
    releaseBuf(sendBuf, memType);
  }
  NCCLCHECK_TEST(ncclCommDeregister(comm, recvHandle));
  releaseBuf(recvBuf, memType);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherPTestInstance,
    AllgatherPTestParam,
    ::testing::Values(
        // algo, inplace, memType
        std::make_tuple(
            NCCL_ALLGATHER_P_ALGO::ctdirect,
            1048576,
            kTestOutOfPlace,
            kMemNcclMemAlloc),
        std::make_tuple(
            NCCL_ALLGATHER_P_ALGO::ctpipeline,
            1048576,
            kTestInPlace,
            kMemNcclMemAlloc)));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  return RUN_ALL_TESTS();
}
