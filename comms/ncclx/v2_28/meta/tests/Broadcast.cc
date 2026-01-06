// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>

#include "checks.h" // NOLINT
#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/colltrace/CollTrace.h"

class BroadcastTestCommon : public ::testing::Test {
 public:
  BroadcastTestCommon() = default;

  enum class MemType {
    CUDA_MALLOC,
    CUDA_HOST_ALLOC,
    MALLOC,
    CUDA_HOST_REGISTER,
  };

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
  }

  void AllocateBuffers(MemType memType, size_t count) {
    if (memType == MemType::CUDA_MALLOC) {
      CUDACHECK_TEST(cudaMalloc(&sendbuff, count * sizeof(int)));
      CUDACHECK_TEST(cudaMalloc(&recvbuff, count * sizeof(int)));
    } else if (memType == MemType::CUDA_HOST_ALLOC) {
      CUDACHECK_TEST(
          cudaHostAlloc(&sendbuff, count * sizeof(int), cudaHostAllocDefault));
      CUDACHECK_TEST(
          cudaHostAlloc(&recvbuff, count * sizeof(int), cudaHostAllocDefault));
    } else if (memType == MemType::MALLOC) {
      sendbuff = malloc(count * sizeof(int));
      recvbuff = malloc(count * sizeof(int));
    } else if (memType == MemType::CUDA_HOST_REGISTER) {
      sendbuff = malloc(count * sizeof(int));
      recvbuff = malloc(count * sizeof(int));
      CUDACHECK_TEST(cudaHostRegister(
          sendbuff, count * sizeof(int), cudaHostRegisterDefault));
      CUDACHECK_TEST(cudaHostRegister(
          recvbuff, count * sizeof(int), cudaHostRegisterDefault));
    }
  }

  void InitializeBuffers(MemType memType, size_t count) {
    if (memType == MemType::CUDA_MALLOC) {
      int* tmpbuff;
      CUDACHECK_TEST(
          cudaHostAlloc(&tmpbuff, count * sizeof(int), cudaHostAllocDefault));

      for (size_t i = 0; i < count; i++) {
        tmpbuff[i] = (int)i;
      }
      CUDACHECK_TEST(cudaMemcpy(
          sendbuff, tmpbuff, count * sizeof(int), cudaMemcpyDefault));

      for (size_t i = 0; i < count; i++) {
        tmpbuff[i] = -1;
      }
      CUDACHECK_TEST(cudaMemcpy(
          recvbuff, tmpbuff, count * sizeof(int), cudaMemcpyDefault));

      CUDACHECK_TEST(cudaFreeHost(tmpbuff));
    } else if (
        memType == MemType::CUDA_HOST_ALLOC || memType == MemType::MALLOC ||
        memType == MemType::CUDA_HOST_REGISTER) {
      for (size_t i = 0; i < count; i++) {
        ((int*)sendbuff)[i] = (int)i;
        ((int*)recvbuff)[i] = -1;
      }
    }
  }

  void DeallocateBuffers(MemType memType) {
    if (memType == MemType::CUDA_MALLOC) {
      CUDACHECK_TEST(cudaFree(sendbuff));
      CUDACHECK_TEST(cudaFree(recvbuff));
    } else if (memType == MemType::CUDA_HOST_ALLOC) {
      CUDACHECK_TEST(cudaFreeHost(sendbuff));
      CUDACHECK_TEST(cudaFreeHost(recvbuff));
    } else if (memType == MemType::MALLOC) {
      free(sendbuff);
      free(recvbuff);
    } else if (memType == MemType::CUDA_HOST_REGISTER) {
      CUDACHECK_TEST(cudaHostUnregister(sendbuff));
      CUDACHECK_TEST(cudaHostUnregister(recvbuff));
      free(sendbuff);
      free(recvbuff);
    }
  }

  void CheckBuffers(MemType memType, size_t count) {
    if (memType == MemType::CUDA_MALLOC) {
      int* tmpbuff;
      CUDACHECK_TEST(
          cudaHostAlloc(&tmpbuff, count * sizeof(int), cudaHostAllocDefault));

      CUDACHECK_TEST(cudaMemcpy(
          tmpbuff, recvbuff, count * sizeof(int), cudaMemcpyDefault));

      for (size_t i = 0; i < count; i++) {
        if (tmpbuff[i] != i) {
          printf("recvbuff[%lu]=%d, expected=%d\n", i, tmpbuff[i], (int)i);
          break;
        }
      }
    } else if (
        memType == MemType::CUDA_HOST_ALLOC || memType == MemType::MALLOC ||
        memType == MemType::CUDA_HOST_REGISTER) {
      for (size_t i = 0; i < count; i++) {
        int val = ((int*)recvbuff)[i];
        if (val != i) {
          printf("recvbuff[%lu]=%d, expected=%d\n", i, val, (int)i);
          break;
        }
      }
    }
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  void* sendbuff{nullptr};
  void* recvbuff{nullptr};
  ncclComm_t comm;
  cudaStream_t stream;
};

class BroadcastTestSuite
    : public BroadcastTestCommon,
      public ::testing::WithParamInterface<
          std::tuple<BroadcastTestCommon::MemType, size_t>> {};

TEST_P(BroadcastTestSuite, Simple) {
  const auto& [memType, count] = GetParam();

  AllocateBuffers(memType, count);
  InitializeBuffers(memType, count);

  auto res = ncclBroadcast(sendbuff, recvbuff, count, ncclInt, 0, comm, stream);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  CheckBuffers(memType, count);
  DeallocateBuffers(memType);
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastTestInstance,
    BroadcastTestSuite,
    ::testing::Values(
        std::make_tuple(BroadcastTestCommon::MemType::CUDA_MALLOC, 1),
        std::make_tuple(BroadcastTestCommon::MemType::CUDA_MALLOC, 1024 * 1024),
        std::make_tuple(
            BroadcastTestCommon::MemType::CUDA_MALLOC,
            64 * 1024 * 1024),
        std::make_tuple(BroadcastTestCommon::MemType::CUDA_MALLOC, 23),
        std::make_tuple(
            BroadcastTestCommon::MemType::CUDA_MALLOC,
            1024 * 1024 + 23),
        std::make_tuple(BroadcastTestCommon::MemType::CUDA_HOST_ALLOC, 1),
        std::make_tuple(
            BroadcastTestCommon::MemType::CUDA_HOST_ALLOC,
            1024 * 1024),
        std::make_tuple(
            BroadcastTestCommon::MemType::CUDA_HOST_ALLOC,
            64 * 1024 * 1024),
        std::make_tuple(BroadcastTestCommon::MemType::CUDA_HOST_ALLOC, 23),
        std::make_tuple(
            BroadcastTestCommon::MemType::CUDA_HOST_ALLOC,
            1024 * 1024 + 23),
        std::make_tuple(BroadcastTestCommon::MemType::MALLOC, 1),
        std::make_tuple(BroadcastTestCommon::MemType::MALLOC, 1024 * 1024),
        std::make_tuple(BroadcastTestCommon::MemType::MALLOC, 64 * 1024 * 1024),
        std::make_tuple(BroadcastTestCommon::MemType::MALLOC, 23),
        std::make_tuple(BroadcastTestCommon::MemType::MALLOC, 1024 * 1024 + 23),
        std::make_tuple(BroadcastTestCommon::MemType::CUDA_HOST_REGISTER, 1),
        std::make_tuple(
            BroadcastTestCommon::MemType::CUDA_HOST_REGISTER,
            1024 * 1024),
        std::make_tuple(
            BroadcastTestCommon::MemType::CUDA_HOST_REGISTER,
            64 * 1024 * 1024),
        std::make_tuple(BroadcastTestCommon::MemType::CUDA_HOST_REGISTER, 23),
        std::make_tuple(
            BroadcastTestCommon::MemType::CUDA_HOST_REGISTER,
            1024 * 1024 + 23)));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
