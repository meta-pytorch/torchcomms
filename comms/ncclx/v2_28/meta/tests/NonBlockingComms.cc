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

class NonBlockingCommsTest : public ::testing::Test {
 public:
  NonBlockingCommsTest() = default;

  ncclResult_t WaitForCompletion() {
    ncclResult_t commStatus;

    if (this->comm) {
      commStatus = ncclInProgress;
      do {
        ncclResult_t res = ncclCommGetAsyncError(this->comm, &commStatus);
        EXPECT_EQ(res, ncclSuccess);

        if (commStatus != ncclInProgress) {
          break;
        }

        sched_yield();
      } while (commStatus == ncclInProgress);
    }

    if (this->splitComm && commStatus == ncclSuccess) {
      commStatus = ncclInProgress;
      do {
        ncclResult_t res = ncclCommGetAsyncError(this->splitComm, &commStatus);
        EXPECT_EQ(res, ncclSuccess);

        if (commStatus != ncclInProgress) {
          break;
        }

        sched_yield();
      } while (commStatus == ncclInProgress);
    }

    return commStatus;
  }

  void SetUp() override {
    ncclResult_t res;

    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;

    this->comm = createNcclComm(
        this->globalRank, this->numRanks, this->localRank, false, &config);
    res = WaitForCompletion();
    EXPECT_EQ(res, ncclSuccess);

    ncclCommSplit(this->comm, 0, this->globalRank, &this->splitComm, &config);
    res = WaitForCompletion();
    EXPECT_EQ(res, ncclSuccess);

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->splitComm));
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
  }

  void AllocateBuffers(size_t count) {
    CUDACHECK_TEST(cudaMalloc(&sendbuff, count * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvbuff, count * sizeof(int)));
  }

  void InitializeBuffers(size_t count) {
    int* tmpbuff;
    CUDACHECK_TEST(
        cudaHostAlloc(&tmpbuff, count * sizeof(int), cudaHostAllocDefault));

    for (size_t i = 0; i < count; i++) {
      tmpbuff[i] = (int)i;
    }
    CUDACHECK_TEST(
        cudaMemcpy(sendbuff, tmpbuff, count * sizeof(int), cudaMemcpyDefault));

    for (size_t i = 0; i < count; i++) {
      tmpbuff[i] = -1;
    }
    CUDACHECK_TEST(
        cudaMemcpy(recvbuff, tmpbuff, count * sizeof(int), cudaMemcpyDefault));

    CUDACHECK_TEST(cudaFreeHost(tmpbuff));
  }

  void DeallocateBuffers() {
    CUDACHECK_TEST(cudaFree(sendbuff));
    CUDACHECK_TEST(cudaFree(recvbuff));
  }

  void CheckBuffers(size_t count) {
    int* tmpbuff;
    CUDACHECK_TEST(
        cudaHostAlloc(&tmpbuff, count * sizeof(int), cudaHostAllocDefault));

    CUDACHECK_TEST(
        cudaMemcpy(tmpbuff, recvbuff, count * sizeof(int), cudaMemcpyDefault));

    for (size_t i = 0; i < count; i++) {
      if (tmpbuff[i] != i) {
        printf("recvbuff[%lu]=%d, expected=%d\n", i, tmpbuff[i], (int)i);
        break;
      }
    }
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  void* sendbuff{nullptr};
  void* recvbuff{nullptr};
  ncclComm_t comm{nullptr};
  ncclComm_t splitComm{nullptr};
  cudaStream_t stream{};
};

TEST_F(NonBlockingCommsTest, Simple) {
  ncclResult_t res;
  constexpr size_t count = 100;

  AllocateBuffers(count);
  InitializeBuffers(count);

  ncclBroadcast(sendbuff, recvbuff, count, ncclInt, 0, comm, stream);
  res = WaitForCompletion();
  EXPECT_EQ(res, ncclSuccess);

  ncclBroadcast(sendbuff, recvbuff, count, ncclInt, 0, splitComm, stream);
  res = WaitForCompletion();
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECKIGNORE(cudaStreamSynchronize(stream));

  CheckBuffers(count);
  DeallocateBuffers();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
