// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>
#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <comm.h>
#include <nccl.h>
#include "checks.h"
#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class AllReduceSparseBlockTest : public ::testing::Test {
 public:
  AllReduceSparseBlockTest() = default;
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

  template <typename T>
  int checkChunkValue(T* buf, size_t count, std::vector<T>& expectedVals) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(), buf, count * sizeof(T), cudaMemcpyDefault));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print failing location
    for (auto i = 0; i < count; ++i) {
      if (observedVals[i] != expectedVals[i]) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              this->globalRank,
              i,
              observedVals[i],
              expectedVals[i]);
        }
        errs++;
      }
    }
    return errs;
  }

  void run(
      std::unordered_map<int, std::vector<int64_t>>& allRankRecvIndices,
      const size_t blockLen,
      const size_t recvCount) {
#ifdef NCCL_ALLREDUCE_SPARSE_BLOCK_SUPPORTED
    int *sendBuf = nullptr, *recvBuf = nullptr;
    std::vector<int> expVals(recvCount, 0);
    size_t blockCount = allRankRecvIndices.count(comm->rank);

    CUDACHECK_TEST(cudaMalloc(&sendBuf, blockCount * blockLen * sizeof(int)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, recvCount * sizeof(int)));

    int64_t* recvIndices = nullptr;
    CUDACHECK_TEST(cudaMalloc(&recvIndices, 4 * sizeof(int64_t)));
    cudaMemcpy(
        recvIndices,
        allRankRecvIndices[comm->rank].data(),
        allRankRecvIndices[comm->rank].size() * sizeof(int64_t),
        cudaMemcpyDefault);

    // Reduce locally to get expected values
    for (auto& it : allRankRecvIndices) {
      auto rank = it.first;
      auto& indices = it.second;
      for (int b = 0; b < blockCount; b++) {
        for (int i = 0; i < blockLen; i++) {
          expVals[indices[b] + i] += rank * 100 + b + 1;
        }
      }
    }

    for (int b = 0; b < blockCount; b++) {
      int expectedVal = comm->rank * 100 + b + 1;
      assignChunkValue(sendBuf + b * blockLen, blockLen, expectedVal);
    }
    assignChunkValue(recvBuf, recvCount, -1);

    for (int i = 0; i < 5; i++) {
      auto res = ncclAllReduceSparseBlock(
          sendBuf,
          recvIndices,
          blockCount,
          blockLen,
          recvBuf,
          recvCount,
          ncclInt,
          ncclSum,
          comm,
          stream);
      ASSERT_EQ(res, ncclSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    // Check results
    int errs = checkChunkValue(recvBuf, recvCount, expVals);
    EXPECT_EQ(errs, 0) << "rank " << comm->rank << " checked recvBuf with "
                       << errs << " errors";

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
    CUDACHECK_TEST(cudaFree(recvIndices));
#endif
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(AllReduceSparseBlockTest, OverlapReceiveIndices) {
  const size_t recvCount = 1048576;
  const size_t blockLen = 48;
  const size_t blockStride = 64;
  const size_t blockCount = 4;
  const size_t offsetShift = 32;

  // rank 0: [0:47], [64:111], [128:175], [192:239]
  // rank 1: [32:79], [96:143], [160:207], [224:271]
  // ...
  // rank N-1: [32*(N-1):32*(N-1)+blockLen-1],
  //           [32*(N-1)+blockStride:32*(N-1)+blockStride+blockLen-1], ...
  std::unordered_map<int, std::vector<int64_t>> allRankRecvIndices;

  for (int r = 0; r < comm->nRanks; r++) {
    allRankRecvIndices[r].resize(blockCount);
    for (int b = 0; b < blockCount; b++) {
      allRankRecvIndices[r][b] = r * offsetShift + b * blockStride;
    }
  }

  run(allRankRecvIndices, blockLen, recvCount);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
