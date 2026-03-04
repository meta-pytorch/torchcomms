// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include "comms/testinfra/TestUtils.h"

// Helper class to prepare required arguments for ncclAllreduceSparseBlock
class AllreduceSparseBlockArgs {
 public:
  AllreduceSparseBlockArgs(
      size_t inBlockCount,
      size_t inBlockLength,
      int64_t* inRecvIndices,
      size_t inRecvCount)
      : blockCount(inBlockCount),
        blockLength(inBlockLength),
        recvCount(inRecvCount) {
    const int dev = 0;
    CUDACHECK_TEST(cudaSetDevice(dev));
    CUDACHECK_TEST(cudaStreamCreate(&stream));
    NCCLCHECK_TEST(ncclCommInitAll(&comm, 1, &dev));
    CUDACHECK_TEST(cudaMalloc(
        (void**)&sendBuff, sizeof(int32_t) * blockCount * blockLength));
    CUDACHECK_TEST(cudaMalloc((void**)&recvBuff, sizeof(int32_t) * recvCount));
    CUDACHECK_TEST(
        cudaMalloc((void**)&recvIndices, sizeof(int64_t) * blockCount));

    CUDACHECK_TEST(cudaMemcpy(
        recvIndices,
        inRecvIndices,
        sizeof(int64_t) * blockCount,
        cudaMemcpyHostToDevice));
  }

  ~AllreduceSparseBlockArgs() {
    CUDACHECK_TEST(cudaSetDevice(0));
    CUDACHECK_TEST(cudaFree(sendBuff));
    CUDACHECK_TEST(cudaFree(recvBuff));
    CUDACHECK_TEST(cudaFree(recvIndices));
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
  }

  // user passed arguments
  const size_t blockCount{0};
  const size_t blockLength{0};
  const size_t recvCount{0};

  // created arguments
  int32_t* sendBuff{nullptr};
  int32_t* recvBuff{nullptr};
  int64_t* recvIndices{nullptr};
  cudaStream_t stream;
  ncclComm_t comm;
};

class AllreduceSparseBlockArgCheckTest : public ::testing::Test {
 public:
  AllreduceSparseBlockArgCheckTest() {
    // Turn on pointer check before executing any argument check test.
    setenv("NCCL_CHECK_POINTERS", "1", 1);
    setenv("NCCL_IGNORE_TOPO_LOAD_FAILURE", "true", 1);
  }
};

TEST_F(AllreduceSparseBlockArgCheckTest, UnsupportedOp) {
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(), 4, recvIndices.data(), 256);
  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          args->sendBuff,
          args->recvIndices,
          args->blockCount,
          args->blockLength,
          args->recvBuff,
          args->recvCount,
          ncclInt,
          ncclMax,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}

TEST_F(AllreduceSparseBlockArgCheckTest, InPlaceBuff) {
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(), 4, recvIndices.data(), 256);
  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          args->recvBuff, /* same buffer for both send and recv */
          args->recvIndices,
          args->blockCount,
          args->blockLength,
          args->recvBuff,
          args->recvCount,
          ncclInt,
          ncclSum,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}

TEST_F(AllreduceSparseBlockArgCheckTest, InvalidSize) {
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(),
      8,
      recvIndices.data(),
      16 /*recv_count < block_count * block_length*/);
  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          args->sendBuff,
          args->recvIndices,
          args->blockCount,
          args->blockLength,
          args->recvBuff,
          args->recvCount,
          ncclInt,
          ncclSum,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}

TEST_F(AllreduceSparseBlockArgCheckTest, InvalidSendBuff) {
  int invalidDevBuff[4];
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(), 4, recvIndices.data(), 256);

  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          (void*)invalidDevBuff,
          args->recvIndices,
          args->blockCount,
          args->blockLength,
          args->recvBuff,
          args->recvCount,
          ncclInt,
          ncclSum,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}

TEST_F(AllreduceSparseBlockArgCheckTest, InvalidRecvBuff) {
  int invalidDevBuff[4];
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(), 4, recvIndices.data(), 256);

  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          args->sendBuff,
          args->recvIndices,
          args->blockCount,
          args->blockLength,
          (void*)invalidDevBuff,
          args->recvCount,
          ncclInt,
          ncclSum,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}

TEST_F(AllreduceSparseBlockArgCheckTest, InvalidRecvIndices) {
  std::vector<int64_t> recvIndices{0, 8, 84, 128};
  auto args = std::make_unique<AllreduceSparseBlockArgs>(
      recvIndices.size(), 4, recvIndices.data(), 256);

  EXPECT_EQ(
      ncclAllReduceSparseBlock(
          args->sendBuff,
          recvIndices.data() /* Invalid recvIndices device pointer */,
          args->blockCount,
          args->blockLength,
          args->recvBuff,
          args->recvCount,
          ncclInt,
          ncclSum,
          args->comm,
          args->stream),
      ncclInvalidArgument);
}
