// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <vector>

#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "meta/lpcoll/p2p_allgather.h"
#include "nccl.h"

class P2PAllGatherTest : public ::testing::Test {
 public:
  P2PAllGatherTest() = default;

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(P2PAllGatherTest, AllGatherFloat) {
  const size_t sendCount = 256;
  const size_t recvCount = sendCount * this->numRanks;
  std::vector<float> hostInput(sendCount);
  std::vector<float> hostOutput(recvCount);

  for (size_t i = 0; i < sendCount; ++i) {
    hostInput[i] = static_cast<float>(this->globalRank * 100 + i);
  }

  float* deviceInput = nullptr;
  float* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, sendCount * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, recvCount * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      sendCount * sizeof(float),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclP2PAllGather(
      deviceInput,
      deviceOutput,
      sendCount,
      ncclFloat,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      recvCount * sizeof(float),
      hipMemcpyDeviceToHost));

  for (int rank = 0; rank < this->numRanks; ++rank) {
    for (size_t i = 0; i < sendCount; ++i) {
      float expected = static_cast<float>(rank * 100 + i);
      size_t idx = rank * sendCount + i;
      EXPECT_FLOAT_EQ(hostOutput[idx], expected)
          << "Mismatch at rank " << rank << " index " << i;
    }
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(P2PAllGatherTest, AllGatherBFloat16) {
  const size_t sendCount = 256;
  const size_t recvCount = sendCount * this->numRanks;
  std::vector<uint16_t> hostInput(sendCount);
  std::vector<uint16_t> hostOutput(recvCount);

  for (size_t i = 0; i < sendCount; ++i) {
    float val = static_cast<float>(this->globalRank * 100 + i);
    uint32_t float_bits = *reinterpret_cast<uint32_t*>(&val);
    hostInput[i] = static_cast<uint16_t>(float_bits >> 16);
  }

  uint16_t* deviceInput = nullptr;
  uint16_t* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, sendCount * sizeof(uint16_t)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, recvCount * sizeof(uint16_t)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      sendCount * sizeof(uint16_t),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclP2PAllGather(
      deviceInput,
      deviceOutput,
      sendCount,
      ncclBfloat16,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      recvCount * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  for (int rank = 0; rank < this->numRanks; ++rank) {
    for (size_t i = 0; i < sendCount; ++i) {
      float expected = static_cast<float>(rank * 100 + i);
      uint32_t expected_bits = *reinterpret_cast<uint32_t*>(&expected);
      uint16_t expected_bf16 = static_cast<uint16_t>(expected_bits >> 16);
      size_t idx = rank * sendCount + i;
      EXPECT_EQ(hostOutput[idx], expected_bf16)
          << "BFloat16 mismatch at rank " << rank << " index " << i;
    }
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(P2PAllGatherTest, AllGatherLargeBuffer) {
  const size_t sendCount = 16 * 1024 * 1024; // 16M elements (~64MB for float)
  const size_t recvCount = sendCount * this->numRanks;
  std::vector<float> hostInput(sendCount);

  for (size_t i = 0; i < sendCount; ++i) {
    hostInput[i] = static_cast<float>(this->globalRank);
  }

  float* deviceInput = nullptr;
  float* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, sendCount * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, recvCount * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      sendCount * sizeof(float),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclP2PAllGather(
      deviceInput,
      deviceOutput,
      sendCount,
      ncclFloat,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));

  std::vector<float> hostOutput(recvCount);
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      recvCount * sizeof(float),
      hipMemcpyDeviceToHost));

  for (int rank = 0; rank < this->numRanks; ++rank) {
    for (size_t i = 0; i < 100; ++i) {
      float expected = static_cast<float>(rank);
      size_t idx = rank * sendCount + i;
      EXPECT_FLOAT_EQ(hostOutput[idx], expected)
          << "Large message mismatch at rank " << rank << " index " << i;
    }
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(P2PAllGatherTest, AllGatherInPlace) {
  const size_t sendCount = 256;
  const size_t recvCount = sendCount * this->numRanks;
  std::vector<float> hostData(recvCount);

  for (size_t i = 0; i < sendCount; ++i) {
    hostData[this->globalRank * sendCount + i] =
        static_cast<float>(this->globalRank * 100 + i);
  }

  float* deviceBuffer = nullptr;
  CUDACHECK_TEST(hipMalloc(&deviceBuffer, recvCount * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceBuffer,
      hostData.data(),
      recvCount * sizeof(float),
      hipMemcpyHostToDevice));

  const float* sendPtr = deviceBuffer + this->globalRank * sendCount;
  float* recvPtr = deviceBuffer;

  ncclResult_t result = ncclP2PAllGather(
      sendPtr, recvPtr, sendCount, ncclFloat, this->comm, this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostData.data(),
      deviceBuffer,
      recvCount * sizeof(float),
      hipMemcpyDeviceToHost));

  for (int rank = 0; rank < this->numRanks; ++rank) {
    for (size_t i = 0; i < sendCount; ++i) {
      float expected = static_cast<float>(rank * 100 + i);
      size_t idx = rank * sendCount + i;
      EXPECT_FLOAT_EQ(hostData[idx], expected)
          << "In-place mismatch at rank " << rank << " index " << i;
    }
  }

  CUDACHECK_TEST(hipFree(deviceBuffer));
}

TEST_F(P2PAllGatherTest, AllGatherInPlaceBFloat16) {
  const size_t sendCount = 256;
  const size_t recvCount = sendCount * this->numRanks;
  std::vector<uint16_t> hostData(recvCount);

  for (size_t i = 0; i < sendCount; ++i) {
    float val = static_cast<float>(this->globalRank * 100 + i);
    uint32_t float_bits = *reinterpret_cast<uint32_t*>(&val);
    hostData[this->globalRank * sendCount + i] =
        static_cast<uint16_t>(float_bits >> 16);
  }

  uint16_t* deviceBuffer = nullptr;
  CUDACHECK_TEST(hipMalloc(&deviceBuffer, recvCount * sizeof(uint16_t)));

  CUDACHECK_TEST(hipMemcpy(
      deviceBuffer,
      hostData.data(),
      recvCount * sizeof(uint16_t),
      hipMemcpyHostToDevice));

  const uint16_t* sendPtr = deviceBuffer + this->globalRank * sendCount;
  uint16_t* recvPtr = deviceBuffer;

  ncclResult_t result = ncclP2PAllGather(
      sendPtr, recvPtr, sendCount, ncclBfloat16, this->comm, this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostData.data(),
      deviceBuffer,
      recvCount * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  for (int rank = 0; rank < this->numRanks; ++rank) {
    for (size_t i = 0; i < sendCount; ++i) {
      float expected = static_cast<float>(rank * 100 + i);
      uint32_t expected_bits = *reinterpret_cast<uint32_t*>(&expected);
      uint16_t expected_bf16 = static_cast<uint16_t>(expected_bits >> 16);
      size_t idx = rank * sendCount + i;
      EXPECT_EQ(hostData[idx], expected_bf16)
          << "In-place BF16 mismatch at rank " << rank << " index " << i;
    }
  }

  CUDACHECK_TEST(hipFree(deviceBuffer));
}

TEST_F(P2PAllGatherTest, MultipleAllGathers) {
  const size_t sendCount = 512;
  const size_t recvCount = sendCount * this->numRanks;
  const int numIterations = 5;

  float* deviceInput = nullptr;
  float* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, sendCount * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, recvCount * sizeof(float)));

  for (int iter = 0; iter < numIterations; ++iter) {
    std::vector<float> hostInput(sendCount);
    for (size_t i = 0; i < sendCount; ++i) {
      hostInput[i] = static_cast<float>(this->globalRank * 100 + iter * 10 + i);
    }

    CUDACHECK_TEST(hipMemcpy(
        deviceInput,
        hostInput.data(),
        sendCount * sizeof(float),
        hipMemcpyHostToDevice));

    ncclResult_t result = ncclP2PAllGather(
        deviceInput,
        deviceOutput,
        sendCount,
        ncclFloat,
        this->comm,
        this->stream);

    ASSERT_EQ(result, ncclSuccess);
    CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
