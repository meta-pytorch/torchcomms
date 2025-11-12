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
#include "meta/lpcoll/low_precision_allgather.h"
#include "meta/lpcoll/low_precision_allreduce.h"
#include "meta/lpcoll/low_precision_alltoall.h"
#include "meta/lpcoll/low_precision_reduce_scatter.h"
#include "nccl.h"

constexpr float EPSILON = 0.2f;

class LowPrecisionCollectivesTest : public ::testing::Test {
 public:
  LowPrecisionCollectivesTest() = default;

  void SetUp() override {
    setenv("RCCL_LOW_PRECISION_ENABLE", "1", 1);

    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    unsetenv("RCCL_LOW_PRECISION_ENABLE");
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(LowPrecisionCollectivesTest, AllReduceFloat) {
  const size_t count = 1024;
  std::vector<float> hostInput(count);

  for (size_t i = 0; i < count; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  float* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      count * sizeof(float),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclLowPrecisionAllReduce(
      deviceInput,
      deviceOutput,
      count,
      ncclFloat,
      ncclSum,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));

  std::vector<float> hostOutput(count);
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      count * sizeof(float),
      hipMemcpyDeviceToHost));

  // Verify result: sum should be numRanks * 1.0f
  float expectedSum = static_cast<float>(this->numRanks);
  for (size_t i = 0; i < count; ++i) {
    EXPECT_NEAR(hostOutput[i], expectedSum, EPSILON * expectedSum);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionCollectivesTest, AllReduceBFloat16) {
  const size_t count = 1024;
  std::vector<uint16_t> hostInput(count);
  std::vector<uint16_t> hostOutput(count);

  for (size_t i = 0; i < count; ++i) {
    float val = 1.0f;
    uint32_t float_bits = *reinterpret_cast<uint32_t*>(&val);
    hostInput[i] = static_cast<uint16_t>(float_bits >> 16);
  }

  uint16_t* deviceInput = nullptr;
  uint16_t* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(uint16_t)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(uint16_t)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      count * sizeof(uint16_t),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclLowPrecisionAllReduce(
      deviceInput,
      deviceOutput,
      count,
      ncclBfloat16,
      ncclSum,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      count * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  float expectedSum = static_cast<float>(this->numRanks);
  for (size_t i = 0; i < count; ++i) {
    uint32_t bf16_as_uint = static_cast<uint32_t>(hostOutput[i]) << 16;
    float result_float = *reinterpret_cast<float*>(&bf16_as_uint);
    EXPECT_NEAR(result_float, expectedSum, EPSILON * expectedSum);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionCollectivesTest, AllReduceLargeBuffer) {
  const size_t count = 1024 * 1024; // 1M elements
  std::vector<float> hostInput(count);
  std::vector<float> hostOutput(count);

  for (size_t i = 0; i < count; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  float* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      count * sizeof(float),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclLowPrecisionAllReduce(
      deviceInput,
      deviceOutput,
      count,
      ncclFloat,
      ncclSum,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      count * sizeof(float),
      hipMemcpyDeviceToHost));

  float expectedSum = static_cast<float>(this->numRanks);
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_NEAR(hostOutput[i], expectedSum, EPSILON * expectedSum);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionCollectivesTest, AllGatherFloat) {
  const size_t sendCount = 256;
  const size_t recvCount = sendCount * this->numRanks;
  std::vector<float> hostInput(sendCount);
  std::vector<float> hostOutput(recvCount);

  for (size_t i = 0; i < sendCount; ++i) {
    hostInput[i] = 1.0f;
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

  ncclResult_t result = ncclLowPrecisionAllGather(
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

  // Verify: all gathered values should be 1.0f
  for (size_t i = 0; i < recvCount; ++i) {
    EXPECT_NEAR(hostOutput[i], 1.0f, EPSILON);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionCollectivesTest, AllGatherBFloat16) {
  const size_t sendCount = 256;
  const size_t recvCount = sendCount * this->numRanks;
  std::vector<uint16_t> hostInput(sendCount);
  std::vector<uint16_t> hostOutput(recvCount);

  for (size_t i = 0; i < sendCount; ++i) {
    float val = 1.0f;
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

  ncclResult_t result = ncclLowPrecisionAllGather(
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

  // Verify: all gathered values should be ~1.0f in bfloat16
  for (size_t i = 0; i < recvCount; ++i) {
    uint32_t bf16_as_uint = static_cast<uint32_t>(hostOutput[i]) << 16;
    float result_float = *reinterpret_cast<float*>(&bf16_as_uint);
    EXPECT_NEAR(result_float, 1.0f, EPSILON);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionCollectivesTest, ReduceScatterFloat) {
  const size_t recvCount = 256;
  const size_t sendCount = recvCount * this->numRanks;
  std::vector<float> hostInput(sendCount);
  std::vector<float> hostOutput(recvCount);

  for (size_t i = 0; i < sendCount; ++i) {
    hostInput[i] = 1.0f;
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

  ncclResult_t result = ncclLowPrecisionReduceScatter(
      deviceInput,
      deviceOutput,
      sendCount,
      ncclFloat,
      ncclSum,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      recvCount * sizeof(float),
      hipMemcpyDeviceToHost));

  float expectedSum = static_cast<float>(this->numRanks);
  for (size_t i = 0; i < recvCount; ++i) {
    EXPECT_NEAR(hostOutput[i], expectedSum, EPSILON * expectedSum);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionCollectivesTest, ReduceScatterBFloat16) {
  const size_t recvCount = 256;
  const size_t sendCount = recvCount * this->numRanks;
  std::vector<uint16_t> hostInput(sendCount);
  std::vector<uint16_t> hostOutput(recvCount);

  for (size_t i = 0; i < sendCount; ++i) {
    float val = 1.0f;
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

  ncclResult_t result = ncclLowPrecisionReduceScatter(
      deviceInput,
      deviceOutput,
      sendCount,
      ncclBfloat16,
      ncclSum,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      recvCount * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  float expectedSum = static_cast<float>(this->numRanks);
  for (size_t i = 0; i < recvCount; ++i) {
    uint32_t bf16_as_uint = static_cast<uint32_t>(hostOutput[i]) << 16;
    float result_float = *reinterpret_cast<float*>(&bf16_as_uint);
    EXPECT_NEAR(result_float, expectedSum, EPSILON * expectedSum);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionCollectivesTest, AllToAllFloat) {
  const size_t count = 256;
  const size_t totalCount = count * this->numRanks;
  std::vector<float> hostInput(totalCount);
  std::vector<float> hostOutput(totalCount);

  for (size_t i = 0; i < totalCount; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  float* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, totalCount * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, totalCount * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      totalCount * sizeof(float),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclLowPrecisionAllToAll(
      deviceInput, deviceOutput, count, ncclFloat, this->comm, this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      totalCount * sizeof(float),
      hipMemcpyDeviceToHost));

  // Verify: all output values should be 1.0f
  for (size_t i = 0; i < totalCount; ++i) {
    EXPECT_NEAR(hostOutput[i], 1.0f, EPSILON);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionCollectivesTest, AllToAllBFloat16) {
  const size_t count = 256;
  const size_t totalCount = count * this->numRanks;
  std::vector<uint16_t> hostInput(totalCount);
  std::vector<uint16_t> hostOutput(totalCount);

  for (size_t i = 0; i < totalCount; ++i) {
    float val = 1.0f;
    uint32_t float_bits = *reinterpret_cast<uint32_t*>(&val);
    hostInput[i] = static_cast<uint16_t>(float_bits >> 16);
  }

  uint16_t* deviceInput = nullptr;
  uint16_t* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, totalCount * sizeof(uint16_t)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, totalCount * sizeof(uint16_t)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      totalCount * sizeof(uint16_t),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclLowPrecisionAllToAll(
      deviceInput, deviceOutput, count, ncclBfloat16, this->comm, this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      totalCount * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  // Verify: all output values should be ~1.0f in bfloat16
  for (size_t i = 0; i < totalCount; ++i) {
    uint32_t bf16_as_uint = static_cast<uint32_t>(hostOutput[i]) << 16;
    float result_float = *reinterpret_cast<float*>(&bf16_as_uint);
    EXPECT_NEAR(result_float, 1.0f, EPSILON);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionCollectivesTest, MultipleAllReduces) {
  const size_t count = 512;
  const int numIterations = 5;

  float* deviceInput = nullptr;
  float* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(float)));

  for (int iter = 0; iter < numIterations; ++iter) {
    std::vector<float> hostInput(count);
    for (size_t i = 0; i < count; ++i) {
      hostInput[i] = 1.0f;
    }

    CUDACHECK_TEST(hipMemcpy(
        deviceInput,
        hostInput.data(),
        count * sizeof(float),
        hipMemcpyHostToDevice));

    ncclResult_t result = ncclLowPrecisionAllReduce(
        deviceInput,
        deviceOutput,
        count,
        ncclFloat,
        ncclSum,
        this->comm,
        this->stream);

    ASSERT_EQ(result, ncclSuccess);
    CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionCollectivesTest, AllReduceInPlace) {
  const size_t count = 1024;
  std::vector<float> hostData(count);

  for (size_t i = 0; i < count; ++i) {
    hostData[i] = 1.0f;
  }

  float* deviceBuffer = nullptr;
  CUDACHECK_TEST(hipMalloc(&deviceBuffer, count * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceBuffer,
      hostData.data(),
      count * sizeof(float),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclLowPrecisionAllReduce(
      deviceBuffer,
      deviceBuffer,
      count,
      ncclFloat,
      ncclSum,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostData.data(),
      deviceBuffer,
      count * sizeof(float),
      hipMemcpyDeviceToHost));

  float expectedSum = static_cast<float>(this->numRanks);
  for (size_t i = 0; i < count; ++i) {
    EXPECT_NEAR(hostData[i], expectedSum, EPSILON * expectedSum);
  }

  CUDACHECK_TEST(hipFree(deviceBuffer));
}

TEST_F(LowPrecisionCollectivesTest, AllReduceInPlaceBFloat16) {
  const size_t count = 1024;
  std::vector<uint16_t> hostData(count);

  for (size_t i = 0; i < count; ++i) {
    float val = 1.0f;
    uint32_t float_bits = *reinterpret_cast<uint32_t*>(&val);
    hostData[i] = static_cast<uint16_t>(float_bits >> 16);
  }

  uint16_t* deviceBuffer = nullptr;
  CUDACHECK_TEST(hipMalloc(&deviceBuffer, count * sizeof(uint16_t)));

  CUDACHECK_TEST(hipMemcpy(
      deviceBuffer,
      hostData.data(),
      count * sizeof(uint16_t),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclLowPrecisionAllReduce(
      deviceBuffer,
      deviceBuffer,
      count,
      ncclBfloat16,
      ncclSum,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostData.data(),
      deviceBuffer,
      count * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  float expectedSum = static_cast<float>(this->numRanks);
  for (size_t i = 0; i < count; ++i) {
    uint32_t bf16_as_uint = static_cast<uint32_t>(hostData[i]) << 16;
    float result_float = *reinterpret_cast<float*>(&bf16_as_uint);
    EXPECT_NEAR(result_float, expectedSum, EPSILON * expectedSum);
  }

  CUDACHECK_TEST(hipFree(deviceBuffer));
}

TEST_F(LowPrecisionCollectivesTest, AllGatherInPlace) {
  const size_t sendCount = 256;
  const size_t recvCount = sendCount * this->numRanks;
  std::vector<float> hostData(recvCount);

  for (size_t i = 0; i < sendCount; ++i) {
    hostData[this->globalRank * sendCount + i] = 1.0f;
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

  ncclResult_t result = ncclLowPrecisionAllGather(
      sendPtr, recvPtr, sendCount, ncclFloat, this->comm, this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostData.data(),
      deviceBuffer,
      recvCount * sizeof(float),
      hipMemcpyDeviceToHost));

  // Verify: all gathered values should be 1.0f
  for (size_t i = 0; i < recvCount; ++i) {
    EXPECT_NEAR(hostData[i], 1.0f, EPSILON);
  }

  CUDACHECK_TEST(hipFree(deviceBuffer));
}

TEST_F(LowPrecisionCollectivesTest, AllGatherInPlaceBFloat16) {
  const size_t sendCount = 256;
  const size_t recvCount = sendCount * this->numRanks;
  std::vector<uint16_t> hostData(recvCount);

  for (size_t i = 0; i < sendCount; ++i) {
    float val = 1.0f;
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

  ncclResult_t result = ncclLowPrecisionAllGather(
      sendPtr, recvPtr, sendCount, ncclBfloat16, this->comm, this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostData.data(),
      deviceBuffer,
      recvCount * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  // Verify: all gathered values should be ~1.0f in bfloat16
  for (size_t i = 0; i < recvCount; ++i) {
    uint32_t bf16_as_uint = static_cast<uint32_t>(hostData[i]) << 16;
    float result_float = *reinterpret_cast<float*>(&bf16_as_uint);
    EXPECT_NEAR(result_float, 1.0f, EPSILON);
  }

  CUDACHECK_TEST(hipFree(deviceBuffer));
}

TEST_F(LowPrecisionCollectivesTest, ReduceScatterInPlace) {
  const size_t recvCount = 256;
  const size_t totalCount = recvCount * this->numRanks;
  std::vector<float> hostData(totalCount);

  for (size_t i = 0; i < totalCount; ++i) {
    hostData[i] = 1.0f;
  }

  float* deviceBuffer = nullptr;
  CUDACHECK_TEST(hipMalloc(&deviceBuffer, totalCount * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceBuffer,
      hostData.data(),
      totalCount * sizeof(float),
      hipMemcpyHostToDevice));

  const float* sendPtr = deviceBuffer;
  float* recvPtr = deviceBuffer + this->globalRank * recvCount;

  ncclResult_t result = ncclLowPrecisionReduceScatter(
      sendPtr,
      recvPtr,
      totalCount,
      ncclFloat,
      ncclSum,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));

  std::vector<float> hostOutput(recvCount);
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      recvPtr,
      recvCount * sizeof(float),
      hipMemcpyDeviceToHost));

  float expectedSum = static_cast<float>(this->numRanks);
  for (size_t i = 0; i < recvCount; ++i) {
    EXPECT_NEAR(hostOutput[i], expectedSum, EPSILON * expectedSum)
        << "In-place ReduceScatter mismatch at index " << i;
  }

  CUDACHECK_TEST(hipFree(deviceBuffer));
}

TEST_F(LowPrecisionCollectivesTest, ReduceScatterInPlaceBFloat16) {
  const size_t recvCount = 256;
  const size_t totalCount = recvCount * this->numRanks;
  std::vector<uint16_t> hostData(totalCount);

  for (size_t i = 0; i < totalCount; ++i) {
    float val = 1.0f;
    uint32_t float_bits = *reinterpret_cast<uint32_t*>(&val);
    hostData[i] = static_cast<uint16_t>(float_bits >> 16);
  }

  uint16_t* deviceBuffer = nullptr;
  CUDACHECK_TEST(hipMalloc(&deviceBuffer, totalCount * sizeof(uint16_t)));

  CUDACHECK_TEST(hipMemcpy(
      deviceBuffer,
      hostData.data(),
      totalCount * sizeof(uint16_t),
      hipMemcpyHostToDevice));

  const uint16_t* sendPtr = deviceBuffer;
  uint16_t* recvPtr = deviceBuffer + this->globalRank * recvCount;

  ncclResult_t result = ncclLowPrecisionReduceScatter(
      sendPtr,
      recvPtr,
      totalCount,
      ncclBfloat16,
      ncclSum,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));

  std::vector<uint16_t> hostOutput(recvCount);
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      recvPtr,
      recvCount * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  float expectedSum = static_cast<float>(this->numRanks);
  for (size_t i = 0; i < recvCount; ++i) {
    uint32_t bf16_as_uint = static_cast<uint32_t>(hostOutput[i]) << 16;
    float result_float = *reinterpret_cast<float*>(&bf16_as_uint);
    EXPECT_NEAR(result_float, expectedSum, EPSILON * expectedSum)
        << "In-place BF16 ReduceScatter mismatch at index " << i;
  }

  CUDACHECK_TEST(hipFree(deviceBuffer));
}

TEST_F(LowPrecisionCollectivesTest, AllToAllInPlace) {
  const size_t count = 256;
  const size_t totalCount = count * this->numRanks;
  std::vector<float> hostData(totalCount);

  for (size_t i = 0; i < totalCount; ++i) {
    hostData[i] = 1.0f;
  }

  float* deviceBuffer = nullptr;
  CUDACHECK_TEST(hipMalloc(&deviceBuffer, totalCount * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceBuffer,
      hostData.data(),
      totalCount * sizeof(float),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclLowPrecisionAllToAll(
      deviceBuffer, deviceBuffer, count, ncclFloat, this->comm, this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostData.data(),
      deviceBuffer,
      totalCount * sizeof(float),
      hipMemcpyDeviceToHost));

  // Verify: all output values should be 1.0f
  for (size_t i = 0; i < totalCount; ++i) {
    EXPECT_NEAR(hostData[i], 1.0f, EPSILON);
  }

  CUDACHECK_TEST(hipFree(deviceBuffer));
}

TEST_F(LowPrecisionCollectivesTest, AllToAllInPlaceBFloat16) {
  const size_t count = 256;
  const size_t totalCount = count * this->numRanks;
  std::vector<uint16_t> hostData(totalCount);

  for (size_t i = 0; i < totalCount; ++i) {
    float val = 1.0f;
    uint32_t float_bits = *reinterpret_cast<uint32_t*>(&val);
    hostData[i] = static_cast<uint16_t>(float_bits >> 16);
  }

  uint16_t* deviceBuffer = nullptr;
  CUDACHECK_TEST(hipMalloc(&deviceBuffer, totalCount * sizeof(uint16_t)));

  CUDACHECK_TEST(hipMemcpy(
      deviceBuffer,
      hostData.data(),
      totalCount * sizeof(uint16_t),
      hipMemcpyHostToDevice));

  ncclResult_t result = ncclLowPrecisionAllToAll(
      deviceBuffer,
      deviceBuffer,
      count,
      ncclBfloat16,
      this->comm,
      this->stream);

  ASSERT_EQ(result, ncclSuccess);

  CUDACHECK_TEST(hipStreamSynchronize(this->stream));
  CUDACHECK_TEST(hipMemcpy(
      hostData.data(),
      deviceBuffer,
      totalCount * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  // Verify: all output values should be ~1.0f in bfloat16
  for (size_t i = 0; i < totalCount; ++i) {
    uint32_t bf16_as_uint = static_cast<uint32_t>(hostData[i]) << 16;
    float result_float = *reinterpret_cast<float*>(&bf16_as_uint);
    EXPECT_NEAR(result_float, 1.0f, EPSILON);
  }

  CUDACHECK_TEST(hipFree(deviceBuffer));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
