// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include "comm.h"
#include "comms/rcclx/develop/meta/testinfra/TestUtils.h"
#include "comms/rcclx/develop/meta/testinfra/TestsDistUtils.h"
#include "meta/lpcoll/low_precision_common.h"
#include "meta/lpcoll/low_precision_kernels.h"
#include "nccl.h"

constexpr float EPSILON = 0.1f;

class LowPrecisionKernelsTest : public ::testing::Test {
 public:
  LowPrecisionKernelsTest() = default;

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

TEST_F(LowPrecisionKernelsTest, QuantizeFloatToFp8Basic) {
  const size_t count = 1024;
  std::vector<float> hostInput(count);

  for (size_t i = 0; i < count; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  rccl_float8* deviceOutput = nullptr;
  std::vector<rccl_float8> hostOutput(count);

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(rccl_float8)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      count * sizeof(float),
      hipMemcpyHostToDevice));

  dim3 blockSize(256);
  dim3 gridSize((count + blockSize.x - 1) / blockSize.x);

  hipLaunchKernelGGL(
      quantizeFloatToFp8Kernel<float>,
      gridSize,
      blockSize,
      0,
      stream,
      deviceInput,
      deviceOutput,
      count,
      0,
      count);

  CUDACHECK_TEST(hipStreamSynchronize(stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      count * sizeof(rccl_float8),
      hipMemcpyDeviceToHost));

  // Verify quantization worked (values should be non-zero where input was
  // non-zero)
  for (size_t i = 0; i < count; ++i) {
    if (hostInput[i] != 0.0f) {
      EXPECT_NE(float(hostOutput[i]), 0.0f);
    }
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionKernelsTest, DequantizeFp8ToFloatBasic) {
  const size_t count = 1024;
  std::vector<float> hostInput(count);

  for (size_t i = 0; i < count; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  rccl_float8* deviceFp8 = nullptr;
  float* deviceOutput = nullptr;
  std::vector<float> hostOutput(count);

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceFp8, count * sizeof(rccl_float8)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      count * sizeof(float),
      hipMemcpyHostToDevice));

  dim3 blockSize(256);
  dim3 gridSize((count + blockSize.x - 1) / blockSize.x);

  // Quantize first
  hipLaunchKernelGGL(
      quantizeFloatToFp8Kernel<float>,
      gridSize,
      blockSize,
      0,
      stream,
      deviceInput,
      deviceFp8,
      count,
      0,
      count);

  // Then dequantize
  hipLaunchKernelGGL(
      dequantizeFp8ToFloatKernel<float>,
      gridSize,
      blockSize,
      0,
      stream,
      deviceFp8,
      deviceOutput,
      count,
      0,
      count);

  CUDACHECK_TEST(hipStreamSynchronize(stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      count * sizeof(float),
      hipMemcpyDeviceToHost));

  // Verify round-trip conversion (allow for quantization error)
  for (size_t i = 0; i < count; ++i) {
    EXPECT_NEAR(hostOutput[i], hostInput[i], EPSILON);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceFp8));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionKernelsTest, QuantizeFloatToFp8LargeArray) {
  const size_t count = 1024 * 1024; // 1M elements
  std::vector<float> hostInput(count);

  for (size_t i = 0; i < count; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  rccl_float8* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(rccl_float8)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      count * sizeof(float),
      hipMemcpyHostToDevice));

  dim3 blockSize(256);
  dim3 gridSize((count + blockSize.x - 1) / blockSize.x);

  hipLaunchKernelGGL(
      quantizeFloatToFp8Kernel<float>,
      gridSize,
      blockSize,
      0,
      stream,
      deviceInput,
      deviceOutput,
      count,
      0,
      count);

  CUDACHECK_TEST(hipStreamSynchronize(stream));

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionKernelsTest, QuantizeBF16ToFp8Basic) {
  const size_t count = 1024;
  std::vector<uint16_t> hostInput(count);

  for (size_t i = 0; i < count; ++i) {
    float val = 1.0f;
    uint32_t float_bits = *reinterpret_cast<uint32_t*>(&val);
    hostInput[i] = static_cast<uint16_t>(float_bits >> 16);
  }

  uint16_t* deviceInput = nullptr;
  rccl_float8* deviceOutput = nullptr;
  std::vector<rccl_float8> hostOutput(count);

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(uint16_t)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(rccl_float8)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      count * sizeof(uint16_t),
      hipMemcpyHostToDevice));

  dim3 blockSize(256);
  dim3 gridSize((count + blockSize.x - 1) / blockSize.x);

  hipLaunchKernelGGL(
      quantizeBF16ToFp8Kernel<uint16_t>,
      gridSize,
      blockSize,
      0,
      stream,
      deviceInput,
      deviceOutput,
      count,
      0,
      count);

  CUDACHECK_TEST(hipStreamSynchronize(stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      count * sizeof(rccl_float8),
      hipMemcpyDeviceToHost));

  for (size_t i = 0; i < count; ++i) {
    if (hostInput[i] != 0) {
      EXPECT_NE(float(hostOutput[i]), 0.0f);
    }
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionKernelsTest, DequantizeFloatToBF16Basic) {
  const size_t count = 1024;
  std::vector<float> hostInput(count);

  for (size_t i = 0; i < count; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  uint16_t* deviceOutput = nullptr;
  std::vector<uint16_t> hostOutput(count);

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(uint16_t)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      count * sizeof(float),
      hipMemcpyHostToDevice));

  dim3 blockSize(256);
  dim3 gridSize((count + blockSize.x - 1) / blockSize.x);

  hipLaunchKernelGGL(
      dequantizeFloatToBF16Kernel<float>,
      gridSize,
      blockSize,
      0,
      stream,
      deviceInput,
      deviceOutput,
      count,
      0,
      count);

  CUDACHECK_TEST(hipStreamSynchronize(stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      count * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  for (size_t i = 0; i < count; ++i) {
    uint32_t float_bits = *reinterpret_cast<const uint32_t*>(&hostInput[i]);
    uint16_t expected_bf16 = static_cast<uint16_t>(float_bits >> 16);
    EXPECT_EQ(hostOutput[i], expected_bf16);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionKernelsTest, DequantizeFp8ToBF16Basic) {
  const size_t count = 1024;
  std::vector<float> hostInput(count);

  for (size_t i = 0; i < count; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  rccl_float8* deviceFp8 = nullptr;
  uint16_t* deviceOutput = nullptr;
  std::vector<uint16_t> hostOutput(count);

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceFp8, count * sizeof(rccl_float8)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(uint16_t)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      count * sizeof(float),
      hipMemcpyHostToDevice));

  dim3 blockSize(256);
  dim3 gridSize((count + blockSize.x - 1) / blockSize.x);

  // Quantize float to FP8
  hipLaunchKernelGGL(
      quantizeFloatToFp8Kernel<float>,
      gridSize,
      blockSize,
      0,
      stream,
      deviceInput,
      deviceFp8,
      count,
      0,
      count);

  // Dequantize FP8 to BF16
  hipLaunchKernelGGL(
      dequantizeFp8ToBF16Kernel<uint16_t>,
      gridSize,
      blockSize,
      0,
      stream,
      deviceFp8,
      deviceOutput,
      count,
      0,
      count);

  CUDACHECK_TEST(hipStreamSynchronize(stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      count * sizeof(uint16_t),
      hipMemcpyDeviceToHost));

  for (size_t i = 0; i < count; ++i) {
    if (hostInput[i] != 0.0f) {
      EXPECT_NE(hostOutput[i], 0);
    }
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceFp8));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionKernelsTest, LocalReductionKernel) {
  const size_t chunkSize = 256;
  const int nRanks = 8;
  const size_t totalSize = chunkSize * nRanks;

  std::vector<float> hostInput(totalSize);
  for (size_t i = 0; i < totalSize; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  rccl_float8* deviceFp8 = nullptr;
  float* deviceOutput = nullptr;
  std::vector<float> hostOutput(chunkSize);

  CUDACHECK_TEST(hipMalloc(&deviceInput, totalSize * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceFp8, totalSize * sizeof(rccl_float8)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, chunkSize * sizeof(float)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      totalSize * sizeof(float),
      hipMemcpyHostToDevice));

  dim3 blockSize(256);
  dim3 gridSize((totalSize + blockSize.x - 1) / blockSize.x);

  // Quantize input
  hipLaunchKernelGGL(
      quantizeFloatToFp8Kernel<float>,
      gridSize,
      blockSize,
      0,
      stream,
      deviceInput,
      deviceFp8,
      totalSize,
      0,
      totalSize);

  // Perform local reduction
  dim3 reductionGridSize((chunkSize + blockSize.x - 1) / blockSize.x);
  hipLaunchKernelGGL(
      localReductionKernel<float>,
      reductionGridSize,
      blockSize,
      0,
      stream,
      deviceFp8,
      deviceOutput,
      totalSize,
      0,
      chunkSize,
      nRanks,
      0);

  CUDACHECK_TEST(hipStreamSynchronize(stream));
  CUDACHECK_TEST(hipMemcpy(
      hostOutput.data(),
      deviceOutput,
      chunkSize * sizeof(float),
      hipMemcpyDeviceToHost));

  // Each element should sum to approximately nRanks
  for (size_t i = 0; i < chunkSize; ++i) {
    EXPECT_NEAR(hostOutput[i], static_cast<float>(nRanks), EPSILON * nRanks);
  }

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceFp8));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionKernelsTest, VectorizedQuantization) {
  const size_t count = 8192; // Multiple of 8 for vectorization
  std::vector<float> hostInput(count);

  for (size_t i = 0; i < count; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  rccl_float8* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, count * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, count * sizeof(rccl_float8)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      count * sizeof(float),
      hipMemcpyHostToDevice));

  dim3 blockSize(256);
  dim3 gridSize((count + blockSize.x - 1) / blockSize.x);

  hipLaunchKernelGGL(
      quantizeFloatToFp8Kernel<float>,
      gridSize,
      blockSize,
      0,
      stream,
      deviceInput,
      deviceOutput,
      count,
      0,
      count);

  CUDACHECK_TEST(hipStreamSynchronize(stream));

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

TEST_F(LowPrecisionKernelsTest, ChunkedProcessing) {
  const size_t totalCount = 4096;
  const size_t chunkSize = 1024;
  std::vector<float> hostInput(totalCount);

  for (size_t i = 0; i < totalCount; ++i) {
    hostInput[i] = 1.0f;
  }

  float* deviceInput = nullptr;
  rccl_float8* deviceOutput = nullptr;

  CUDACHECK_TEST(hipMalloc(&deviceInput, totalCount * sizeof(float)));
  CUDACHECK_TEST(hipMalloc(&deviceOutput, totalCount * sizeof(rccl_float8)));

  CUDACHECK_TEST(hipMemcpy(
      deviceInput,
      hostInput.data(),
      totalCount * sizeof(float),
      hipMemcpyHostToDevice));

  dim3 blockSize(256);
  dim3 gridSize((chunkSize + blockSize.x - 1) / blockSize.x);

  // Process in chunks
  for (size_t chunkStart = 0; chunkStart < totalCount;
       chunkStart += chunkSize) {
    hipLaunchKernelGGL(
        quantizeFloatToFp8Kernel<float>,
        gridSize,
        blockSize,
        0,
        stream,
        deviceInput,
        deviceOutput,
        totalCount,
        chunkStart,
        chunkSize);
  }

  CUDACHECK_TEST(hipStreamSynchronize(stream));

  CUDACHECK_TEST(hipFree(deviceInput));
  CUDACHECK_TEST(hipFree(deviceOutput));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
