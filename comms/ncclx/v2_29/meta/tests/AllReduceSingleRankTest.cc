// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This test verifies allreduce behavior with a single rank (world_size=1).
// In this case, the output should equal the input since there's nothing to
// reduce. This test targets the same class of bugs as the reduce_scatter
// single-rank test where the oneRankReduce kernel could miss processing
// some elements.

#include <comm.h>
#include <fmt/core.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include <random>
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class AllReduceSingleRankTest : public NcclxBaseTest {
 public:
  AllReduceSingleRankTest() = default;

  void SetUp() override {
    NcclxBaseTest::SetUp();
    // This test is specifically for single-rank scenarios
    if (numRanks != 1) {
      GTEST_SKIP() << "This test requires exactly 1 rank, got " << numRanks;
    }
    comm = createNcclComm(globalRank, numRanks, localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    if (comm != nullptr) {
      NCCLCHECK_TEST(ncclCommDestroy(comm));
    }
    if (stream != nullptr) {
      CUDACHECK_TEST(cudaStreamDestroy(stream));
    }
    NcclxBaseTest::TearDown();
  }

 protected:
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
};

// Test parameters: <ncclRedOp_t, ncclDataType_t, count>
class AllReduceSingleRankTestParam
    : public AllReduceSingleRankTest,
      public ::testing::WithParamInterface<
          std::tuple<ncclRedOp_t, ncclDataType_t, size_t>> {};

TEST_P(AllReduceSingleRankTestParam, OutputEqualsInput) {
  const auto& [redOp, dataType, count] = GetParam();

  // Skip if single rank requirement not met (handled in SetUp but double-check)
  if (numRanks != 1) {
    GTEST_SKIP() << "Test requires exactly 1 rank";
  }

  // Determine element size based on data type
  size_t elemSize;
  switch (dataType) {
    case ncclFloat16:
    case ncclBfloat16:
      elemSize = 2;
      break;
    case ncclFloat32:
      elemSize = 4;
      break;
    case ncclFloat64:
      elemSize = 8;
      break;
    default:
      elemSize = 4; // Default to 4 bytes
      break;
  }

  size_t bufSize = count * elemSize;

  // Allocate input and output buffers
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, bufSize));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, bufSize));

  // Allocate host buffers for verification
  std::vector<uint8_t> hostInput(bufSize);
  std::vector<uint8_t> hostOutput(bufSize);

  // Initialize input with valid floating point values
  // We generate normalized float values and convert them to the target type
  // to avoid issues with NaN/Inf/denormalized numbers where x*1.0 != x
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  switch (dataType) {
    case ncclFloat16: {
      uint16_t* ptr = reinterpret_cast<uint16_t*>(hostInput.data());
      for (size_t i = 0; i < count; ++i) {
        float val = dist(rng);
        // Convert float to fp16 representation
        uint32_t floatBits;
        std::memcpy(&floatBits, &val, sizeof(float));
        // Simple float32 to float16 conversion (truncate mantissa)
        uint16_t sign = (floatBits >> 16) & 0x8000;
        int32_t exp = ((floatBits >> 23) & 0xFF) - 127 + 15;
        uint16_t mant = (floatBits >> 13) & 0x3FF;
        if (exp <= 0) {
          ptr[i] = sign; // Flush to zero
        } else if (exp >= 31) {
          ptr[i] = sign | 0x7C00; // Infinity
        } else {
          ptr[i] = sign | (exp << 10) | mant;
        }
      }
      break;
    }
    case ncclBfloat16: {
      uint16_t* ptr = reinterpret_cast<uint16_t*>(hostInput.data());
      for (size_t i = 0; i < count; ++i) {
        float val = dist(rng);
        uint32_t floatBits;
        std::memcpy(&floatBits, &val, sizeof(float));
        ptr[i] = static_cast<uint16_t>(floatBits >> 16);
      }
      break;
    }
    case ncclFloat32: {
      float* ptr = reinterpret_cast<float*>(hostInput.data());
      for (size_t i = 0; i < count; ++i) {
        ptr[i] = dist(rng);
      }
      break;
    }
    case ncclFloat64: {
      double* ptr = reinterpret_cast<double*>(hostInput.data());
      std::uniform_real_distribution<double> ddist(-1.0, 1.0);
      for (size_t i = 0; i < count; ++i) {
        ptr[i] = ddist(rng);
      }
      break;
    }
    default: {
      // For integer types, use random bytes
      for (size_t i = 0; i < bufSize; ++i) {
        hostInput[i] = static_cast<uint8_t>(rng() % 256);
      }
      break;
    }
  }

  // Copy input to device
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostInput.data(), bufSize, cudaMemcpyHostToDevice));

  // Initialize output with different pattern to detect if it's overwritten
  std::memset(hostOutput.data(), 0xAB, bufSize);
  CUDACHECK_TEST(
      cudaMemcpy(recvBuf, hostOutput.data(), bufSize, cudaMemcpyHostToDevice));

  // Perform allreduce - with single rank, output should equal input
  constexpr int numIterations = 10;
  for (int iter = 0; iter < numIterations; ++iter) {
    auto res =
        ncclAllReduce(sendBuf, recvBuf, count, dataType, redOp, comm, stream);
    ASSERT_EQ(res, ncclSuccess) << "ncclAllReduce failed on iteration " << iter;
    CUDACHECK_TEST(cudaStreamSynchronize(stream));

    // Copy output back to host
    CUDACHECK_TEST(cudaMemcpy(
        hostOutput.data(), recvBuf, bufSize, cudaMemcpyDeviceToHost));

    // Verify output equals input
    int errs = 0;
    for (size_t i = 0; i < bufSize; ++i) {
      if (hostInput[i] != hostOutput[i]) {
        if (errs < 10) {
          printf(
              "[iter %d] Mismatch at byte %zu: input=0x%02x, output=0x%02x\n",
              iter,
              i,
              hostInput[i],
              hostOutput[i]);
        }
        errs++;
      }
    }
    EXPECT_EQ(errs, 0) << "Iteration " << iter << ": Found " << errs
                       << " byte mismatches between input and output. "
                       << "For single-rank allreduce, output must equal input.";
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceSingleRankTestInstance,
    AllReduceSingleRankTestParam,
    ::testing::Combine(
        ::testing::Values(ncclSum, ncclAvg),
        ::testing::Values(ncclBfloat16, ncclFloat32),
        ::testing::Values(8191, 8192, 16777213, 16777216)),
    [](const testing::TestParamInfo<AllReduceSingleRankTestParam::ParamType>&
           info) {
      const char* opName;
      switch (std::get<0>(info.param)) {
        case ncclSum:
          opName = "Sum";
          break;
        case ncclAvg:
          opName = "Avg";
          break;
        case ncclMax:
          opName = "Max";
          break;
        case ncclMin:
          opName = "Min";
          break;
        default:
          opName = "Unknown";
          break;
      }
      const char* typeName;
      switch (std::get<1>(info.param)) {
        case ncclFloat16:
          typeName = "Float16";
          break;
        case ncclBfloat16:
          typeName = "Bfloat16";
          break;
        case ncclFloat32:
          typeName = "Float32";
          break;
        case ncclFloat64:
          typeName = "Float64";
          break;
        default:
          typeName = "Unknown";
          break;
      }
      return fmt::format(
          "{}_{}_{}count", opName, typeName, std::get<2>(info.param));
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
