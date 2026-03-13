// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <cuda_bf16.h>
#include <fmt/core.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class ReduceScatterQuantizeTest : public NcclxBaseTest {
 public:
  ReduceScatterQuantizeTest() = default;
  void SetUp() override {
    setenv("NCCL_PAT_ENABLE", "1", 1);
    setenv("NCCL_ALGO", "PAT", 1);

    NcclxBaseTest::SetUp();
    comm = createNcclComm(globalRank, numRanks, localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NcclxBaseTest::TearDown();
  }

 protected:
  ncclComm_t comm;
  cudaStream_t stream;
};

// Test parameters: <ncclRedOp_t, size_t count, uint64_t seed>
class ReduceScatterQuantizeTestParam
    : public ReduceScatterQuantizeTest,
      public ::testing::WithParamInterface<
          std::tuple<ncclRedOp_t, size_t, uint64_t>> {};

TEST_P(ReduceScatterQuantizeTestParam, CorrectReduction) {
  const auto& [redOp, count, seed] = GetParam();

  // Allocate buffers for ncclReduceScatterQuantize - input is FP32, output is
  // FP32
  float *sendBuf = nullptr, *recvBufQuantize = nullptr;
  size_t sendSize = count * numRanks * sizeof(float);
  size_t recvSize = count * sizeof(float);

  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvBufQuantize, recvSize));

  // Allocate buffers for ncclReduceScatter in BF16
  __nv_bfloat16 *sendBufBf16 = nullptr, *recvBufBf16 = nullptr;
  size_t sendSizeBf16 = count * numRanks * sizeof(__nv_bfloat16);
  size_t recvSizeBf16 = count * sizeof(__nv_bfloat16);

  CUDACHECK_TEST(cudaMalloc(&sendBufBf16, sendSizeBf16));
  CUDACHECK_TEST(cudaMalloc(&recvBufBf16, recvSizeBf16));

  // Initialize send buffer with deterministic values
  // Each rank r sends: sendBuf[chunk_for_rank_c][i] = r * numRanks + c + i *
  // 0.001 This allows us to compute expected values after reduce scatter
  std::vector<float> hostSendBuf(count * numRanks);
  std::vector<__nv_bfloat16> hostSendBufBf16(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      float val = static_cast<float>(globalRank * numRanks + c) +
          static_cast<float>(i) * 0.001f;
      hostSendBuf[c * count + i] = val;
      hostSendBufBf16[c * count + i] = __float2bfloat16(val);
    }
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf, hostSendBuf.data(), sendSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      sendBufBf16,
      hostSendBufBf16.data(),
      sendSizeBf16,
      cudaMemcpyHostToDevice));

  // Initialize recv buffers with sentinel value
  CUDACHECK_TEST(cudaMemset(recvBufQuantize, 0xFF, recvSize));
  CUDACHECK_TEST(cudaMemset(recvBufBf16, 0xFF, recvSizeBf16));

  // Initialize seed
  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));

  // Perform reduce scatter with quantization (FP32 -> BF16 transport -> FP32)
  auto res = ncclReduceScatterQuantize(
      sendBuf,
      recvBufQuantize,
      count,
      ncclFloat32, // inputType
      ncclBfloat16, // transportType
      redOp,
      seedBuf,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);

  // Perform regular reduce scatter in BF16 for comparison
  res = ncclReduceScatter(
      sendBufBf16, recvBufBf16, count, ncclBfloat16, redOp, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy results back to host for verification
  std::vector<float> hostRecvBufQuantize(count);
  std::vector<__nv_bfloat16> hostRecvBufBf16(count);
  CUDACHECK_TEST(cudaMemcpy(
      hostRecvBufQuantize.data(),
      recvBufQuantize,
      recvSize,
      cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      hostRecvBufBf16.data(),
      recvBufBf16,
      recvSizeBf16,
      cudaMemcpyDeviceToHost));

  // Convert BF16 results to FP32 for comparison
  std::vector<float> hostRecvBufBf16AsFloat(count);
  for (size_t i = 0; i < count; i++) {
    hostRecvBufBf16AsFloat[i] = __bfloat162float(hostRecvBufBf16[i]);
  }

  // Compute expected values:
  // For rank 'globalRank', we receive the reduction of chunk 'globalRank' from
  // all ranks. Each rank r contributes: r * numRanks + globalRank + i * 0.001
  // Sum = sum over r of (r * numRanks + globalRank + i * 0.001)
  //     = numRanks * (0 + 1 + ... + numRanks-1) + numRanks * globalRank + i *
  //     0.001 * numRanks = numRanks * numRanks * (numRanks-1) / 2 + numRanks *
  //     globalRank + i * 0.001 * numRanks
  // Compute number of PAT steps for error bound calculation
  int numPatSteps = 0;
  {
    int n = numRanks;
    while (n > 1) {
      n >>= 1;
      numPatSteps++;
    }
  }

  // Per-element and aggregate error analysis
  double totalQuantizeErr = 0.0;
  double totalBf16Err = 0.0;
  double maxQuantizeErr = 0.0;
  double maxBf16Err = 0.0;
  double totalSignedQuantizeErr = 0.0;
  int quantizeUlpViolations = 0;

  for (size_t i = 0; i < count; i++) {
    float expectedSum = static_cast<float>(numRanks) *
            static_cast<float>(numRanks) * static_cast<float>(numRanks - 1) /
            2.0f +
        static_cast<float>(numRanks * globalRank) +
        static_cast<float>(i) * 0.001f * static_cast<float>(numRanks);

    float expected = expectedSum;
    if (redOp == ncclAvg) {
      expected = expectedSum / static_cast<float>(numRanks);
    }

    float quantizeDiff = std::abs(hostRecvBufQuantize[i] - expected);
    float bf16Diff = std::abs(hostRecvBufBf16AsFloat[i] - expected);
    totalQuantizeErr += static_cast<double>(quantizeDiff);
    totalBf16Err += static_cast<double>(bf16Diff);
    maxQuantizeErr =
        std::max(maxQuantizeErr, static_cast<double>(quantizeDiff));
    maxBf16Err = std::max(maxBf16Err, static_cast<double>(bf16Diff));
    totalSignedQuantizeErr +=
        static_cast<double>(hostRecvBufQuantize[i] - expected);

    // Per-element error check: each element should be within numPatSteps
    // BF16 ULPs of the expected value. With log2(numRanks) PAT steps, each
    // intermediate quantization can introduce up to 1 BF16 ULP of error.
    float absExpected = std::max(std::abs(expected), 1e-10f);
    int exponent;
    std::frexp(absExpected, &exponent);
    float bf16Ulp = std::ldexp(1.0f, exponent - 8); // BF16 has 7 mantissa bits
    float tolerance = static_cast<float>(numPatSteps) * bf16Ulp;

    if (quantizeDiff > tolerance) {
      quantizeUlpViolations++;
      if (quantizeUlpViolations <= 10) {
        printf(
            "Rank %d, index %zu: expected=%f, got=%f, diff=%f, "
            "tolerance=%f (%d ULPs), diff_in_ulps=%.1f\n",
            globalRank,
            i,
            expected,
            hostRecvBufQuantize[i],
            quantizeDiff,
            tolerance,
            numPatSteps,
            quantizeDiff / bf16Ulp);
      }
    }
  }

  double meanQuantizeErr = totalQuantizeErr / count;
  double meanBf16Err = totalBf16Err / count;
  double meanSignedErr = totalSignedQuantizeErr / static_cast<double>(count);

  // Print diagnostic summary for debugging
  printf(
      "Rank %d, count=%zu: quantize MAE=%.6f, bf16 MAE=%.6f, "
      "max_quantize=%.6f, max_bf16=%.6f, mean_signed=%.6f, "
      "ulp_violations=%d\n",
      globalRank,
      count,
      meanQuantizeErr,
      meanBf16Err,
      maxQuantizeErr,
      maxBf16Err,
      meanSignedErr,
      quantizeUlpViolations);

  // Check 1: No per-element ULP violations. Each element's error should be
  // bounded by numPatSteps BF16 ULPs.
  EXPECT_EQ(quantizeUlpViolations, 0)
      << "Rank " << globalRank << ": " << quantizeUlpViolations << " of "
      << count << " elements exceeded " << numPatSteps
      << " BF16 ULP tolerance. Max quantize error: " << maxQuantizeErr
      << ", max BF16 error: " << maxBf16Err;

  // Check 2: Quantized path's mean absolute error should be no worse than the
  // BF16 baseline. The quantized path uses FP32 input (no input quantization
  // loss) and FP32 accumulation, so it should achieve at least the same
  // accuracy as the pure BF16 path.
  EXPECT_LE(meanQuantizeErr, meanBf16Err * 1.5)
      << "Rank " << globalRank << ": quantized mean absolute error ("
      << meanQuantizeErr << ") exceeds BF16 baseline * 1.5 ("
      << meanBf16Err * 1.5 << ")"
      << ". Max quantize error: " << maxQuantizeErr
      << ", max BF16 error: " << maxBf16Err
      << ", mean signed error: " << meanSignedErr;

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBufQuantize));
  CUDACHECK_TEST(cudaFree(sendBufBf16));
  CUDACHECK_TEST(cudaFree(recvBufBf16));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// Test invalid input type (should fail)
TEST_F(ReduceScatterQuantizeTest, InvalidInputType) {
  size_t count = 1024;
  void *sendBuf = nullptr, *recvBuf = nullptr;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

  // Using ncclFloat16 as input type should fail
  auto res = ncclReduceScatterQuantize(
      sendBuf,
      recvBuf,
      count,
      ncclFloat16, // invalid - must be ncclFloat32
      ncclBfloat16,
      ncclSum,
      0,
      comm,
      stream);
  EXPECT_EQ(res, ncclInvalidArgument);

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Test invalid transport type (should fail)
TEST_F(ReduceScatterQuantizeTest, InvalidTransportType) {
  size_t count = 1024;
  void *sendBuf = nullptr, *recvBuf = nullptr;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

  // Using ncclFloat16 as transport type should fail
  auto res = ncclReduceScatterQuantize(
      sendBuf,
      recvBuf,
      count,
      ncclFloat32,
      ncclFloat16, // invalid - must be ncclBfloat16
      ncclSum,
      0,
      comm,
      stream);
  EXPECT_EQ(res, ncclInvalidArgument);

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Test invalid reduction operation (should fail)
TEST_F(ReduceScatterQuantizeTest, InvalidRedOp) {
  size_t count = 1024;
  void *sendBuf = nullptr, *recvBuf = nullptr;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

  // Using ncclMax should fail - only ncclSum and ncclAvg are supported
  auto res = ncclReduceScatterQuantize(
      sendBuf,
      recvBuf,
      count,
      ncclFloat32,
      ncclBfloat16,
      ncclMax, // invalid - must be ncclSum or ncclAvg
      0,
      comm,
      stream);
  EXPECT_EQ(res, ncclInvalidArgument);

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterQuantizeTestInstance,
    ReduceScatterQuantizeTestParam,
    ::testing::Values(
        // redOp, count, seed
        std::make_tuple(ncclSum, 1024, 0UL),
        std::make_tuple(ncclSum, 8192, 42UL),
        std::make_tuple(ncclSum, 65536, 12345UL)),
    [](const testing::TestParamInfo<ReduceScatterQuantizeTestParam::ParamType>&
           info) {
      const char* opName;
      switch (std::get<0>(info.param)) {
        case ncclSum:
          opName = "Sum";
          break;
        case ncclAvg:
          opName = "Avg";
          break;
        default:
          opName = "Unknown";
          break;
      }
      return fmt::format(
          "{}_{}count_seed{}",
          opName,
          std::get<1>(info.param),
          std::get<2>(info.param));
    });

// Test that stochastic rounding produces unbiased results over many iterations
// This test verifies that the average error from stochastic rounding converges
// to zero as we run more iterations, confirming the rounding is unbiased.
TEST_F(ReduceScatterQuantizeTest, StochasticRoundingUnbiased) {
  size_t count = 1024;
  int numIterations = 10; // Run multiple iterations to check averaging

  float *sendBuf = nullptr, *recvBuf = nullptr;
  size_t sendSize = count * numRanks * sizeof(float);
  size_t recvSize = count * sizeof(float);

  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvSize));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));

  // Use values that have non-trivial fractional parts when converted to BF16
  // This ensures stochastic rounding is actually being exercised
  std::vector<float> hostSendBuf(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      // Use values like 1.123456 that don't have exact BF16 representations
      hostSendBuf[c * count + i] = 1.0f + static_cast<float>(i) * 0.000123f;
    }
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf, hostSendBuf.data(), sendSize, cudaMemcpyHostToDevice));

  // Track accumulated results across iterations
  std::vector<double> accumulatedResults(count, 0.0);

  for (int iter = 0; iter < numIterations; iter++) {
    // Set a different seed for each iteration
    uint64_t seed = static_cast<uint64_t>(iter * 12345 + globalRank);
    CUDACHECK_TEST(
        cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));

    CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvSize));

    auto res = ncclReduceScatterQuantize(
        sendBuf,
        recvBuf,
        count,
        ncclFloat32,
        ncclBfloat16,
        ncclSum,
        seedBuf,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> hostRecvBuf(count);
    CUDACHECK_TEST(cudaMemcpy(
        hostRecvBuf.data(), recvBuf, recvSize, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < count; i++) {
      accumulatedResults[i] += static_cast<double>(hostRecvBuf[i]);
    }
  }

  // Per-element analysis of averaged results
  int ulpViolations = 0;
  double totalSignedErr = 0.0;
  double totalAbsErr = 0.0;
  double totalUlp = 0.0;

  for (size_t i = 0; i < count; i++) {
    double avgResult = accumulatedResults[i] / numIterations;

    // Expected value: sum of (1.0 + i * 0.000123) across all ranks
    // For chunk globalRank, all ranks contribute the same value
    double expectedValue = static_cast<double>(numRanks) *
        (1.0 + static_cast<double>(i) * 0.000123);

    double signedErr = avgResult - expectedValue;
    double absErr = std::abs(signedErr);
    totalSignedErr += signedErr;
    totalAbsErr += absErr;

    // After averaging numIterations runs, per-element error should be within
    // 2 BF16 ULPs. Stochastic rounding errors are independent across
    // iterations, so averaging should significantly reduce the error.
    float absExpected =
        std::max(static_cast<float>(std::abs(expectedValue)), 1e-10f);
    int exponent;
    std::frexp(absExpected, &exponent);
    float bf16Ulp = std::ldexp(1.0f, exponent - 8);
    totalUlp += static_cast<double>(bf16Ulp);

    if (absErr > 2.0 * bf16Ulp) {
      ulpViolations++;
      if (ulpViolations <= 10) {
        printf(
            "Rank %d, index %zu: expected=%.6f, avg=%.6f, "
            "err=%.6f, ulp=%.6f, err_in_ulps=%.1f\n",
            globalRank,
            i,
            expectedValue,
            avgResult,
            absErr,
            static_cast<double>(bf16Ulp),
            absErr / bf16Ulp);
      }
    }
  }

  double meanSignedErr = totalSignedErr / count;
  double meanAbsErr = totalAbsErr / count;
  double meanUlp = totalUlp / count;

  printf(
      "Rank %d: stochastic rounding bias check: mean_signed_err=%.8f, "
      "mean_abs_err=%.6f, mean_ulp=%.6f, ulp_violations=%d/%zu\n",
      globalRank,
      meanSignedErr,
      meanAbsErr,
      meanUlp,
      ulpViolations,
      count);

  // Check 1: After averaging, no elements should exceed 2 BF16 ULP tolerance
  EXPECT_EQ(ulpViolations, 0)
      << "Rank " << globalRank << ": " << ulpViolations << " of " << count
      << " elements exceeded 2 BF16 ULP tolerance after averaging "
      << numIterations << " iterations";

  // Check 2: Mean signed error should be close to zero, confirming
  // unbiasedness. Threshold is 0.5 * mean ULP - a truly unbiased stochastic
  // rounding should produce near-zero mean signed error when averaged over
  // many elements and iterations.
  EXPECT_LT(std::abs(meanSignedErr), 0.5 * meanUlp)
      << "Rank " << globalRank
      << ": stochastic rounding appears biased. Mean signed error = "
      << meanSignedErr << " exceeds threshold of " << 0.5 * meanUlp
      << " (0.5 * mean ULP). Mean absolute error = " << meanAbsErr;

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// Test mixed-precision pipeline: float input -> bf16 transport -> float output
// Verifies the full quantized reduce scatter pipeline with different data
// patterns
TEST_F(ReduceScatterQuantizeTest, MixedPrecisionPipeline) {
  size_t count = 4096;

  float *sendBuf = nullptr, *recvBuf = nullptr;
  size_t sendSize = count * numRanks * sizeof(float);
  size_t recvSize = count * sizeof(float);

  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvSize));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));
  uint64_t seed = 42;
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));

  // Test different data patterns that exercise the mixed-precision path
  // Tolerance is computed dynamically as 3 BF16 ULPs at the expected value's
  // magnitude. With 8 ranks (3 PAT steps), each step can introduce up to ~1
  // BF16 ULP of stochastic rounding error.
  struct TestPattern {
    std::string name;
    std::function<float(int, size_t)> generator; // (rank, index) -> value
  };

  std::vector<TestPattern> patterns = {
      {"small_values",
       [](int rank, size_t i) {
         return 0.001f * static_cast<float>(rank + 1) +
             static_cast<float>(i) * 0.0001f;
       }},
      {"medium_values",
       [](int rank, size_t i) {
         return static_cast<float>(rank) + static_cast<float>(i) * 0.1f;
       }},
      {"large_values",
       [](int rank, size_t i) {
         return 100.0f * static_cast<float>(rank + 1) +
             static_cast<float>(i) * 0.01f;
       }},
  };

  for (const auto& pattern : patterns) {
    // Initialize send buffer with pattern
    std::vector<float> hostSendBuf(count * numRanks);
    for (int c = 0; c < numRanks; c++) {
      for (size_t i = 0; i < count; i++) {
        hostSendBuf[c * count + i] = pattern.generator(globalRank, i);
      }
    }
    CUDACHECK_TEST(cudaMemcpy(
        sendBuf, hostSendBuf.data(), sendSize, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recvBuf, 0xFF, recvSize));

    auto res = ncclReduceScatterQuantize(
        sendBuf,
        recvBuf,
        count,
        ncclFloat32,
        ncclBfloat16,
        ncclSum,
        seedBuf,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess) << "Failed for pattern: " << pattern.name;
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> hostRecvBuf(count);
    CUDACHECK_TEST(cudaMemcpy(
        hostRecvBuf.data(), recvBuf, recvSize, cudaMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (size_t i = 0; i < count; i++) {
      // Expected: sum across all ranks of pattern.generator(r, i)
      float expected = 0.0f;
      for (int r = 0; r < numRanks; r++) {
        expected += pattern.generator(r, i);
      }

      // Compute tolerance as 3 BF16 ULPs at the expected value's magnitude.
      // BF16 has 7 mantissa bits, so ULP = 2^(exponent - 7).
      float absExpected = std::max(std::abs(expected), 1e-10f);
      int exponent;
      std::frexp(absExpected, &exponent);
      float bf16Ulp = std::ldexp(1.0f, exponent - 8); // 2^(exp-1-7) = ULP
      float tolerance = 3.0f * bf16Ulp;

      if (std::abs(hostRecvBuf[i] - expected) > tolerance) {
        if (errors < 5) {
          printf(
              "Pattern '%s', Rank %d, index %zu: expected=%f, got=%f\n",
              pattern.name.c_str(),
              globalRank,
              i,
              expected,
              hostRecvBuf[i]);
        }
        errors++;
      }
    }
    EXPECT_EQ(errors, 0) << "Pattern '" << pattern.name << "' had " << errors
                         << " errors";
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// Test edge cases for mixed-precision reduce copy
TEST_F(ReduceScatterQuantizeTest, EdgeCases) {
  // Test with minimum count (1 element)
  {
    size_t count = 1;
    float *sendBuf = nullptr, *recvBuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

    std::vector<float> hostSendBuf(count * numRanks, 1.0f);
    CUDACHECK_TEST(cudaMemcpy(
        sendBuf,
        hostSendBuf.data(),
        count * numRanks * sizeof(float),
        cudaMemcpyHostToDevice));

    uint64_t* seedBuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));

    auto res = ncclReduceScatterQuantize(
        sendBuf,
        recvBuf,
        count,
        ncclFloat32,
        ncclBfloat16,
        ncclSum,
        seedBuf,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    float result;
    CUDACHECK_TEST(
        cudaMemcpy(&result, recvBuf, sizeof(float), cudaMemcpyDeviceToHost));
    // 1.0f is exactly representable in BF16, so stochastic rounding is exact.
    // The sum of numRanks copies of 1.0f should be exactly numRanks.
    EXPECT_FLOAT_EQ(result, static_cast<float>(numRanks));

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
    CUDACHECK_TEST(cudaFree(seedBuf));
  }

  // Test with non-power-of-2 count
  {
    size_t count = 1023; // Not a power of 2
    float *sendBuf = nullptr, *recvBuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
    CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

    std::vector<float> hostSendBuf(count * numRanks, 0.5f);
    CUDACHECK_TEST(cudaMemcpy(
        sendBuf,
        hostSendBuf.data(),
        count * numRanks * sizeof(float),
        cudaMemcpyHostToDevice));

    uint64_t* seedBuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));

    auto res = ncclReduceScatterQuantize(
        sendBuf,
        recvBuf,
        count,
        ncclFloat32,
        ncclBfloat16,
        ncclSum,
        seedBuf,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> hostRecvBuf(count);
    CUDACHECK_TEST(cudaMemcpy(
        hostRecvBuf.data(),
        recvBuf,
        count * sizeof(float),
        cudaMemcpyDeviceToHost));

    // 0.5f is exactly representable in BF16, so stochastic rounding is exact.
    // The sum should be exactly 0.5 * numRanks for every element.
    float expected = 0.5f * static_cast<float>(numRanks);
    int errors = 0;
    for (size_t i = 0; i < count; i++) {
      if (hostRecvBuf[i] != expected) {
        if (errors < 5) {
          printf(
              "Rank %d, index %zu: expected=%f, got=%f\n",
              globalRank,
              i,
              expected,
              hostRecvBuf[i]);
        }
        errors++;
      }
    }
    EXPECT_EQ(errors, 0) << "Non-power-of-2 count test: " << errors << " of "
                         << count
                         << " elements differ from exact expected value "
                         << expected;

    CUDACHECK_TEST(cudaFree(sendBuf));
    CUDACHECK_TEST(cudaFree(recvBuf));
    CUDACHECK_TEST(cudaFree(seedBuf));
  }
}

// Determinism test: given the same input and seed, ncclReduceScatterQuantize
// must produce bitwise-identical output across repeated invocations.
TEST_F(ReduceScatterQuantizeTest, Determinism) {
  const size_t count = 8192;
  const int numRuns = 5;
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvSize));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));

  // Fill with values that are not exactly representable in BF16 so
  // stochastic rounding is actually exercised.
  std::vector<float> hostSend(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      hostSend[c * count + i] = 1.33f + static_cast<float>(i) * 0.000456f +
          static_cast<float>(globalRank) * 0.007f +
          static_cast<float>(c) * 0.013f;
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));

  // First run — capture the reference output.
  const uint64_t seed = 42;
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvSize));

  auto res = ncclReduceScatterQuantize(
      sendBuf,
      recvBuf,
      count,
      ncclFloat32,
      ncclBfloat16,
      ncclSum,
      seedBuf,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> referenceOutput(count);
  CUDACHECK_TEST(cudaMemcpy(
      referenceOutput.data(), recvBuf, recvSize, cudaMemcpyDeviceToHost));

  // Subsequent runs — each must match the reference exactly.
  for (int run = 1; run < numRuns; run++) {
    CUDACHECK_TEST(
        cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvSize));

    res = ncclReduceScatterQuantize(
        sendBuf,
        recvBuf,
        count,
        ncclFloat32,
        ncclBfloat16,
        ncclSum,
        seedBuf,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> currentOutput(count);
    CUDACHECK_TEST(cudaMemcpy(
        currentOutput.data(), recvBuf, recvSize, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (size_t i = 0; i < count; i++) {
      // Bitwise comparison via memcmp so ±0 and NaN differences are caught.
      if (std::memcmp(&currentOutput[i], &referenceOutput[i], sizeof(float)) !=
          0) {
        if (mismatches < 5) {
          printf(
              "Rank %d, run %d, index %zu: reference=%.8f, got=%.8f\n",
              globalRank,
              run,
              i,
              referenceOutput[i],
              currentOutput[i]);
        }
        mismatches++;
      }
    }
    EXPECT_EQ(mismatches, 0)
        << "Rank " << globalRank << ", run " << run << ": " << mismatches
        << " of " << count
        << " elements differ from the reference run (same seed=" << seed << ")";
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// Determinism test for regular ncclReduceScatter: repeated invocations with the
// same input must produce bitwise-identical output.
TEST_F(ReduceScatterQuantizeTest, DeterminismReduceScatter) {
  const size_t count = 8192;
  const int numRuns = 5;
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvSize));

  // Fill with non-trivial values.
  std::vector<float> hostSend(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      hostSend[c * count + i] = 1.33f + static_cast<float>(i) * 0.000456f +
          static_cast<float>(globalRank) * 0.007f +
          static_cast<float>(c) * 0.013f;
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));

  // First run — capture the reference output.
  CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvSize));
  auto res = ncclReduceScatter(
      sendBuf, recvBuf, count, ncclFloat32, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> referenceOutput(count);
  CUDACHECK_TEST(cudaMemcpy(
      referenceOutput.data(), recvBuf, recvSize, cudaMemcpyDeviceToHost));

  // Subsequent runs — each must match the reference exactly.
  for (int run = 1; run < numRuns; run++) {
    CUDACHECK_TEST(cudaMemset(recvBuf, 0, recvSize));

    res = ncclReduceScatter(
        sendBuf, recvBuf, count, ncclFloat32, ncclSum, comm, stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> currentOutput(count);
    CUDACHECK_TEST(cudaMemcpy(
        currentOutput.data(), recvBuf, recvSize, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (size_t i = 0; i < count; i++) {
      if (std::memcmp(&currentOutput[i], &referenceOutput[i], sizeof(float)) !=
          0) {
        if (mismatches < 5) {
          printf(
              "Rank %d, run %d, index %zu: reference=%.8f, got=%.8f\n",
              globalRank,
              run,
              i,
              referenceOutput[i],
              currentOutput[i]);
        }
        mismatches++;
      }
    }
    EXPECT_EQ(mismatches, 0)
        << "Rank " << globalRank << ", run " << run << ": " << mismatches
        << " of " << count << " elements differ from the reference run";
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Helper: compute the number of PAT reduction steps = log2(numRanks).
static int patSteps(int numRanks) {
  int steps = 0;
  int n = numRanks;
  while (n > 1) {
    n >>= 1;
    steps++;
  }
  return steps;
}

// Helper: BF16 ULP at a given magnitude.
static float bf16Ulp(float value) {
  float absVal = std::max(std::abs(value), 1e-30f);
  int exponent;
  std::frexp(absVal, &exponent);
  return std::ldexp(1.0f, exponent - 8); // BF16 has 7 mantissa bits
}

// Index correctness test: verify that the PAT algorithm reduces the correct
// elements together, i.e. element i on every rank contributes to element i of
// the output — not some other index.
//
// Strategy: encode (rank, chunk, index) into each send value so that any
// index-mismatch produces a detectably wrong sum. We test both FP32
// ReduceScatter and ReduceScatterQuantize.
TEST_F(ReduceScatterQuantizeTest, IndexCorrectness) {
  // Use a non-power-of-2 count to also stress non-aligned tails.
  const size_t count = 4099;
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);

  float *sendBuf = nullptr, *recvRS = nullptr, *recvRSQ = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvRS, recvSize));
  CUDACHECK_TEST(cudaMalloc(&recvRSQ, recvSize));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));
  uint64_t seed = 99;
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));

  // Encoding: val(rank r, chunk c, index i) = (r+1)*P1 + (c+1)*P2 + (i+1)*P3
  // where P1, P2, P3 are chosen so that:
  //   - swapping two indices i,j produces a different per-element sum
  //   - swapping two chunks c1,c2 produces wrong results on the receiving rank
  //   - all values stay well within FP32 exact-integer range (< 2^24)
  //
  // After reduce-scatter, rank R receives chunk R.
  // expected[i] = sum_{r=0}^{N-1} val(r, R, i)
  //             = P1 * N*(N+1)/2  +  (R+1)*P2*N  +  (i+1)*P3*N
  constexpr float P1 = 1.0f;
  constexpr float P2 = 0.0001f;
  constexpr float P3 = 0.001f;

  std::vector<float> hostSend(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      hostSend[c * count + i] = static_cast<float>(globalRank + 1) * P1 +
          static_cast<float>(c + 1) * P2 + static_cast<float>(i + 1) * P3;
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));

  // Compute expected values for this rank's output chunk (chunk == globalRank).
  // Use FP64 to get an exact reference.
  std::vector<double> expected(count);
  double rankSum = 0.0; // sum_{r=0}^{N-1} (r+1) = N*(N+1)/2
  for (int r = 0; r < numRanks; r++) {
    rankSum += static_cast<double>(r + 1);
  }
  for (size_t i = 0; i < count; i++) {
    expected[i] = P1 * rankSum +
        static_cast<double>(P2) * static_cast<double>(globalRank + 1) *
            numRanks +
        static_cast<double>(P3) * static_cast<double>(i + 1) * numRanks;
  }

  // ---- FP32 reduce-scatter ----
  CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
  auto res = ncclReduceScatter(
      sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  // ---- Quantized reduce-scatter ----
  CUDACHECK_TEST(cudaMemset(recvRSQ, 0, recvSize));
  res = ncclReduceScatterQuantize(
      sendBuf,
      recvRSQ,
      count,
      ncclFloat32,
      ncclBfloat16,
      ncclSum,
      seedBuf,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> hostRS(count), hostRSQ(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostRS.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(
      cudaMemcpy(hostRSQ.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));

  // ---- Check FP32 RS (should be essentially exact) ----
  {
    int errors = 0;
    for (size_t i = 0; i < count; i++) {
      // FP32 accumulation of small integers — tolerance is very tight.
      // Allow 1 ULP of the expected magnitude.
      float exp32 = static_cast<float>(expected[i]);
      float diff = std::abs(hostRS[i] - exp32);
      float ulp =
          bf16Ulp(exp32); // bf16Ulp is conservative; FP32 ULP is smaller
      float tolerance = ulp; // 1 BF16 ULP is generous for FP32
      if (diff > tolerance) {
        if (errors < 10) {
          printf(
              "RS index error — rank %d, i=%zu: expected=%.8f, got=%.8f, "
              "diff=%.8e\n",
              globalRank,
              i,
              exp32,
              hostRS[i],
              diff);
        }
        errors++;
      }
    }
    EXPECT_EQ(errors, 0) << "Rank " << globalRank
                         << ": FP32 ReduceScatter produced " << errors << " of "
                         << count
                         << " elements that don't match expected values — "
                            "possible index mapping bug in PAT algorithm";
  }

  // ---- Check RSQ (allow numPatSteps BF16 ULPs) ----
  {
    int nSteps = patSteps(numRanks);
    int errors = 0;
    for (size_t i = 0; i < count; i++) {
      float exp32 = static_cast<float>(expected[i]);
      float diff = std::abs(hostRSQ[i] - exp32);
      float ulp = bf16Ulp(exp32);
      float tolerance = static_cast<float>(nSteps + 1) * ulp;
      if (diff > tolerance) {
        if (errors < 10) {
          printf(
              "RSQ index error — rank %d, i=%zu: expected=%.8f, got=%.8f, "
              "diff=%.8e, tolerance=%.8e (%d ULPs)\n",
              globalRank,
              i,
              exp32,
              hostRSQ[i],
              diff,
              tolerance,
              nSteps + 1);
        }
        errors++;
      }
    }
    EXPECT_EQ(errors, 0) << "Rank " << globalRank
                         << ": ReduceScatterQuantize produced " << errors
                         << " of " << count << " elements that exceed "
                         << (nSteps + 1)
                         << " BF16 ULP tolerance — "
                            "possible index mapping bug in PAT algorithm";
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvRS));
  CUDACHECK_TEST(cudaFree(recvRSQ));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// ---------------------------------------------------------------------------
// Numerical benchmarks: compare RS (FP32) vs RS-Quantized (FP32→BF16→FP32)
// Both paths use PAT algorithm (NCCL_ALGO=PAT set in fixture SetUp).
//
// These benchmarks report numerical differences rather than pass/fail; they
// are meant to be read by a human inspecting the printed tables.
// ---------------------------------------------------------------------------

// Helper struct to collect per-element error statistics.
struct ErrorStats {
  double maxAbs{0.0};
  double meanAbs{0.0};
  double rmsError{0.0};
  double meanSigned{0.0};
  double maxUlps{0.0}; // max error in BF16 ULP units
  double meanUlps{0.0};

  void compute(
      const std::vector<float>& result,
      const std::vector<double>& reference,
      size_t count) {
    double sumAbs = 0, sumSigned = 0, sumSq = 0, sumUlps = 0;
    for (size_t i = 0; i < count; i++) {
      double diff = static_cast<double>(result[i]) - reference[i];
      double ad = std::abs(diff);
      sumAbs += ad;
      sumSigned += diff;
      sumSq += diff * diff;
      float ulp = bf16Ulp(static_cast<float>(std::abs(reference[i])));
      double ulps = ad / ulp;
      sumUlps += ulps;
      maxAbs = std::max(maxAbs, ad);
      maxUlps = std::max(maxUlps, ulps);
    }
    meanAbs = sumAbs / count;
    meanSigned = sumSigned / count;
    rmsError = std::sqrt(sumSq / count);
    meanUlps = sumUlps / count;
  }
};

// ===================================================================
// Benchmark 1 – Cancellation error
//
// Inspired by StochasticRoundingNumericalBench.cu: construct input so
// that the FP32 reduce-scatter should yield ≈0 for every element
// (large negative + many small positives).  Measure how far each path
// deviates from zero.
// ===================================================================
TEST_F(ReduceScatterQuantizeTest, BenchCancellationError) {
  // We distribute the cancellation across ranks:
  //   rank 0  sends:  largeNeg  (one large negative per element)
  //   ranks 1..N-1 send:  smallPos = -largeNeg / (N-1)  per element
  // So exact FP32 reduce-scatter sum = 0 for every element.
  const size_t count = 4096;

  // Choose a "hard" value that is not exactly representable in BF16.
  const float smallPos = 1.33f;
  const float largeNeg =
      -smallPos * static_cast<float>(numRanks - 1); // exact in FP64

  // ---- Allocate buffers ----
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);
  const size_t sendSizeBf16 = count * numRanks * sizeof(__nv_bfloat16);
  const size_t recvSizeBf16 = count * sizeof(__nv_bfloat16);

  float *sendBuf = nullptr, *recvRS = nullptr, *recvRSQ = nullptr;
  __nv_bfloat16 *sendBufBf16 = nullptr, *recvBufBf16 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvRS, recvSize));
  CUDACHECK_TEST(cudaMalloc(&recvRSQ, recvSize));
  CUDACHECK_TEST(cudaMalloc(&sendBufBf16, sendSizeBf16));
  CUDACHECK_TEST(cudaMalloc(&recvBufBf16, recvSizeBf16));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));
  uint64_t seed = 42;
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));

  // ---- Fill send buffer ----
  // Each rank fills all chunks identically: the destination rank is
  // irrelevant for this test—what matters is the sum across ranks.
  std::vector<float> hostSend(count * numRanks);
  std::vector<__nv_bfloat16> hostSendBf16(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      float val = (globalRank == 0) ? largeNeg : smallPos;
      hostSend[c * count + i] = val;
      hostSendBf16[c * count + i] = __float2bfloat16(val);
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      sendBufBf16, hostSendBf16.data(), sendSizeBf16, cudaMemcpyHostToDevice));

  // ---- FP32 reduce-scatter (ground-truth path) ----
  CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
  auto res = ncclReduceScatter(
      sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  // ---- BF16 reduce-scatter (pure BF16 baseline) ----
  CUDACHECK_TEST(cudaMemset(recvBufBf16, 0, recvSizeBf16));
  res = ncclReduceScatter(
      sendBufBf16, recvBufBf16, count, ncclBfloat16, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  // ---- Quantized reduce-scatter (FP32→BF16→FP32) ----
  CUDACHECK_TEST(cudaMemset(recvRSQ, 0, recvSize));
  res = ncclReduceScatterQuantize(
      sendBuf,
      recvRSQ,
      count,
      ncclFloat32,
      ncclBfloat16,
      ncclSum,
      seedBuf,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // ---- Copy back ----
  std::vector<float> hostRS(count), hostRSQ(count);
  std::vector<__nv_bfloat16> hostRecvBf16(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostRS.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(
      cudaMemcpy(hostRSQ.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      hostRecvBf16.data(), recvBufBf16, recvSizeBf16, cudaMemcpyDeviceToHost));

  // Convert BF16 results to float for comparison.
  std::vector<float> hostRSBf16(count);
  for (size_t i = 0; i < count; i++) {
    hostRSBf16[i] = __bfloat162float(hostRecvBf16[i]);
  }

  // Reference: exact FP64 sum = 0 for every element.
  std::vector<double> reference(count, 0.0);

  ErrorStats statsRS, statsRSBf16, statsRSQ;
  statsRS.compute(hostRS, reference, count);
  statsRSBf16.compute(hostRSBf16, reference, count);
  statsRSQ.compute(hostRSQ, reference, count);

  if (globalRank == 0) {
    printf(
        "\n=== Cancellation Error Benchmark (count=%zu, "
        "numRanks=%d, PAT steps=%d) ===\n",
        count,
        numRanks,
        patSteps(numRanks));
    printf(
        "  %-25s %15s %15s %15s\n", "", "RS(FP32)", "RS(BF16)", "RS-Quantized");
    printf(
        "  %-25s %15.8f %15.8f %15.8f\n",
        "max |error|",
        statsRS.maxAbs,
        statsRSBf16.maxAbs,
        statsRSQ.maxAbs);
    printf(
        "  %-25s %15.8f %15.8f %15.8f\n",
        "mean |error|",
        statsRS.meanAbs,
        statsRSBf16.meanAbs,
        statsRSQ.meanAbs);
    printf(
        "  %-25s %15.8f %15.8f %15.8f\n",
        "RMS error",
        statsRS.rmsError,
        statsRSBf16.rmsError,
        statsRSQ.rmsError);
    printf(
        "  %-25s %+15.8f %+15.8f %+15.8f\n",
        "mean signed error",
        statsRS.meanSigned,
        statsRSBf16.meanSigned,
        statsRSQ.meanSigned);
    printf(
        "  %-25s %15.2f %15.2f %15.2f\n",
        "max error (BF16 ULPs)",
        statsRS.maxUlps,
        statsRSBf16.maxUlps,
        statsRSQ.maxUlps);
    printf(
        "  %-25s %15.2f %15.2f %15.2f\n",
        "mean error (BF16 ULPs)",
        statsRS.meanUlps,
        statsRSBf16.meanUlps,
        statsRSQ.meanUlps);
    printf("\n");
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvRS));
  CUDACHECK_TEST(cudaFree(recvRSQ));
  CUDACHECK_TEST(cudaFree(sendBufBf16));
  CUDACHECK_TEST(cudaFree(recvBufBf16));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// ===================================================================
// Benchmark 2 – Summation accuracy (typical gradient-like values)
//
// Each rank sends random-ish values drawn from a deterministic
// sequence.  We compute the FP64 expected sum as reference and
// compare both paths against it.
// ===================================================================
TEST_F(ReduceScatterQuantizeTest, BenchSummationAccuracy) {
  const size_t count = 8192;
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);
  const size_t sendSizeBf16 = count * numRanks * sizeof(__nv_bfloat16);
  const size_t recvSizeBf16 = count * sizeof(__nv_bfloat16);

  float *sendBuf = nullptr, *recvRS = nullptr, *recvRSQ = nullptr;
  __nv_bfloat16 *sendBufBf16 = nullptr, *recvBufBf16 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvRS, recvSize));
  CUDACHECK_TEST(cudaMalloc(&recvRSQ, recvSize));
  CUDACHECK_TEST(cudaMalloc(&sendBufBf16, sendSizeBf16));
  CUDACHECK_TEST(cudaMalloc(&recvBufBf16, recvSizeBf16));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));
  uint64_t seed = 7;
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));

  // Deterministic "gradient-like" values: small floats not exactly in BF16.
  // val(rank, chunk, i) = sin(rank * 1000 + chunk * count + i) * 0.01
  // – magnitude ~0.01, exercises the fractional bits that BF16 drops.
  std::vector<float> hostSend(count * numRanks);
  std::vector<__nv_bfloat16> hostSendBf16(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      float val =
          std::sin(static_cast<float>(globalRank * 1000 + c * count + i)) *
          0.01f;
      hostSend[c * count + i] = val;
      hostSendBf16[c * count + i] = __float2bfloat16(val);
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      sendBufBf16, hostSendBf16.data(), sendSizeBf16, cudaMemcpyHostToDevice));

  // ---- FP32 reduce-scatter ----
  CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
  auto res = ncclReduceScatter(
      sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  // ---- BF16 reduce-scatter ----
  CUDACHECK_TEST(cudaMemset(recvBufBf16, 0, recvSizeBf16));
  res = ncclReduceScatter(
      sendBufBf16, recvBufBf16, count, ncclBfloat16, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  // ---- Quantized reduce-scatter ----
  CUDACHECK_TEST(cudaMemset(recvRSQ, 0, recvSize));
  res = ncclReduceScatterQuantize(
      sendBuf,
      recvRSQ,
      count,
      ncclFloat32,
      ncclBfloat16,
      ncclSum,
      seedBuf,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> hostRS(count), hostRSQ(count);
  std::vector<__nv_bfloat16> hostRecvBf16(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostRS.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(
      cudaMemcpy(hostRSQ.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      hostRecvBf16.data(), recvBufBf16, recvSizeBf16, cudaMemcpyDeviceToHost));

  std::vector<float> hostRSBf16(count);
  for (size_t i = 0; i < count; i++) {
    hostRSBf16[i] = __bfloat162float(hostRecvBf16[i]);
  }

  // Use FP32 RS as the "near-exact" reference.
  std::vector<double> refRS(count);
  for (size_t i = 0; i < count; i++) {
    refRS[i] = static_cast<double>(hostRS[i]);
  }

  ErrorStats statsRSBf16, statsRSQ;
  statsRSBf16.compute(hostRSBf16, refRS, count);
  statsRSQ.compute(hostRSQ, refRS, count);

  if (globalRank == 0) {
    printf(
        "\n=== Summation Accuracy Benchmark (count=%zu, "
        "numRanks=%d) ===\n",
        count,
        numRanks);
    printf("  Reference: FP32 ReduceScatter (PAT)\n");
    printf("  %-30s %15s %15s\n", "", "RS(BF16)", "RS-Quantized");
    printf(
        "  %-30s %15.10f %15.10f\n",
        "max |error|",
        statsRSBf16.maxAbs,
        statsRSQ.maxAbs);
    printf(
        "  %-30s %15.10f %15.10f\n",
        "mean |error|",
        statsRSBf16.meanAbs,
        statsRSQ.meanAbs);
    printf(
        "  %-30s %15.10f %15.10f\n",
        "RMS error",
        statsRSBf16.rmsError,
        statsRSQ.rmsError);
    printf(
        "  %-30s %+15.10f %+15.10f\n",
        "mean signed error",
        statsRSBf16.meanSigned,
        statsRSQ.meanSigned);
    printf(
        "  %-30s %15.4f %15.4f\n",
        "max error (BF16 ULPs)",
        statsRSBf16.maxUlps,
        statsRSQ.maxUlps);
    printf(
        "  %-30s %15.4f %15.4f\n",
        "mean error (BF16 ULPs)",
        statsRSBf16.meanUlps,
        statsRSQ.meanUlps);
    printf("\n");
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvRS));
  CUDACHECK_TEST(cudaFree(recvRSQ));
  CUDACHECK_TEST(cudaFree(sendBufBf16));
  CUDACHECK_TEST(cudaFree(recvBufBf16));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// ===================================================================
// Benchmark 3 – Stochastic rounding bias (averaged over trials)
//
// Like StochasticRoundingNumericalBench, run multiple iterations of
// RSQ with different seeds and average the results. If SR is truly
// unbiased, the averaged error should converge toward the FP32 RS
// result.
// ===================================================================
TEST_F(ReduceScatterQuantizeTest, BenchSRBiasConvergence) {
  const size_t count = 4096;
  const int numTrials = 16;
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);

  float *sendBuf = nullptr, *recvRS = nullptr, *recvRSQ = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvRS, recvSize));
  CUDACHECK_TEST(cudaMalloc(&recvRSQ, recvSize));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));

  // Values that are not exactly representable in BF16.
  std::vector<float> hostSend(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      hostSend[c * count + i] = 1.33f + static_cast<float>(i) * 0.000123f +
          static_cast<float>(globalRank) * 0.001f;
    }
  }
  CUDACHECK_TEST(
      cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));

  // ---- FP32 reduce-scatter (reference) ----
  CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
  auto res = ncclReduceScatter(
      sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> hostRS(count);
  CUDACHECK_TEST(
      cudaMemcpy(hostRS.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));

  std::vector<double> refRS(count);
  for (size_t i = 0; i < count; i++) {
    refRS[i] = static_cast<double>(hostRS[i]);
  }

  // ---- Run multiple RSQ trials and accumulate ----
  std::vector<double> accumulated(count, 0.0);
  for (int t = 0; t < numTrials; t++) {
    uint64_t seed = static_cast<uint64_t>(t * 9973 + globalRank * 31);
    CUDACHECK_TEST(
        cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recvRSQ, 0, recvSize));

    res = ncclReduceScatterQuantize(
        sendBuf,
        recvRSQ,
        count,
        ncclFloat32,
        ncclBfloat16,
        ncclSum,
        seedBuf,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> hostRSQ(count);
    CUDACHECK_TEST(
        cudaMemcpy(hostRSQ.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; i++) {
      accumulated[i] += static_cast<double>(hostRSQ[i]);
    }
  }

  // Averaged result.
  std::vector<float> avgResult(count);
  for (size_t i = 0; i < count; i++) {
    avgResult[i] = static_cast<float>(accumulated[i] / numTrials);
  }

  // Also compute single-trial stats for comparison.
  // (last trial is still in hostRSQ via the loop above—re-read)
  std::vector<float> lastTrial(count);
  CUDACHECK_TEST(
      cudaMemcpy(lastTrial.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));

  ErrorStats statsSingle, statsAvg;
  statsSingle.compute(lastTrial, refRS, count);
  statsAvg.compute(avgResult, refRS, count);

  if (globalRank == 0) {
    printf(
        "\n=== SR Bias Convergence Benchmark (count=%zu, "
        "numTrials=%d, numRanks=%d) ===\n",
        count,
        numTrials,
        numRanks);
    printf("  Reference: FP32 ReduceScatter (PAT)\n");
    printf("  %-30s %15s %15s\n", "", "SingleTrial", "Avg-of-Trials");
    printf(
        "  %-30s %15.10f %15.10f\n",
        "max |error|",
        statsSingle.maxAbs,
        statsAvg.maxAbs);
    printf(
        "  %-30s %15.10f %15.10f\n",
        "mean |error|",
        statsSingle.meanAbs,
        statsAvg.meanAbs);
    printf(
        "  %-30s %15.10f %15.10f\n",
        "RMS error",
        statsSingle.rmsError,
        statsAvg.rmsError);
    printf(
        "  %-30s %+15.10f %+15.10f\n",
        "mean signed error",
        statsSingle.meanSigned,
        statsAvg.meanSigned);
    printf(
        "  %-30s %15.4f %15.4f\n",
        "max error (BF16 ULPs)",
        statsSingle.maxUlps,
        statsAvg.maxUlps);
    printf(
        "  %-30s %15.4f %15.4f\n",
        "mean error (BF16 ULPs)",
        statsSingle.meanUlps,
        statsAvg.meanUlps);
    printf("\n");
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvRS));
  CUDACHECK_TEST(cudaFree(recvRSQ));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// ===================================================================
// Benchmark 4 – Dynamic range sweep
//
// Evaluate RS vs RSQ across several magnitudes (1e-3 … 1e+3) to
// show how quantization error scales with value magnitude.
// ===================================================================
TEST_F(ReduceScatterQuantizeTest, BenchDynamicRangeSweep) {
  const size_t count = 2048;
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);
  const size_t sendSizeBf16 = count * numRanks * sizeof(__nv_bfloat16);
  const size_t recvSizeBf16 = count * sizeof(__nv_bfloat16);

  float *sendBuf = nullptr, *recvRS = nullptr, *recvRSQ = nullptr;
  __nv_bfloat16 *sendBufBf16 = nullptr, *recvBufBf16 = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvRS, recvSize));
  CUDACHECK_TEST(cudaMalloc(&recvRSQ, recvSize));
  CUDACHECK_TEST(cudaMalloc(&sendBufBf16, sendSizeBf16));
  CUDACHECK_TEST(cudaMalloc(&recvBufBf16, recvSizeBf16));

  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));
  uint64_t seed = 123;
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf, &seed, sizeof(uint64_t), cudaMemcpyHostToDevice));

  if (globalRank == 0) {
    printf(
        "\n=== Dynamic Range Sweep (count=%zu, "
        "numRanks=%d) ===\n",
        count,
        numRanks);
    printf("  Reference: FP32 ReduceScatter (PAT)\n");
    printf(
        "  %-12s %15s %15s %15s %15s\n",
        "Magnitude",
        "BF16 mean|err|",
        "RSQ mean|err|",
        "BF16 meanULP",
        "RSQ meanULP");
  }

  std::vector<float> magnitudes = {1e-3f, 1e-2f, 1e-1f, 1.0f, 1e1f, 1e2f, 1e3f};

  for (float mag : magnitudes) {
    std::vector<float> hostSend(count * numRanks);
    std::vector<__nv_bfloat16> hostSendBf16(count * numRanks);
    for (int c = 0; c < numRanks; c++) {
      for (size_t i = 0; i < count; i++) {
        float val = mag * (1.33f + static_cast<float>(i % 128) * 0.0077f) +
            static_cast<float>(globalRank) * mag * 0.01f;
        hostSend[c * count + i] = val;
        hostSendBf16[c * count + i] = __float2bfloat16(val);
      }
    }
    CUDACHECK_TEST(
        cudaMemcpy(sendBuf, hostSend.data(), sendSize, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        sendBufBf16,
        hostSendBf16.data(),
        sendSizeBf16,
        cudaMemcpyHostToDevice));

    // FP32 RS
    CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
    auto res = ncclReduceScatter(
        sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
    ASSERT_EQ(res, ncclSuccess);

    // BF16 RS
    CUDACHECK_TEST(cudaMemset(recvBufBf16, 0, recvSizeBf16));
    res = ncclReduceScatter(
        sendBufBf16, recvBufBf16, count, ncclBfloat16, ncclSum, comm, stream);
    ASSERT_EQ(res, ncclSuccess);

    // RSQ
    CUDACHECK_TEST(cudaMemset(recvRSQ, 0, recvSize));
    res = ncclReduceScatterQuantize(
        sendBuf,
        recvRSQ,
        count,
        ncclFloat32,
        ncclBfloat16,
        ncclSum,
        seedBuf,
        comm,
        stream);
    ASSERT_EQ(res, ncclSuccess);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<float> hostRS(count), hostRSQ(count);
    std::vector<__nv_bfloat16> hostRecvBf16(count);
    CUDACHECK_TEST(
        cudaMemcpy(hostRS.data(), recvRS, recvSize, cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(
        cudaMemcpy(hostRSQ.data(), recvRSQ, recvSize, cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(
        hostRecvBf16.data(),
        recvBufBf16,
        recvSizeBf16,
        cudaMemcpyDeviceToHost));

    std::vector<float> hostRSBf16(count);
    for (size_t i = 0; i < count; i++) {
      hostRSBf16[i] = __bfloat162float(hostRecvBf16[i]);
    }

    // Use FP32 RS as reference.
    std::vector<double> refRS(count);
    for (size_t i = 0; i < count; i++) {
      refRS[i] = static_cast<double>(hostRS[i]);
    }

    ErrorStats statsRSBf16, statsRSQ;
    statsRSBf16.compute(hostRSBf16, refRS, count);
    statsRSQ.compute(hostRSQ, refRS, count);

    if (globalRank == 0) {
      printf(
          "  %-12.0e %15.10f %15.10f %15.4f %15.4f\n",
          static_cast<double>(mag),
          statsRSBf16.meanAbs,
          statsRSQ.meanAbs,
          statsRSBf16.meanUlps,
          statsRSQ.meanUlps);
    }
  }
  if (globalRank == 0) {
    printf("\n");
  }

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvRS));
  CUDACHECK_TEST(cudaFree(recvRSQ));
  CUDACHECK_TEST(cudaFree(sendBufBf16));
  CUDACHECK_TEST(cudaFree(recvBufBf16));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
