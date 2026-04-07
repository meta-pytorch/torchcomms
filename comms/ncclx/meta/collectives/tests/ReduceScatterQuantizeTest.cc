// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// API validation and basic functionality tests for ncclReduceScatterQuantize.
// Numerical correctness tests are in ReduceScatterQuantizeNumericalTest.cc.
// Numerical benchmarks are in
// benchmarks/ReduceScatterQuantizeNumericalBench.cc.

#include <comm.h>
#include <cuda_bf16.h>
#include <fmt/core.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include "comms/ncclx/meta/collectives/tests/ReduceScatterQuantizeTestUtils.h"

// ---------------------------------------------------------------------------
// Parameterized test class for CorrectReduction.
// ---------------------------------------------------------------------------
class ReduceScatterQuantizeTestParam
    : public ReduceScatterQuantizeTest,
      public ::testing::WithParamInterface<
          std::tuple<ncclRedOp_t, size_t, uint64_t>> {};

// ---------------------------------------------------------------------------
// CorrectReduction: verify that ncclReduceScatterQuantize produces results
// within numPatSteps BF16 ULPs of the analytically computed expected value,
// and that its mean absolute error is no worse than 1.5x the pure BF16
// reduce-scatter baseline.
// ---------------------------------------------------------------------------
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

  int numPatSteps = patSteps(numRanks);

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
    float ulp = bf16Ulp(expected);
    float tolerance = static_cast<float>(numPatSteps) * ulp;

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
            quantizeDiff / ulp);
      }
    }
  }

  double meanQuantizeErr = totalQuantizeErr / count;
  double meanBf16Err = totalBf16Err / count;
  double meanSignedErr = totalSignedQuantizeErr / static_cast<double>(count);

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

  // Check 1: No per-element ULP violations.
  EXPECT_EQ(quantizeUlpViolations, 0)
      << "Rank " << globalRank << ": " << quantizeUlpViolations << " of "
      << count << " elements exceeded " << numPatSteps
      << " BF16 ULP tolerance. Max quantize error: " << maxQuantizeErr
      << ", max BF16 error: " << maxBf16Err;

  // Check 2: Quantized path's MAE should be no worse than the BF16 baseline.
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

// ---------------------------------------------------------------------------
// InvalidInputType: passing ncclFloat16 as inputType must return
// ncclInvalidArgument (only ncclFloat32 is supported).
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, InvalidInputType) {
  size_t count = 1024;
  void *sendBuf = nullptr, *recvBuf = nullptr;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

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

// ---------------------------------------------------------------------------
// InvalidTransportType: passing ncclFloat16 as transportType must return
// ncclInvalidArgument (only ncclBfloat16 is supported).
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, InvalidTransportType) {
  size_t count = 1024;
  void *sendBuf = nullptr, *recvBuf = nullptr;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

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

// ---------------------------------------------------------------------------
// InvalidRedOp: passing ncclMax must return ncclInvalidArgument
// (only ncclSum and ncclAvg are supported).
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, InvalidRedOp) {
  size_t count = 1024;
  void *sendBuf = nullptr, *recvBuf = nullptr;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

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

// ---------------------------------------------------------------------------
// MixedPrecisionPipeline: verify the full FP32→BF16→FP32 pipeline with
// three data patterns (small, medium, large values). Each output element
// must be within 3 BF16 ULPs of the analytically expected sum.
// ---------------------------------------------------------------------------
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

  struct TestPattern {
    std::string name;
    std::function<float(int, size_t)> generator;
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

    int errors = 0;
    for (size_t i = 0; i < count; i++) {
      float expected = 0.0f;
      for (int r = 0; r < numRanks; r++) {
        expected += pattern.generator(r, i);
      }

      float tolerance = 3.0f * bf16Ulp(expected);
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

// ---------------------------------------------------------------------------
// MinimumCount: the smallest possible reduce-scatter (count=1) with an
// exactly BF16-representable value. Stochastic rounding is exact, so the
// result must match exactly.
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, MinimumCount) {
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

// ---------------------------------------------------------------------------
// NonPowerOfTwoCount: a non-power-of-2 element count (1023) to stress
// non-aligned tail handling in the PAT kernel. Uses exactly
// BF16-representable values so the result must match exactly.
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, NonPowerOfTwoCount) {
  size_t count = 1023;
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
                       << count << " elements differ from exact expected value "
                       << expected;

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
  CUDACHECK_TEST(cudaFree(seedBuf));
}

// ---------------------------------------------------------------------------
// Determinism: given the same input and seed, ncclReduceScatterQuantize
// must produce bitwise-identical output across repeated invocations.
// ---------------------------------------------------------------------------
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
  const float kBase = 1.33f;
  const float kUlp = bf16Ulp(kBase);
  std::vector<float> hostSend(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      hostSend[c * count + i] = kBase + static_cast<float>(i) * (kUlp / 16.0f) +
          static_cast<float>(globalRank) * kUlp +
          static_cast<float>(c) * (kUlp * 2.0f);
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

// ---------------------------------------------------------------------------
// DeterminismReduceScatter: repeated invocations of regular ncclReduceScatter
// (FP32, no quantization) with the same input must produce bitwise-identical
// output. This validates the PAT algorithm itself is deterministic.
// ---------------------------------------------------------------------------
TEST_F(ReduceScatterQuantizeTest, DeterminismReduceScatter) {
  const size_t count = 8192;
  const int numRuns = 5;
  const size_t sendSize = count * numRanks * sizeof(float);
  const size_t recvSize = count * sizeof(float);

  float* sendBuf = nullptr;
  float* recvBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvSize));

  const float kBase = 1.33f;
  const float kUlp = bf16Ulp(kBase);
  std::vector<float> hostSend(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      hostSend[c * count + i] = kBase + static_cast<float>(i) * (kUlp / 16.0f) +
          static_cast<float>(globalRank) * kUlp +
          static_cast<float>(c) * (kUlp * 2.0f);
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

// ---------------------------------------------------------------------------
// IndexCorrectness: verify that the PAT algorithm reduces the correct
// elements together, i.e. element i on every rank contributes to element i
// of the output — not some other index.
//
// Strategy: encode (rank, chunk, index) into each send value with distinct
// coefficients P1, P2, P3 so that any index-mismatch produces a detectably
// wrong sum. Tests both FP32 ReduceScatter and ReduceScatterQuantize.
// ---------------------------------------------------------------------------
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

  // Compute expected values in FP64 for this rank's output chunk.
  std::vector<double> expected(count);
  double rankSum = 0.0;
  for (int r = 0; r < numRanks; r++) {
    rankSum += static_cast<double>(r + 1);
  }
  for (size_t i = 0; i < count; i++) {
    expected[i] = P1 * rankSum +
        static_cast<double>(P2) * static_cast<double>(globalRank + 1) *
            numRanks +
        static_cast<double>(P3) * static_cast<double>(i + 1) * numRanks;
  }

  // FP32 reduce-scatter
  CUDACHECK_TEST(cudaMemset(recvRS, 0, recvSize));
  auto res = ncclReduceScatter(
      sendBuf, recvRS, count, ncclFloat32, ncclSum, comm, stream);
  ASSERT_EQ(res, ncclSuccess);

  // Quantized reduce-scatter
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

  // Check FP32 RS (should be essentially exact)
  {
    int errors = 0;
    for (size_t i = 0; i < count; i++) {
      float exp32 = static_cast<float>(expected[i]);
      float diff = std::abs(hostRS[i] - exp32);
      float tolerance = bf16Ulp(exp32);
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

  // Check RSQ (allow numPatSteps + 1 BF16 ULPs)
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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
