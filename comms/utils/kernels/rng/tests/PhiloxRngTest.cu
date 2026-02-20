// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

// The device header under test
#include "comms/utils/kernels/rng/philox_rng.cuh"

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Test Kernels
// =============================================================================

// Kernel: run philox_randint4x for a single (seed, offset) and store results
__global__ void
philoxSingleKernel(uint64_t seed, uint64_t offset, uint32_t* out) {
  uint32_t r0, r1, r2, r3;
  philox_randint4x(seed, offset, r0, r1, r2, r3);
  out[0] = r0;
  out[1] = r1;
  out[2] = r2;
  out[3] = r3;
}

// Kernel: run philox_randint4x for N different offsets
__global__ void
philoxBatchKernel(uint64_t seed, uint64_t baseOffset, int n, uint32_t* out) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  uint32_t r0, r1, r2, r3;
  philox_randint4x(seed, baseOffset + idx, r0, r1, r2, r3);
  out[idx * 4 + 0] = r0;
  out[idx * 4 + 1] = r1;
  out[idx * 4 + 2] = r2;
  out[idx * 4 + 3] = r3;
}

// Kernel: test philox4x32 directly with specified round count
template <int N_ROUNDS>
__global__ void philoxRoundsKernel(
    uint32_t c0_in,
    uint32_t c1_in,
    uint32_t c2_in,
    uint32_t c3_in,
    uint32_t k0,
    uint32_t k1,
    uint32_t* out) {
  uint32_t c0 = c0_in, c1 = c1_in, c2 = c2_in, c3 = c3_in;
  philox4x32<N_ROUNDS>(c0, c1, c2, c3, k0, k1);
  out[0] = c0;
  out[1] = c1;
  out[2] = c2;
  out[3] = c3;
}

// =============================================================================
// Test Fixture
// =============================================================================

class PhiloxRngTest : public ::testing::Test {
 protected:
  uint32_t* d_out = nullptr;
  static constexpr int kMaxOutputSize = 4096 * 4; // 4096 offsets * 4 values

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_out, kMaxOutputSize * sizeof(uint32_t)));
    CUDACHECK(cudaMemset(d_out, 0, kMaxOutputSize * sizeof(uint32_t)));
  }

  void TearDown() override {
    if (d_out) {
      CUDACHECK(cudaFree(d_out));
    }
  }

  std::vector<uint32_t> readOutput(int count) {
    std::vector<uint32_t> h_out(count);
    EXPECT_EQ(
        cudaMemcpy(
            h_out.data(),
            d_out,
            count * sizeof(uint32_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    return h_out;
  }
};

// =============================================================================
// Tests
// =============================================================================

// Test determinism: same (seed, offset) produces same output
TEST_F(PhiloxRngTest, Deterministic) {
  uint64_t seed = 12345ULL;
  uint64_t offset = 67890ULL;

  philoxSingleKernel<<<1, 1>>>(seed, offset, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result1 = readOutput(4);

  CUDACHECK(cudaMemset(d_out, 0, 4 * sizeof(uint32_t)));
  philoxSingleKernel<<<1, 1>>>(seed, offset, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result2 = readOutput(4);

  EXPECT_EQ(result1[0], result2[0]);
  EXPECT_EQ(result1[1], result2[1]);
  EXPECT_EQ(result1[2], result2[2]);
  EXPECT_EQ(result1[3], result2[3]);
}

// Test that different offsets produce different outputs
TEST_F(PhiloxRngTest, DifferentOffsetsProduceDifferentValues) {
  uint64_t seed = 42ULL;

  philoxSingleKernel<<<1, 1>>>(seed, 0, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result0 = readOutput(4);

  philoxSingleKernel<<<1, 1>>>(seed, 1, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result1 = readOutput(4);

  // At least one of the 4 outputs should differ
  bool anyDifferent = false;
  for (int i = 0; i < 4; i++) {
    if (result0[i] != result1[i]) {
      anyDifferent = true;
    }
  }
  EXPECT_TRUE(anyDifferent)
      << "Different offsets should produce different values";
}

// Test that different seeds produce different outputs
TEST_F(PhiloxRngTest, DifferentSeedsProduceDifferentValues) {
  uint64_t offset = 0;

  philoxSingleKernel<<<1, 1>>>(0, offset, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result0 = readOutput(4);

  philoxSingleKernel<<<1, 1>>>(1, offset, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result1 = readOutput(4);

  bool anyDifferent = false;
  for (int i = 0; i < 4; i++) {
    if (result0[i] != result1[i]) {
      anyDifferent = true;
    }
  }
  EXPECT_TRUE(anyDifferent)
      << "Different seeds should produce different values";
}

// Test uniformity: output values should be roughly uniformly distributed
// Use a chi-squared test approach with buckets
TEST_F(PhiloxRngTest, UniformDistribution) {
  constexpr int N = 4096;
  uint64_t seed = 0xDEADBEEFULL;

  philoxBatchKernel<<<(N + 255) / 256, 256>>>(seed, 0, N, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto results = readOutput(N * 4);

  // Split uint32 range into 16 buckets and check distribution
  constexpr int nBuckets = 16;
  int buckets[nBuckets] = {};
  int totalValues = N * 4;

  for (int i = 0; i < totalValues; i++) {
    int bucket = results[i] / (UINT32_MAX / nBuckets + 1);
    if (bucket >= nBuckets) {
      bucket = nBuckets - 1;
    }
    buckets[bucket]++;
  }

  double expected = (double)totalValues / nBuckets;
  double chiSquared = 0.0;
  for (int b = 0; b < nBuckets; b++) {
    double diff = buckets[b] - expected;
    chiSquared += (diff * diff) / expected;
  }

  // Chi-squared critical value for 15 df at p=0.001 is ~30.58
  // Use a generous threshold since we have finite samples
  EXPECT_LT(chiSquared, 50.0)
      << "Distribution does not look uniform, chi-squared = " << chiSquared;
}

// Test that all 4 output channels are independent (low correlation)
TEST_F(PhiloxRngTest, OutputChannelIndependence) {
  constexpr int N = 1024;
  uint64_t seed = 0xCAFEBABEULL;

  philoxBatchKernel<<<(N + 255) / 256, 256>>>(seed, 0, N, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto results = readOutput(N * 4);

  // Check that channels r0 and r1 are not correlated
  // Compute Pearson correlation coefficient
  auto computeCorrelation = [&](int ch_a, int ch_b) -> double {
    double sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
    for (int i = 0; i < N; i++) {
      double a = (double)results[i * 4 + ch_a];
      double b = (double)results[i * 4 + ch_b];
      sumA += a;
      sumB += b;
      sumAB += a * b;
      sumA2 += a * a;
      sumB2 += b * b;
    }
    double meanA = sumA / N, meanB = sumB / N;
    double covAB = sumAB / N - meanA * meanB;
    double stdA = std::sqrt(sumA2 / N - meanA * meanA);
    double stdB = std::sqrt(sumB2 / N - meanB * meanB);
    if (stdA < 1e-10 || stdB < 1e-10) {
      return 0.0;
    }
    return covAB / (stdA * stdB);
  };

  // Test all pairs of output channels
  for (int a = 0; a < 4; a++) {
    for (int b = a + 1; b < 4; b++) {
      double corr = computeCorrelation(a, b);
      EXPECT_LT(std::abs(corr), 0.1) << "Channels " << a << " and " << b
                                     << " appear correlated (r=" << corr << ")";
    }
  }
}

// Test with zero seed and offset
TEST_F(PhiloxRngTest, ZeroInputs) {
  philoxSingleKernel<<<1, 1>>>(0, 0, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result = readOutput(4);

  // Should produce non-zero output even with zero inputs
  bool anyNonZero = false;
  for (int i = 0; i < 4; i++) {
    if (result[i] != 0) {
      anyNonZero = true;
    }
  }
  EXPECT_TRUE(anyNonZero)
      << "Philox with zero inputs should produce non-zero output";
}

// Test with maximum seed and offset values
TEST_F(PhiloxRngTest, MaxInputs) {
  philoxSingleKernel<<<1, 1>>>(UINT64_MAX, UINT64_MAX, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result = readOutput(4);

  // Should not crash and should produce valid output
  // No specific value check, just verify it runs
  EXPECT_TRUE(true);
}

// Test that 64-bit seed uses both halves
TEST_F(PhiloxRngTest, SeedBothHalves) {
  uint64_t offset = 42;

  // Seeds that differ only in upper 32 bits
  uint64_t seedA = 0x00000001ULL;
  uint64_t seedB = 0x0000000100000001ULL;

  philoxSingleKernel<<<1, 1>>>(seedA, offset, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto resultA = readOutput(4);

  philoxSingleKernel<<<1, 1>>>(seedB, offset, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto resultB = readOutput(4);

  bool anyDifferent = false;
  for (int i = 0; i < 4; i++) {
    if (resultA[i] != resultB[i]) {
      anyDifferent = true;
    }
  }
  EXPECT_TRUE(anyDifferent) << "Seed upper bits should affect output";
}

// Test that 64-bit offset uses both halves
TEST_F(PhiloxRngTest, OffsetBothHalves) {
  uint64_t seed = 42;

  uint64_t offsetA = 0x00000001ULL;
  uint64_t offsetB = 0x0000000100000001ULL;

  philoxSingleKernel<<<1, 1>>>(seed, offsetA, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto resultA = readOutput(4);

  philoxSingleKernel<<<1, 1>>>(seed, offsetB, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto resultB = readOutput(4);

  bool anyDifferent = false;
  for (int i = 0; i < 4; i++) {
    if (resultA[i] != resultB[i]) {
      anyDifferent = true;
    }
  }
  EXPECT_TRUE(anyDifferent) << "Offset upper bits should affect output";
}

// Test uniqueness: consecutive offsets produce unique 4-tuples
TEST_F(PhiloxRngTest, ConsecutiveOffsetsUnique) {
  constexpr int N = 256;
  uint64_t seed = 999;

  philoxBatchKernel<<<(N + 255) / 256, 256>>>(seed, 0, N, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto results = readOutput(N * 4);

  // Check each 4-tuple is unique by creating string keys
  std::unordered_set<std::string> seen;
  for (int i = 0; i < N; i++) {
    std::string key = std::to_string(results[i * 4]) + "," +
        std::to_string(results[i * 4 + 1]) + "," +
        std::to_string(results[i * 4 + 2]) + "," +
        std::to_string(results[i * 4 + 3]);
    EXPECT_EQ(seen.count(key), 0u) << "Duplicate 4-tuple found at offset " << i;
    seen.insert(key);
  }
}

// Test that different round counts produce different (and presumably
// weaker/stronger) outputs
TEST_F(PhiloxRngTest, DifferentRoundCounts) {
  uint32_t c0 = 1, c1 = 2, c2 = 0, c3 = 0;
  uint32_t k0 = 0xDEAD, k1 = 0xBEEF;

  philoxRoundsKernel<1><<<1, 1>>>(c0, c1, c2, c3, k0, k1, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result1 = readOutput(4);

  philoxRoundsKernel<7><<<1, 1>>>(c0, c1, c2, c3, k0, k1, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result7 = readOutput(4);

  philoxRoundsKernel<10><<<1, 1>>>(c0, c1, c2, c3, k0, k1, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result10 = readOutput(4);

  // Different round counts should produce different results
  bool diff_1_7 = false, diff_7_10 = false;
  for (int i = 0; i < 4; i++) {
    if (result1[i] != result7[i]) {
      diff_1_7 = true;
    }
    if (result7[i] != result10[i]) {
      diff_7_10 = true;
    }
  }
  EXPECT_TRUE(diff_1_7) << "1 round vs 7 rounds should differ";
  EXPECT_TRUE(diff_7_10) << "7 rounds vs 10 rounds should differ";
}

// Test multi-threaded consistency: each thread gets its own unique output
TEST_F(PhiloxRngTest, MultiThreadedConsistency) {
  constexpr int N = 512;
  uint64_t seed = 0x1234;

  // Launch with many threads
  philoxBatchKernel<<<2, 256>>>(seed, 0, N, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto resultsMulti = readOutput(N * 4);

  // Verify against single-threaded execution
  CUDACHECK(cudaMemset(d_out, 0, N * 4 * sizeof(uint32_t)));
  philoxBatchKernel<<<N, 1>>>(seed, 0, N, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto resultsSingle = readOutput(N * 4);

  for (int i = 0; i < N * 4; i++) {
    EXPECT_EQ(resultsMulti[i], resultsSingle[i])
        << "Mismatch at index " << i
        << " between multi-thread and single-thread execution";
  }
}
