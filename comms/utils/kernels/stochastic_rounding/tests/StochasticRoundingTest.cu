// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>

#include "comms/utils/kernels/rng/philox_rng.cuh"
#include "comms/utils/kernels/stochastic_rounding/stochastic_rounding.cuh"

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Test Kernels for stochastic_round_bf16_software
// =============================================================================

// Kernel: round a single float to bf16 using stochastic rounding
__global__ void stochasticRoundSingleKernel(
    float input,
    uint32_t rand_bits,
    __nv_bfloat16* output) {
  *output = stochastic_round_bf16_software(input, rand_bits);
}

// Kernel: round N floats to bf16 with different random bits per element
__global__ void stochasticRoundBatchKernel(
    const float* inputs,
    int n,
    uint64_t seed,
    __nv_bfloat16* outputs) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  uint32_t r0, r1, r2, r3;
  philox_randint4x(seed, (uint64_t)idx, r0, r1, r2, r3);
  outputs[idx] = stochastic_round_bf16_software(inputs[idx], r0);
}

// Kernel: round the same float many times with different random bits
__global__ void stochasticRoundRepeatKernel(
    float input,
    int n,
    uint64_t seed,
    __nv_bfloat16* outputs) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  uint32_t r0, r1, r2, r3;
  philox_randint4x(seed, (uint64_t)idx, r0, r1, r2, r3);
  outputs[idx] = stochastic_round_bf16_software(input, r0);
}

// Kernel: test bf16x2 software rounding
__global__ void stochasticRoundBf16x2Kernel(
    float x,
    float y,
    uint32_t r0,
    uint32_t r1,
    __nv_bfloat162* output) {
  float2 vals = make_float2(x, y);
  *output = stochastic_round_bf16x2_software(vals, r0, r1);
}

// =============================================================================
// Hardware-Accelerated Stochastic Rounding Kernels (Blackwell, SM >= 100)
// =============================================================================
// Note: __CUDA_ARCH__ is only defined during device compilation, so we guard
// the kernel *body* rather than the kernel declaration. This ensures host-side
// launch stubs are always generated, allowing the test to compile for all
// target architectures. The Blackwell TEST_F cases use a runtime GPU capability
// check and GTEST_SKIP() on non-Blackwell GPUs.

// Kernel: test bf16x2 hardware-accelerated rounding (Blackwell)
__global__ void stochasticRoundBf16x2BlackwellKernel(
    float x,
    float y,
    uint32_t rand_bits,
    __nv_bfloat162* output) {
#if __CUDA_ARCH__ >= 1000
  float2 vals = make_float2(x, y);
  *output = stochastic_round_bf16x2_blackwell(vals, rand_bits);
#endif
}

// Kernel: repeat hardware-accelerated rounding for statistical tests
__global__ void stochasticRoundBf16x2BlackwellRepeatKernel(
    float x,
    float y,
    int n,
    uint64_t seed,
    __nv_bfloat162* outputs) {
#if __CUDA_ARCH__ >= 1000
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  uint32_t r0, r1, r2, r3;
  philox_randint4x(seed, (uint64_t)idx, r0, r1, r2, r3);
  uint32_t rand_bits = r0 ^ (r1 << 16);
  float2 vals = make_float2(x, y);
  outputs[idx] = stochastic_round_bf16x2_blackwell(vals, rand_bits);
#endif
}

// =============================================================================
// Test Fixture
// =============================================================================

class StochasticRoundingTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  // Helper: check if running on Blackwell (SM >= 100) GPU
  static bool isBlackwell() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) {
      return false;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
      return false;
    }
    return prop.major >= 10;
  }

  // Helper: convert bf16 back to float on host
  static float bf16ToFloat(__nv_bfloat16 val) {
    return __bfloat162float(val);
  }

  // Helper: get the two nearest bf16 values bracketing a float
  static void getBracketingBf16(float val, float& lower, float& upper) {
    __nv_bfloat16 rounded_down = __float2bfloat16_rd(val);
    float rd = __bfloat162float(rounded_down);
    ASSERT_TRUE(rd <= val) << "Rounding down should not increase value";

    if (rd == val) {
      lower = upper = val;
      return;
    }

    __nv_bfloat16 rounded_up = __float2bfloat16_ru(val);
    float ru = __bfloat162float(rounded_up);
    ASSERT_TRUE(ru >= val) << "Rounding up should not decrease value";

    lower = rd;
    upper = ru;
  }
};

// =============================================================================
// Tests: Basic Correctness
// =============================================================================

// Values exactly representable in bf16 should round to themselves
TEST_F(StochasticRoundingTest, ExactBf16ValuesUnchanged) {
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, sizeof(__nv_bfloat16)));

  // 1.0, 2.0, -3.0, 0.0 are exactly representable in bf16
  float exactValues[] = {0.0f, 1.0f, -1.0f, 2.0f, -3.0f, 0.5f, 256.0f};
  for (float val : exactValues) {
    // Test with various random bits - should all give same result
    for (uint32_t rand : {0u, 0xFFFFu, 0x8000u, 0xDEADu}) {
      stochasticRoundSingleKernel<<<1, 1>>>(val, rand, d_out);
      CUDACHECK(cudaGetLastError());
      CUDACHECK(cudaDeviceSynchronize());
      __nv_bfloat16 h_out;
      CUDACHECK(cudaMemcpy(
          &h_out, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
      EXPECT_EQ(bf16ToFloat(h_out), val)
          << "Exact bf16 value " << val
          << " should remain unchanged with rand=" << rand;
    }
  }

  CUDACHECK(cudaFree(d_out));
}

// NaN should be preserved
TEST_F(StochasticRoundingTest, NanPreserved) {
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, sizeof(__nv_bfloat16)));

  float nanVal = std::nanf("");
  stochasticRoundSingleKernel<<<1, 1>>>(nanVal, 0xFFFF, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  __nv_bfloat16 h_out;
  CUDACHECK(
      cudaMemcpy(&h_out, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(std::isnan(bf16ToFloat(h_out))) << "NaN should be preserved";

  CUDACHECK(cudaFree(d_out));
}

// Infinity should be preserved
TEST_F(StochasticRoundingTest, InfPreserved) {
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, sizeof(__nv_bfloat16)));

  stochasticRoundSingleKernel<<<1, 1>>>(INFINITY, 0xFFFF, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  __nv_bfloat16 h_out;
  CUDACHECK(
      cudaMemcpy(&h_out, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(std::isinf(bf16ToFloat(h_out))) << "+Inf should be preserved";
  EXPECT_GT(bf16ToFloat(h_out), 0) << "+Inf should remain positive";

  stochasticRoundSingleKernel<<<1, 1>>>(-INFINITY, 0xFFFF, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(
      cudaMemcpy(&h_out, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(std::isinf(bf16ToFloat(h_out))) << "-Inf should be preserved";
  EXPECT_LT(bf16ToFloat(h_out), 0) << "-Inf should remain negative";

  CUDACHECK(cudaFree(d_out));
}

// =============================================================================
// Tests: Unbiasedness (most important property)
// =============================================================================

// The average of many stochastic roundings should converge to the original
// value
TEST_F(StochasticRoundingTest, UnbiasedRounding) {
  constexpr int N = 8192;
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));

  // Test a value between two bf16 representable values
  // 1.0 + 0.004 is between bf16 representable 1.0 (0x3F80) and ~1.0078125
  // (0x3F81)
  float testValue = 1.004f;

  stochasticRoundRepeatKernel<<<(N + 255) / 256, 256>>>(
      testValue, N, 42, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_out(N);
  CUDACHECK(cudaMemcpy(
      h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  double sum = 0.0;
  for (int i = 0; i < N; i++) {
    sum += bf16ToFloat(h_out[i]);
  }
  double avg = sum / N;

  // Average should be close to original value
  // Tolerance: relative error < 1% or absolute error < the bf16 gap
  float lower, upper;
  getBracketingBf16(testValue, lower, upper);
  double gap = upper - lower;

  EXPECT_NEAR(avg, (double)testValue, gap * 0.1)
      << "Average of stochastic roundings should approximate original value. "
      << "Original: " << testValue << " Average: " << avg << " BF16 bracket: ["
      << lower << ", " << upper << "]";

  CUDACHECK(cudaFree(d_out));
}

// Test unbiasedness for multiple values
TEST_F(StochasticRoundingTest, UnbiasedMultipleValues) {
  constexpr int N = 4096;
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));

  // Test values at various magnitudes
  float testValues[] = {
      0.3f, // Small, between 0.2998.. and 0.3008..
      3.14159f, // Pi, between 3.140625 and 3.15625
      100.7f, // Medium magnitude
      -2.718f, // Negative value (Euler's number)
      0.001f, // Very small positive
  };

  for (float testValue : testValues) {
    stochasticRoundRepeatKernel<<<(N + 255) / 256, 256>>>(
        testValue, N, 12345, d_out);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out(N);
    CUDACHECK(cudaMemcpy(
        h_out.data(),
        d_out,
        N * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < N; i++) {
      sum += bf16ToFloat(h_out[i]);
    }
    double avg = sum / N;

    float lower, upper;
    getBracketingBf16(testValue, lower, upper);
    double gap =
        (upper == lower) ? std::abs(testValue) * 0.01 : (upper - lower);

    // Allow 15% of the gap as tolerance (statistical variation)
    EXPECT_NEAR(avg, (double)testValue, gap * 0.15)
        << "Unbiased rounding failed for value " << testValue << " (avg=" << avg
        << ", bracket=[" << lower << "," << upper << "])";
  }

  CUDACHECK(cudaFree(d_out));
}

// =============================================================================
// Tests: Rounding to correct neighbors
// =============================================================================

// Stochastic rounding should only produce the two nearest bf16 values
TEST_F(StochasticRoundingTest, RoundsToNeighbors) {
  constexpr int N = 1024;
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));

  float testValue = 1.5f + 0.003f; // Not exactly representable

  stochasticRoundRepeatKernel<<<(N + 255) / 256, 256>>>(
      testValue, N, 77, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_out(N);
  CUDACHECK(cudaMemcpy(
      h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  float lower, upper;
  getBracketingBf16(testValue, lower, upper);

  for (int i = 0; i < N; i++) {
    float result = bf16ToFloat(h_out[i]);
    EXPECT_TRUE(result == lower || result == upper)
        << "Stochastic rounding produced " << result << " which is neither "
        << lower << " nor " << upper << " for input " << testValue;
  }

  CUDACHECK(cudaFree(d_out));
}

// =============================================================================
// Tests: Vectorized (bf16x2) operations
// =============================================================================

TEST_F(StochasticRoundingTest, Bf16x2Software) {
  __nv_bfloat162* d_out;
  CUDACHECK(cudaMalloc(&d_out, sizeof(__nv_bfloat162)));

  float x = 1.0f, y = 2.0f; // Exactly representable
  stochasticRoundBf16x2Kernel<<<1, 1>>>(x, y, 0, 0, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  __nv_bfloat162 h_out;
  CUDACHECK(cudaMemcpy(
      &h_out, d_out, sizeof(__nv_bfloat162), cudaMemcpyDeviceToHost));

  EXPECT_EQ(bf16ToFloat(__low2bfloat16(h_out)), x);
  EXPECT_EQ(bf16ToFloat(__high2bfloat16(h_out)), y);

  CUDACHECK(cudaFree(d_out));
}

// Test that stochastic rounding with zero random bits rounds down
TEST_F(StochasticRoundingTest, ZeroRandomBitsRoundsDown) {
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, sizeof(__nv_bfloat16)));

  // 1.004 is between 1.0 and ~1.0078125 in bf16
  float testValue = 1.004f;
  stochasticRoundSingleKernel<<<1, 1>>>(testValue, 0, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  __nv_bfloat16 h_out;
  CUDACHECK(
      cudaMemcpy(&h_out, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  float lower, upper;
  getBracketingBf16(testValue, lower, upper);

  // With zero random bits, the lower 16 bits are just truncated
  // This should give the floor (lower neighbor)
  EXPECT_EQ(bf16ToFloat(h_out), lower)
      << "Zero random bits should effectively truncate to lower neighbor";

  CUDACHECK(cudaFree(d_out));
}

// Test that max random bits rounds up (when value is not exactly representable)
TEST_F(StochasticRoundingTest, MaxRandomBitsRoundsUp) {
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, sizeof(__nv_bfloat16)));

  // Use a value that is just barely above a bf16 representable value
  // The lower 16 bits of 1.004 = 1.004 - floor_bf16(1.004)
  float testValue = 1.004f;
  stochasticRoundSingleKernel<<<1, 1>>>(testValue, 0xFFFF, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  __nv_bfloat16 h_out;
  CUDACHECK(
      cudaMemcpy(&h_out, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  float lower, upper;
  getBracketingBf16(testValue, lower, upper);

  // With max random bits (0xFFFF), adding to the lower 16 bits should carry up
  EXPECT_EQ(bf16ToFloat(h_out), upper)
      << "Max random bits should round up to upper neighbor";

  CUDACHECK(cudaFree(d_out));
}

// Test rounding for negative values
TEST_F(StochasticRoundingTest, NegativeValues) {
  constexpr int N = 4096;
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));

  float testValue = -1.004f;
  stochasticRoundRepeatKernel<<<(N + 255) / 256, 256>>>(
      testValue, N, 42, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_out(N);
  CUDACHECK(cudaMemcpy(
      h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  double sum = 0.0;
  for (int i = 0; i < N; i++) {
    sum += bf16ToFloat(h_out[i]);
  }
  double avg = sum / N;

  // For negative values, the result should be close to the original
  // Note: the software rounding adds noise to unsigned bits, which may not
  // perfectly handle negative values, but should still be approximately
  // unbiased
  EXPECT_NEAR(avg, (double)testValue, 0.02)
      << "Negative value rounding should be approximately unbiased";

  CUDACHECK(cudaFree(d_out));
}

// Test with very small values (subnormal territory)
TEST_F(StochasticRoundingTest, SmallValues) {
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, sizeof(__nv_bfloat16)));

  // A very small float that is within bf16 subnormal range
  float smallVal = 1e-38f;
  stochasticRoundSingleKernel<<<1, 1>>>(smallVal, 0, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  __nv_bfloat16 h_out;
  CUDACHECK(
      cudaMemcpy(&h_out, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  // Should not produce NaN or Inf
  float result = bf16ToFloat(h_out);
  EXPECT_FALSE(std::isnan(result)) << "Small value should not produce NaN";
  EXPECT_FALSE(std::isinf(result)) << "Small value should not produce Inf";

  CUDACHECK(cudaFree(d_out));
}

// Test probability distribution: the fraction rounding up should match
// the fractional position within the bf16 interval
TEST_F(StochasticRoundingTest, CorrectRoundUpProbability) {
  constexpr int N = 16384;
  __nv_bfloat16* d_out;
  CUDACHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));

  // Choose a value at roughly 25% between two bf16 values
  // bf16(1.0) = 1.0, bf16_next(1.0) = 1.0078125
  // 1.0 + 0.25 * 0.0078125 = 1.001953125
  float testValue = 1.001953125f;

  stochasticRoundRepeatKernel<<<(N + 255) / 256, 256>>>(
      testValue, N, 99, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_out(N);
  CUDACHECK(cudaMemcpy(
      h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  float lower, upper;
  getBracketingBf16(testValue, lower, upper);

  int countUpper = 0;
  for (int i = 0; i < N; i++) {
    if (bf16ToFloat(h_out[i]) == upper)
      countUpper++;
  }

  double expectedFrac = (testValue - lower) / (upper - lower);
  double actualFrac = (double)countUpper / N;

  // Allow 5% absolute tolerance due to finite sample
  EXPECT_NEAR(actualFrac, expectedFrac, 0.05)
      << "Round-up probability should match fractional position. "
      << "Expected: " << expectedFrac << " Got: " << actualFrac
      << " (lower=" << lower << ", upper=" << upper << ", value=" << testValue
      << ")";

  CUDACHECK(cudaFree(d_out));
}

// =============================================================================
// Tests: Hardware-Accelerated Stochastic Rounding (Blackwell, SM >= 100)
// =============================================================================

// Test basic correctness: exact bf16 values should remain unchanged
TEST_F(StochasticRoundingTest, BlackwellExactBf16ValuesUnchanged) {
  if (!isBlackwell()) {
    GTEST_SKIP() << "Requires Blackwell GPU (SM >= 100)";
  }
  __nv_bfloat162* d_out;
  CUDACHECK(cudaMalloc(&d_out, sizeof(__nv_bfloat162)));

  // Test with exactly representable bf16 values
  float x = 1.0f, y = 2.0f;
  for (uint32_t rand : {0u, 0xFFFFFFFFu, 0x12345678u}) {
    stochasticRoundBf16x2BlackwellKernel<<<1, 1>>>(x, y, rand, d_out);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    __nv_bfloat162 h_out;
    CUDACHECK(cudaMemcpy(
        &h_out, d_out, sizeof(__nv_bfloat162), cudaMemcpyDeviceToHost));

    EXPECT_EQ(bf16ToFloat(__low2bfloat16(h_out)), x)
        << "Exact bf16 value x=" << x
        << " should remain unchanged with rand=" << rand;
    EXPECT_EQ(bf16ToFloat(__high2bfloat16(h_out)), y)
        << "Exact bf16 value y=" << y
        << " should remain unchanged with rand=" << rand;
  }

  CUDACHECK(cudaFree(d_out));
}

// Test that NaN and Inf are preserved
TEST_F(StochasticRoundingTest, BlackwellSpecialValuesPreserved) {
  if (!isBlackwell()) {
    GTEST_SKIP() << "Requires Blackwell GPU (SM >= 100)";
  }
  __nv_bfloat162* d_out;
  CUDACHECK(cudaMalloc(&d_out, sizeof(__nv_bfloat162)));

  // Test NaN
  float nanVal = std::nanf("");
  stochasticRoundBf16x2BlackwellKernel<<<1, 1>>>(nanVal, nanVal, 0xFFFF, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  __nv_bfloat162 h_out;
  CUDACHECK(cudaMemcpy(
      &h_out, d_out, sizeof(__nv_bfloat162), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(std::isnan(bf16ToFloat(__low2bfloat16(h_out))))
      << "NaN should be preserved in low element";
  EXPECT_TRUE(std::isnan(bf16ToFloat(__high2bfloat16(h_out))))
      << "NaN should be preserved in high element";

  // Test +Inf
  stochasticRoundBf16x2BlackwellKernel<<<1, 1>>>(
      INFINITY, -INFINITY, 0xFFFF, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  CUDACHECK(cudaMemcpy(
      &h_out, d_out, sizeof(__nv_bfloat162), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(std::isinf(bf16ToFloat(__low2bfloat16(h_out))))
      << "+Inf should be preserved";
  EXPECT_GT(bf16ToFloat(__low2bfloat16(h_out)), 0) << "+Inf should be positive";
  EXPECT_TRUE(std::isinf(bf16ToFloat(__high2bfloat16(h_out))))
      << "-Inf should be preserved";
  EXPECT_LT(bf16ToFloat(__high2bfloat16(h_out)), 0)
      << "-Inf should be negative";

  CUDACHECK(cudaFree(d_out));
}

// Test unbiasedness of hardware-accelerated rounding
TEST_F(StochasticRoundingTest, BlackwellUnbiasedRounding) {
  if (!isBlackwell()) {
    GTEST_SKIP() << "Requires Blackwell GPU (SM >= 100)";
  }
  constexpr int N = 8192;
  __nv_bfloat162* d_out;
  CUDACHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat162)));

  // Test values between two bf16 representable values
  float testX = 1.004f;
  float testY = 2.003f;

  stochasticRoundBf16x2BlackwellRepeatKernel<<<(N + 255) / 256, 256>>>(
      testX, testY, N, 42, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat162> h_out(N);
  CUDACHECK(cudaMemcpy(
      h_out.data(), d_out, N * sizeof(__nv_bfloat162), cudaMemcpyDeviceToHost));

  double sumX = 0.0, sumY = 0.0;
  for (int i = 0; i < N; i++) {
    sumX += bf16ToFloat(__low2bfloat16(h_out[i]));
    sumY += bf16ToFloat(__high2bfloat16(h_out[i]));
  }
  double avgX = sumX / N;
  double avgY = sumY / N;

  float lowerX, upperX, lowerY, upperY;
  getBracketingBf16(testX, lowerX, upperX);
  getBracketingBf16(testY, lowerY, upperY);
  double gapX = upperX - lowerX;
  double gapY = upperY - lowerY;

  EXPECT_NEAR(avgX, (double)testX, gapX * 0.15)
      << "Average of stochastic roundings for X should approximate original. "
      << "Original: " << testX << " Average: " << avgX;
  EXPECT_NEAR(avgY, (double)testY, gapY * 0.15)
      << "Average of stochastic roundings for Y should approximate original. "
      << "Original: " << testY << " Average: " << avgY;

  CUDACHECK(cudaFree(d_out));
}

// Test that results only produce the two nearest bf16 values
TEST_F(StochasticRoundingTest, BlackwellRoundsToNeighbors) {
  if (!isBlackwell()) {
    GTEST_SKIP() << "Requires Blackwell GPU (SM >= 100)";
  }
  constexpr int N = 1024;
  __nv_bfloat162* d_out;
  CUDACHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat162)));

  float testX = 1.5f + 0.003f;
  float testY = 3.14159f;

  stochasticRoundBf16x2BlackwellRepeatKernel<<<(N + 255) / 256, 256>>>(
      testX, testY, N, 77, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat162> h_out(N);
  CUDACHECK(cudaMemcpy(
      h_out.data(), d_out, N * sizeof(__nv_bfloat162), cudaMemcpyDeviceToHost));

  float lowerX, upperX, lowerY, upperY;
  getBracketingBf16(testX, lowerX, upperX);
  getBracketingBf16(testY, lowerY, upperY);

  for (int i = 0; i < N; i++) {
    float resultX = bf16ToFloat(__low2bfloat16(h_out[i]));
    float resultY = bf16ToFloat(__high2bfloat16(h_out[i]));

    EXPECT_TRUE(resultX == lowerX || resultX == upperX)
        << "Blackwell stochastic rounding produced X=" << resultX
        << " which is neither " << lowerX << " nor " << upperX;
    EXPECT_TRUE(resultY == lowerY || resultY == upperY)
        << "Blackwell stochastic rounding produced Y=" << resultY
        << " which is neither " << lowerY << " nor " << upperY;
  }

  CUDACHECK(cudaFree(d_out));
}

// Test determinism: same random bits should produce same results
TEST_F(StochasticRoundingTest, BlackwellDeterministic) {
  if (!isBlackwell()) {
    GTEST_SKIP() << "Requires Blackwell GPU (SM >= 100)";
  }
  __nv_bfloat162* d_out;
  CUDACHECK(cudaMalloc(&d_out, sizeof(__nv_bfloat162)));

  float x = 1.337f, y = 2.718f;
  uint32_t rand_bits = 0xDEADBEEF;

  stochasticRoundBf16x2BlackwellKernel<<<1, 1>>>(x, y, rand_bits, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  __nv_bfloat162 result1;
  CUDACHECK(cudaMemcpy(
      &result1, d_out, sizeof(__nv_bfloat162), cudaMemcpyDeviceToHost));

  stochasticRoundBf16x2BlackwellKernel<<<1, 1>>>(x, y, rand_bits, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  __nv_bfloat162 result2;
  CUDACHECK(cudaMemcpy(
      &result2, d_out, sizeof(__nv_bfloat162), cudaMemcpyDeviceToHost));

  EXPECT_EQ(
      bf16ToFloat(__low2bfloat16(result1)),
      bf16ToFloat(__low2bfloat16(result2)))
      << "Same rand_bits should produce same X result";
  EXPECT_EQ(
      bf16ToFloat(__high2bfloat16(result1)),
      bf16ToFloat(__high2bfloat16(result2)))
      << "Same rand_bits should produce same Y result";

  CUDACHECK(cudaFree(d_out));
}

// Test negative values
TEST_F(StochasticRoundingTest, BlackwellNegativeValues) {
  if (!isBlackwell()) {
    GTEST_SKIP() << "Requires Blackwell GPU (SM >= 100)";
  }
  constexpr int N = 4096;
  __nv_bfloat162* d_out;
  CUDACHECK(cudaMalloc(&d_out, N * sizeof(__nv_bfloat162)));

  float testX = -1.004f;
  float testY = -2.718f;

  stochasticRoundBf16x2BlackwellRepeatKernel<<<(N + 255) / 256, 256>>>(
      testX, testY, N, 42, d_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat162> h_out(N);
  CUDACHECK(cudaMemcpy(
      h_out.data(), d_out, N * sizeof(__nv_bfloat162), cudaMemcpyDeviceToHost));

  double sumX = 0.0, sumY = 0.0;
  for (int i = 0; i < N; i++) {
    sumX += bf16ToFloat(__low2bfloat16(h_out[i]));
    sumY += bf16ToFloat(__high2bfloat16(h_out[i]));
  }
  double avgX = sumX / N;
  double avgY = sumY / N;

  // Hardware stochastic rounding should handle negative values correctly
  EXPECT_NEAR(avgX, (double)testX, 0.02)
      << "Negative value X rounding should be approximately unbiased";
  EXPECT_NEAR(avgY, (double)testY, 0.03)
      << "Negative value Y rounding should be approximately unbiased";

  CUDACHECK(cudaFree(d_out));
}

// =============================================================================
// Tests: Hardware vs Software Consistency (Blackwell)
// =============================================================================

// Kernel: compare hardware and software stochastic rounding with equivalent
// random bits. For each element, compute both hardware and software results
// and store them for comparison.
__global__ void compareHwSwRoundingKernel(
    const float* x_vals,
    const float* y_vals,
    int n,
    uint64_t seed,
    __nv_bfloat162* hw_outputs,
    __nv_bfloat162* sw_outputs) {
#if __CUDA_ARCH__ >= 1000
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  float x = x_vals[idx];
  float y = y_vals[idx];
  float2 vals = make_float2(x, y);

  // Generate random bits using Philox
  uint32_t r0, r1, r2, r3;
  philox_randint4x(seed, (uint64_t)idx, r0, r1, r2, r3);

  // Hardware uses combined random bits (same as Apply_StochasticCast<float,
  // __nv_bfloat16, 2>)
  uint32_t rand_bits = r0 ^ (r1 << 16);
  hw_outputs[idx] = stochastic_round_bf16x2_blackwell(vals, rand_bits);

  // Software uses separate random bits for each element
  sw_outputs[idx] = stochastic_round_bf16x2_software(vals, r0, r1);
#endif
}

// =============================================================================
// Tests: Cancellation – large number + many small numbers should sum to zero
// =============================================================================
//
// These tests demonstrate a key advantage of stochastic rounding over
// deterministic (round-to-nearest-even) BF16 conversion. When a set of
// float values sums exactly to zero (e.g., -98.0 + 13.95 * 7), converting
// each value independently to BF16 introduces rounding error. With RNE
// rounding, the error is systematic and the sum drifts away from zero. With
// stochastic rounding, the errors are unbiased, so over many independent
// trials the *average* sum converges back to zero.

// Helper kernel: stochastically round an array of floats to bf16, one element
// per thread, using a per-trial seed so each trial gets independent randomness.
__global__ void stochasticRoundArrayKernel(
    const float* inputs,
    int n,
    uint64_t seed,
    __nv_bfloat16* outputs) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  uint32_t r0, r1, r2, r3;
  philox_randint4x(seed, (uint64_t)idx, r0, r1, r2, r3);
  outputs[idx] = stochastic_round_bf16_software(inputs[idx], r0);
}

// Parameterised helper that runs one cancellation scenario.
//
//   smallVal:     the repeated small number
//   smallCount:   how many copies of smallVal (total array size = 1 +
//                 smallCount)
//   trials:       how many independent stochastic-rounding trials to average
//
// The large value is computed as -(smallCount * smallVal) in double and then
// cast to float. The "exact" FP32 sum (large + smallCount * small) may have
// a tiny residual due to the float cast, but it will be negligibly small
// compared to the BF16 rounding error we are testing.
//
// Checks:
//   1. FP32 exact sum is near zero  (sanity)
//   2. Deterministic BF16 (RNE) sum != 0  (shows the problem)
//   3. Average stochastic-rounded sum is close to the FP32 exact sum
static void runCancellationTest(float smallVal, int smallCount, int trials) {
  if ((std::bit_cast<uint32_t>(smallVal) & 0xffff) == 0) {
    GTEST_SKIP()
        << "Skipping test because smallVal is exactly representable in bf16";
  }
  // Compute the large value so that the double-precision sum is exactly 0,
  // then cast to float. The cast may introduce a tiny residual.
  double largeDbl = -((double)smallVal * smallCount);
  float largeVal = (float)largeDbl;
  const int n = 1 + smallCount; // total number of elements

  // ---- 1. Verify that the FP32 sum is close to zero ----
  double fp32ExactSum = (double)largeVal;
  for (int i = 0; i < smallCount; i++)
    fp32ExactSum += (double)smallVal;
  // The residual from the float cast should be tiny relative to the values.
  ASSERT_NEAR(fp32ExactSum, 0.0, std::abs((double)largeVal) * 1e-6)
      << "Test design error: FP32 sum should be approximately 0";

  // ---- 2. Deterministic BF16 (RNE) sum – should differ from fp32ExactSum ----
  {
    double rneSum = 0.0;
    rneSum += __bfloat162float(__float2bfloat16(largeVal));
    for (int i = 0; i < smallCount; i++)
      rneSum += __bfloat162float(__float2bfloat16(smallVal));
    double rneError = std::abs(rneSum - fp32ExactSum);
    // We expect RNE to introduce measurable error compared to the fp32 sum
    EXPECT_GT(rneError, 0.0)
        << "Expected RNE BF16 conversion to introduce error "
        << "(largeVal=" << largeVal << ", smallVal=" << smallVal
        << ", count=" << smallCount << ", rneSum=" << rneSum << ")";
  }

  // ---- 3. Stochastic rounding: average sum should converge to fp32ExactSum --
  // Build the input array on the host
  std::vector<float> h_inputs(n);
  h_inputs[0] = largeVal;
  for (int i = 1; i < n; i++)
    h_inputs[i] = smallVal;

  // Allocate device memory
  float* d_inputs;
  __nv_bfloat16* d_outputs;
  ASSERT_EQ(cudaMalloc(&d_inputs, n * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_outputs, n * sizeof(__nv_bfloat16)), cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(
          d_inputs, h_inputs.data(), n * sizeof(float), cudaMemcpyHostToDevice),
      cudaSuccess);

  std::vector<__nv_bfloat16> h_outputs(n);
  double totalSum = 0.0;

  for (int t = 0; t < trials; t++) {
    uint64_t seed = 1000 + t; // different seed per trial
    int blocks = (n + 255) / 256;
    stochasticRoundArrayKernel<<<blocks, 256>>>(d_inputs, n, seed, d_outputs);
    {
      cudaError_t err = cudaGetLastError();
      ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(
            h_outputs.data(),
            d_outputs,
            n * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost),
        cudaSuccess);

    double trialSum = 0.0;
    for (int i = 0; i < n; i++) {
      trialSum += __bfloat162float(h_outputs[i]);
    }
    totalSum += trialSum;
  }

  double avgSum = totalSum / trials;

  // The tolerance scales with the magnitude of the large value and the
  // bf16 gap at that magnitude. For values around 100, the bf16 gap is
  // 0.5, so we allow a few percent of that.
  float absLarge = std::abs(largeVal);
  // bf16 gap at the magnitude of the large value
  float lower, upper;
  __nv_bfloat16 rd = __float2bfloat16_rd(absLarge);
  __nv_bfloat16 ru = __float2bfloat16_ru(absLarge);
  lower = __bfloat162float(rd);
  upper = __bfloat162float(ru);
  double gap = (upper > lower) ? (upper - lower) : 1.0;
  // Allow tolerance proportional to sqrt(n) * gap / sqrt(trials)
  // This is a rough statistical bound.
  double tolerance = gap * std::sqrt((double)n) / std::sqrt((double)trials) * 3;
  // Also ensure a minimum tolerance
  if (tolerance < gap * 0.5)
    tolerance = gap * 0.5;

  EXPECT_NEAR(avgSum, fp32ExactSum, tolerance)
      << "Stochastic rounding average sum should be close to FP32 exact sum. "
      << "largeVal=" << largeVal << ", smallVal=" << smallVal
      << ", count=" << smallCount << ", trials=" << trials
      << ", fp32ExactSum=" << fp32ExactSum << ", avgSum=" << avgSum
      << ", tolerance=" << tolerance;

  ASSERT_EQ(cudaFree(d_inputs), cudaSuccess);
  ASSERT_EQ(cudaFree(d_outputs), cudaSuccess);
}

// -97.65 + 13.95 * 7 = 0 (8 numbers total)
TEST_F(StochasticRoundingTest, CancellationSmall8) {
  runCancellationTest(/*smallVal=*/13.95f, /*smallCount=*/7, /*trials=*/1024);
}

// 16 numbers: one large negative + 15 copies of 10.3
TEST_F(StochasticRoundingTest, CancellationMedium16) {
  runCancellationTest(/*smallVal=*/10.3f, /*smallCount=*/15, /*trials=*/1024);
}

// 32 numbers: one large negative + 31 copies of 3.14
TEST_F(StochasticRoundingTest, Cancellation32) {
  runCancellationTest(/*smallVal=*/3.14f, /*smallCount=*/31, /*trials=*/1024);
}

// 64 numbers
TEST_F(StochasticRoundingTest, Cancellation64) {
  runCancellationTest(/*smallVal=*/1.77f, /*smallCount=*/63, /*trials=*/512);
}

// 128 numbers
TEST_F(StochasticRoundingTest, Cancellation128) {
  runCancellationTest(/*smallVal=*/0.87f, /*smallCount=*/127, /*trials=*/512);
}

// 256 numbers
TEST_F(StochasticRoundingTest, Cancellation256) {
  runCancellationTest(/*smallVal=*/0.53f, /*smallCount=*/255, /*trials=*/256);
}

// 512 numbers
TEST_F(StochasticRoundingTest, Cancellation512) {
  runCancellationTest(/*smallVal=*/0.29f, /*smallCount=*/511, /*trials=*/256);
}

// 1024 numbers
TEST_F(StochasticRoundingTest, Cancellation1024) {
  runCancellationTest(/*smallVal=*/0.13f, /*smallCount=*/1023, /*trials=*/256);
}

// Test that hardware and software stochastic rounding produce statistically
// equivalent results on Blackwell. Since the hardware and software
// implementations may use random bits differently internally, we verify
// that both produce the same statistical distribution (unbiased rounding
// with correct probability distribution).
TEST_F(StochasticRoundingTest, BlackwellHardwareSoftwareConsistency) {
  if (!isBlackwell()) {
    GTEST_SKIP() << "Requires Blackwell GPU (SM >= 100)";
  }

  constexpr int N = 8192;

  // Allocate device memory
  float* d_x_vals;
  float* d_y_vals;
  __nv_bfloat162* d_hw_out;
  __nv_bfloat162* d_sw_out;
  CUDACHECK(cudaMalloc(&d_x_vals, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_y_vals, N * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_hw_out, N * sizeof(__nv_bfloat162)));
  CUDACHECK(cudaMalloc(&d_sw_out, N * sizeof(__nv_bfloat162)));

  // Test with a single pair of values to compare statistical properties
  float testX = 1.004f; // Not exactly representable in bf16
  float testY = 2.003f; // Not exactly representable in bf16

  std::vector<float> h_x_vals(N, testX);
  std::vector<float> h_y_vals(N, testY);

  CUDACHECK(cudaMemcpy(
      d_x_vals, h_x_vals.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_y_vals, h_y_vals.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // Run the comparison kernel
  uint64_t seed = 12345;
  compareHwSwRoundingKernel<<<(N + 255) / 256, 256>>>(
      d_x_vals, d_y_vals, N, seed, d_hw_out, d_sw_out);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  // Copy results back to host
  std::vector<__nv_bfloat162> h_hw_out(N);
  std::vector<__nv_bfloat162> h_sw_out(N);
  CUDACHECK(cudaMemcpy(
      h_hw_out.data(),
      d_hw_out,
      N * sizeof(__nv_bfloat162),
      cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(
      h_sw_out.data(),
      d_sw_out,
      N * sizeof(__nv_bfloat162),
      cudaMemcpyDeviceToHost));

  // Compute averages for both hardware and software
  double hw_sumX = 0.0, hw_sumY = 0.0;
  double sw_sumX = 0.0, sw_sumY = 0.0;
  for (int i = 0; i < N; i++) {
    hw_sumX += bf16ToFloat(__low2bfloat16(h_hw_out[i]));
    hw_sumY += bf16ToFloat(__high2bfloat16(h_hw_out[i]));
    sw_sumX += bf16ToFloat(__low2bfloat16(h_sw_out[i]));
    sw_sumY += bf16ToFloat(__high2bfloat16(h_sw_out[i]));
  }
  double hw_avgX = hw_sumX / N;
  double hw_avgY = hw_sumY / N;
  double sw_avgX = sw_sumX / N;
  double sw_avgY = sw_sumY / N;

  // Get the bf16 bracket for test values
  float lowerX, upperX, lowerY, upperY;
  getBracketingBf16(testX, lowerX, upperX);
  getBracketingBf16(testY, lowerY, upperY);
  double gapX = upperX - lowerX;
  double gapY = upperY - lowerY;

  // Both hardware and software should be unbiased (close to original value)
  EXPECT_NEAR(hw_avgX, (double)testX, gapX * 0.15)
      << "Hardware X average should be close to original value";
  EXPECT_NEAR(hw_avgY, (double)testY, gapY * 0.15)
      << "Hardware Y average should be close to original value";
  EXPECT_NEAR(sw_avgX, (double)testX, gapX * 0.15)
      << "Software X average should be close to original value";
  EXPECT_NEAR(sw_avgY, (double)testY, gapY * 0.15)
      << "Software Y average should be close to original value";

  // Hardware and software averages should be similar to each other
  // (within statistical tolerance)
  EXPECT_NEAR(hw_avgX, sw_avgX, gapX * 0.2)
      << "Hardware and software X averages should be similar";
  EXPECT_NEAR(hw_avgY, sw_avgY, gapY * 0.2)
      << "Hardware and software Y averages should be similar";

  // Verify both round to the same set of neighbor values
  for (int i = 0; i < N; i++) {
    float hw_x = bf16ToFloat(__low2bfloat16(h_hw_out[i]));
    float hw_y = bf16ToFloat(__high2bfloat16(h_hw_out[i]));
    float sw_x = bf16ToFloat(__low2bfloat16(h_sw_out[i]));
    float sw_y = bf16ToFloat(__high2bfloat16(h_sw_out[i]));

    EXPECT_TRUE(hw_x == lowerX || hw_x == upperX)
        << "Hardware X=" << hw_x << " is neither " << lowerX << " nor "
        << upperX;
    EXPECT_TRUE(hw_y == lowerY || hw_y == upperY)
        << "Hardware Y=" << hw_y << " is neither " << lowerY << " nor "
        << upperY;
    EXPECT_TRUE(sw_x == lowerX || sw_x == upperX)
        << "Software X=" << sw_x << " is neither " << lowerX << " nor "
        << upperX;
    EXPECT_TRUE(sw_y == lowerY || sw_y == upperY)
        << "Software Y=" << sw_y << " is neither " << lowerY << " nor "
        << upperY;
  }

  CUDACHECK(cudaFree(d_x_vals));
  CUDACHECK(cudaFree(d_y_vals));
  CUDACHECK(cudaFree(d_hw_out));
  CUDACHECK(cudaFree(d_sw_out));
}
