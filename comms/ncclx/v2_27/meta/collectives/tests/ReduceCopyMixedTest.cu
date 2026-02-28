// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "meta/collectives/kernels/common_kernel_quantize.cuh" // @manual

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Test Kernels
// =============================================================================

// Single-warp kernel for testing basic correctness
template <
    int Unroll,
    typename AccumType,
    typename TransportType,
    bool Src0IsAccumType,
    bool Src1IsAccumType,
    bool Dst0IsAccumType>
__global__ void reduceCopyMixedKernel_SingleWarp(
    int nSrcs,
    void** srcs,
    int nDsts,
    void** dsts,
    int64_t nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  reduceCopyMixedImpl<
      Unroll,
      FuncSum<AccumType>,
      AccumType,
      TransportType,
      Src0IsAccumType,
      Src1IsAccumType,
      Dst0IsAccumType,
      int64_t>(
      threadIdx.x, // thread
      blockDim.x, // nThreads
      (uint64_t)0, // redArg
      nSrcs,
      srcs,
      nDsts,
      dsts,
      nElts,
      randomSeed,
      randomBaseOffset);
}

// Multi-threaded kernel using dispatch function
template <int Unroll, typename AccumType, typename TransportType>
__global__ void reduceCopyMixedKernel_MultiThread(
    int nSrcs,
    void** srcs,
    int nDsts,
    void** dsts,
    int64_t nElts,
    bool src0IsAccumType,
    bool src1IsAccumType,
    bool dst0IsAccumType,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  reduceCopyMixed<Unroll, FuncSum<AccumType>, AccumType, TransportType>(
      thread,
      nThreads,
      0, // redArg
      nSrcs,
      srcs,
      nDsts,
      dsts,
      nElts,
      src0IsAccumType,
      src1IsAccumType,
      dst0IsAccumType,
      randomSeed,
      randomBaseOffset);
}

// =============================================================================
// Test Fixture
// =============================================================================

class ReduceCopyMixedTest : public ::testing::Test {
 protected:
  static constexpr int kMaxElts = 65536;
  static constexpr int kBlockSize = 256;

  float* d_srcFloat0 = nullptr;
  float* d_srcFloat1 = nullptr;
  float* d_dstFloat = nullptr;
  __nv_bfloat16* d_srcBf16_0 = nullptr;
  __nv_bfloat16* d_srcBf16_1 = nullptr;
  __nv_bfloat16* d_dstBf16 = nullptr;
  void** d_srcs = nullptr;
  void** d_dsts = nullptr;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_srcFloat0, kMaxElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcFloat1, kMaxElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_dstFloat, kMaxElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcBf16_0, kMaxElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_srcBf16_1, kMaxElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_dstBf16, kMaxElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_srcs, 2 * sizeof(void*)));
    CUDACHECK(cudaMalloc(&d_dsts, sizeof(void*)));
  }

  void TearDown() override {
    cudaFree(d_srcFloat0);
    cudaFree(d_srcFloat1);
    cudaFree(d_dstFloat);
    cudaFree(d_srcBf16_0);
    cudaFree(d_srcBf16_1);
    cudaFree(d_dstBf16);
    cudaFree(d_srcs);
    cudaFree(d_dsts);
  }

  void setSources(void* src0, void* src1 = nullptr) {
    void* h_srcs[2] = {src0, src1};
    CUDACHECK(
        cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  }

  void setDestination(void* dst) {
    CUDACHECK(cudaMemcpy(d_dsts, &dst, sizeof(void*), cudaMemcpyHostToDevice));
  }

  std::vector<float> readFloatOutput(int n) {
    std::vector<float> result(n);
    auto res = cudaMemcpy(
        result.data(), d_dstFloat, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (res != cudaSuccess) {
      std::cout << "Error reading output: " << cudaGetErrorString(res)
                << std::endl;
      throw std::runtime_error("Error reading output");
    }
    return result;
  }

  std::vector<float> readBf16AsFloat(int n) {
    std::vector<__nv_bfloat16> bf16Result(n);
    auto res = cudaMemcpy(
        bf16Result.data(),
        d_dstBf16,
        n * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost);
    if (res != cudaSuccess) {
      std::cout << "Error reading output: " << cudaGetErrorString(res)
                << std::endl;
      throw std::runtime_error("Error reading output");
    }
    std::vector<float> result(n);
    for (int i = 0; i < n; i++) {
      result[i] = __bfloat162float(bf16Result[i]);
    }
    return result;
  }
};

// =============================================================================
// Basic Correctness Tests
// =============================================================================

// Test: Single float source -> float destination (no conversion)
TEST_F(ReduceCopyMixedTest, SingleSourceFloatToFloat) {
  constexpr int N = 1024;
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = static_cast<float>(i) * 0.5f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcFloat0);
  setDestination(d_dstFloat);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, true>
      <<<1, 32>>>(
          1, // nSrcs
          d_srcs,
          1, // nDsts
          d_dsts,
          N,
          12345ULL, // seed
          0 // baseOffset
      );
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(result[i], h_src[i]) << "Mismatch at index " << i;
  }
}

// Test: Two float sources -> float destination (sum reduction)
TEST_F(ReduceCopyMixedTest, TwoSourceFloatSumReduction) {
  constexpr int N = 512;
  std::vector<float> h_src0(N), h_src1(N);
  for (int i = 0; i < N; i++) {
    h_src0[i] = static_cast<float>(i);
    h_src1[i] = static_cast<float>(i * 2);
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1, h_src1.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcFloat0, d_srcFloat1);
  setDestination(d_dstFloat);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, true>
      <<<1, 32>>>(2, d_srcs, 1, d_dsts, N, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    float expected = h_src0[i] + h_src1[i];
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

// Test: Float source -> BF16 destination (with stochastic rounding)
TEST_F(ReduceCopyMixedTest, FloatToBf16WithStochasticRounding) {
  constexpr int N = 1024;
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = 1.0f + static_cast<float>(i) * 1e-4f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));

  setSources(d_srcFloat0);
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, 42ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readBf16AsFloat(N);
  for (int i = 0; i < N; i++) {
    float lower = std::floor(h_src[i] * 128.0f) / 128.0f;
    float upper = std::ceil(h_src[i] * 128.0f) / 128.0f;
    bool valid = (result[i] >= lower - 1e-6f && result[i] <= upper + 1e-6f);
    EXPECT_TRUE(valid) << "Value " << result[i] << " at index " << i
                       << " not in valid range [" << lower << ", " << upper
                       << "]"
                       << " for input " << h_src[i];
  }
}

// Test: BF16 source -> Float destination
TEST_F(ReduceCopyMixedTest, Bf16ToFloat) {
  constexpr int N = 256;
  std::vector<float> h_srcFloat(N);
  std::vector<__nv_bfloat16> h_srcBf16(N);
  for (int i = 0; i < N; i++) {
    h_srcFloat[i] = static_cast<float>(i) * 0.25f;
    h_srcBf16[i] = __float2bfloat16(h_srcFloat[i]);
  }

  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_srcBf16.data(),
      N * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcBf16_0);
  setDestination(d_dstFloat);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, false, false, true>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    float expected = __bfloat162float(h_srcBf16[i]);
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

// Test: Empty input (nElts = 0)
TEST_F(ReduceCopyMixedTest, EmptyInput) {
  std::vector<float> h_initial = {999.0f, 888.0f, 777.0f};
  CUDACHECK(cudaMemcpy(
      d_dstFloat, h_initial.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

  setSources(d_srcFloat0);
  setDestination(d_dstFloat);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, true>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, 0, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(3);
  EXPECT_FLOAT_EQ(result[0], 999.0f);
  EXPECT_FLOAT_EQ(result[1], 888.0f);
  EXPECT_FLOAT_EQ(result[2], 777.0f);
}

// Test: Single element
TEST_F(ReduceCopyMixedTest, SingleElement) {
  float h_src = 42.5f;
  CUDACHECK(
      cudaMemcpy(d_srcFloat0, &h_src, sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, sizeof(float)));

  setSources(d_srcFloat0);
  setDestination(d_dstFloat);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, true>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, 1, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(1);
  EXPECT_FLOAT_EQ(result[0], 42.5f);
}

// Test: Non-power-of-2 size
TEST_F(ReduceCopyMixedTest, NonPowerOf2Size) {
  constexpr int N = 1000; // Not a power of 2
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = static_cast<float>(i + 1);
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcFloat0);
  setDestination(d_dstFloat);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, true>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(result[i], h_src[i]) << "Mismatch at index " << i;
  }
}

// Test: Large input
TEST_F(ReduceCopyMixedTest, LargeInput) {
  constexpr int N = 65536;
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = static_cast<float>(i % 1000);
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcFloat0);
  setDestination(d_dstFloat);

  int nBlocks = (N + kBlockSize - 1) / kBlockSize;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          1,
          d_srcs,
          1,
          d_dsts,
          N,
          true, // src0IsAccumType
          true, // src1IsAccumType
          true, // dst0IsAccumType
          0,
          0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(result[i], h_src[i]) << "Mismatch at index " << i;
  }
}

// =============================================================================
// Stochastic Rounding Tests
// =============================================================================

// Test: Same seed produces same results (determinism)
TEST_F(ReduceCopyMixedTest, StochasticRoundingDeterminism) {
  constexpr int N = 1024;
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = 1.0f + static_cast<float>(i) * 1.5e-4f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // First run
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));
  setSources(d_srcFloat0);
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, 12345ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result1 = readBf16AsFloat(N);

  // Second run with same seed
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, 12345ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result2 = readBf16AsFloat(N);

  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(result1[i], result2[i])
        << "Non-deterministic result at index " << i;
  }
}

// Test: Different seeds produce different results
TEST_F(ReduceCopyMixedTest, DifferentSeedsDifferentResults) {
  constexpr int N = 1024;
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = 1.0f + static_cast<float>(i) * 1.5e-4f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // First run with seed 1
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));
  setSources(d_srcFloat0);
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, 11111ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result1 = readBf16AsFloat(N);

  // Second run with different seed
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, 99999ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result2 = readBf16AsFloat(N);

  int diffCount = 0;
  for (int i = 0; i < N; i++) {
    if (result1[i] != result2[i]) {
      diffCount++;
    }
  }
  EXPECT_GT(diffCount, N / 4)
      << "Different seeds should produce different rounding patterns";
}

// Test: Stochastic rounding is statistically unbiased
TEST_F(ReduceCopyMixedTest, StochasticRoundingUnbiased) {
  constexpr int N = 10000;
  constexpr int kTrials = 100;

  float testValue = 1.0f + 0.00390625f * 0.5f;

  std::vector<float> h_src(N, testValue);
  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  double sumRounded = 0.0;

  for (int trial = 0; trial < kTrials; trial++) {
    CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));
    setSources(d_srcFloat0);
    setDestination(d_dstBf16);

    uint64_t seed = 1000ULL + trial;
    reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
        <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, seed, 0);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    auto result = readBf16AsFloat(N);
    for (int i = 0; i < N; i++) {
      sumRounded += result[i];
    }
  }

  double avgRounded = sumRounded / (N * kTrials);
  double relError = std::abs(avgRounded - testValue) / testValue;
  EXPECT_LT(relError, 0.01)
      << "Stochastic rounding appears biased: avg=" << avgRounded
      << ", expected=" << testValue;
}

// =============================================================================
// Special Value Tests
// =============================================================================

// Test: Handle infinity correctly
TEST_F(ReduceCopyMixedTest, InfinityHandling) {
  std::vector<float> h_src = {
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      1.0f,
      -1.0f};

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, 4 * sizeof(__nv_bfloat16)));

  setSources(d_srcFloat0);
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, 4, 12345ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readBf16AsFloat(4);
  EXPECT_TRUE(std::isinf(result[0]) && result[0] > 0)
      << "Positive infinity not preserved";
  EXPECT_TRUE(std::isinf(result[1]) && result[1] < 0)
      << "Negative infinity not preserved";
}

// Test: Handle NaN correctly
TEST_F(ReduceCopyMixedTest, NaNHandling) {
  std::vector<float> h_src = {std::numeric_limits<float>::quiet_NaN(), 1.0f};

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), 2 * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, 2 * sizeof(__nv_bfloat16)));

  setSources(d_srcFloat0);
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, 2, 12345ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readBf16AsFloat(2);
  EXPECT_TRUE(std::isnan(result[0])) << "NaN not preserved, got " << result[0];
  EXPECT_FALSE(std::isnan(result[1])) << "Normal value corrupted";
}

// Test: Handle zero correctly
TEST_F(ReduceCopyMixedTest, ZeroHandling) {
  std::vector<float> h_src = {0.0f, -0.0f, 1.0f};

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, 3 * sizeof(__nv_bfloat16)));

  setSources(d_srcFloat0);
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, 3, 12345ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readBf16AsFloat(3);
  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[2], 1.0f);
}

// =============================================================================
// Type Configuration Tests
// =============================================================================

// Test: BF16 -> BF16 (transport type passthrough)
TEST_F(ReduceCopyMixedTest, Bf16ToBf16Passthrough) {
  constexpr int N = 256;
  std::vector<__nv_bfloat16> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = __float2bfloat16(static_cast<float>(i));
  }

  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_src.data(),
      N * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));

  setSources(d_srcBf16_0);
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, false, false, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readBf16AsFloat(N);
  for (int i = 0; i < N; i++) {
    float expected = __bfloat162float(h_src[i]);
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

// Test: Mixed types - BF16 + Float -> Float
TEST_F(ReduceCopyMixedTest, MixedBf16FloatToFloat) {
  constexpr int N = 256;
  std::vector<__nv_bfloat16> h_srcBf16(N);
  std::vector<float> h_srcFloat(N);

  for (int i = 0; i < N; i++) {
    h_srcBf16[i] = __float2bfloat16(static_cast<float>(i));
    h_srcFloat[i] = static_cast<float>(i * 2);
  }

  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_srcBf16.data(),
      N * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1,
      h_srcFloat.data(),
      N * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstFloat);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, false, true, true>
      <<<1, 32>>>(2, d_srcs, 1, d_dsts, N, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    float bf16AsFloat = __bfloat162float(h_srcBf16[i]);
    float expected = bf16AsFloat + h_srcFloat[i];
    EXPECT_NEAR(result[i], expected, 0.5f) << "Mismatch at index " << i;
  }
}

// =============================================================================
// Multi-threaded Tests
// =============================================================================

// Test: Multi-threaded correctness
TEST_F(ReduceCopyMixedTest, MultiThreadedCorrectness) {
  constexpr int N = 8192;
  std::vector<float> h_src0(N), h_src1(N);
  for (int i = 0; i < N; i++) {
    h_src0[i] = static_cast<float>(i);
    h_src1[i] = static_cast<float>(i * 3);
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1, h_src1.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcFloat0, d_srcFloat1);
  setDestination(d_dstFloat);

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2,
          d_srcs,
          1,
          d_dsts,
          N,
          true, // src0IsAccumType
          true, // src1IsAccumType
          true, // dst0IsAccumType
          0,
          0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    float expected = h_src0[i] + h_src1[i];
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

// Test: Multi-threaded with stochastic rounding
TEST_F(ReduceCopyMixedTest, MultiThreadedStochasticRounding) {
  constexpr int N = 4096;
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = 1.0f + static_cast<float>(i) * 1e-4f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));

  setSources(d_srcFloat0);
  setDestination(d_dstBf16);

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          1,
          d_srcs,
          1,
          d_dsts,
          N,
          true, // src0IsAccumType
          true, // src1IsAccumType
          false, // dst0IsAccumType (BF16 output)
          12345ULL,
          0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readBf16AsFloat(N);
  for (int i = 0; i < N; i++) {
    float lower = std::floor(h_src[i] * 128.0f) / 128.0f;
    float upper = std::ceil(h_src[i] * 128.0f) / 128.0f;
    bool valid = (result[i] >= lower - 1e-6f && result[i] <= upper + 1e-6f);
    EXPECT_TRUE(valid) << "Value " << result[i] << " at index " << i
                       << " not in valid range [" << lower << ", " << upper
                       << "]";
  }
}

// =============================================================================
// Offset Tests
// =============================================================================

// Test: Different offsets produce different results
TEST_F(ReduceCopyMixedTest, DifferentOffsetsProduceDifferentResults) {
  constexpr int N = 512;
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = 1.0f + static_cast<float>(i) * 1.5e-4f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // Run with offset 0
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));
  setSources(d_srcFloat0);
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, 12345ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result1 = readBf16AsFloat(N);

  // Run with offset 10000
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, 12345ULL, 10000ULL);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result2 = readBf16AsFloat(N);

  int diffCount = 0;
  for (int i = 0; i < N; i++) {
    if (result1[i] != result2[i]) {
      diffCount++;
    }
  }
  EXPECT_GT(diffCount, N / 4)
      << "Different offsets should produce different rounding patterns";
}

// =============================================================================
// Vectorized Path Tests (exercise PackElts=4 packed loads/stores)
// =============================================================================

// Test: nElts not a multiple of ElemsPerHunk exercises the scalar tail path.
// With Unroll=4, WARP_SIZE=32, PackElts=4: ElemsPerHunk = 4*32*4 = 512.
// Using N=700 means 1 complete hunk (512 elts) + 188 elts in scalar tail.
TEST_F(ReduceCopyMixedTest, ScalarTailPath) {
  constexpr int N = 700; // 512 + 188
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = static_cast<float>(i) * 0.1f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcFloat0);
  setDestination(d_dstFloat);

  int nBlocks = 2;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          1, d_srcs, 1, d_dsts, N, true, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(result[i], h_src[i]) << "Mismatch at index " << i;
  }
}

// Test: Multi-block with many hunks exercises the hunk-strided loop.
// N=32768 with 4 blocks of 256 threads = 32 warps.
// ElemsPerHunk=512, nHunks=64, each warp processes 2 hunks.
TEST_F(ReduceCopyMixedTest, MultiBlockManyHunks) {
  constexpr int N = 32768;
  std::vector<float> h_src0(N), h_src1(N);
  for (int i = 0; i < N; i++) {
    h_src0[i] = static_cast<float>(i % 500);
    h_src1[i] = static_cast<float>((i + 100) % 500);
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1, h_src1.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcFloat0, d_srcFloat1);
  setDestination(d_dstFloat);

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2, d_srcs, 1, d_dsts, N, true, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    float expected = h_src0[i] + h_src1[i];
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

// Test: BF16 sources with vectorized path (tests convertPackToAccum).
// Uses multiple blocks to ensure packed bf16 loads are exercised.
TEST_F(ReduceCopyMixedTest, Bf16ToFloatMultiBlock) {
  constexpr int N = 16384;
  std::vector<float> h_srcFloat(N);
  std::vector<__nv_bfloat16> h_srcBf16(N);
  for (int i = 0; i < N; i++) {
    h_srcFloat[i] = static_cast<float>(i % 1000) * 0.1f;
    h_srcBf16[i] = __float2bfloat16(h_srcFloat[i]);
  }

  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_srcBf16.data(),
      N * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcBf16_0);
  setDestination(d_dstFloat);

  int nBlocks = 8;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          1, d_srcs, 1, d_dsts, N, false, false, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    float expected = __bfloat162float(h_srcBf16[i]);
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

// Test: Mixed bf16+float sum with stochastic rounding to bf16 at scale.
// Exercises: packed bf16 loads, packed float loads, packed reduction,
// and batched Apply_StochasticCast<..., 4>.
TEST_F(ReduceCopyMixedTest, MixedReductionWithSRMultiBlock) {
  constexpr int N = 8192;
  std::vector<__nv_bfloat16> h_srcBf16(N);
  std::vector<float> h_srcFloat(N);
  for (int i = 0; i < N; i++) {
    h_srcBf16[i] = __float2bfloat16(1.0f + static_cast<float>(i % 100) * 0.01f);
    h_srcFloat[i] = 2.0f + static_cast<float>(i % 100) * 0.01f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_srcBf16.data(),
      N * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1,
      h_srcFloat.data(),
      N * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));

  void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstBf16);

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2, d_srcs, 1, d_dsts, N, false, true, false, 54321ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readBf16AsFloat(N);
  for (int i = 0; i < N; i++) {
    float bf16AsFloat = __bfloat162float(h_srcBf16[i]);
    float expected = bf16AsFloat + h_srcFloat[i];
    // Find the two adjacent BF16 values bracketing 'expected' using float32
    // bit manipulation. BF16 is float32 with the bottom 16 mantissa bits
    // zeroed, so truncating gives the lower BF16 bound and incrementing
    // by 0x10000 gives the upper bound.
    uint32_t f32bits;
    memcpy(&f32bits, &expected, sizeof(uint32_t));
    uint32_t lower_bits = f32bits & 0xFFFF0000u; // truncate to BF16
    uint32_t upper_bits = lower_bits + 0x00010000u; // next BF16 value
    float lower_val, upper_val;
    memcpy(&lower_val, &lower_bits, sizeof(float));
    memcpy(&upper_val, &upper_bits, sizeof(float));
    bool valid =
        (result[i] >= lower_val - 1e-6f && result[i] <= upper_val + 1e-6f);
    EXPECT_TRUE(valid) << "Value " << result[i] << " at index " << i
                       << " not in valid range [" << lower_val << ", "
                       << upper_val << "] for expected " << expected;
  }
}

// Test: nElts smaller than one hunk (all work in scalar tail).
// ElemsPerHunk = 4*32*4 = 512 with 1 warp. With 256 threads = 8 warps,
// ElemsPerHunk = 512, but with 8 warps the first hunk needs 8*512 = 4096
// elements. N=100 < 4096, so nHunksTotal=0, all scalar tail.
TEST_F(ReduceCopyMixedTest, AllScalarTailSmallN) {
  constexpr int N = 100;
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = static_cast<float>(i) * 0.5f + 1.0f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcFloat0);
  setDestination(d_dstFloat);

  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<1, kBlockSize>>>(1, d_srcs, 1, d_dsts, N, true, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(result[i], h_src[i]) << "Mismatch at index " << i;
  }
}

// Test: Same seed and offset produce same results across runs
TEST_F(ReduceCopyMixedTest, SameSeedAndOffsetDeterminism) {
  constexpr int N = 512;
  std::vector<float> h_src(N);
  for (int i = 0; i < N; i++) {
    h_src[i] = 1.0f + static_cast<float>(i) * 1.5e-4f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  uint64_t seed = 0xDEADBEEFULL;
  uint64_t offset = 100000ULL;

  // First run
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));
  setSources(d_srcFloat0);
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, seed, offset);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result1 = readBf16AsFloat(N);

  // Second run with same seed and offset
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));
  setDestination(d_dstBf16);

  reduceCopyMixedKernel_SingleWarp<4, float, __nv_bfloat16, true, true, false>
      <<<1, 32>>>(1, d_srcs, 1, d_dsts, N, seed, offset);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());
  auto result2 = readBf16AsFloat(N);

  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(result1[i], result2[i])
        << "Non-deterministic at index " << i;
  }
}

// =============================================================================
// Alignment Correctness Tests
// =============================================================================

// Test: src1 offset by 1 element, src0 and dst0 aligned. Float->Float.
TEST_F(ReduceCopyMixedTest, MisalignedSrc1Only_FloatToFloat) {
  constexpr int N = 4096;
  constexpr int kSrc1Offset = 1;
  std::vector<float> h_src0(N), h_src1(N);
  for (int i = 0; i < N; i++) {
    h_src0[i] = static_cast<float>(i) * 0.5f;
    h_src1[i] = static_cast<float>(i) * 0.25f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1 + kSrc1Offset,
      h_src1.data(),
      N * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  void* h_srcs[2] = {d_srcFloat0, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstFloat);

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2, d_srcs, 1, d_dsts, N, true, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    float expected = h_src0[i] + h_src1[i];
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

// Test: src0 is bf16 (aligned), src1 is float (offset by 1), dst0 is float.
TEST_F(ReduceCopyMixedTest, MisalignedSrc1Only_Bf16ToFloat) {
  constexpr int N = 4096;
  constexpr int kSrc1Offset = 1;
  std::vector<__nv_bfloat16> h_srcBf16(N);
  std::vector<float> h_srcFloat(N);
  for (int i = 0; i < N; i++) {
    h_srcBf16[i] = __float2bfloat16(static_cast<float>(i) * 0.5f);
    h_srcFloat[i] = static_cast<float>(i) * 0.25f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_srcBf16.data(),
      N * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1 + kSrc1Offset,
      h_srcFloat.data(),
      N * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstFloat);

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2, d_srcs, 1, d_dsts, N, false, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    float expected = __bfloat162float(h_srcBf16[i]) + h_srcFloat[i];
    EXPECT_NEAR(result[i], expected, 0.5f) << "Mismatch at index " << i;
  }
}

// Test: src0 bf16, src1 float (offset by 1), dst0 bf16 with SR.
TEST_F(ReduceCopyMixedTest, MisalignedSrc1Only_FloatToBf16WithSR) {
  constexpr int N = 4096;
  constexpr int kSrc1Offset = 1;
  std::vector<__nv_bfloat16> h_srcBf16(N);
  std::vector<float> h_srcFloat(N);
  for (int i = 0; i < N; i++) {
    h_srcBf16[i] = __float2bfloat16(1.0f + static_cast<float>(i % 100) * 0.01f);
    h_srcFloat[i] = 2.0f + static_cast<float>(i % 100) * 0.01f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_srcBf16.data(),
      N * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1 + kSrc1Offset,
      h_srcFloat.data(),
      N * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, N * sizeof(__nv_bfloat16)));

  void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstBf16);

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2, d_srcs, 1, d_dsts, N, false, true, false, 54321ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readBf16AsFloat(N);
  for (int i = 0; i < N; i++) {
    float bf16AsFloat = __bfloat162float(h_srcBf16[i]);
    float expected = bf16AsFloat + h_srcFloat[i];
    uint32_t f32bits;
    memcpy(&f32bits, &expected, sizeof(uint32_t));
    uint32_t lower_bits = f32bits & 0xFFFF0000u;
    uint32_t upper_bits = lower_bits + 0x00010000u;
    float lower_val, upper_val;
    memcpy(&lower_val, &lower_bits, sizeof(float));
    memcpy(&upper_val, &upper_bits, sizeof(float));
    bool valid =
        (result[i] >= lower_val - 1e-6f && result[i] <= upper_val + 1e-6f);
    EXPECT_TRUE(valid) << "Value " << result[i] << " at index " << i
                       << " not in valid range [" << lower_val << ", "
                       << upper_val << "] for expected " << expected;
  }
}

// Test: Sources aligned, dst0 offset by 1. Float->Float.
TEST_F(ReduceCopyMixedTest, MisalignedDst0Only_FloatToFloat) {
  constexpr int N = 4096;
  constexpr int kDst0Offset = 1;
  std::vector<float> h_src0(N), h_src1(N);
  for (int i = 0; i < N; i++) {
    h_src0[i] = static_cast<float>(i) * 0.5f;
    h_src1[i] = static_cast<float>(i) * 0.25f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1, h_src1.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, (N + kDst0Offset) * sizeof(float)));

  setSources(d_srcFloat0, d_srcFloat1);
  void* dstOffset = d_dstFloat + kDst0Offset;
  CUDACHECK(
      cudaMemcpy(d_dsts, &dstOffset, sizeof(void*), cudaMemcpyHostToDevice));

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2, d_srcs, 1, d_dsts, N, true, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<float> result(N);
  CUDACHECK(cudaMemcpy(
      result.data(),
      d_dstFloat + kDst0Offset,
      N * sizeof(float),
      cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    float expected = h_src0[i] + h_src1[i];
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

// Test: Sources aligned, dst0 bf16 offset by 1, with SR.
TEST_F(ReduceCopyMixedTest, MisalignedDst0Only_FloatToBf16WithSR) {
  constexpr int N = 4096;
  constexpr int kDst0Offset = 1;
  std::vector<float> h_src0(N), h_src1(N);
  for (int i = 0; i < N; i++) {
    h_src0[i] = 1.0f + static_cast<float>(i % 100) * 0.01f;
    h_src1[i] = 2.0f + static_cast<float>(i % 100) * 0.01f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1, h_src1.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(
      cudaMemset(d_dstBf16, 0, (N + kDst0Offset) * sizeof(__nv_bfloat16)));

  setSources(d_srcFloat0, d_srcFloat1);
  void* dstOffset = d_dstBf16 + kDst0Offset;
  CUDACHECK(
      cudaMemcpy(d_dsts, &dstOffset, sizeof(void*), cudaMemcpyHostToDevice));

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2, d_srcs, 1, d_dsts, N, true, true, false, 12345ULL, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> bf16Result(N);
  CUDACHECK(cudaMemcpy(
      bf16Result.data(),
      d_dstBf16 + kDst0Offset,
      N * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    float result = __bfloat162float(bf16Result[i]);
    float expected = h_src0[i] + h_src1[i];
    uint32_t f32bits;
    memcpy(&f32bits, &expected, sizeof(uint32_t));
    uint32_t lower_bits = f32bits & 0xFFFF0000u;
    uint32_t upper_bits = lower_bits + 0x00010000u;
    float lower_val, upper_val;
    memcpy(&lower_val, &lower_bits, sizeof(float));
    memcpy(&upper_val, &upper_bits, sizeof(float));
    bool valid = (result >= lower_val - 1e-6f && result <= upper_val + 1e-6f);
    EXPECT_TRUE(valid) << "Value " << result << " at index " << i
                       << " not in valid range [" << lower_val << ", "
                       << upper_val << "] for expected " << expected;
  }
}

// Test: All buffers misaligned with different offsets. Float->Float.
TEST_F(ReduceCopyMixedTest, MisalignedAllBuffers) {
  constexpr int N = 4096;
  constexpr int kSrc0Offset = 1;
  constexpr int kSrc1Offset = 2;
  constexpr int kDst0Offset = 3;
  std::vector<float> h_src0(N), h_src1(N);
  for (int i = 0; i < N; i++) {
    h_src0[i] = static_cast<float>(i) * 0.5f;
    h_src1[i] = static_cast<float>(i) * 0.25f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0 + kSrc0Offset,
      h_src0.data(),
      N * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1 + kSrc1Offset,
      h_src1.data(),
      N * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, (N + kDst0Offset) * sizeof(float)));

  void* h_srcs[2] = {d_srcFloat0 + kSrc0Offset, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  void* dstOffset = d_dstFloat + kDst0Offset;
  CUDACHECK(
      cudaMemcpy(d_dsts, &dstOffset, sizeof(void*), cudaMemcpyHostToDevice));

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2, d_srcs, 1, d_dsts, N, true, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<float> result(N);
  CUDACHECK(cudaMemcpy(
      result.data(),
      d_dstFloat + kDst0Offset,
      N * sizeof(float),
      cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    float expected = h_src0[i] + h_src1[i];
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

// Test: Small N=2 with src1 offset by 1.
TEST_F(ReduceCopyMixedTest, MisalignedSmallN) {
  constexpr int N = 2;
  constexpr int kSrc1Offset = 1;
  std::vector<float> h_src0 = {3.0f, 7.0f};
  std::vector<float> h_src1 = {4.0f, 8.0f};

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1 + kSrc1Offset,
      h_src1.data(),
      N * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  void* h_srcs[2] = {d_srcFloat0, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstFloat);

  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<1, kBlockSize>>>(2, d_srcs, 1, d_dsts, N, true, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  EXPECT_FLOAT_EQ(result[0], 7.0f);
  EXPECT_FLOAT_EQ(result[1], 15.0f);
}

// Test: Single element N=1 with src1 offset by 1.
TEST_F(ReduceCopyMixedTest, MisalignedSmallN_SingleElement) {
  constexpr int N = 1;
  constexpr int kSrc1Offset = 1;
  float h_src0 = 5.0f;
  float h_src1 = 11.0f;

  CUDACHECK(
      cudaMemcpy(d_srcFloat0, &h_src0, sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1 + kSrc1Offset,
      &h_src1,
      sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, sizeof(float)));

  void* h_srcs[2] = {d_srcFloat0, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstFloat);

  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<1, kBlockSize>>>(2, d_srcs, 1, d_dsts, N, true, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(1);
  EXPECT_FLOAT_EQ(result[0], 16.0f);
}

// Test: Aligned regression test. N=4096, all pointers aligned.
TEST_F(ReduceCopyMixedTest, AlignedUnchanged) {
  constexpr int N = 4096;
  std::vector<float> h_src0(N), h_src1(N);
  for (int i = 0; i < N; i++) {
    h_src0[i] = static_cast<float>(i);
    h_src1[i] = static_cast<float>(i * 2);
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1, h_src1.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  setSources(d_srcFloat0, d_srcFloat1);
  setDestination(d_dstFloat);

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2, d_srcs, 1, d_dsts, N, true, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    float expected = h_src0[i] + h_src1[i];
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}

// Test: Large N=16384 with 4 blocks, src1 offset by 3.
TEST_F(ReduceCopyMixedTest, MisalignedMultiBlock) {
  constexpr int N = 16384;
  constexpr int kSrc1Offset = 3;
  std::vector<float> h_src0(N), h_src1(N);
  for (int i = 0; i < N; i++) {
    h_src0[i] = static_cast<float>(i % 1000) * 0.1f;
    h_src1[i] = static_cast<float>((i + 500) % 1000) * 0.1f;
  }

  CUDACHECK(cudaMemcpy(
      d_srcFloat0, h_src0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1 + kSrc1Offset,
      h_src1.data(),
      N * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, N * sizeof(float)));

  void* h_srcs[2] = {d_srcFloat0, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstFloat);

  int nBlocks = 4;
  reduceCopyMixedKernel_MultiThread<4, float, __nv_bfloat16>
      <<<nBlocks, kBlockSize>>>(
          2, d_srcs, 1, d_dsts, N, true, true, true, 0, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  auto result = readFloatOutput(N);
  for (int i = 0; i < N; i++) {
    float expected = h_src0[i] + h_src1[i];
    EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at index " << i;
  }
}
