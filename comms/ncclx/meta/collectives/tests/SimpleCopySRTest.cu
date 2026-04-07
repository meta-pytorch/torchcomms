// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "meta/collectives/kernels/reduce_copy.cuh"
#include "meta/collectives/kernels/reduce_copy_sr.cuh"

// =============================================================================
// Wrapper Kernels
// =============================================================================

// 1-src reduceCopySR wrapper
template <int Unroll, typename AccType, typename DstType, typename Src0Type>
__global__ __launch_bounds__(256, 1) void reduce_copy_sr_1src_kernel(
    DstType* dst,
    const Src0Type* src0,
    ssize_t nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopySR<Unroll, AccType>(
      thread, nThreads, dst, nElts, randomSeed, randomBaseOffset, src0);
}

// 2-src reduceCopySR wrapper
template <
    int Unroll,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(256, 1) void reduce_copy_sr_2src_kernel(
    DstType* dst,
    const Src0Type* src0,
    const Src1Type* src1,
    ssize_t nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopySR<Unroll, AccType>(
      thread, nThreads, dst, nElts, randomSeed, randomBaseOffset, src0, src1);
}

// 1-src reduceCopyMixed wrapper (for baseline comparison)
template <int Unroll, typename AccType, typename DstType, typename Src0Type>
__global__ __launch_bounds__(256, 1) void reduce_copy_mixed_1src_kernel(
    DstType* dst,
    const Src0Type* src0,
    ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopyMixed<Unroll, AccType>(
      thread, nThreads, dst, nElts, src0);
}

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Test Fixture
// =============================================================================

class SimpleCopySRTest : public ::testing::Test {
 protected:
  static constexpr int64_t kMaxN = 4L * 1024L * 1024L + 16;
  static constexpr int kBlockSize = 256;
  static constexpr int kDefaultBlocks = 32;
  static constexpr uint64_t kSeed = 42;
  static constexpr uint64_t kBaseOffset = 0;

  float* d_srcFloat0 = nullptr;
  float* d_srcFloat1 = nullptr;
  float* d_dstFloat = nullptr;
  __nv_bfloat16* d_srcBf16_0 = nullptr;
  __nv_bfloat16* d_srcBf16_1 = nullptr;
  __nv_bfloat16* d_dstBf16 = nullptr;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_srcFloat0, kMaxN * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcFloat1, kMaxN * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_dstFloat, kMaxN * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcBf16_0, kMaxN * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_srcBf16_1, kMaxN * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_dstBf16, kMaxN * sizeof(__nv_bfloat16)));
  }

  void TearDown() override {
    CUDACHECK(cudaFree(d_srcFloat0));
    CUDACHECK(cudaFree(d_srcFloat1));
    CUDACHECK(cudaFree(d_dstFloat));
    CUDACHECK(cudaFree(d_srcBf16_0));
    CUDACHECK(cudaFree(d_srcBf16_1));
    CUDACHECK(cudaFree(d_dstBf16));
  }

  // Get the two BF16 values that bracket a given FP32 value.
  static void getBracketingBf16(float val, float& lo, float& hi) {
    __nv_bfloat16 bf = __float2bfloat16(val);
    float truncated = __bfloat162float(bf);
    if (truncated == val) {
      lo = hi = val;
      return;
    }
    if (truncated < val) {
      lo = truncated;
      uint16_t bits = __bfloat16_as_ushort(bf);
      hi = __bfloat162float(__ushort_as_bfloat16(bits + 1));
    } else {
      hi = truncated;
      uint16_t bits = __bfloat16_as_ushort(bf);
      lo = __bfloat162float(__ushort_as_bfloat16(bits - 1));
    }
  }
};

// =============================================================================
// SameType_NoSR: FP32→FP32 (AccType==DstType), SR path is skipped.
// Output must be bitwise identical to reduceCopyMixed.
// =============================================================================

TEST_F(SimpleCopySRTest, SameType_NoSR) {
  constexpr int64_t sizes[] = {32, 1024, 4L * 1024L * 1024L, 1000, 3000007};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));

    std::vector<float> h_src(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src[i] = static_cast<float>(i) * 0.5f + 1.0f;
    }
    CUDACHECK(cudaMemcpy(
        d_srcFloat0,
        h_src.data(),
        nElts * sizeof(float),
        cudaMemcpyHostToDevice));

    // Run reduceCopyMixed as baseline
    CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));
    reduce_copy_mixed_1src_kernel<4, float, float, float>
        <<<kDefaultBlocks, kBlockSize>>>(d_dstFloat, d_srcFloat0, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<float> h_expected(nElts);
    CUDACHECK(cudaMemcpy(
        h_expected.data(),
        d_dstFloat,
        nElts * sizeof(float),
        cudaMemcpyDeviceToHost));

    // Run reduceCopySR with same types
    CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));
    reduce_copy_sr_1src_kernel<4, float, float, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstFloat, d_srcFloat0, nElts, kSeed, kBaseOffset);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<float> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstFloat,
        nElts * sizeof(float),
        cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_dst, h_expected);
  }
}

// =============================================================================
// Determinism: Same seed/offset → same output; different seed → different.
// =============================================================================

TEST_F(SimpleCopySRTest, Determinism) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  // Use values that are not exactly representable in BF16 so SR has effect
  std::vector<float> h_src(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src[i] = 1.0f + static_cast<float>(i) * 1e-4f;
  }
  CUDACHECK(cudaMemcpy(
      d_srcFloat0,
      h_src.data(),
      nElts * sizeof(float),
      cudaMemcpyHostToDevice));

  // Run 1: seed=42, offset=0
  CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));
  reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(d_dstBf16, d_srcFloat0, nElts, 42, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_run1(nElts);
  CUDACHECK(cudaMemcpy(
      h_run1.data(),
      d_dstBf16,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  // Run 2: same seed=42, offset=0
  CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));
  reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(d_dstBf16, d_srcFloat0, nElts, 42, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_run2(nElts);
  CUDACHECK(cudaMemcpy(
      h_run2.data(),
      d_dstBf16,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  // Same seed → identical output
  for (int64_t i = 0; i < nElts; i++) {
    EXPECT_EQ(__bfloat16_as_ushort(h_run1[i]), __bfloat16_as_ushort(h_run2[i]))
        << "Determinism failure at index " << i;
  }

  // Run 3: different seed=999
  CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));
  reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(d_dstBf16, d_srcFloat0, nElts, 999, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_run3(nElts);
  CUDACHECK(cudaMemcpy(
      h_run3.data(),
      d_dstBf16,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  // Different seed → at least some elements should differ
  int nDiff = 0;
  for (int64_t i = 0; i < nElts; i++) {
    if (__bfloat16_as_ushort(h_run1[i]) != __bfloat16_as_ushort(h_run3[i])) {
      nDiff++;
    }
  }
  EXPECT_GT(nDiff, 0)
      << "Different seed should produce at least some different outputs";
}

// =============================================================================
// Neighbor_Correctness: FP32→BF16, 1-src. Each output must be one of the
// two BF16 values bracketing the FP32 input.
// =============================================================================

TEST_F(SimpleCopySRTest, Neighbor_Correctness) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<float> h_src(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src[i] = 1.0f + static_cast<float>(i) * 1e-4f;
  }
  CUDACHECK(cudaMemcpy(
      d_srcFloat0,
      h_src.data(),
      nElts * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));

  reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstBf16, d_srcFloat0, nElts, kSeed, kBaseOffset);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_dst(nElts);
  CUDACHECK(cudaMemcpy(
      h_dst.data(),
      d_dstBf16,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < nElts; i++) {
    float result = __bfloat162float(h_dst[i]);
    float lo, hi;
    getBracketingBf16(h_src[i], lo, hi);
    EXPECT_TRUE(result == lo || result == hi)
        << "Index " << i << ": src=" << h_src[i] << " result=" << result
        << " lo=" << lo << " hi=" << hi;
  }
}

// =============================================================================
// Unbiasedness: Average over many seeds ≈ original FP32 value.
// =============================================================================

TEST_F(SimpleCopySRTest, Unbiasedness) {
  // Use a small number of elements and many seeds for statistical power
  constexpr int64_t nElts = 1024;
  constexpr int kNumSeeds = 200;

  std::vector<float> h_src(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    // Values not exactly representable in BF16
    h_src[i] = 1.0f + static_cast<float>(i) * 0.001f;
  }
  CUDACHECK(cudaMemcpy(
      d_srcFloat0,
      h_src.data(),
      nElts * sizeof(float),
      cudaMemcpyHostToDevice));

  std::vector<double> accum(nElts, 0.0);

  for (int s = 0; s < kNumSeeds; s++) {
    uint64_t seed = static_cast<uint64_t>(s * 12345 + 67);
    CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));

    reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstBf16, d_srcFloat0, nElts, seed, 0);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstBf16,
        nElts * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < nElts; i++) {
      accum[i] += static_cast<double>(__bfloat162float(h_dst[i]));
    }
  }

  for (int64_t i = 0; i < nElts; i++) {
    double avg = accum[i] / kNumSeeds;
    double expected = static_cast<double>(h_src[i]);
    // BF16 has ~7-bit mantissa, so the ULP is about 2^-7 ≈ 0.0078 for
    // values near 1.0. With 200 samples, the standard error is about
    // ULP / sqrt(200) ≈ 0.00055. Use a generous tolerance.
    double tolerance = 0.005;
    EXPECT_NEAR(avg, expected, tolerance)
        << "Unbiasedness failure at index " << i;
  }
}

// =============================================================================
// ReduceSum_2src_SR: BF16+BF16→BF16 with AccType=FP32.
// =============================================================================

TEST_F(SimpleCopySRTest, ReduceSum_2src_SR) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<__nv_bfloat16> h_src0(nElts), h_src1(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src0[i] = __float2bfloat16(static_cast<float>(i % 500) * 0.1f);
    h_src1[i] = __float2bfloat16(static_cast<float>(i % 700) * 0.2f);
  }
  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_src0.data(),
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcBf16_1,
      h_src1.data(),
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));

  reduce_copy_sr_2src_kernel<
      4,
      float,
      __nv_bfloat16,
      __nv_bfloat16,
      __nv_bfloat16><<<kDefaultBlocks, kBlockSize>>>(
      d_dstBf16, d_srcBf16_0, d_srcBf16_1, nElts, kSeed, kBaseOffset);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_dst(nElts);
  CUDACHECK(cudaMemcpy(
      h_dst.data(),
      d_dstBf16,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < nElts; i++) {
    float result = __bfloat162float(h_dst[i]);
    float sum = __bfloat162float(h_src0[i]) + __bfloat162float(h_src1[i]);
    float lo, hi;
    getBracketingBf16(sum, lo, hi);
    EXPECT_TRUE(result == lo || result == hi)
        << "Index " << i << ": sum=" << sum << " result=" << result
        << " lo=" << lo << " hi=" << hi;
  }
}

// =============================================================================
// ReduceSum_2src_FloatFloat_Bf16Dst: FP32+FP32→BF16 with SR.
// =============================================================================

TEST_F(SimpleCopySRTest, ReduceSum_2src_FloatFloat_Bf16Dst) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<float> h_src0(nElts), h_src1(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src0[i] = static_cast<float>(i % 500) * 0.1f;
    h_src1[i] = static_cast<float>(i % 700) * 0.2f;
  }
  CUDACHECK(cudaMemcpy(
      d_srcFloat0,
      h_src0.data(),
      nElts * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1,
      h_src1.data(),
      nElts * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));

  reduce_copy_sr_2src_kernel<4, float, __nv_bfloat16, float, float>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstBf16, d_srcFloat0, d_srcFloat1, nElts, kSeed, kBaseOffset);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_dst(nElts);
  CUDACHECK(cudaMemcpy(
      h_dst.data(),
      d_dstBf16,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < nElts; i++) {
    float result = __bfloat162float(h_dst[i]);
    float sum = h_src0[i] + h_src1[i];
    float lo, hi;
    getBracketingBf16(sum, lo, hi);
    EXPECT_TRUE(result == lo || result == hi)
        << "Index " << i << ": sum=" << sum << " result=" << result
        << " lo=" << lo << " hi=" << hi;
  }
}

// =============================================================================
// SmallSizes: Edge cases for tail logic paths.
// =============================================================================

TEST_F(SimpleCopySRTest, SmallSizes) {
  constexpr int64_t sizes[] = {0, 1, 2, 31, 32, 33, 127, 128, 129};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));
    if (nElts == 0) {
      reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
          <<<kDefaultBlocks, kBlockSize>>>(
              d_dstBf16, d_srcFloat0, 0, kSeed, kBaseOffset);
      CUDACHECK(cudaGetLastError());
      CUDACHECK(cudaDeviceSynchronize());
      continue;
    }

    std::vector<float> h_src(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src[i] = 1.0f + static_cast<float>(i) * 0.001f;
    }
    CUDACHECK(cudaMemcpy(
        d_srcFloat0,
        h_src.data(),
        nElts * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));

    reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstBf16, d_srcFloat0, nElts, kSeed, kBaseOffset);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstBf16,
        nElts * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < nElts; i++) {
      float result = __bfloat162float(h_dst[i]);
      float lo, hi;
      getBracketingBf16(h_src[i], lo, hi);
      EXPECT_TRUE(result == lo || result == hi)
          << "Index " << i << ": src=" << h_src[i] << " result=" << result
          << " lo=" << lo << " hi=" << hi;
    }
  }
}

// =============================================================================
// UnrollVariants: Verify all Unroll values work with SR.
// =============================================================================

TEST_F(SimpleCopySRTest, UnrollVariants) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<__nv_bfloat16> h_src0(nElts);
  std::vector<float> h_src1(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src0[i] = __float2bfloat16(static_cast<float>(i % 500));
    h_src1[i] = static_cast<float>(i % 700);
  }
  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_src0.data(),
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1,
      h_src1.data(),
      nElts * sizeof(float),
      cudaMemcpyHostToDevice));

  auto test = [&](auto unrollTag) {
    constexpr int U = decltype(unrollTag)::value;
    SCOPED_TRACE("Unroll=" + std::to_string(U));
    CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));

    reduce_copy_sr_2src_kernel<U, float, __nv_bfloat16, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstBf16, d_srcBf16_0, d_srcFloat1, nElts, kSeed, kBaseOffset);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstBf16,
        nElts * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < nElts; i++) {
      float result = __bfloat162float(h_dst[i]);
      float sum = __bfloat162float(h_src0[i]) + h_src1[i];
      float lo, hi;
      getBracketingBf16(sum, lo, hi);
      EXPECT_TRUE(result == lo || result == hi)
          << "Unroll=" << U << " index " << i << ": sum=" << sum
          << " result=" << result << " lo=" << lo << " hi=" << hi;
    }
  };

  test(std::integral_constant<int, 1>{});
  test(std::integral_constant<int, 2>{});
  test(std::integral_constant<int, 4>{});
  test(std::integral_constant<int, 8>{});
}

// =============================================================================
// CrossIndexContamination: Unique per-element values, verify no mixing.
// =============================================================================

TEST_F(SimpleCopySRTest, CrossIndexContamination) {
  constexpr int64_t sizes[] = {1, 2, 31, 33, 127, 129, 1000, 4100000};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));

    std::vector<float> h_src(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src[i] = static_cast<float>(i * 7 + 3);
    }
    CUDACHECK(cudaMemcpy(
        d_srcFloat0,
        h_src.data(),
        nElts * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));

    reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstBf16, d_srcFloat0, nElts, kSeed, kBaseOffset);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstBf16,
        nElts * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < nElts; i++) {
      float result = __bfloat162float(h_dst[i]);
      float lo, hi;
      getBracketingBf16(h_src[i], lo, hi);
      EXPECT_TRUE(result == lo || result == hi)
          << "Index " << i << ": src=" << h_src[i] << " result=" << result
          << " lo=" << lo << " hi=" << hi;
    }
  }
}
