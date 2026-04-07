// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "meta/collectives/kernels/reduce_copy.cuh"

// =============================================================================
// Wrapper Kernels
// =============================================================================

// 1-src reduceCopyMixed: AccType accumulation, DstType output.
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

// 2-src reduceCopyMixed: AccType accumulation, DstType output.
template <
    int Unroll,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(256, 1) void reduce_copy_mixed_2src_kernel(
    DstType* dst,
    const Src0Type* src0,
    const Src1Type* src1,
    ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopyMixed<Unroll, AccType>(
      thread, nThreads, dst, nElts, src0, src1);
}

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Test Fixture
// =============================================================================

class SimpleCopyMixedTest : public ::testing::Test {
 protected:
  static constexpr int64_t kMaxN = 4L * 1024L * 1024L + 16;
  static constexpr int kBlockSize = 256;
  static constexpr int kDefaultBlocks = 32;

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
};

// =============================================================================
// 1-src tests: FP32 -> FP32 (AccType=FP32, DstType=FP32, Src=FP32)
// Degenerates to plain copy; verifies no cross-index contamination.
// =============================================================================

TEST_F(SimpleCopyMixedTest, Float_Float_1src_Aligned) {
  constexpr int64_t sizes[] = {
      32, 1024, 4L * 1024L * 1024L, 1000, 3000007, 4100000};

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
    CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));

    reduce_copy_mixed_1src_kernel<4, float, float, float>
        <<<kDefaultBlocks, kBlockSize>>>(d_dstFloat, d_srcFloat0, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<float> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstFloat,
        nElts * sizeof(float),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(h_src, h_dst);
  }
}

// =============================================================================
// 1-src tests: BF16 -> FP32 (AccType=FP32, DstType=FP32, Src=BF16)
// =============================================================================

TEST_F(SimpleCopyMixedTest, Bf16Src_FloatDst_1src) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<float> h_srcFloat(nElts);
  std::vector<__nv_bfloat16> h_srcBf16(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_srcFloat[i] = static_cast<float>(i % 1000) * 0.25f;
    h_srcBf16[i] = __float2bfloat16(h_srcFloat[i]);
  }
  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_srcBf16.data(),
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));

  reduce_copy_mixed_1src_kernel<4, float, float, __nv_bfloat16>
      <<<kDefaultBlocks, kBlockSize>>>(d_dstFloat, d_srcBf16_0, nElts);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<float> h_dst(nElts);
  CUDACHECK(cudaMemcpy(
      h_dst.data(), d_dstFloat, nElts * sizeof(float), cudaMemcpyDeviceToHost));

  // Each output should be the float representation of the bf16 value.
  for (int64_t i = 0; i < nElts; i++) {
    float expected = __bfloat162float(h_srcBf16[i]);
    EXPECT_FLOAT_EQ(h_dst[i], expected) << "Mismatch at index " << i;
  }
}

// =============================================================================
// 1-src tests: FP32 -> BF16 (AccType=FP32, DstType=BF16, Src=FP32)
// =============================================================================

TEST_F(SimpleCopyMixedTest, FloatSrc_Bf16Dst_1src) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<float> h_src(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src[i] = static_cast<float>(i % 1000) * 0.25f;
  }
  CUDACHECK(cudaMemcpy(
      d_srcFloat0,
      h_src.data(),
      nElts * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));

  reduce_copy_mixed_1src_kernel<4, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(d_dstBf16, d_srcFloat0, nElts);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_dstBf16(nElts);
  CUDACHECK(cudaMemcpy(
      h_dstBf16.data(),
      d_dstBf16,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < nElts; i++) {
    float result = __bfloat162float(h_dstBf16[i]);
    // applyCast truncates FP32 -> BF16, so the result should be the truncated
    // bf16 representation of the source.
    float expected = __bfloat162float(__float2bfloat16(h_src[i]));
    EXPECT_FLOAT_EQ(result, expected) << "Mismatch at index " << i;
  }
}

// =============================================================================
// 1-src tests: BF16 -> BF16 (AccType=FP32, DstType=BF16, Src=BF16)
// =============================================================================

TEST_F(SimpleCopyMixedTest, Bf16Src_Bf16Dst_1src) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<__nv_bfloat16> h_src(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src[i] = __float2bfloat16(static_cast<float>(i % 1000) * 0.25f);
  }
  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_src.data(),
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));

  reduce_copy_mixed_1src_kernel<4, float, __nv_bfloat16, __nv_bfloat16>
      <<<kDefaultBlocks, kBlockSize>>>(d_dstBf16, d_srcBf16_0, nElts);
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
    float expected = __bfloat162float(h_src[i]);
    EXPECT_FLOAT_EQ(result, expected) << "Mismatch at index " << i;
  }
}

// =============================================================================
// 2-src tests: FP32 + FP32 -> FP32 (AccType=FP32)
// Verifies sum and no cross-index contamination.
// =============================================================================

TEST_F(SimpleCopyMixedTest, FloatFloat_FloatDst_2src) {
  constexpr int64_t sizes[] = {32, 1024, 4L * 1024L * 1024L, 1000, 3000007};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));

    std::vector<float> h_src0(nElts), h_src1(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src0[i] = static_cast<float>(i);
      h_src1[i] = static_cast<float>(i * 2 + 1);
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
    CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));

    reduce_copy_mixed_2src_kernel<4, float, float, float, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstFloat, d_srcFloat0, d_srcFloat1, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<float> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstFloat,
        nElts * sizeof(float),
        cudaMemcpyDeviceToHost));

    std::vector<float> expected(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      expected[i] = h_src0[i] + h_src1[i];
    }
    EXPECT_EQ(h_dst, expected);
  }
}

// =============================================================================
// 2-src tests: BF16 + BF16 -> FP32 (AccType=FP32, DstType=FP32)
// =============================================================================

TEST_F(SimpleCopyMixedTest, Bf16Bf16_FloatDst_2src) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<float> h_src0Float(nElts), h_src1Float(nElts);
  std::vector<__nv_bfloat16> h_src0(nElts), h_src1(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src0Float[i] = static_cast<float>(i % 500) * 0.1f;
    h_src1Float[i] = static_cast<float>(i % 700) * 0.2f;
    h_src0[i] = __float2bfloat16(h_src0Float[i]);
    h_src1[i] = __float2bfloat16(h_src1Float[i]);
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
  CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));

  reduce_copy_mixed_2src_kernel<4, float, float, __nv_bfloat16, __nv_bfloat16>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstFloat, d_srcBf16_0, d_srcBf16_1, nElts);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<float> h_dst(nElts);
  CUDACHECK(cudaMemcpy(
      h_dst.data(), d_dstFloat, nElts * sizeof(float), cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < nElts; i++) {
    // Accumulation happens in FP32 after upcast from BF16
    float expected = __bfloat162float(h_src0[i]) + __bfloat162float(h_src1[i]);
    EXPECT_FLOAT_EQ(h_dst[i], expected) << "Mismatch at index " << i;
  }
}

// =============================================================================
// 2-src tests: BF16 + FP32 -> FP32 (AccType=FP32, mixed source types)
// =============================================================================

TEST_F(SimpleCopyMixedTest, Bf16Float_FloatDst_2src) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<__nv_bfloat16> h_srcBf16(nElts);
  std::vector<float> h_srcFloat(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_srcBf16[i] = __float2bfloat16(static_cast<float>(i % 500) * 0.1f);
    h_srcFloat[i] = static_cast<float>(i % 700) * 0.2f;
  }
  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_srcBf16.data(),
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1,
      h_srcFloat.data(),
      nElts * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));

  reduce_copy_mixed_2src_kernel<4, float, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstFloat, d_srcBf16_0, d_srcFloat1, nElts);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<float> h_dst(nElts);
  CUDACHECK(cudaMemcpy(
      h_dst.data(), d_dstFloat, nElts * sizeof(float), cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < nElts; i++) {
    float expected = __bfloat162float(h_srcBf16[i]) + h_srcFloat[i];
    EXPECT_FLOAT_EQ(h_dst[i], expected) << "Mismatch at index " << i;
  }
}

// =============================================================================
// 2-src tests: FP32 + FP32 -> BF16 (AccType=FP32, DstType=BF16)
// =============================================================================

TEST_F(SimpleCopyMixedTest, FloatFloat_Bf16Dst_2src) {
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

  reduce_copy_mixed_2src_kernel<4, float, __nv_bfloat16, float, float>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstBf16, d_srcFloat0, d_srcFloat1, nElts);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_dstBf16(nElts);
  CUDACHECK(cudaMemcpy(
      h_dstBf16.data(),
      d_dstBf16,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < nElts; i++) {
    float result = __bfloat162float(h_dstBf16[i]);
    float sum = h_src0[i] + h_src1[i];
    // applyCast truncates FP32->BF16
    float expected = __bfloat162float(__float2bfloat16(sum));
    EXPECT_FLOAT_EQ(result, expected) << "Mismatch at index " << i;
  }
}

// =============================================================================
// 2-src tests: BF16 + BF16 -> BF16 (AccType=FP32, all BF16 I/O)
// =============================================================================

TEST_F(SimpleCopyMixedTest, Bf16Bf16_Bf16Dst_2src) {
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

  reduce_copy_mixed_2src_kernel<
      4,
      float,
      __nv_bfloat16,
      __nv_bfloat16,
      __nv_bfloat16><<<kDefaultBlocks, kBlockSize>>>(
      d_dstBf16, d_srcBf16_0, d_srcBf16_1, nElts);
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
    // Accumulate in FP32, then truncate to BF16
    float sum = __bfloat162float(h_src0[i]) + __bfloat162float(h_src1[i]);
    float expected = __bfloat162float(__float2bfloat16(sum));
    EXPECT_FLOAT_EQ(result, expected) << "Mismatch at index " << i;
  }
}

// =============================================================================
// 2-src tests: BF16 + FP32 -> BF16 (AccType=FP32, mixed src, BF16 dst)
// =============================================================================

TEST_F(SimpleCopyMixedTest, Bf16Float_Bf16Dst_2src) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<__nv_bfloat16> h_srcBf16(nElts);
  std::vector<float> h_srcFloat(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_srcBf16[i] = __float2bfloat16(static_cast<float>(i % 500) * 0.1f);
    h_srcFloat[i] = static_cast<float>(i % 700) * 0.2f;
  }
  CUDACHECK(cudaMemcpy(
      d_srcBf16_0,
      h_srcBf16.data(),
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcFloat1,
      h_srcFloat.data(),
      nElts * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstBf16, 0, nElts * sizeof(__nv_bfloat16)));

  reduce_copy_mixed_2src_kernel<4, float, __nv_bfloat16, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstBf16, d_srcBf16_0, d_srcFloat1, nElts);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_dstBf16(nElts);
  CUDACHECK(cudaMemcpy(
      h_dstBf16.data(),
      d_dstBf16,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < nElts; i++) {
    float result = __bfloat162float(h_dstBf16[i]);
    float sum = __bfloat162float(h_srcBf16[i]) + h_srcFloat[i];
    float expected = __bfloat162float(__float2bfloat16(sum));
    EXPECT_FLOAT_EQ(result, expected) << "Mismatch at index " << i;
  }
}

// =============================================================================
// 2-src tests: FP32 + BF16 -> FP32 (AccType=FP32, reversed order from above)
// =============================================================================

TEST_F(SimpleCopyMixedTest, FloatBf16_FloatDst_2src) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<float> h_srcFloat(nElts);
  std::vector<__nv_bfloat16> h_srcBf16(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_srcFloat[i] = static_cast<float>(i % 500) * 0.1f;
    h_srcBf16[i] = __float2bfloat16(static_cast<float>(i % 700) * 0.2f);
  }
  CUDACHECK(cudaMemcpy(
      d_srcFloat0,
      h_srcFloat.data(),
      nElts * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemcpy(
      d_srcBf16_1,
      h_srcBf16.data(),
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));

  reduce_copy_mixed_2src_kernel<4, float, float, float, __nv_bfloat16>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstFloat, d_srcFloat0, d_srcBf16_1, nElts);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<float> h_dst(nElts);
  CUDACHECK(cudaMemcpy(
      h_dst.data(), d_dstFloat, nElts * sizeof(float), cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < nElts; i++) {
    float expected = h_srcFloat[i] + __bfloat162float(h_srcBf16[i]);
    EXPECT_FLOAT_EQ(h_dst[i], expected) << "Mismatch at index " << i;
  }
}

// =============================================================================
// Cross-index contamination test: Use unique per-element values to verify
// that element i is never mixed with element j.
// =============================================================================

TEST_F(SimpleCopyMixedTest, NoIndexCrossContamination_2src) {
  // Use small non-power-of-2 sizes to stress tail paths.
  constexpr int64_t sizes[] = {1, 2, 31, 33, 127, 129, 1000, 4100000};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));

    // Use primes to make collisions between indices extremely unlikely.
    std::vector<float> h_src0(nElts), h_src1(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src0[i] = static_cast<float>(i * 7 + 3);
      h_src1[i] = static_cast<float>(i * 13 + 5);
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
    CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));

    reduce_copy_mixed_2src_kernel<4, float, float, float, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstFloat, d_srcFloat0, d_srcFloat1, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<float> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstFloat,
        nElts * sizeof(float),
        cudaMemcpyDeviceToHost));

    std::vector<float> expected(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      expected[i] = h_src0[i] + h_src1[i];
    }
    EXPECT_EQ(h_dst, expected);
  }
}

// =============================================================================
// Cross-index contamination test with mixed BF16+FP32 sources.
// =============================================================================

TEST_F(SimpleCopyMixedTest, NoIndexCrossContamination_Bf16Float_2src) {
  constexpr int64_t sizes[] = {1, 33, 127, 1000, 4100000};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));

    std::vector<__nv_bfloat16> h_src0(nElts);
    std::vector<float> h_src1(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src0[i] = __float2bfloat16(static_cast<float>(i * 7 + 3));
      h_src1[i] = static_cast<float>(i * 13 + 5);
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
    CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));

    reduce_copy_mixed_2src_kernel<4, float, float, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstFloat, d_srcBf16_0, d_srcFloat1, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<float> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstFloat,
        nElts * sizeof(float),
        cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < nElts; i++) {
      float expected = __bfloat162float(h_src0[i]) + h_src1[i];
      EXPECT_FLOAT_EQ(h_dst[i], expected) << "Mismatch at index " << i;
    }
  }
}

// =============================================================================
// Small sizes: edge cases for tail logic
// =============================================================================

TEST_F(SimpleCopyMixedTest, SmallSizes_MixedTypes) {
  constexpr int64_t sizes[] = {0, 1, 2, 31, 32, 33, 127, 128, 129};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));
    if (nElts == 0) {
      reduce_copy_mixed_2src_kernel<4, float, float, __nv_bfloat16, float>
          <<<kDefaultBlocks, kBlockSize>>>(
              d_dstFloat, d_srcBf16_0, d_srcFloat1, 0);
      CUDACHECK(cudaGetLastError());
      CUDACHECK(cudaDeviceSynchronize());
      continue;
    }

    std::vector<__nv_bfloat16> h_src0(nElts);
    std::vector<float> h_src1(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src0[i] = __float2bfloat16(static_cast<float>(i + 1));
      h_src1[i] = static_cast<float>(i * 3);
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
    CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));

    reduce_copy_mixed_2src_kernel<4, float, float, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstFloat, d_srcBf16_0, d_srcFloat1, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<float> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstFloat,
        nElts * sizeof(float),
        cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < nElts; i++) {
      float expected = __bfloat162float(h_src0[i]) + h_src1[i];
      EXPECT_FLOAT_EQ(h_dst[i], expected) << "Mismatch at index " << i;
    }
  }
}

// =============================================================================
// Unroll variants: verify all Unroll values work.
// =============================================================================

TEST_F(SimpleCopyMixedTest, UnrollVariants_Bf16Float_FloatDst) {
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
    CUDACHECK(cudaMemset(d_dstFloat, 0, nElts * sizeof(float)));

    reduce_copy_mixed_2src_kernel<U, float, float, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstFloat, d_srcBf16_0, d_srcFloat1, nElts);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<float> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstFloat,
        nElts * sizeof(float),
        cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < nElts; i++) {
      float expected = __bfloat162float(h_src0[i]) + h_src1[i];
      EXPECT_FLOAT_EQ(h_dst[i], expected) << "Mismatch at index " << i;
    }
  };

  test(std::integral_constant<int, 1>{});
  test(std::integral_constant<int, 2>{});
  test(std::integral_constant<int, 4>{});
  test(std::integral_constant<int, 8>{});
}
