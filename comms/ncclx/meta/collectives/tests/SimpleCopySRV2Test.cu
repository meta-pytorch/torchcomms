// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "meta/collectives/kernels/reduce_copy_sr.cuh"
#include "meta/collectives/kernels/reduce_copy_sr_v2.cuh"

// =============================================================================
// Wrapper Kernels — V2
// =============================================================================

template <int Unroll, typename AccType, typename DstType, typename SrcType>
__global__ __launch_bounds__(256, 1) void v2_reduce_copy_sr_1src_kernel(
    DstType* dst,
    const SrcType* src,
    ssize_t nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy_v2::reduceCopySR<Unroll, AccType>(
      thread, nThreads, dst, nElts, randomSeed, randomBaseOffset, src);
}

template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename SrcType>
__global__ __launch_bounds__(256, 1) void v2_reduce_copy_packs_sr_kernel(
    DstType* dst,
    const SrcType* src,
    ssize_t nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  ssize_t nEltsBehind = 0;
  ssize_t nEltsAhead = nElts;
  meta::comms::ncclx::kernels::simplecopy_v2::
      reduceCopyPacksSR<Unroll, EltPerPack, AccType, 1>(
          nThreads,
          thread,
          nEltsBehind,
          nEltsAhead,
          randomSeed,
          randomBaseOffset,
          src,
          dst);
}

// V1 wrapper (for cross-version comparison of neighbor correctness)
template <int Unroll, typename AccType, typename DstType, typename SrcType>
__global__ __launch_bounds__(256, 1) void v1_reduce_copy_sr_1src_kernel(
    DstType* dst,
    const SrcType* src,
    ssize_t nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopySR<Unroll, AccType>(
      thread, nThreads, dst, nElts, randomSeed, randomBaseOffset, src);
}

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Test Fixture
// =============================================================================

class SimpleCopySRV2Test : public ::testing::Test {
 protected:
  static constexpr int64_t kMaxN = 4L * 1024L * 1024L + 16;
  static constexpr int kBlockSize = 256;
  static constexpr int kDefaultBlocks = 32;
  static constexpr uint64_t kSeed = 42;
  static constexpr uint64_t kBaseOffset = 0;

  float* d_srcFloat = nullptr;
  __nv_bfloat16* d_dstBf16_A = nullptr;
  __nv_bfloat16* d_dstBf16_B = nullptr;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_srcFloat, kMaxN * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_dstBf16_A, kMaxN * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_dstBf16_B, kMaxN * sizeof(__nv_bfloat16)));
  }

  void TearDown() override {
    CUDACHECK(cudaFree(d_srcFloat));
    CUDACHECK(cudaFree(d_dstBf16_A));
    CUDACHECK(cudaFree(d_dstBf16_B));
  }

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

  void initSrc(int64_t nElts) {
    std::vector<float> h_src(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src[i] = 1.0f + static_cast<float>(i) * 1e-4f;
    }
    CUDACHECK(cudaMemcpy(
        d_srcFloat,
        h_src.data(),
        nElts * sizeof(float),
        cudaMemcpyHostToDevice));
  }
};

// =============================================================================
// CrossUnrollDeterminism: Different Unroll values → identical output
// =============================================================================

TEST_F(SimpleCopySRV2Test, CrossUnrollDeterminism) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;
  initSrc(nElts);

  // Run with Unroll=1 as reference
  CUDACHECK(cudaMemset(d_dstBf16_A, 0, nElts * sizeof(__nv_bfloat16)));
  v2_reduce_copy_sr_1src_kernel<1, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstBf16_A, d_srcFloat, nElts, kSeed, kBaseOffset);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_ref(nElts);
  CUDACHECK(cudaMemcpy(
      h_ref.data(),
      d_dstBf16_A,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  auto testUnroll = [&](auto unrollTag) {
    constexpr int U = decltype(unrollTag)::value;
    SCOPED_TRACE("Unroll=" + std::to_string(U));

    CUDACHECK(cudaMemset(d_dstBf16_B, 0, nElts * sizeof(__nv_bfloat16)));
    v2_reduce_copy_sr_1src_kernel<U, float, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstBf16_B, d_srcFloat, nElts, kSeed, kBaseOffset);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_test(nElts);
    CUDACHECK(cudaMemcpy(
        h_test.data(),
        d_dstBf16_B,
        nElts * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < nElts; i++) {
      EXPECT_EQ(__bfloat16_as_ushort(h_ref[i]), __bfloat16_as_ushort(h_test[i]))
          << "Mismatch at index " << i << " (U=1 vs U=" << U << ")";
    }
  };

  testUnroll(std::integral_constant<int, 2>{});
  testUnroll(std::integral_constant<int, 4>{});
  testUnroll(std::integral_constant<int, 8>{});
}

// =============================================================================
// CrossConfigDeterminism_Packs: Different (Unroll, EltPerPack) → identical
// output when calling reduceCopyPacksSR directly with aligned buffers.
// =============================================================================

TEST_F(SimpleCopySRV2Test, CrossConfigDeterminism_Packs) {
  // Use a size that is a multiple of the largest hunk (U=8, EPP=4: 1024 elts)
  constexpr int64_t nElts = 1024L * 1024L;
  initSrc(nElts);

  // Reference: (U=1, EPP=4) — simple path, no exchange
  CUDACHECK(cudaMemset(d_dstBf16_A, 0, nElts * sizeof(__nv_bfloat16)));
  v2_reduce_copy_packs_sr_kernel<1, 4, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstBf16_A, d_srcFloat, nElts, kSeed, kBaseOffset);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_ref(nElts);
  CUDACHECK(cudaMemcpy(
      h_ref.data(),
      d_dstBf16_A,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  auto testConfig = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;
    SCOPED_TRACE("U=" + std::to_string(U) + " EPP=" + std::to_string(EPP));

    CUDACHECK(cudaMemset(d_dstBf16_B, 0, nElts * sizeof(__nv_bfloat16)));
    v2_reduce_copy_packs_sr_kernel<U, EPP, float, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstBf16_B, d_srcFloat, nElts, kSeed, kBaseOffset);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_test(nElts);
    CUDACHECK(cudaMemcpy(
        h_test.data(),
        d_dstBf16_B,
        nElts * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < nElts; i++) {
      EXPECT_EQ(__bfloat16_as_ushort(h_ref[i]), __bfloat16_as_ushort(h_test[i]))
          << "Mismatch at index " << i << " (ref U=1,EPP=4 vs U=" << U
          << ",EPP=" << EPP << ")";
    }
  };

  // EPP=4 with different Unroll (exchange activates at U>=2)
  testConfig(
      std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{});
  testConfig(
      std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  testConfig(
      std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{});

  // EPP=2 with different Unroll (exchange activates at U>=4)
  testConfig(
      std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
  testConfig(
      std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{});
  testConfig(
      std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{});
  testConfig(
      std::integral_constant<int, 8>{}, std::integral_constant<int, 2>{});

  // EPP=1 with different Unroll (exchange activates at U>=8)
  testConfig(
      std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
  testConfig(
      std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
  testConfig(
      std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{});
  testConfig(
      std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
}

// =============================================================================
// Neighbor_Correctness: Each output is one of the two BF16 brackets.
// =============================================================================

TEST_F(SimpleCopySRV2Test, Neighbor_Correctness) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;
  initSrc(nElts);

  std::vector<float> h_src(nElts);
  CUDACHECK(cudaMemcpy(
      h_src.data(), d_srcFloat, nElts * sizeof(float), cudaMemcpyDeviceToHost));

  CUDACHECK(cudaMemset(d_dstBf16_A, 0, nElts * sizeof(__nv_bfloat16)));
  v2_reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstBf16_A, d_srcFloat, nElts, kSeed, kBaseOffset);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_dst(nElts);
  CUDACHECK(cudaMemcpy(
      h_dst.data(),
      d_dstBf16_A,
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
// Determinism: Same seed → same output; different seed → different output.
// =============================================================================

TEST_F(SimpleCopySRV2Test, Determinism) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;
  initSrc(nElts);

  // Run 1
  CUDACHECK(cudaMemset(d_dstBf16_A, 0, nElts * sizeof(__nv_bfloat16)));
  v2_reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(d_dstBf16_A, d_srcFloat, nElts, 42, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_run1(nElts);
  CUDACHECK(cudaMemcpy(
      h_run1.data(),
      d_dstBf16_A,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  // Run 2: same seed
  CUDACHECK(cudaMemset(d_dstBf16_A, 0, nElts * sizeof(__nv_bfloat16)));
  v2_reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(d_dstBf16_A, d_srcFloat, nElts, 42, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_run2(nElts);
  CUDACHECK(cudaMemcpy(
      h_run2.data(),
      d_dstBf16_A,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < nElts; i++) {
    EXPECT_EQ(__bfloat16_as_ushort(h_run1[i]), __bfloat16_as_ushort(h_run2[i]))
        << "Determinism failure at index " << i;
  }

  // Run 3: different seed
  CUDACHECK(cudaMemset(d_dstBf16_A, 0, nElts * sizeof(__nv_bfloat16)));
  v2_reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
      <<<kDefaultBlocks, kBlockSize>>>(d_dstBf16_A, d_srcFloat, nElts, 999, 0);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_run3(nElts);
  CUDACHECK(cudaMemcpy(
      h_run3.data(),
      d_dstBf16_A,
      nElts * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  int nDiff = 0;
  for (int64_t i = 0; i < nElts; i++) {
    if (__bfloat16_as_ushort(h_run1[i]) != __bfloat16_as_ushort(h_run3[i]))
      nDiff++;
  }
  EXPECT_GT(nDiff, 0) << "Different seed should produce different output";
}

// =============================================================================
// SameType_NoSR: FP32→FP32, SR path is skipped. Must match V1 exactly.
// =============================================================================

TEST_F(SimpleCopySRV2Test, SameType_NoSR) {
  constexpr int64_t nElts = 4L * 1024L * 1024L;

  std::vector<float> h_src(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src[i] = static_cast<float>(i) * 0.5f + 1.0f;
  }
  CUDACHECK(cudaMemcpy(
      d_srcFloat, h_src.data(), nElts * sizeof(float), cudaMemcpyHostToDevice));

  // Allocate float dst
  float* d_dstFloat_v1 = nullptr;
  float* d_dstFloat_v2 = nullptr;
  CUDACHECK(cudaMalloc(&d_dstFloat_v1, nElts * sizeof(float)));
  CUDACHECK(cudaMalloc(&d_dstFloat_v2, nElts * sizeof(float)));

  // V1
  CUDACHECK(cudaMemset(d_dstFloat_v1, 0, nElts * sizeof(float)));
  v1_reduce_copy_sr_1src_kernel<4, float, float, float>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstFloat_v1, d_srcFloat, nElts, kSeed, kBaseOffset);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  // V2
  CUDACHECK(cudaMemset(d_dstFloat_v2, 0, nElts * sizeof(float)));
  v2_reduce_copy_sr_1src_kernel<4, float, float, float>
      <<<kDefaultBlocks, kBlockSize>>>(
          d_dstFloat_v2, d_srcFloat, nElts, kSeed, kBaseOffset);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaDeviceSynchronize());

  std::vector<float> h_v1(nElts), h_v2(nElts);
  CUDACHECK(cudaMemcpy(
      h_v1.data(),
      d_dstFloat_v1,
      nElts * sizeof(float),
      cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(
      h_v2.data(),
      d_dstFloat_v2,
      nElts * sizeof(float),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(h_v1, h_v2);

  CUDACHECK(cudaFree(d_dstFloat_v1));
  CUDACHECK(cudaFree(d_dstFloat_v2));
}

// =============================================================================
// SmallSizes: Edge cases for tail logic paths.
// =============================================================================

TEST_F(SimpleCopySRV2Test, SmallSizes) {
  constexpr int64_t sizes[] = {0, 1, 2, 31, 32, 33, 127, 128, 129};

  for (int64_t nElts : sizes) {
    SCOPED_TRACE("nElts=" + std::to_string(nElts));
    if (nElts == 0) {
      v2_reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
          <<<kDefaultBlocks, kBlockSize>>>(
              d_dstBf16_A, d_srcFloat, 0, kSeed, kBaseOffset);
      CUDACHECK(cudaGetLastError());
      CUDACHECK(cudaDeviceSynchronize());
      continue;
    }

    std::vector<float> h_src(nElts);
    for (int64_t i = 0; i < nElts; i++) {
      h_src[i] = 1.0f + static_cast<float>(i) * 0.001f;
    }
    CUDACHECK(cudaMemcpy(
        d_srcFloat,
        h_src.data(),
        nElts * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_dstBf16_A, 0, nElts * sizeof(__nv_bfloat16)));

    v2_reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstBf16_A, d_srcFloat, nElts, kSeed, kBaseOffset);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstBf16_A,
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
// Unbiasedness: Average over many seeds ≈ original FP32 value.
// =============================================================================

TEST_F(SimpleCopySRV2Test, Unbiasedness) {
  constexpr int64_t nElts = 1024;
  constexpr int kNumSeeds = 200;

  std::vector<float> h_src(nElts);
  for (int64_t i = 0; i < nElts; i++) {
    h_src[i] = 1.0f + static_cast<float>(i) * 0.001f;
  }
  CUDACHECK(cudaMemcpy(
      d_srcFloat, h_src.data(), nElts * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<double> accum(nElts, 0.0);

  for (int s = 0; s < kNumSeeds; s++) {
    uint64_t seed = static_cast<uint64_t>(s * 12345 + 67);
    CUDACHECK(cudaMemset(d_dstBf16_A, 0, nElts * sizeof(__nv_bfloat16)));

    v2_reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
        <<<kDefaultBlocks, kBlockSize>>>(
            d_dstBf16_A, d_srcFloat, nElts, seed, 0);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_dst(nElts);
    CUDACHECK(cudaMemcpy(
        h_dst.data(),
        d_dstBf16_A,
        nElts * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToHost));

    for (int64_t i = 0; i < nElts; i++) {
      accum[i] += static_cast<double>(__bfloat162float(h_dst[i]));
    }
  }

  for (int64_t i = 0; i < nElts; i++) {
    double avg = accum[i] / kNumSeeds;
    double expected = static_cast<double>(h_src[i]);
    double tolerance = 0.005;
    EXPECT_NEAR(avg, expected, tolerance)
        << "Unbiasedness failure at index " << i;
  }
}
