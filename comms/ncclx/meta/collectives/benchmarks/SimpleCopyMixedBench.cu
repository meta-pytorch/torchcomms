// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "meta/collectives/kernels/reduce_copy.cuh"

// =============================================================================
// Wrapper Kernels
// =============================================================================

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
// Benchmark Fixture
// =============================================================================

class SimpleCopyMixedBench : public ::testing::Test {
 protected:
  static constexpr int64_t kN = 4L * 1024L * 1024L; // 4M elements
  static constexpr int kBlockSize = 256;
  static constexpr int kDefaultBlocks = 32;
  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;

  float* d_srcFloat0 = nullptr;
  float* d_srcFloat1 = nullptr;
  float* d_dstFloat = nullptr;
  __nv_bfloat16* d_srcBf16_0 = nullptr;
  __nv_bfloat16* d_srcBf16_1 = nullptr;
  __nv_bfloat16* d_dstBf16 = nullptr;

  cudaEvent_t startEvent, stopEvent;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_srcFloat0, kN * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcFloat1, kN * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_dstFloat, kN * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcBf16_0, kN * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_srcBf16_1, kN * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_dstBf16, kN * sizeof(__nv_bfloat16)));

    // Initialize with simple pattern
    std::vector<float> h_init(kN);
    for (int64_t i = 0; i < kN; i++) {
      h_init[i] = 1.0f + static_cast<float>(i % 1000) * 1e-4f;
    }
    CUDACHECK(cudaMemcpy(
        d_srcFloat0,
        h_init.data(),
        kN * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(
        d_srcFloat1,
        h_init.data(),
        kN * sizeof(float),
        cudaMemcpyHostToDevice));

    std::vector<__nv_bfloat16> h_bf16(kN);
    for (int64_t i = 0; i < kN; i++) {
      h_bf16[i] = __float2bfloat16(h_init[i]);
    }
    CUDACHECK(cudaMemcpy(
        d_srcBf16_0,
        h_bf16.data(),
        kN * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(
        d_srcBf16_1,
        h_bf16.data(),
        kN * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));

    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&stopEvent));
  }

  void TearDown() override {
    CUDACHECK(cudaEventDestroy(startEvent));
    CUDACHECK(cudaEventDestroy(stopEvent));
    CUDACHECK(cudaFree(d_srcFloat0));
    CUDACHECK(cudaFree(d_srcFloat1));
    CUDACHECK(cudaFree(d_dstFloat));
    CUDACHECK(cudaFree(d_srcBf16_0));
    CUDACHECK(cudaFree(d_srcBf16_1));
    CUDACHECK(cudaFree(d_dstBf16));
  }

  template <typename LaunchFn>
  void runBenchCore(
      int64_t nElts,
      int nBlocks,
      LaunchFn launchFn,
      const char* label,
      size_t totalBytes) {
    // Warmup
    for (int i = 0; i < kWarmupIters; i++) {
      launchFn(nBlocks, kBlockSize, nElts);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Timed iterations
    CUDACHECK(cudaEventRecord(startEvent));
    for (int i = 0; i < kBenchIters; i++) {
      launchFn(nBlocks, kBlockSize, nElts);
    }
    CUDACHECK(cudaEventRecord(stopEvent));
    CUDACHECK(cudaDeviceSynchronize());

    float elapsedMs = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    float avgMs = elapsedMs / kBenchIters;

    double gbPerSec = (double)totalBytes / (avgMs * 1e6);
    printf(
        "  %-55s  nBlocks=%4d  nElts=%10ld  avg=%.3f ms  BW=%.2f GB/s\n",
        label,
        nBlocks,
        (long)nElts,
        avgMs,
        gbPerSec);
  }
};

// =============================================================================
// 1-src benchmarks (32 blocks, 4M elements)
// =============================================================================

// FP32 -> FP32 (1 src): 1 read + 1 write = 2 * 4M * 4 bytes
TEST_F(SimpleCopyMixedBench, OneSrc_FloatToFloat) {
  printf(
      "\n--- reduceCopyMixed 1-src: FP32 -> FP32 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = kN * sizeof(float) + kN * sizeof(float);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_1src_kernel<4, float, float, float>
            <<<nBlk, blockSize>>>(d_dstFloat, d_srcFloat0, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "1-src f32 -> f32",
      totalBytes);
}

// BF16 -> FP32 (1 src): 1 read of BF16 + 1 write of FP32
TEST_F(SimpleCopyMixedBench, OneSrc_Bf16ToFloat) {
  printf(
      "\n--- reduceCopyMixed 1-src: BF16 -> FP32 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = kN * sizeof(__nv_bfloat16) + kN * sizeof(float);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_1src_kernel<4, float, float, __nv_bfloat16>
            <<<nBlk, blockSize>>>(d_dstFloat, d_srcBf16_0, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "1-src bf16 -> f32",
      totalBytes);
}

// FP32 -> BF16 (1 src): 1 read of FP32 + 1 write of BF16
TEST_F(SimpleCopyMixedBench, OneSrc_FloatToBf16) {
  printf(
      "\n--- reduceCopyMixed 1-src: FP32 -> BF16 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = kN * sizeof(float) + kN * sizeof(__nv_bfloat16);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_1src_kernel<4, float, __nv_bfloat16, float>
            <<<nBlk, blockSize>>>(d_dstBf16, d_srcFloat0, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "1-src f32 -> bf16",
      totalBytes);
}

// BF16 -> BF16 (1 src): 1 read of BF16 + 1 write of BF16
TEST_F(SimpleCopyMixedBench, OneSrc_Bf16ToBf16) {
  printf(
      "\n--- reduceCopyMixed 1-src: BF16 -> BF16 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = kN * sizeof(__nv_bfloat16) + kN * sizeof(__nv_bfloat16);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_1src_kernel<4, float, __nv_bfloat16, __nv_bfloat16>
            <<<nBlk, blockSize>>>(d_dstBf16, d_srcBf16_0, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "1-src bf16 -> bf16",
      totalBytes);
}

// =============================================================================
// 2-src benchmarks (32 blocks, 4M elements)
// =============================================================================

// FP32 + FP32 -> FP32: 2 reads of FP32 + 1 write of FP32
TEST_F(SimpleCopyMixedBench, TwoSrc_FloatFloat_FloatDst) {
  printf(
      "\n--- reduceCopyMixed 2-src: FP32+FP32 -> FP32 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = 2 * kN * sizeof(float) + kN * sizeof(float);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_2src_kernel<4, float, float, float, float>
            <<<nBlk, blockSize>>>(
                d_dstFloat, d_srcFloat0, d_srcFloat1, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "2-src f32+f32 -> f32",
      totalBytes);
}

// FP32 + FP32 -> BF16: 2 reads of FP32 + 1 write of BF16
TEST_F(SimpleCopyMixedBench, TwoSrc_FloatFloat_Bf16Dst) {
  printf(
      "\n--- reduceCopyMixed 2-src: FP32+FP32 -> BF16 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = 2 * kN * sizeof(float) + kN * sizeof(__nv_bfloat16);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_2src_kernel<4, float, __nv_bfloat16, float, float>
            <<<nBlk, blockSize>>>(
                d_dstBf16, d_srcFloat0, d_srcFloat1, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "2-src f32+f32 -> bf16",
      totalBytes);
}

// BF16 + BF16 -> FP32: 2 reads of BF16 + 1 write of FP32
TEST_F(SimpleCopyMixedBench, TwoSrc_Bf16Bf16_FloatDst) {
  printf(
      "\n--- reduceCopyMixed 2-src: BF16+BF16 -> FP32 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = 2 * kN * sizeof(__nv_bfloat16) + kN * sizeof(float);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_2src_kernel<
            4,
            float,
            float,
            __nv_bfloat16,
            __nv_bfloat16><<<nBlk, blockSize>>>(
            d_dstFloat, d_srcBf16_0, d_srcBf16_1, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "2-src bf16+bf16 -> f32",
      totalBytes);
}

// BF16 + BF16 -> BF16: 2 reads of BF16 + 1 write of BF16
TEST_F(SimpleCopyMixedBench, TwoSrc_Bf16Bf16_Bf16Dst) {
  printf(
      "\n--- reduceCopyMixed 2-src: BF16+BF16 -> BF16 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = 3 * kN * sizeof(__nv_bfloat16);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_2src_kernel<
            4,
            float,
            __nv_bfloat16,
            __nv_bfloat16,
            __nv_bfloat16><<<nBlk, blockSize>>>(
            d_dstBf16, d_srcBf16_0, d_srcBf16_1, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "2-src bf16+bf16 -> bf16",
      totalBytes);
}

// BF16 + FP32 -> FP32: 1 read of BF16 + 1 read of FP32 + 1 write of FP32
TEST_F(SimpleCopyMixedBench, TwoSrc_Bf16Float_FloatDst) {
  printf(
      "\n--- reduceCopyMixed 2-src: BF16+FP32 -> FP32 (32 blocks, 4M elts) ---\n");
  size_t totalBytes =
      kN * sizeof(__nv_bfloat16) + kN * sizeof(float) + kN * sizeof(float);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_2src_kernel<4, float, float, __nv_bfloat16, float>
            <<<nBlk, blockSize>>>(
                d_dstFloat, d_srcBf16_0, d_srcFloat1, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "2-src bf16+f32 -> f32",
      totalBytes);
}

// BF16 + FP32 -> BF16: 1 read of BF16 + 1 read of FP32 + 1 write of BF16
TEST_F(SimpleCopyMixedBench, TwoSrc_Bf16Float_Bf16Dst) {
  printf(
      "\n--- reduceCopyMixed 2-src: BF16+FP32 -> BF16 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = kN * sizeof(__nv_bfloat16) + kN * sizeof(float) +
      kN * sizeof(__nv_bfloat16);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_2src_kernel<
            4,
            float,
            __nv_bfloat16,
            __nv_bfloat16,
            float><<<nBlk, blockSize>>>(
            d_dstBf16, d_srcBf16_0, d_srcFloat1, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "2-src bf16+f32 -> bf16",
      totalBytes);
}

// FP32 + BF16 -> FP32: 1 read of FP32 + 1 read of BF16 + 1 write of FP32
TEST_F(SimpleCopyMixedBench, TwoSrc_FloatBf16_FloatDst) {
  printf(
      "\n--- reduceCopyMixed 2-src: FP32+BF16 -> FP32 (32 blocks, 4M elts) ---\n");
  size_t totalBytes =
      kN * sizeof(float) + kN * sizeof(__nv_bfloat16) + kN * sizeof(float);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_2src_kernel<4, float, float, float, __nv_bfloat16>
            <<<nBlk, blockSize>>>(
                d_dstFloat, d_srcFloat0, d_srcBf16_1, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "2-src f32+bf16 -> f32",
      totalBytes);
}

// FP32 + BF16 -> BF16: 1 read of FP32 + 1 read of BF16 + 1 write of BF16
TEST_F(SimpleCopyMixedBench, TwoSrc_FloatBf16_Bf16Dst) {
  printf(
      "\n--- reduceCopyMixed 2-src: FP32+BF16 -> BF16 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = kN * sizeof(float) + kN * sizeof(__nv_bfloat16) +
      kN * sizeof(__nv_bfloat16);
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_2src_kernel<
            4,
            float,
            __nv_bfloat16,
            float,
            __nv_bfloat16><<<nBlk, blockSize>>>(
            d_dstBf16, d_srcFloat0, d_srcBf16_1, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "2-src f32+bf16 -> bf16",
      totalBytes);
}
