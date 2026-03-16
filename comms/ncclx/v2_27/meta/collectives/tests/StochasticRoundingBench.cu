// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "device.h" // @manual needed for op128.h
#include "op128.h" // @manual

#include "meta/collectives/kernels/stochastic_cast.cuh" // @manual

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Benchmark Kernels
// =============================================================================

// Stochastic rounding benchmark: read FP32, write BF16 with stochastic rounding
template <int Pack>
__global__ void benchSRCast(
    float* src,
    __nv_bfloat16* dst,
    int64_t nElts,
    uint64_t seed,
    uint64_t offset) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  if constexpr (Pack == 1) {
    for (int64_t i = tid; i < nElts; i += nThreads) {
      BytePack<4> pack = toPack(src[i]);
      BytePack<2> result = Apply_StochasticCast<float, __nv_bfloat16, 1>::cast(
          pack, seed, offset + i);
      dst[i] = fromPack<__nv_bfloat16>(result);
    }
  } else if constexpr (Pack == 2) {
    for (int64_t i = tid * 2; i < nElts; i += nThreads * 2) {
      if (i + 1 < nElts) {
        float2 vals = make_float2(src[i], src[i + 1]);
        BytePack<8> pack = toPack(vals);
        BytePack<4> result =
            Apply_StochasticCast<float, __nv_bfloat16, 2>::cast(
                pack, seed, offset + i);
        __nv_bfloat162 bf16_pair = fromPack<__nv_bfloat162>(result);
        dst[i] = __low2bfloat16(bf16_pair);
        dst[i + 1] = __high2bfloat16(bf16_pair);
      }
    }
  } else if constexpr (Pack == 4) {
    for (int64_t i = tid * 4; i < nElts; i += nThreads * 4) {
      if (i + 3 < nElts) {
        float4 vals = make_float4(src[i], src[i + 1], src[i + 2], src[i + 3]);
        BytePack<16> pack = toPack(vals);
        BytePack<8> result =
            Apply_StochasticCast<float, __nv_bfloat16, 4>::cast(
                pack, seed, offset + i);
        __nv_bfloat162 lo = fromPack<__nv_bfloat162>(result.half[0]);
        __nv_bfloat162 hi = fromPack<__nv_bfloat162>(result.half[1]);
        dst[i] = __low2bfloat16(lo);
        dst[i + 1] = __high2bfloat16(lo);
        dst[i + 2] = __low2bfloat16(hi);
        dst[i + 3] = __high2bfloat16(hi);
      }
    }
  }
}

// Naive truncation baseline: read FP32, write BF16 with simple truncation
__global__ void benchNaiveCast(float* src, __nv_bfloat16* dst, int64_t nElts) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  for (int64_t i = tid; i < nElts; i += nThreads) {
    dst[i] = __float2bfloat16(src[i]);
  }
}

// FP32 identity copy (ceiling): Just copy FP32→FP32 to measure pure bandwidth
__global__ void benchIdentityCopy(float* src, float* dst, int64_t nElts) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  for (int64_t i = tid; i < nElts; i += nThreads) {
    dst[i] = src[i];
  }
}

// =============================================================================
// Benchmark Fixture
// =============================================================================

class StochasticRoundingBench : public ::testing::Test {
 protected:
  static constexpr int64_t kMaxElts = 16 * 1024 * 1024; // 16M elements
  static constexpr int kBlockSize = 256;
  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;

  float* d_srcFloat = nullptr;
  __nv_bfloat16* d_dstBf16 = nullptr;
  float* d_dstFloat = nullptr;
  cudaEvent_t startEvent, stopEvent;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_srcFloat, kMaxElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_dstBf16, kMaxElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_dstFloat, kMaxElts * sizeof(float)));
    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&stopEvent));

    // Initialize source data with simple pattern
    std::vector<float> h_init(kMaxElts);
    for (int64_t i = 0; i < kMaxElts; i++) {
      h_init[i] = 1.0f + static_cast<float>(i % 1000) * 1e-4f;
    }
    CUDACHECK(cudaMemcpy(
        d_srcFloat,
        h_init.data(),
        kMaxElts * sizeof(float),
        cudaMemcpyHostToDevice));
  }

  void TearDown() override {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_srcFloat);
    cudaFree(d_dstBf16);
    cudaFree(d_dstFloat);
  }

  // Compute the maximum number of blocks for a given element count
  int maxBlocks(int64_t nElts) {
    return std::min((int)((nElts + kBlockSize - 1) / kBlockSize), 1024);
  }

  // Generate a sequence of block counts: 1, 2, 4, 8, ..., up to maxBlk
  static std::vector<int> blockCountSweep(int maxBlk) {
    std::vector<int> counts;
    for (int b = 1; b <= maxBlk; b *= 2) {
      counts.push_back(b);
    }
    // Always include the true max if it wasn't a power of 2
    if (counts.empty() || counts.back() != maxBlk) {
      counts.push_back(maxBlk);
    }
    return counts;
  }

  // Core benchmark runner with explicit block count
  template <typename LaunchFn>
  void runBenchCore(
      int64_t nElts,
      int nBlocks,
      size_t totalBytesPerIter,
      LaunchFn launchFn,
      const char* label) {
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

    double gbPerSec = (double)totalBytesPerIter / (avgMs * 1e6);
    printf(
        "  %-45s  nBlocks=%4d  nElts=%10ld  avg=%.3f ms  BW=%.2f GB/s\n",
        label,
        nBlocks,
        (long)nElts,
        avgMs,
        gbPerSec);
  }

  // Generic benchmark runner (max blocks)
  template <typename LaunchFn>
  void runBench(
      int64_t nElts,
      size_t totalBytesPerIter,
      LaunchFn launchFn,
      const char* label) {
    runBenchCore(nElts, maxBlocks(nElts), totalBytesPerIter, launchFn, label);
  }
};

// =============================================================================
// Benchmarks: FP32→BF16 Stochastic Rounding vs Naive Truncation
// =============================================================================

TEST_F(StochasticRoundingBench, FloatToBf16_Pack1_SizesSweep) {
  printf(
      "\n--- Stochastic Rounding: FP32→BF16 (pack=1) vs Naive Truncation ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  uint64_t seed = 12345ULL;
  uint64_t offset = 0;

  for (int64_t n : sizes) {
    size_t totalBytes = n * (sizeof(float) + sizeof(__nv_bfloat16));

    // Stochastic rounding
    runBench(
        n,
        totalBytes,
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchSRCast<1><<<nBlocks, blockSize>>>(
              d_srcFloat, d_dstBf16, nElts, seed, offset);
        },
        "f32→bf16(SR,pack=1)");

    // Naive truncation
    runBench(
        n,
        totalBytes,
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchNaiveCast<<<nBlocks, blockSize>>>(d_srcFloat, d_dstBf16, nElts);
        },
        "f32→bf16(naive)");
  }
}

TEST_F(StochasticRoundingBench, FloatToBf16_Pack2_SizesSweep) {
  printf("\n--- Stochastic Rounding: FP32→BF16 (pack=2) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  uint64_t seed = 12345ULL;
  uint64_t offset = 0;

  for (int64_t n : sizes) {
    size_t totalBytes = n * (sizeof(float) + sizeof(__nv_bfloat16));
    runBench(
        n,
        totalBytes,
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchSRCast<2><<<nBlocks, blockSize>>>(
              d_srcFloat, d_dstBf16, nElts, seed, offset);
        },
        "f32→bf16(SR,pack=2)");
  }
}

TEST_F(StochasticRoundingBench, FloatToBf16_Pack4_SizesSweep) {
  printf("\n--- Stochastic Rounding: FP32→BF16 (pack=4) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  uint64_t seed = 12345ULL;
  uint64_t offset = 0;

  for (int64_t n : sizes) {
    size_t totalBytes = n * (sizeof(float) + sizeof(__nv_bfloat16));
    runBench(
        n,
        totalBytes,
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchSRCast<4><<<nBlocks, blockSize>>>(
              d_srcFloat, d_dstBf16, nElts, seed, offset);
        },
        "f32→bf16(SR,pack=4)");
  }
}

// =============================================================================
// Benchmarks: FP32→FP32 Identity Copy (Bandwidth Ceiling)
// =============================================================================

TEST_F(StochasticRoundingBench, IdentityCopy_FloatToFloat) {
  printf("\n--- FP32→FP32 Identity Copy (bandwidth ceiling) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};

  for (int64_t n : sizes) {
    size_t totalBytes = n * 2 * sizeof(float); // read + write
    runBench(
        n,
        totalBytes,
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchIdentityCopy<<<nBlocks, blockSize>>>(
              d_srcFloat, d_dstFloat, nElts);
        },
        "f32→f32(identity)");
  }
}

// =============================================================================
// Benchmarks: Block count sweep (1 to max)
// =============================================================================

TEST_F(StochasticRoundingBench, BlockSweep_StochasticRounding) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- Stochastic Rounding block sweep: FP32→BF16 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  uint64_t seed = 12345ULL;
  uint64_t offset = 0;
  size_t totalBytes = N * (sizeof(float) + sizeof(__nv_bfloat16));

  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        totalBytes,
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchSRCast<4><<<nBlocks, blockSize>>>(
              d_srcFloat, d_dstBf16, nElts, seed, offset);
        },
        "f32→bf16(SR,pack=4)");
  }
}

TEST_F(StochasticRoundingBench, BlockSweep_NaiveTruncation) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- Naive Truncation block sweep: FP32→BF16 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  size_t totalBytes = N * (sizeof(float) + sizeof(__nv_bfloat16));

  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        totalBytes,
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchNaiveCast<<<nBlocks, blockSize>>>(d_srcFloat, d_dstBf16, nElts);
        },
        "f32→bf16(naive)");
  }
}

TEST_F(StochasticRoundingBench, BlockSweep_IdentityCopy) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- Identity Copy block sweep: FP32→FP32 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  size_t totalBytes = N * 2 * sizeof(float);

  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        totalBytes,
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchIdentityCopy<<<nBlocks, blockSize>>>(
              d_srcFloat, d_dstFloat, nElts);
        },
        "f32→f32(identity)");
  }
}
