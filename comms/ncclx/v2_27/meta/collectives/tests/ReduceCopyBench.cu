// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common_kernel.h" // @manual

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Benchmark Kernels
// =============================================================================

// Benchmark kernel: 1 source, 1 destination (simple copy with reduction fn)
template <int Unroll, typename RedFn, typename T>
__global__ void
benchReduceCopy_1Src1Dst(void* src0, void* dst0, int64_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  reduceCopy<
      Unroll,
      RedFn,
      T,
      /*MultimemSrcs=*/0,
      /*MinSrcs=*/1,
      /*MaxSrcs=*/1,
      /*MultimemDsts=*/0,
      /*MinDsts=*/1,
      /*MaxDsts=*/1,
      /*PreOpSrcs=*/0,
      int64_t>(
      thread,
      nThreads,
      (uint64_t)0, // redArg
      nullptr, // preOpArgs
      false, // postOp
      1, // nSrcs
      [=] __device__(int) { return src0; },
      1, // nDsts
      [=] __device__(int) { return dst0; },
      nElts);
}

// Benchmark kernel: 2 sources, 1 destination (sum reduction)
template <int Unroll, typename RedFn, typename T>
__global__ void
benchReduceCopy_2Src1Dst(void* src0, void* src1, void* dst0, int64_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  void* srcs[2] = {src0, src1};
  reduceCopy<
      Unroll,
      RedFn,
      T,
      /*MultimemSrcs=*/0,
      /*MinSrcs=*/2,
      /*MaxSrcs=*/2,
      /*MultimemDsts=*/0,
      /*MinDsts=*/1,
      /*MaxDsts=*/1,
      /*PreOpSrcs=*/0,
      int64_t>(
      thread,
      nThreads,
      (uint64_t)0, // redArg
      nullptr, // preOpArgs
      false, // postOp
      2, // nSrcs
      [=] __device__(int i) { return srcs[i]; },
      1, // nDsts
      [=] __device__(int) { return dst0; },
      nElts);
}

// Benchmark kernel: 1 source, 2 destinations (broadcast)
template <int Unroll, typename RedFn, typename T>
__global__ void
benchReduceCopy_1Src2Dst(void* src0, void* dst0, void* dst1, int64_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  void* dsts[2] = {dst0, dst1};
  reduceCopy<
      Unroll,
      RedFn,
      T,
      /*MultimemSrcs=*/0,
      /*MinSrcs=*/1,
      /*MaxSrcs=*/1,
      /*MultimemDsts=*/0,
      /*MinDsts=*/2,
      /*MaxDsts=*/2,
      /*PreOpSrcs=*/0,
      int64_t>(
      thread,
      nThreads,
      (uint64_t)0, // redArg
      nullptr, // preOpArgs
      false, // postOp
      1, // nSrcs
      [=] __device__(int) { return src0; },
      2, // nDsts
      [=] __device__(int i) { return dsts[i]; },
      nElts);
}

// =============================================================================
// Benchmark Fixture
// =============================================================================

class ReduceCopyBench : public ::testing::Test {
 protected:
  static constexpr int64_t kMaxElts = 16 * 1024 * 1024; // 16M elements
  static constexpr int kBlockSize = 256;
  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;

  // Buffers for float
  float* d_srcFloat0 = nullptr;
  float* d_srcFloat1 = nullptr;
  float* d_dstFloat0 = nullptr;
  float* d_dstFloat1 = nullptr;

  // Buffers for bf16
  __nv_bfloat16* d_srcBf16_0 = nullptr;
  __nv_bfloat16* d_srcBf16_1 = nullptr;
  __nv_bfloat16* d_dstBf16_0 = nullptr;
  __nv_bfloat16* d_dstBf16_1 = nullptr;

  // Buffers for half
  __half* d_srcHalf0 = nullptr;
  __half* d_dstHalf0 = nullptr;

  cudaEvent_t startEvent, stopEvent;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_srcFloat0, kMaxElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcFloat1, kMaxElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_dstFloat0, kMaxElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_dstFloat1, kMaxElts * sizeof(float)));

    CUDACHECK(cudaMalloc(&d_srcBf16_0, kMaxElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_srcBf16_1, kMaxElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_dstBf16_0, kMaxElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_dstBf16_1, kMaxElts * sizeof(__nv_bfloat16)));

    CUDACHECK(cudaMalloc(&d_srcHalf0, kMaxElts * sizeof(__half)));
    CUDACHECK(cudaMalloc(&d_dstHalf0, kMaxElts * sizeof(__half)));

    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&stopEvent));

    // Initialize float sources
    std::vector<float> h_init(kMaxElts);
    for (int64_t i = 0; i < kMaxElts; i++) {
      h_init[i] = static_cast<float>(i % 1000) * 0.001f;
    }
    CUDACHECK(cudaMemcpy(
        d_srcFloat0,
        h_init.data(),
        kMaxElts * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(
        d_srcFloat1,
        h_init.data(),
        kMaxElts * sizeof(float),
        cudaMemcpyHostToDevice));

    // Initialize bf16 sources
    std::vector<__nv_bfloat16> h_bf16(kMaxElts);
    for (int64_t i = 0; i < kMaxElts; i++) {
      h_bf16[i] = __float2bfloat16(h_init[i]);
    }
    CUDACHECK(cudaMemcpy(
        d_srcBf16_0,
        h_bf16.data(),
        kMaxElts * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(
        d_srcBf16_1,
        h_bf16.data(),
        kMaxElts * sizeof(__nv_bfloat16),
        cudaMemcpyHostToDevice));

    // Initialize half sources
    std::vector<__half> h_half(kMaxElts);
    for (int64_t i = 0; i < kMaxElts; i++) {
      h_half[i] = __float2half(h_init[i]);
    }
    CUDACHECK(cudaMemcpy(
        d_srcHalf0,
        h_half.data(),
        kMaxElts * sizeof(__half),
        cudaMemcpyHostToDevice));
  }

  void TearDown() override {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_srcFloat0);
    cudaFree(d_srcFloat1);
    cudaFree(d_dstFloat0);
    cudaFree(d_dstFloat1);
    cudaFree(d_srcBf16_0);
    cudaFree(d_srcBf16_1);
    cudaFree(d_dstBf16_0);
    cudaFree(d_dstBf16_1);
    cudaFree(d_srcHalf0);
    cudaFree(d_dstHalf0);
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

  // Generic benchmark runner for 1-src 1-dst kernels (max blocks)
  template <typename T, typename LaunchFn>
  void runBench1Src1Dst(
      int64_t nElts,
      size_t srcEltBytes,
      size_t dstEltBytes,
      LaunchFn launchFn,
      const char* label) {
    size_t totalBytes = nElts * (srcEltBytes + dstEltBytes);
    runBenchCore(nElts, maxBlocks(nElts), totalBytes, launchFn, label);
  }

  // Generic benchmark runner for 2-src 1-dst kernels (max blocks)
  template <typename T, typename LaunchFn>
  void runBench2Src1Dst(
      int64_t nElts,
      size_t srcEltBytes,
      size_t dstEltBytes,
      LaunchFn launchFn,
      const char* label) {
    size_t totalBytes = nElts * (2 * srcEltBytes + dstEltBytes);
    runBenchCore(nElts, maxBlocks(nElts), totalBytes, launchFn, label);
  }
};

// =============================================================================
// Benchmarks: float, 1 source -> 1 destination (FuncSum copy)
// =============================================================================

TEST_F(ReduceCopyBench, Float_1Src1Dst_Copy) {
  printf("\n--- reduceCopy: float, 1 src -> 1 dst (FuncSum copy) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    runBench1Src1Dst<float>(
        n,
        sizeof(float),
        sizeof(float),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_1Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(d_srcFloat0, d_dstFloat0, nElts);
        },
        "f32: 1src->1dst (FuncSum)");
  }
}

// =============================================================================
// Benchmarks: float, 2 sources -> 1 destination (FuncSum reduce)
// =============================================================================

TEST_F(ReduceCopyBench, Float_2Src1Dst_Sum) {
  printf("\n--- reduceCopy: float, 2 src -> 1 dst (FuncSum reduce) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    runBench2Src1Dst<float>(
        n,
        sizeof(float),
        sizeof(float),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_2Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0, d_srcFloat1, d_dstFloat0, nElts);
        },
        "f32: 2src->1dst (FuncSum)");
  }
}

// =============================================================================
// Benchmarks: float, 1 source -> 2 destinations (broadcast)
// =============================================================================

TEST_F(ReduceCopyBench, Float_1Src2Dst_Broadcast) {
  printf("\n--- reduceCopy: float, 1 src -> 2 dst (broadcast) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    runBench1Src1Dst<float>(
        n,
        sizeof(float),
        2 * sizeof(float),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_1Src2Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0, d_dstFloat0, d_dstFloat1, nElts);
        },
        "f32: 1src->2dst (broadcast)");
  }
}

// =============================================================================
// Benchmarks: __nv_bfloat16 type
// =============================================================================

TEST_F(ReduceCopyBench, Bf16_1Src1Dst_Copy) {
  printf("\n--- reduceCopy: bf16, 1 src -> 1 dst (FuncSum copy) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    runBench1Src1Dst<__nv_bfloat16>(
        n,
        sizeof(__nv_bfloat16),
        sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_1Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(d_srcBf16_0, d_dstBf16_0, nElts);
        },
        "bf16: 1src->1dst (FuncSum)");
  }
}

TEST_F(ReduceCopyBench, Bf16_2Src1Dst_Sum) {
  printf("\n--- reduceCopy: bf16, 2 src -> 1 dst (FuncSum reduce) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    runBench2Src1Dst<__nv_bfloat16>(
        n,
        sizeof(__nv_bfloat16),
        sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_2Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(
                  d_srcBf16_0, d_srcBf16_1, d_dstBf16_0, nElts);
        },
        "bf16: 2src->1dst (FuncSum)");
  }
}

// =============================================================================
// Benchmarks: half (fp16) type
// =============================================================================

TEST_F(ReduceCopyBench, Half_1Src1Dst_Copy) {
  printf("\n--- reduceCopy: half, 1 src -> 1 dst (FuncSum copy) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    runBench1Src1Dst<__half>(
        n,
        sizeof(__half),
        sizeof(__half),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_1Src1Dst<4, FuncSum<__half>, __half>
              <<<nBlocks, blockSize>>>(d_srcHalf0, d_dstHalf0, nElts);
        },
        "half: 1src->1dst (FuncSum)");
  }
}

// =============================================================================
// Benchmarks: Unroll factor comparison (float)
// =============================================================================

TEST_F(ReduceCopyBench, Float_UnrollComparison) {
  printf(
      "\n--- reduceCopy: float, unroll factor comparison (1M elements) ---\n");
  int64_t n = 1024 * 1024;

  auto runWithUnroll = [&](auto unrollTag, const char* label) {
    constexpr int U = decltype(unrollTag)::value;
    runBench1Src1Dst<float>(
        n,
        sizeof(float),
        sizeof(float),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_1Src1Dst<U, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(d_srcFloat0, d_dstFloat0, nElts);
        },
        label);
  };

  runWithUnroll(std::integral_constant<int, 1>{}, "f32: Unroll=1");
  runWithUnroll(std::integral_constant<int, 2>{}, "f32: Unroll=2");
  runWithUnroll(std::integral_constant<int, 4>{}, "f32: Unroll=4");
  runWithUnroll(std::integral_constant<int, 8>{}, "f32: Unroll=8");
}

// =============================================================================
// Benchmarks: Block count sweep (1 to max)
// =============================================================================

TEST_F(ReduceCopyBench, BlockSweep_Float_1Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopy block sweep: float 1src->1dst (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        N * 2 * sizeof(float),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_1Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(d_srcFloat0, d_dstFloat0, nElts);
        },
        "f32: 1src->1dst (FuncSum)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_Float_2Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopy block sweep: float 2src->1dst (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        N * 3 * sizeof(float),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_2Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0, d_srcFloat1, d_dstFloat0, nElts);
        },
        "f32: 2src->1dst (FuncSum)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_Float_1Src2Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopy block sweep: float 1src->2dst (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        N * 3 * sizeof(float),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_1Src2Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0, d_dstFloat0, d_dstFloat1, nElts);
        },
        "f32: 1src->2dst (broadcast)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_Bf16_1Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopy block sweep: bf16 1src->1dst (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        N * 2 * sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_1Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(d_srcBf16_0, d_dstBf16_0, nElts);
        },
        "bf16: 1src->1dst (FuncSum)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_Bf16_2Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopy block sweep: bf16 2src->1dst (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        N * 3 * sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_2Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(
                  d_srcBf16_0, d_srcBf16_1, d_dstBf16_0, nElts);
        },
        "bf16: 2src->1dst (FuncSum)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_Half_1Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopy block sweep: half 1src->1dst (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        N * 2 * sizeof(__half),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_1Src1Dst<4, FuncSum<__half>, __half>
              <<<nBlocks, blockSize>>>(d_srcHalf0, d_dstHalf0, nElts);
        },
        "half: 1src->1dst (FuncSum)");
  }
}

// =============================================================================
// Benchmarks: Misaligned Pointer Tests (float)
// =============================================================================

TEST_F(ReduceCopyBench, AlignedBaseline_Float_2Src1Dst) {
  printf("\n--- reduceCopy: float 2src->1dst, ALIGNED baseline ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    runBench2Src1Dst<float>(
        n,
        sizeof(float),
        sizeof(float),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_2Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0, d_srcFloat1, d_dstFloat0, nElts);
        },
        "f32: 2src->1dst (aligned baseline)");
  }
}

TEST_F(ReduceCopyBench, MisalignedSrc1_Float_2Src1Dst) {
  printf("\n--- reduceCopy: float 2src->1dst, src1 misaligned by 1 ---\n");
  constexpr int kSrc1Offset = 1;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kSrc1Offset;
    runBench2Src1Dst<float>(
        nElts,
        sizeof(float),
        sizeof(float),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0, d_srcFloat1 + kSrc1Offset, d_dstFloat0, ne);
        },
        "f32: 2src->1dst (src1+1)");
  }
}

TEST_F(ReduceCopyBench, MisalignedAll_Float_2Src1Dst) {
  printf(
      "\n--- reduceCopy: float 2src->1dst, all misaligned (src0+1, src1+2, dst+3) ---\n");
  constexpr int kSrc0Offset = 1;
  constexpr int kSrc1Offset = 2;
  constexpr int kDst0Offset = 3;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kDst0Offset;
    runBench2Src1Dst<float>(
        nElts,
        sizeof(float),
        sizeof(float),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0 + kSrc0Offset,
                  d_srcFloat1 + kSrc1Offset,
                  d_dstFloat0 + kDst0Offset,
                  ne);
        },
        "f32: 2src->1dst (all misaligned)");
  }
}

TEST_F(ReduceCopyBench, MisalignedDst0_Float_2Src1Dst) {
  printf("\n--- reduceCopy: float 2src->1dst, dst misaligned by 1 ---\n");
  constexpr int kDst0Offset = 1;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kDst0Offset;
    runBench2Src1Dst<float>(
        nElts,
        sizeof(float),
        sizeof(float),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0, d_srcFloat1, d_dstFloat0 + kDst0Offset, ne);
        },
        "f32: 2src->1dst (dst+1)");
  }
}

// =============================================================================
// Benchmarks: Misaligned Pointer Tests (bf16)
// =============================================================================

TEST_F(ReduceCopyBench, AlignedBaseline_Bf16_2Src1Dst) {
  printf("\n--- reduceCopy: bf16 2src->1dst, ALIGNED baseline ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    runBench2Src1Dst<__nv_bfloat16>(
        n,
        sizeof(__nv_bfloat16),
        sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_2Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(
                  d_srcBf16_0, d_srcBf16_1, d_dstBf16_0, nElts);
        },
        "bf16: 2src->1dst (aligned baseline)");
  }
}

TEST_F(ReduceCopyBench, MisalignedSrc1_Bf16_2Src1Dst) {
  printf("\n--- reduceCopy: bf16 2src->1dst, src1 misaligned by 1 ---\n");
  constexpr int kSrc1Offset = 1;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kSrc1Offset;
    runBench2Src1Dst<__nv_bfloat16>(
        nElts,
        sizeof(__nv_bfloat16),
        sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(
                  d_srcBf16_0, d_srcBf16_1 + kSrc1Offset, d_dstBf16_0, ne);
        },
        "bf16: 2src->1dst (src1+1)");
  }
}

TEST_F(ReduceCopyBench, MisalignedAll_Bf16_2Src1Dst) {
  printf(
      "\n--- reduceCopy: bf16 2src->1dst, all misaligned (src0+1, src1+2, dst+3) ---\n");
  constexpr int kSrc0Offset = 1;
  constexpr int kSrc1Offset = 2;
  constexpr int kDst0Offset = 3;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kDst0Offset;
    runBench2Src1Dst<__nv_bfloat16>(
        nElts,
        sizeof(__nv_bfloat16),
        sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(
                  d_srcBf16_0 + kSrc0Offset,
                  d_srcBf16_1 + kSrc1Offset,
                  d_dstBf16_0 + kDst0Offset,
                  ne);
        },
        "bf16: 2src->1dst (all misaligned)");
  }
}

TEST_F(ReduceCopyBench, MisalignedDst0_Bf16_2Src1Dst) {
  printf("\n--- reduceCopy: bf16 2src->1dst, dst misaligned by 1 ---\n");
  constexpr int kDst0Offset = 1;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kDst0Offset;
    runBench2Src1Dst<__nv_bfloat16>(
        nElts,
        sizeof(__nv_bfloat16),
        sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(
                  d_srcBf16_0, d_srcBf16_1, d_dstBf16_0 + kDst0Offset, ne);
        },
        "bf16: 2src->1dst (dst+1)");
  }
}

// =============================================================================
// Benchmarks: Misaligned Pointer Block Count Sweeps (float)
// =============================================================================

TEST_F(ReduceCopyBench, BlockSweep_AlignedBaseline_Float_2Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- block sweep: float 2src->1dst, ALIGNED baseline (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        N * 3 * sizeof(float),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_2Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0, d_srcFloat1, d_dstFloat0, nElts);
        },
        "f32: 2src->1dst (aligned baseline)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_MisalignedSrc1_Float_2Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kSrc1Offset = 1;
  int64_t nElts = N - kSrc1Offset;
  printf(
      "\n--- block sweep: float 2src->1dst, src1 misaligned by 1 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchCore(
        nElts,
        b,
        nElts * 3 * sizeof(float),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0, d_srcFloat1 + kSrc1Offset, d_dstFloat0, ne);
        },
        "f32: 2src->1dst (src1+1)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_MisalignedAll_Float_2Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kSrc0Offset = 1;
  constexpr int kSrc1Offset = 2;
  constexpr int kDst0Offset = 3;
  int64_t nElts = N - kDst0Offset;
  printf(
      "\n--- block sweep: float 2src->1dst, all misaligned (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchCore(
        nElts,
        b,
        nElts * 3 * sizeof(float),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0 + kSrc0Offset,
                  d_srcFloat1 + kSrc1Offset,
                  d_dstFloat0 + kDst0Offset,
                  ne);
        },
        "f32: 2src->1dst (all misaligned)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_MisalignedDst0_Float_2Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kDst0Offset = 1;
  int64_t nElts = N - kDst0Offset;
  printf(
      "\n--- block sweep: float 2src->1dst, dst misaligned by 1 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchCore(
        nElts,
        b,
        nElts * 3 * sizeof(float),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<float>, float>
              <<<nBlocks, blockSize>>>(
                  d_srcFloat0, d_srcFloat1, d_dstFloat0 + kDst0Offset, ne);
        },
        "f32: 2src->1dst (dst+1)");
  }
}

// =============================================================================
// Benchmarks: Misaligned Pointer Block Count Sweeps (bf16)
// =============================================================================

TEST_F(ReduceCopyBench, BlockSweep_AlignedBaseline_Bf16_2Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- block sweep: bf16 2src->1dst, ALIGNED baseline (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchCore(
        N,
        b,
        N * 3 * sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchReduceCopy_2Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(
                  d_srcBf16_0, d_srcBf16_1, d_dstBf16_0, nElts);
        },
        "bf16: 2src->1dst (aligned baseline)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_MisalignedSrc1_Bf16_2Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kSrc1Offset = 1;
  int64_t nElts = N - kSrc1Offset;
  printf(
      "\n--- block sweep: bf16 2src->1dst, src1 misaligned by 1 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchCore(
        nElts,
        b,
        nElts * 3 * sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(
                  d_srcBf16_0, d_srcBf16_1 + kSrc1Offset, d_dstBf16_0, ne);
        },
        "bf16: 2src->1dst (src1+1)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_MisalignedAll_Bf16_2Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kSrc0Offset = 1;
  constexpr int kSrc1Offset = 2;
  constexpr int kDst0Offset = 3;
  int64_t nElts = N - kDst0Offset;
  printf(
      "\n--- block sweep: bf16 2src->1dst, all misaligned (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchCore(
        nElts,
        b,
        nElts * 3 * sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(
                  d_srcBf16_0 + kSrc0Offset,
                  d_srcBf16_1 + kSrc1Offset,
                  d_dstBf16_0 + kDst0Offset,
                  ne);
        },
        "bf16: 2src->1dst (all misaligned)");
  }
}

TEST_F(ReduceCopyBench, BlockSweep_MisalignedDst0_Bf16_2Src1Dst) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kDst0Offset = 1;
  int64_t nElts = N - kDst0Offset;
  printf(
      "\n--- block sweep: bf16 2src->1dst, dst misaligned by 1 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchCore(
        nElts,
        b,
        nElts * 3 * sizeof(__nv_bfloat16),
        [&](int nBlocks, int blockSize, int64_t ne) {
          benchReduceCopy_2Src1Dst<4, FuncSum<__nv_bfloat16>, __nv_bfloat16>
              <<<nBlocks, blockSize>>>(
                  d_srcBf16_0, d_srcBf16_1, d_dstBf16_0 + kDst0Offset, ne);
        },
        "bf16: 2src->1dst (dst+1)");
  }
}
