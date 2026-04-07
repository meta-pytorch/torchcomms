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
    CUDACHECK(cudaEventDestroy(startEvent));
    CUDACHECK(cudaEventDestroy(stopEvent));
    CUDACHECK(cudaFree(d_srcFloat0));
    CUDACHECK(cudaFree(d_srcFloat1));
    CUDACHECK(cudaFree(d_dstFloat0));
    CUDACHECK(cudaFree(d_dstFloat1));
    CUDACHECK(cudaFree(d_srcBf16_0));
    CUDACHECK(cudaFree(d_srcBf16_1));
    CUDACHECK(cudaFree(d_dstBf16_0));
    CUDACHECK(cudaFree(d_dstBf16_1));
    CUDACHECK(cudaFree(d_srcHalf0));
    CUDACHECK(cudaFree(d_dstHalf0));
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
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
          CUDACHECK(cudaGetLastError());
        },
        "bf16: 2src->1dst (dst+1)");
  }
}

// clang-format off
// We want to keep the format for the result below

/*
--- reduceCopy: float, 1 src -> 1 dst (FuncSum copy) ---
  f32: 1src->1dst (FuncSum)                      nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.99 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=15.92 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=127.79 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=1020.89 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=5453.95 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=1024  nElts=  16777216  avg=0.021 ms  BW=6426.97 GB/s
[       OK ] ReduceCopyBench.Float_1Src1Dst_Copy (255 ms)
[ RUN      ] ReduceCopyBench.Float_2Src1Dst_Sum

--- reduceCopy: float, 2 src -> 1 dst (FuncSum reduce) ---
  f32: 2src->1dst (FuncSum)                      nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=2.99 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=24.06 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=190.90 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=1530.98 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=1024  nElts=   4194304  avg=0.007 ms  BW=7007.32 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=1024  nElts=  16777216  avg=0.031 ms  BW=6527.63 GB/s
[       OK ] ReduceCopyBench.Float_2Src1Dst_Sum (67 ms)
[ RUN      ] ReduceCopyBench.Float_1Src2Dst_Broadcast

--- reduceCopy: float, 1 src -> 2 dst (broadcast) ---
  f32: 1src->2dst (broadcast)                    nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=2.99 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=23.95 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=191.33 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=1523.86 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=6134.41 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=1024  nElts=  16777216  avg=0.035 ms  BW=5752.71 GB/s
[       OK ] ReduceCopyBench.Float_1Src2Dst_Broadcast (67 ms)
[ RUN      ] ReduceCopyBench.Bf16_1Src1Dst_Copy

--- reduceCopy: bf16, 1 src -> 1 dst (FuncSum copy) ---
  bf16: 1src->1dst (FuncSum)                     nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.00 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=7.98 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=63.80 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=509.69 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=1024  nElts=   4194304  avg=0.005 ms  BW=3285.42 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=1024  nElts=  16777216  avg=0.008 ms  BW=8165.84 GB/s
[       OK ] ReduceCopyBench.Bf16_1Src1Dst_Copy (64 ms)
[ RUN      ] ReduceCopyBench.Bf16_2Src1Dst_Sum

--- reduceCopy: bf16, 2 src -> 1 dst (FuncSum reduce) ---
  bf16: 2src->1dst (FuncSum)                     nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.49 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=11.97 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=95.86 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=764.42 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=4083.45 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=1024  nElts=  16777216  avg=0.012 ms  BW=8181.56 GB/s
[       OK ] ReduceCopyBench.Bf16_2Src1Dst_Sum (65 ms)
[ RUN      ] ReduceCopyBench.Half_1Src1Dst_Copy

--- reduceCopy: half, 1 src -> 1 dst (FuncSum copy) ---
  half: 1src->1dst (FuncSum)                     nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.00 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=7.97 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=63.89 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=510.80 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=1024  nElts=   4194304  avg=0.005 ms  BW=3281.31 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=1024  nElts=  16777216  avg=0.008 ms  BW=8164.57 GB/s
[       OK ] ReduceCopyBench.Half_1Src1Dst_Copy (63 ms)
[ RUN      ] ReduceCopyBench.Float_UnrollComparison

--- reduceCopy: float, unroll factor comparison (1M elements) ---
  f32: Unroll=1                                  nBlocks=1024  nElts=   1048576  avg=0.004 ms  BW=2036.39 GB/s
  f32: Unroll=2                                  nBlocks=1024  nElts=   1048576  avg=0.004 ms  BW=2040.83 GB/s
  f32: Unroll=4                                  nBlocks=1024  nElts=   1048576  avg=0.004 ms  BW=2034.49 GB/s
  f32: Unroll=8                                  nBlocks=1024  nElts=   1048576  avg=0.004 ms  BW=2034.81 GB/s
[       OK ] ReduceCopyBench.Float_UnrollComparison (62 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_Float_1Src1Dst

--- reduceCopy block sweep: float 1src->1dst (4M elts) ---
  f32: 1src->1dst (FuncSum)                      nBlocks=   1  nElts=   4194304  avg=0.332 ms  BW=101.03 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=   2  nElts=   4194304  avg=0.168 ms  BW=199.62 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=   4  nElts=   4194304  avg=0.088 ms  BW=383.04 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=   8  nElts=   4194304  avg=0.045 ms  BW=744.88 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=  16  nElts=   4194304  avg=0.025 ms  BW=1364.27 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=  32  nElts=   4194304  avg=0.014 ms  BW=2339.11 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=  64  nElts=   4194304  avg=0.008 ms  BW=4049.03 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks= 128  nElts=   4194304  avg=0.006 ms  BW=5432.19 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks= 256  nElts=   4194304  avg=0.006 ms  BW=5448.85 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks= 512  nElts=   4194304  avg=0.006 ms  BW=5441.49 GB/s
  f32: 1src->1dst (FuncSum)                      nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=5449.98 GB/s
[       OK ] ReduceCopyBench.BlockSweep_Float_1Src1Dst (138 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_Float_2Src1Dst

--- reduceCopy block sweep: float 2src->1dst (4M elts) ---
  f32: 2src->1dst (FuncSum)                      nBlocks=   1  nElts=   4194304  avg=0.410 ms  BW=122.84 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=   2  nElts=   4194304  avg=0.208 ms  BW=242.22 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=   4  nElts=   4194304  avg=0.106 ms  BW=472.62 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=   8  nElts=   4194304  avg=0.053 ms  BW=942.50 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=  16  nElts=   4194304  avg=0.029 ms  BW=1752.92 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=  32  nElts=   4194304  avg=0.016 ms  BW=3067.39 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=  64  nElts=   4194304  avg=0.010 ms  BW=4901.11 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks= 128  nElts=   4194304  avg=0.007 ms  BW=6818.38 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=6579.09 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=6682.23 GB/s
  f32: 2src->1dst (FuncSum)                      nBlocks=1024  nElts=   4194304  avg=0.007 ms  BW=6928.61 GB/s
[       OK ] ReduceCopyBench.BlockSweep_Float_2Src1Dst (156 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_Float_1Src2Dst

--- reduceCopy block sweep: float 1src->2dst (4M elts) ---
  f32: 1src->2dst (broadcast)                    nBlocks=   1  nElts=   4194304  avg=0.589 ms  BW=85.50 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=   2  nElts=   4194304  avg=0.297 ms  BW=169.57 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=   4  nElts=   4194304  avg=0.152 ms  BW=330.99 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=   8  nElts=   4194304  avg=0.078 ms  BW=647.09 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=  16  nElts=   4194304  avg=0.041 ms  BW=1228.55 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=  32  nElts=   4194304  avg=0.023 ms  BW=2234.15 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=  64  nElts=   4194304  avg=0.013 ms  BW=3738.77 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=6119.62 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=6126.77 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=6124.38 GB/s
  f32: 1src->2dst (broadcast)                    nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=6126.29 GB/s
[       OK ] ReduceCopyBench.BlockSweep_Float_1Src2Dst (196 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_Bf16_1Src1Dst

--- reduceCopy block sweep: bf16 1src->1dst (4M elts) ---
  bf16: 1src->1dst (FuncSum)                     nBlocks=   1  nElts=   4194304  avg=0.168 ms  BW=99.91 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=   2  nElts=   4194304  avg=0.086 ms  BW=195.07 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=   4  nElts=   4194304  avg=0.045 ms  BW=372.26 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=   8  nElts=   4194304  avg=0.025 ms  BW=682.69 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=  16  nElts=   4194304  avg=0.014 ms  BW=1169.16 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=  32  nElts=   4194304  avg=0.008 ms  BW=2040.98 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=  64  nElts=   4194304  avg=0.006 ms  BW=2724.71 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks= 128  nElts=   4194304  avg=0.005 ms  BW=3691.91 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks= 256  nElts=   4194304  avg=0.004 ms  BW=4069.29 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks= 512  nElts=   4194304  avg=0.004 ms  BW=4067.72 GB/s
  bf16: 1src->1dst (FuncSum)                     nBlocks=1024  nElts=   4194304  avg=0.005 ms  BW=3269.44 GB/s
[       OK ] ReduceCopyBench.BlockSweep_Bf16_1Src1Dst (103 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_Bf16_2Src1Dst

--- reduceCopy block sweep: bf16 2src->1dst (4M elts) ---
  bf16: 2src->1dst (FuncSum)                     nBlocks=   1  nElts=   4194304  avg=0.204 ms  BW=123.29 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=   2  nElts=   4194304  avg=0.104 ms  BW=241.02 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=   4  nElts=   4194304  avg=0.053 ms  BW=472.39 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=   8  nElts=   4194304  avg=0.029 ms  BW=876.45 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=  16  nElts=   4194304  avg=0.016 ms  BW=1533.81 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=  32  nElts=   4194304  avg=0.010 ms  BW=2456.14 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=  64  nElts=   4194304  avg=0.007 ms  BW=3801.39 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks= 128  nElts=   4194304  avg=0.006 ms  BW=4089.61 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks= 256  nElts=   4194304  avg=0.006 ms  BW=4086.85 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks= 512  nElts=   4194304  avg=0.006 ms  BW=4091.10 GB/s
  bf16: 2src->1dst (FuncSum)                     nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=4084.72 GB/s
[       OK ] ReduceCopyBench.BlockSweep_Bf16_2Src1Dst (112 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_Half_1Src1Dst

--- reduceCopy block sweep: half 1src->1dst (4M elts) ---
  half: 1src->1dst (FuncSum)                     nBlocks=   1  nElts=   4194304  avg=0.168 ms  BW=99.90 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=   2  nElts=   4194304  avg=0.086 ms  BW=194.94 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=   4  nElts=   4194304  avg=0.045 ms  BW=371.77 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=   8  nElts=   4194304  avg=0.025 ms  BW=682.53 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=  16  nElts=   4194304  avg=0.014 ms  BW=1169.61 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=  32  nElts=   4194304  avg=0.008 ms  BW=2040.03 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=  64  nElts=   4194304  avg=0.006 ms  BW=2724.56 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks= 128  nElts=   4194304  avg=0.005 ms  BW=3618.52 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks= 256  nElts=   4194304  avg=0.004 ms  BW=4068.66 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks= 512  nElts=   4194304  avg=0.004 ms  BW=4069.61 GB/s
  half: 1src->1dst (FuncSum)                     nBlocks=1024  nElts=   4194304  avg=0.005 ms  BW=3264.35 GB/s
[       OK ] ReduceCopyBench.BlockSweep_Half_1Src1Dst (104 ms)
[ RUN      ] ReduceCopyBench.AlignedBaseline_Float_2Src1Dst

--- reduceCopy: float 2src->1dst, ALIGNED baseline ---
  f32: 2src->1dst (aligned baseline)             nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=2.99 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=23.86 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=191.25 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=1523.38 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=1024  nElts=   4194304  avg=0.007 ms  BW=7010.76 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=1024  nElts=  16777216  avg=0.031 ms  BW=6518.90 GB/s
[       OK ] ReduceCopyBench.AlignedBaseline_Float_2Src1Dst (69 ms)
[ RUN      ] ReduceCopyBench.MisalignedSrc1_Float_2Src1Dst

--- reduceCopy: float 2src->1dst, src1 misaligned by 1 ---
  f32: 2src->1dst (src1+1)                       nBlocks=   4  nElts=      1023  avg=0.004 ms  BW=2.99 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=  32  nElts=      8191  avg=0.004 ms  BW=23.95 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks= 256  nElts=     65535  avg=0.004 ms  BW=191.31 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=1024  nElts=    524287  avg=0.004 ms  BW=1524.80 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=1024  nElts=   4194303  avg=0.008 ms  BW=6124.38 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=1024  nElts=  16777215  avg=0.034 ms  BW=5907.53 GB/s
[       OK ] ReduceCopyBench.MisalignedSrc1_Float_2Src1Dst (68 ms)
[ RUN      ] ReduceCopyBench.MisalignedAll_Float_2Src1Dst

--- reduceCopy: float 2src->1dst, all misaligned (src0+1, src1+2, dst+3) ---
  f32: 2src->1dst (all misaligned)               nBlocks=   4  nElts=      1021  avg=0.004 ms  BW=2.97 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=  32  nElts=      8189  avg=0.004 ms  BW=23.90 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks= 256  nElts=     65533  avg=0.004 ms  BW=191.66 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=1024  nElts=    524285  avg=0.004 ms  BW=1524.32 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=1024  nElts=   4194301  avg=0.008 ms  BW=6123.66 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=1024  nElts=  16777213  avg=0.035 ms  BW=5782.48 GB/s
[       OK ] ReduceCopyBench.MisalignedAll_Float_2Src1Dst (68 ms)
[ RUN      ] ReduceCopyBench.MisalignedDst0_Float_2Src1Dst

--- reduceCopy: float 2src->1dst, dst misaligned by 1 ---
  f32: 2src->1dst (dst+1)                        nBlocks=   4  nElts=      1023  avg=0.004 ms  BW=2.98 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=  32  nElts=      8191  avg=0.004 ms  BW=23.94 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks= 256  nElts=     65535  avg=0.004 ms  BW=188.35 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=1024  nElts=    524287  avg=0.004 ms  BW=1525.27 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=1024  nElts=   4194303  avg=0.008 ms  BW=6137.29 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=1024  nElts=  16777215  avg=0.034 ms  BW=5878.60 GB/s
[       OK ] ReduceCopyBench.MisalignedDst0_Float_2Src1Dst (67 ms)
[ RUN      ] ReduceCopyBench.AlignedBaseline_Bf16_2Src1Dst

--- reduceCopy: bf16 2src->1dst, ALIGNED baseline ---
  bf16: 2src->1dst (aligned baseline)            nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.50 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=11.92 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=95.76 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=762.99 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=4089.82 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=1024  nElts=  16777216  avg=0.012 ms  BW=8184.97 GB/s
[       OK ] ReduceCopyBench.AlignedBaseline_Bf16_2Src1Dst (66 ms)
[ RUN      ] ReduceCopyBench.MisalignedSrc1_Bf16_2Src1Dst

--- reduceCopy: bf16 2src->1dst, src1 misaligned by 1 ---
  bf16: 2src->1dst (src1+1)                      nBlocks=   4  nElts=      1023  avg=0.004 ms  BW=1.49 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=  32  nElts=      8191  avg=0.004 ms  BW=11.97 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks= 256  nElts=     65535  avg=0.004 ms  BW=95.40 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=1024  nElts=    524287  avg=0.004 ms  BW=762.58 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=1024  nElts=   4194303  avg=0.006 ms  BW=4075.41 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=1024  nElts=  16777215  avg=0.020 ms  BW=4912.97 GB/s
[       OK ] ReduceCopyBench.MisalignedSrc1_Bf16_2Src1Dst (66 ms)
[ RUN      ] ReduceCopyBench.MisalignedAll_Bf16_2Src1Dst

--- reduceCopy: bf16 2src->1dst, all misaligned (src0+1, src1+2, dst+3) ---
  bf16: 2src->1dst (all misaligned)              nBlocks=   4  nElts=      1021  avg=0.004 ms  BW=1.48 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=  32  nElts=      8189  avg=0.004 ms  BW=11.96 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks= 256  nElts=     65533  avg=0.004 ms  BW=95.73 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=1024  nElts=    524285  avg=0.004 ms  BW=762.16 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=1024  nElts=   4194301  avg=0.008 ms  BW=3068.16 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=1024  nElts=  16777213  avg=0.021 ms  BW=4909.37 GB/s
[       OK ] ReduceCopyBench.MisalignedAll_Bf16_2Src1Dst (66 ms)
[ RUN      ] ReduceCopyBench.MisalignedDst0_Bf16_2Src1Dst

--- reduceCopy: bf16 2src->1dst, dst misaligned by 1 ---
  bf16: 2src->1dst (dst+1)                       nBlocks=   4  nElts=      1023  avg=0.004 ms  BW=1.49 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=  32  nElts=      8191  avg=0.004 ms  BW=11.98 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks= 256  nElts=     65535  avg=0.004 ms  BW=95.73 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=1024  nElts=    524287  avg=0.004 ms  BW=762.99 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=1024  nElts=   4194303  avg=0.007 ms  BW=3464.61 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=1024  nElts=  16777215  avg=0.019 ms  BW=5439.52 GB/s
[       OK ] ReduceCopyBench.MisalignedDst0_Bf16_2Src1Dst (66 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_AlignedBaseline_Float_2Src1Dst

--- block sweep: float 2src->1dst, ALIGNED baseline (4M elts) ---
  f32: 2src->1dst (aligned baseline)             nBlocks=   1  nElts=   4194304  avg=0.410 ms  BW=122.81 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=   2  nElts=   4194304  avg=0.208 ms  BW=241.50 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=   4  nElts=   4194304  avg=0.106 ms  BW=472.71 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=   8  nElts=   4194304  avg=0.054 ms  BW=940.30 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=  16  nElts=   4194304  avg=0.029 ms  BW=1754.20 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=  32  nElts=   4194304  avg=0.016 ms  BW=3066.85 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=  64  nElts=   4194304  avg=0.010 ms  BW=4902.48 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks= 128  nElts=   4194304  avg=0.007 ms  BW=6757.16 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=6526.14 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=6596.48 GB/s
  f32: 2src->1dst (aligned baseline)             nBlocks=1024  nElts=   4194304  avg=0.007 ms  BW=6959.88 GB/s
[       OK ] ReduceCopyBench.BlockSweep_AlignedBaseline_Float_2Src1Dst (156 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_MisalignedSrc1_Float_2Src1Dst

--- block sweep: float 2src->1dst, src1 misaligned by 1 (4M elts) ---
  f32: 2src->1dst (src1+1)                       nBlocks=   1  nElts=   4194303  avg=0.529 ms  BW=95.08 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=   2  nElts=   4194303  avg=0.273 ms  BW=184.62 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=   4  nElts=   4194303  avg=0.145 ms  BW=347.19 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=   8  nElts=   4194303  avg=0.072 ms  BW=700.61 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=  16  nElts=   4194303  avg=0.038 ms  BW=1320.39 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=  32  nElts=   4194303  avg=0.021 ms  BW=2428.80 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=  64  nElts=   4194303  avg=0.013 ms  BW=3783.83 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks= 128  nElts=   4194303  avg=0.008 ms  BW=5943.18 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks= 256  nElts=   4194303  avg=0.008 ms  BW=6135.13 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks= 512  nElts=   4194303  avg=0.008 ms  BW=6129.15 GB/s
  f32: 2src->1dst (src1+1)                       nBlocks=1024  nElts=   4194303  avg=0.008 ms  BW=6130.83 GB/s
[       OK ] ReduceCopyBench.BlockSweep_MisalignedSrc1_Float_2Src1Dst (186 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_MisalignedAll_Float_2Src1Dst

--- block sweep: float 2src->1dst, all misaligned (4M elts) ---
  f32: 2src->1dst (all misaligned)               nBlocks=   1  nElts=   4194301  avg=0.569 ms  BW=88.47 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=   2  nElts=   4194301  avg=0.290 ms  BW=173.77 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=   4  nElts=   4194301  avg=0.156 ms  BW=323.15 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=   8  nElts=   4194301  avg=0.078 ms  BW=646.91 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=  16  nElts=   4194301  avg=0.041 ms  BW=1228.84 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=  32  nElts=   4194301  avg=0.023 ms  BW=2232.34 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=  64  nElts=   4194301  avg=0.014 ms  BW=3503.74 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks= 128  nElts=   4194301  avg=0.010 ms  BW=5014.23 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks= 256  nElts=   4194301  avg=0.008 ms  BW=6106.55 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks= 512  nElts=   4194301  avg=0.008 ms  BW=5998.49 GB/s
  f32: 2src->1dst (all misaligned)               nBlocks=1024  nElts=   4194301  avg=0.008 ms  BW=6131.06 GB/s
[       OK ] ReduceCopyBench.BlockSweep_MisalignedAll_Float_2Src1Dst (194 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_MisalignedDst0_Float_2Src1Dst

--- block sweep: float 2src->1dst, dst misaligned by 1 (4M elts) ---
  f32: 2src->1dst (dst+1)                        nBlocks=   1  nElts=   4194303  avg=0.531 ms  BW=94.83 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=   2  nElts=   4194303  avg=0.271 ms  BW=185.53 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=   4  nElts=   4194303  avg=0.142 ms  BW=354.93 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=   8  nElts=   4194303  avg=0.070 ms  BW=715.40 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=  16  nElts=   4194303  avg=0.038 ms  BW=1339.21 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=  32  nElts=   4194303  avg=0.021 ms  BW=2447.50 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=  64  nElts=   4194303  avg=0.012 ms  BW=4057.75 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks= 128  nElts=   4194303  avg=0.008 ms  BW=6118.19 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks= 256  nElts=   4194303  avg=0.008 ms  BW=6137.29 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks= 512  nElts=   4194303  avg=0.008 ms  BW=6139.68 GB/s
  f32: 2src->1dst (dst+1)                        nBlocks=1024  nElts=   4194303  avg=0.008 ms  BW=6135.13 GB/s
[       OK ] ReduceCopyBench.BlockSweep_MisalignedDst0_Float_2Src1Dst (185 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_AlignedBaseline_Bf16_2Src1Dst

--- block sweep: bf16 2src->1dst, ALIGNED baseline (4M elts) ---
  bf16: 2src->1dst (aligned baseline)            nBlocks=   1  nElts=   4194304  avg=0.204 ms  BW=123.17 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=   2  nElts=   4194304  avg=0.104 ms  BW=240.96 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=   4  nElts=   4194304  avg=0.053 ms  BW=472.15 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=   8  nElts=   4194304  avg=0.029 ms  BW=877.05 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=  16  nElts=   4194304  avg=0.016 ms  BW=1534.89 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=  32  nElts=   4194304  avg=0.010 ms  BW=2451.93 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=  64  nElts=   4194304  avg=0.007 ms  BW=3725.05 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks= 128  nElts=   4194304  avg=0.006 ms  BW=4091.74 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks= 256  nElts=   4194304  avg=0.006 ms  BW=4086.63 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks= 512  nElts=   4194304  avg=0.006 ms  BW=4087.70 GB/s
  bf16: 2src->1dst (aligned baseline)            nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=4087.27 GB/s
[       OK ] ReduceCopyBench.BlockSweep_AlignedBaseline_Bf16_2Src1Dst (110 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_MisalignedSrc1_Bf16_2Src1Dst

--- block sweep: bf16 2src->1dst, src1 misaligned by 1 (4M elts) ---
  bf16: 2src->1dst (src1+1)                      nBlocks=   1  nElts=   4194303  avg=0.377 ms  BW=66.75 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=   2  nElts=   4194303  avg=0.193 ms  BW=130.71 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=   4  nElts=   4194303  avg=0.097 ms  BW=258.29 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=   8  nElts=   4194303  avg=0.053 ms  BW=476.14 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=  16  nElts=   4194303  avg=0.029 ms  BW=876.54 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=  32  nElts=   4194303  avg=0.016 ms  BW=1534.74 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=  64  nElts=   4194303  avg=0.010 ms  BW=2454.30 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks= 128  nElts=   4194303  avg=0.008 ms  BW=3064.22 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks= 256  nElts=   4194303  avg=0.006 ms  BW=3872.90 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks= 512  nElts=   4194303  avg=0.006 ms  BW=4077.31 GB/s
  bf16: 2src->1dst (src1+1)                      nBlocks=1024  nElts=   4194303  avg=0.006 ms  BW=4077.52 GB/s
[       OK ] ReduceCopyBench.BlockSweep_MisalignedSrc1_Bf16_2Src1Dst (149 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_MisalignedAll_Bf16_2Src1Dst

--- block sweep: bf16 2src->1dst, all misaligned (4M elts) ---
  bf16: 2src->1dst (all misaligned)              nBlocks=   1  nElts=   4194301  avg=0.482 ms  BW=52.19 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=   2  nElts=   4194301  avg=0.244 ms  BW=103.17 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=   4  nElts=   4194301  avg=0.123 ms  BW=204.22 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=   8  nElts=   4194301  avg=0.065 ms  BW=387.97 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=  16  nElts=   4194301  avg=0.035 ms  BW=722.89 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=  32  nElts=   4194301  avg=0.019 ms  BW=1306.36 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=  64  nElts=   4194301  avg=0.012 ms  BW=2045.92 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks= 128  nElts=   4194301  avg=0.008 ms  BW=3066.61 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks= 256  nElts=   4194301  avg=0.008 ms  BW=3061.71 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks= 512  nElts=   4194301  avg=0.008 ms  BW=3061.95 GB/s
  bf16: 2src->1dst (all misaligned)              nBlocks=1024  nElts=   4194301  avg=0.008 ms  BW=3069.72 GB/s
[       OK ] ReduceCopyBench.BlockSweep_MisalignedAll_Bf16_2Src1Dst (173 ms)
[ RUN      ] ReduceCopyBench.BlockSweep_MisalignedDst0_Bf16_2Src1Dst

--- block sweep: bf16 2src->1dst, dst misaligned by 1 (4M elts) ---
  bf16: 2src->1dst (dst+1)                       nBlocks=   1  nElts=   4194303  avg=0.441 ms  BW=57.00 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=   2  nElts=   4194303  avg=0.228 ms  BW=110.55 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=   4  nElts=   4194303  avg=0.118 ms  BW=212.41 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=   8  nElts=   4194303  avg=0.061 ms  BW=410.89 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=  16  nElts=   4194303  avg=0.033 ms  BW=767.15 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=  32  nElts=   4194303  avg=0.018 ms  BW=1364.41 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=  64  nElts=   4194303  avg=0.011 ms  BW=2245.73 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks= 128  nElts=   4194303  avg=0.008 ms  BW=3069.48 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks= 256  nElts=   4194303  avg=0.008 ms  BW=3069.12 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks= 512  nElts=   4194303  avg=0.008 ms  BW=3066.85 GB/s
  bf16: 2src->1dst (dst+1)                       nBlocks=1024  nElts=   4194303  avg=0.007 ms  BW=3488.12 GB/s
*/
