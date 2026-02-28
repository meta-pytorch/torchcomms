// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "meta/collectives/kernels/common_kernel_quantize.cuh" // @manual

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Benchmark Kernels
// =============================================================================

// Multi-block kernel wrapping reduceCopyMixed dispatch
template <int Unroll, typename AccumType, typename TransportType>
__global__ void benchReduceCopyMixedKernel(
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
// Benchmark Fixture
// =============================================================================

class ReduceCopyMixedBench : public ::testing::Test {
 protected:
  // Max elements for the largest benchmark size
  static constexpr int64_t kMaxElts = 16 * 1024 * 1024; // 16M elements
  static constexpr int kBlockSize = 256;
  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;

  float* d_srcFloat0 = nullptr;
  float* d_srcFloat1 = nullptr;
  float* d_dstFloat = nullptr;
  __nv_bfloat16* d_srcBf16_0 = nullptr;
  __nv_bfloat16* d_srcBf16_1 = nullptr;
  __nv_bfloat16* d_dstBf16 = nullptr;
  void** d_srcs = nullptr;
  void** d_dsts = nullptr;

  cudaEvent_t startEvent, stopEvent;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_srcFloat0, kMaxElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcFloat1, kMaxElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_dstFloat, kMaxElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcBf16_0, kMaxElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_srcBf16_1, kMaxElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_dstBf16, kMaxElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_srcs, 2 * sizeof(void*)));
    CUDACHECK(cudaMalloc(&d_dsts, sizeof(void*)));

    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&stopEvent));

    // Initialize source data with simple pattern
    std::vector<float> h_init(kMaxElts);
    for (int64_t i = 0; i < kMaxElts; i++) {
      h_init[i] = 1.0f + static_cast<float>(i % 1000) * 1e-4f;
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
  }

  void TearDown() override {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
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

  // Compute the maximum number of blocks for a given element count
  int maxBlocks(int64_t nElts) {
    return std::min((int)((nElts + kBlockSize - 1) / kBlockSize), 1024);
  }

  // Run a benchmark with a specific block count and report throughput
  void runBenchWithBlocks(
      int64_t nElts,
      int nBlocks,
      int nSrcs,
      bool src0IsAccumType,
      bool src1IsAccumType,
      bool dst0IsAccumType,
      uint64_t randomSeed,
      const char* label) {
    // Warmup
    for (int i = 0; i < kWarmupIters; i++) {
      benchReduceCopyMixedKernel<4, float, __nv_bfloat16>
          <<<nBlocks, kBlockSize>>>(
              nSrcs,
              d_srcs,
              1,
              d_dsts,
              nElts,
              src0IsAccumType,
              src1IsAccumType,
              dst0IsAccumType,
              randomSeed,
              0);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Timed iterations
    CUDACHECK(cudaEventRecord(startEvent));
    for (int i = 0; i < kBenchIters; i++) {
      benchReduceCopyMixedKernel<4, float, __nv_bfloat16>
          <<<nBlocks, kBlockSize>>>(
              nSrcs,
              d_srcs,
              1,
              d_dsts,
              nElts,
              src0IsAccumType,
              src1IsAccumType,
              dst0IsAccumType,
              randomSeed,
              static_cast<uint64_t>(i) * nElts);
    }
    CUDACHECK(cudaEventRecord(stopEvent));
    CUDACHECK(cudaDeviceSynchronize());

    float elapsedMs = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    float avgMs = elapsedMs / kBenchIters;

    // Calculate bytes:
    // reads: nSrcs * nElts * srcEltSize
    // writes: 1 * nElts * dstEltSize
    size_t srcEltSize0 =
        src0IsAccumType ? sizeof(float) : sizeof(__nv_bfloat16);
    size_t srcEltSize1 =
        src1IsAccumType ? sizeof(float) : sizeof(__nv_bfloat16);
    size_t dstEltSize = dst0IsAccumType ? sizeof(float) : sizeof(__nv_bfloat16);

    size_t totalBytes = nElts * srcEltSize0;
    if (nSrcs > 1) {
      totalBytes += nElts * srcEltSize1;
    }
    totalBytes += nElts * dstEltSize;

    double gbPerSec = (double)totalBytes / (avgMs * 1e6);
    printf(
        "  %-45s  nBlocks=%4d  nElts=%10ld  avg=%.3f ms  BW=%.2f GB/s\n",
        label,
        nBlocks,
        (long)nElts,
        avgMs,
        gbPerSec);
  }

  // Run a benchmark at max blocks (convenience wrapper for size sweeps)
  void runBench(
      int64_t nElts,
      int nSrcs,
      bool src0IsAccumType,
      bool src1IsAccumType,
      bool dst0IsAccumType,
      uint64_t randomSeed,
      const char* label) {
    runBenchWithBlocks(
        nElts,
        maxBlocks(nElts),
        nSrcs,
        src0IsAccumType,
        src1IsAccumType,
        dst0IsAccumType,
        randomSeed,
        label);
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
};

// =============================================================================
// Benchmarks: Single Source, varying sizes
// =============================================================================

TEST_F(ReduceCopyMixedBench, FloatToFloat_SingleSource) {
  printf("\n--- reduceCopyMixed: Float -> Float (single source) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    setSources(d_srcFloat0);
    setDestination(d_dstFloat);
    runBench(n, 1, true, true, true, 0, "f32->f32");
  }
}

TEST_F(ReduceCopyMixedBench, FloatToBf16_SingleSource) {
  printf(
      "\n--- reduceCopyMixed: Float -> BF16 (single source, stochastic rounding) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    setSources(d_srcFloat0);
    setDestination(d_dstBf16);
    runBench(n, 1, true, true, false, 12345ULL, "f32->bf16(SR)");
  }
}

TEST_F(ReduceCopyMixedBench, Bf16ToFloat_SingleSource) {
  printf("\n--- reduceCopyMixed: BF16 -> Float (single source) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    setSources(d_srcBf16_0);
    setDestination(d_dstFloat);
    runBench(n, 1, false, false, true, 0, "bf16->f32");
  }
}

TEST_F(ReduceCopyMixedBench, Bf16ToBf16_SingleSource) {
  printf(
      "\n--- reduceCopyMixed: BF16 -> BF16 (single source, stochastic rounding) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    setSources(d_srcBf16_0);
    setDestination(d_dstBf16);
    runBench(n, 1, false, false, false, 12345ULL, "bf16->bf16(SR)");
  }
}

// =============================================================================
// Benchmarks: Two-source reduction
// =============================================================================

TEST_F(ReduceCopyMixedBench, TwoFloatToFloat) {
  printf("\n--- reduceCopyMixed: 2x Float -> Float (sum reduction) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    setSources(d_srcFloat0, d_srcFloat1);
    setDestination(d_dstFloat);
    runBench(n, 2, true, true, true, 0, "2xf32->f32");
  }
}

TEST_F(ReduceCopyMixedBench, TwoFloatToBf16) {
  printf(
      "\n--- reduceCopyMixed: 2x Float -> BF16 (sum + stochastic rounding) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    setSources(d_srcFloat0, d_srcFloat1);
    setDestination(d_dstBf16);
    runBench(n, 2, true, true, false, 12345ULL, "2xf32->bf16(SR)");
  }
}

TEST_F(ReduceCopyMixedBench, MixedBf16FloatToFloat) {
  printf("\n--- reduceCopyMixed: BF16 + Float -> Float (mixed sum) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1};
    CUDACHECK(
        cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    setDestination(d_dstFloat);
    runBench(n, 2, false, true, true, 0, "bf16+f32->f32");
  }
}

TEST_F(ReduceCopyMixedBench, MixedBf16FloatToBf16) {
  printf(
      "\n--- reduceCopyMixed: BF16 + Float -> BF16 (mixed sum + stochastic rounding) ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1};
    CUDACHECK(
        cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    setDestination(d_dstBf16);
    runBench(n, 2, false, true, false, 12345ULL, "bf16+f32->bf16(SR)");
  }
}

// =============================================================================
// Benchmarks: Block count sweep (1 to max)
// =============================================================================

TEST_F(ReduceCopyMixedBench, BlockSweep_FloatToFloat) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopyMixed block sweep: Float -> Float (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  setSources(d_srcFloat0);
  setDestination(d_dstFloat);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(N, b, 1, true, true, true, 0, "f32->f32");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_FloatToBf16) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopyMixed block sweep: Float -> BF16 SR (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  setSources(d_srcFloat0);
  setDestination(d_dstBf16);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(N, b, 1, true, true, false, 12345ULL, "f32->bf16(SR)");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_Bf16ToFloat) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopyMixed block sweep: BF16 -> Float (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  setSources(d_srcBf16_0);
  setDestination(d_dstFloat);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(N, b, 1, false, false, true, 0, "bf16->f32");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_TwoFloatToFloat) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopyMixed block sweep: 2xFloat -> Float (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  setSources(d_srcFloat0, d_srcFloat1);
  setDestination(d_dstFloat);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(N, b, 2, true, true, true, 0, "2xf32->f32");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_TwoFloatToBf16) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopyMixed block sweep: 2xFloat -> BF16 SR (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  setSources(d_srcFloat0, d_srcFloat1);
  setDestination(d_dstBf16);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(N, b, 2, true, true, false, 12345ULL, "2xf32->bf16(SR)");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_MixedBf16FloatToFloat) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- reduceCopyMixed block sweep: BF16+Float -> Float (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstFloat);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(N, b, 2, false, true, true, 0, "bf16+f32->f32");
  }
}

// =============================================================================
// Benchmarks: PAT Step Type-Config Specific Tests
// =============================================================================
// These test cases mirror the actual type configurations that occur during
// a PAT reduce-scatter quantized run. They enable apples-to-apples comparison
// with the non-quantized PAT baseline.

TEST_F(ReduceCopyMixedBench, PatStep_Initial_FloatToBf16) {
  printf(
      "\n--- PAT Step: Initial (no recv) - FP32 -> BF16 with SR ---\n"
      "Type config case 6: First step quantizes local data\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    setSources(d_srcFloat0);
    setDestination(d_dstBf16);
    runBench(n, 1, true, true, false, 12345ULL, "PAT-Initial: f32->bf16(SR)");
  }
}

TEST_F(ReduceCopyMixedBench, PatStep_Intermediate_Bf16FloatToBf16) {
  printf(
      "\n--- PAT Step: Intermediate - BF16 + FP32 -> BF16 with SR ---\n"
      "Type config case 2: Most common step - recv + reduce + send\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1};
    CUDACHECK(
        cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    setDestination(d_dstBf16);
    runBench(
        n,
        2,
        false,
        true,
        false,
        12345ULL,
        "PAT-Intermediate: bf16+f32->bf16(SR)");
  }
}

TEST_F(ReduceCopyMixedBench, PatStep_Reaccumulate_Bf16ToBf16) {
  printf(
      "\n--- PAT Step: Re-accumulate - BF16 + BF16 -> BF16 with SR ---\n"
      "Type config case 0: Re-accumulation from send buffer\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    setSources(d_srcBf16_0, d_srcBf16_1);
    setDestination(d_dstBf16);
    runBench(
        n,
        2,
        false,
        false,
        false,
        12345ULL,
        "PAT-Reaccumulate: bf16+bf16->bf16(SR)");
  }
}

TEST_F(ReduceCopyMixedBench, PatStep_Final_Bf16FloatToFloat) {
  printf(
      "\n--- PAT Step: Final output - BF16 + FP32 -> FP32 ---\n"
      "Type config case 3: Final step writes FP32 result\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1};
    CUDACHECK(
        cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    setDestination(d_dstFloat);
    runBench(n, 2, false, true, true, 0, "PAT-Final: bf16+f32->f32");
  }
}

// =============================================================================
// Benchmarks: PAT Step Block Count Sweeps
// =============================================================================

TEST_F(ReduceCopyMixedBench, BlockSweep_PatStep_Initial) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- PAT Step block sweep: Initial f32->bf16(SR) (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  setSources(d_srcFloat0);
  setDestination(d_dstBf16);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(N, b, 1, true, true, false, 12345ULL, "PAT-Initial");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_PatStep_Intermediate) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- PAT Step block sweep: Intermediate bf16+f32->bf16(SR) (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstBf16);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(
        N, b, 2, false, true, false, 12345ULL, "PAT-Intermediate");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_PatStep_Reaccumulate) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- PAT Step block sweep: Reaccumulate bf16+bf16->bf16(SR) (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  setSources(d_srcBf16_0, d_srcBf16_1);
  setDestination(d_dstBf16);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(
        N, b, 2, false, false, false, 12345ULL, "PAT-Reaccumulate");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_PatStep_Final) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- PAT Step block sweep: Final bf16+f32->f32 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstFloat);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(N, b, 2, false, true, true, 0, "PAT-Final");
  }
}

// =============================================================================
// Benchmarks: Misaligned Pointer Tests
// =============================================================================

TEST_F(ReduceCopyMixedBench, AlignedBaseline_2xFloatToFloat) {
  printf("\n--- reduceCopyMixed: 2x Float -> Float, ALIGNED baseline ---\n");
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    setSources(d_srcFloat0, d_srcFloat1);
    setDestination(d_dstFloat);
    runBench(n, 2, true, true, true, 0, "2xf32->f32 (aligned baseline)");
  }
}

TEST_F(ReduceCopyMixedBench, MisalignedSrc1_2xFloatToFloat) {
  printf(
      "\n--- reduceCopyMixed: 2x Float -> Float, src1 misaligned by 1 ---\n");
  constexpr int kSrc1Offset = 1;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kSrc1Offset;
    void* h_srcs[2] = {d_srcFloat0, d_srcFloat1 + kSrc1Offset};
    CUDACHECK(
        cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    setDestination(d_dstFloat);
    runBench(nElts, 2, true, true, true, 0, "2xf32->f32 (src1+1)");
  }
}

TEST_F(ReduceCopyMixedBench, MisalignedSrc1_Bf16FloatToFloat) {
  printf(
      "\n--- reduceCopyMixed: BF16 + Float -> Float, src1(float) misaligned by 1 ---\n");
  constexpr int kSrc1Offset = 1;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kSrc1Offset;
    void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1 + kSrc1Offset};
    CUDACHECK(
        cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    setDestination(d_dstFloat);
    runBench(nElts, 2, false, true, true, 0, "bf16+f32->f32 (src1+1)");
  }
}

TEST_F(ReduceCopyMixedBench, MisalignedSrc1_Bf16FloatToBf16SR) {
  printf(
      "\n--- reduceCopyMixed: BF16 + Float -> BF16 SR, src1(float) misaligned by 1 ---\n");
  constexpr int kSrc1Offset = 1;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kSrc1Offset;
    void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1 + kSrc1Offset};
    CUDACHECK(
        cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    setDestination(d_dstBf16);
    runBench(
        nElts, 2, false, true, false, 12345ULL, "bf16+f32->bf16SR (src1+1)");
  }
}

TEST_F(ReduceCopyMixedBench, MisalignedAll_2xFloatToFloat) {
  printf(
      "\n--- reduceCopyMixed: 2x Float -> Float, all misaligned (src0+1, src1+2, dst+3) ---\n");
  constexpr int kSrc0Offset = 1;
  constexpr int kSrc1Offset = 2;
  constexpr int kDst0Offset = 3;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kDst0Offset;
    void* h_srcs[2] = {d_srcFloat0 + kSrc0Offset, d_srcFloat1 + kSrc1Offset};
    CUDACHECK(
        cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
    void* dstOffset = d_dstFloat + kDst0Offset;
    CUDACHECK(
        cudaMemcpy(d_dsts, &dstOffset, sizeof(void*), cudaMemcpyHostToDevice));
    runBench(nElts, 2, true, true, true, 0, "2xf32->f32 (all misaligned)");
  }
}

TEST_F(ReduceCopyMixedBench, MisalignedDst0_FloatToBf16SR) {
  printf(
      "\n--- reduceCopyMixed: 2x Float -> BF16 SR, dst misaligned by 1 ---\n");
  constexpr int kDst0Offset = 1;
  int64_t sizes[] = {
      1024, 8192, 65536, 524288, 4 * 1024 * 1024, 16 * 1024 * 1024};
  for (int64_t n : sizes) {
    int64_t nElts = n - kDst0Offset;
    setSources(d_srcFloat0, d_srcFloat1);
    void* dstOffset = d_dstBf16 + kDst0Offset;
    CUDACHECK(
        cudaMemcpy(d_dsts, &dstOffset, sizeof(void*), cudaMemcpyHostToDevice));
    runBench(nElts, 2, true, true, false, 12345ULL, "2xf32->bf16SR (dst+1)");
  }
}

// =============================================================================
// Benchmarks: Misaligned Pointer Block Count Sweeps
// =============================================================================

TEST_F(ReduceCopyMixedBench, BlockSweep_AlignedBaseline_2xFloatToFloat) {
  constexpr int64_t N = 4 * 1024 * 1024;
  printf(
      "\n--- block sweep: 2x Float -> Float, ALIGNED baseline (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  setSources(d_srcFloat0, d_srcFloat1);
  setDestination(d_dstFloat);
  for (int b : blockCountSweep(maxBlocks(N))) {
    runBenchWithBlocks(
        N, b, 2, true, true, true, 0, "2xf32->f32 (aligned baseline)");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_MisalignedSrc1_2xFloatToFloat) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kSrc1Offset = 1;
  int64_t nElts = N - kSrc1Offset;
  printf(
      "\n--- block sweep: 2x Float -> Float, src1 misaligned by 1 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  void* h_srcs[2] = {d_srcFloat0, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstFloat);
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchWithBlocks(nElts, b, 2, true, true, true, 0, "2xf32->f32 (src1+1)");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_MisalignedSrc1_Bf16FloatToFloat) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kSrc1Offset = 1;
  int64_t nElts = N - kSrc1Offset;
  printf(
      "\n--- block sweep: BF16 + Float -> Float, src1 misaligned by 1 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstFloat);
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchWithBlocks(
        nElts, b, 2, false, true, true, 0, "bf16+f32->f32 (src1+1)");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_MisalignedSrc1_Bf16FloatToBf16SR) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kSrc1Offset = 1;
  int64_t nElts = N - kSrc1Offset;
  printf(
      "\n--- block sweep: BF16 + Float -> BF16 SR, src1 misaligned by 1 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  void* h_srcs[2] = {d_srcBf16_0, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  setDestination(d_dstBf16);
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchWithBlocks(
        nElts, b, 2, false, true, false, 12345ULL, "bf16+f32->bf16SR (src1+1)");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_MisalignedAll_2xFloatToFloat) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kSrc0Offset = 1;
  constexpr int kSrc1Offset = 2;
  constexpr int kDst0Offset = 3;
  int64_t nElts = N - kDst0Offset;
  printf(
      "\n--- block sweep: 2x Float -> Float, all misaligned (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  void* h_srcs[2] = {d_srcFloat0 + kSrc0Offset, d_srcFloat1 + kSrc1Offset};
  CUDACHECK(
      cudaMemcpy(d_srcs, h_srcs, 2 * sizeof(void*), cudaMemcpyHostToDevice));
  void* dstOffset = d_dstFloat + kDst0Offset;
  CUDACHECK(
      cudaMemcpy(d_dsts, &dstOffset, sizeof(void*), cudaMemcpyHostToDevice));
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchWithBlocks(
        nElts, b, 2, true, true, true, 0, "2xf32->f32 (all misaligned)");
  }
}

TEST_F(ReduceCopyMixedBench, BlockSweep_MisalignedDst0_FloatToBf16SR) {
  constexpr int64_t N = 4 * 1024 * 1024;
  constexpr int kDst0Offset = 1;
  int64_t nElts = N - kDst0Offset;
  printf(
      "\n--- block sweep: 2x Float -> BF16 SR, dst misaligned by 1 (%ldM elts) ---\n",
      (long)(N / (1024 * 1024)));
  setSources(d_srcFloat0, d_srcFloat1);
  void* dstOffset = d_dstBf16 + kDst0Offset;
  CUDACHECK(
      cudaMemcpy(d_dsts, &dstOffset, sizeof(void*), cudaMemcpyHostToDevice));
  for (int b : blockCountSweep(maxBlocks(nElts))) {
    runBenchWithBlocks(
        nElts, b, 2, true, true, false, 12345ULL, "2xf32->bf16SR (dst+1)");
  }
}

// clang-format off
// We want to keep the format for the result below

/*
[==========] Running 34 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 34 tests from ReduceCopyMixedBench
[ RUN      ] ReduceCopyMixedBench.FloatToFloat_SingleSource

--- reduceCopyMixed: Float -> Float (single source) ---
  f32->f32                                       nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.98 GB/s
  f32->f32                                       nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=15.95 GB/s
  f32->f32                                       nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=128.21 GB/s
  f32->f32                                       nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=1020.25 GB/s
  f32->f32                                       nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=5436.13 GB/s
  f32->f32                                       nBlocks=1024  nElts=  16777216  avg=0.023 ms  BW=5902.40 GB/s
[       OK ] ReduceCopyMixedBench.FloatToFloat_SingleSource (213 ms)
[ RUN      ] ReduceCopyMixedBench.FloatToBf16_SingleSource

--- reduceCopyMixed: Float -> BF16 (single source, stochastic rounding) ---
  f32->bf16(SR)                                  nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.49 GB/s
  f32->bf16(SR)                                  nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=11.95 GB/s
  f32->bf16(SR)                                  nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=95.75 GB/s
  f32->bf16(SR)                                  nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=764.95 GB/s
  f32->bf16(SR)                                  nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=3067.81 GB/s
  f32->bf16(SR)                                  nBlocks=1024  nElts=  16777216  avg=0.017 ms  BW=5867.03 GB/s
[       OK ] ReduceCopyMixedBench.FloatToBf16_SingleSource (41 ms)
[ RUN      ] ReduceCopyMixedBench.Bf16ToFloat_SingleSource

--- reduceCopyMixed: BF16 -> Float (single source) ---
  bf16->f32                                      nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.50 GB/s
  bf16->f32                                      nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=11.95 GB/s
  bf16->f32                                      nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=95.76 GB/s
  bf16->f32                                      nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=763.17 GB/s
  bf16->f32                                      nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=4082.82 GB/s
  bf16->f32                                      nBlocks=1024  nElts=  16777216  avg=0.018 ms  BW=5512.34 GB/s
[       OK ] ReduceCopyMixedBench.Bf16ToFloat_SingleSource (40 ms)
[ RUN      ] ReduceCopyMixedBench.Bf16ToBf16_SingleSource

--- reduceCopyMixed: BF16 -> BF16 (single source, stochastic rounding) ---
  bf16->bf16(SR)                                 nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=0.99 GB/s
  bf16->bf16(SR)                                 nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=7.96 GB/s
  bf16->bf16(SR)                                 nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=63.53 GB/s
  bf16->bf16(SR)                                 nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=507.76 GB/s
  bf16->bf16(SR)                                 nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=2044.41 GB/s
  bf16->bf16(SR)                                 nBlocks=1024  nElts=  16777216  avg=0.018 ms  BW=3638.24 GB/s
[       OK ] ReduceCopyMixedBench.Bf16ToBf16_SingleSource (41 ms)
[ RUN      ] ReduceCopyMixedBench.TwoFloatToFloat

--- reduceCopyMixed: 2x Float -> Float (sum reduction) ---
  2xf32->f32                                     nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=2.98 GB/s
  2xf32->f32                                     nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=23.92 GB/s
  2xf32->f32                                     nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=191.21 GB/s
  2xf32->f32                                     nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=1523.74 GB/s
  2xf32->f32                                     nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=6130.35 GB/s
  2xf32->f32                                     nBlocks=1024  nElts=  16777216  avg=0.033 ms  BW=6140.64 GB/s
[       OK ] ReduceCopyMixedBench.TwoFloatToFloat (42 ms)
[ RUN      ] ReduceCopyMixedBench.TwoFloatToBf16

--- reduceCopyMixed: 2x Float -> BF16 (sum + stochastic rounding) ---
  2xf32->bf16(SR)                                nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=2.48 GB/s
  2xf32->bf16(SR)                                nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=19.86 GB/s
  2xf32->bf16(SR)                                nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=158.72 GB/s
  2xf32->bf16(SR)                                nBlocks=1024  nElts=    524288  avg=0.005 ms  BW=1017.51 GB/s
  2xf32->bf16(SR)                                nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=5101.67 GB/s
  2xf32->bf16(SR)                                nBlocks=1024  nElts=  16777216  avg=0.031 ms  BW=5460.99 GB/s
[       OK ] ReduceCopyMixedBench.TwoFloatToBf16 (43 ms)
[ RUN      ] ReduceCopyMixedBench.MixedBf16FloatToFloat

--- reduceCopyMixed: BF16 + Float -> Float (mixed sum) ---
  bf16+f32->f32                                  nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=2.48 GB/s
  bf16+f32->f32                                  nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=19.86 GB/s
  bf16+f32->f32                                  nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=158.88 GB/s
  bf16+f32->f32                                  nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=1269.78 GB/s
  bf16+f32->f32                                  nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=5111.02 GB/s
  bf16+f32->f32                                  nBlocks=1024  nElts=  16777216  avg=0.036 ms  BW=4652.11 GB/s
[       OK ] ReduceCopyMixedBench.MixedBf16FloatToFloat (44 ms)
[ RUN      ] ReduceCopyMixedBench.MixedBf16FloatToBf16

--- reduceCopyMixed: BF16 + Float -> BF16 (mixed sum + stochastic rounding) ---
  bf16+f32->bf16(SR)                             nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.99 GB/s
  bf16+f32->bf16(SR)                             nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=15.93 GB/s
  bf16+f32->bf16(SR)                             nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=125.51 GB/s
  bf16+f32->bf16(SR)                             nBlocks=1024  nElts=    524288  avg=0.005 ms  BW=813.91 GB/s
  bf16+f32->bf16(SR)                             nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=4081.49 GB/s
  bf16+f32->bf16(SR)                             nBlocks=1024  nElts=  16777216  avg=0.031 ms  BW=4358.81 GB/s
[       OK ] ReduceCopyMixedBench.MixedBf16FloatToBf16 (44 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_FloatToFloat

--- reduceCopyMixed block sweep: Float -> Float (4M elts) ---
  f32->f32                                       nBlocks=   1  nElts=   4194304  avg=0.343 ms  BW=97.92 GB/s
  f32->f32                                       nBlocks=   2  nElts=   4194304  avg=0.178 ms  BW=188.20 GB/s
  f32->f32                                       nBlocks=   4  nElts=   4194304  avg=0.100 ms  BW=336.86 GB/s
  f32->f32                                       nBlocks=   8  nElts=   4194304  avg=0.053 ms  BW=631.50 GB/s
  f32->f32                                       nBlocks=  16  nElts=   4194304  avg=0.029 ms  BW=1139.21 GB/s
  f32->f32                                       nBlocks=  32  nElts=   4194304  avg=0.017 ms  BW=1921.56 GB/s
  f32->f32                                       nBlocks=  64  nElts=   4194304  avg=0.011 ms  BW=3123.36 GB/s
  f32->f32                                       nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=4091.05 GB/s
  f32->f32                                       nBlocks= 256  nElts=   4194304  avg=0.007 ms  BW=4729.71 GB/s
  f32->f32                                       nBlocks= 512  nElts=   4194304  avg=0.006 ms  BW=5444.60 GB/s
  f32->f32                                       nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=5429.94 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_FloatToFloat (121 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_FloatToBf16

--- reduceCopyMixed block sweep: Float -> BF16 SR (4M elts) ---
  f32->bf16(SR)                                  nBlocks=   1  nElts=   4194304  avg=0.465 ms  BW=54.12 GB/s
  f32->bf16(SR)                                  nBlocks=   2  nElts=   4194304  avg=0.235 ms  BW=107.15 GB/s
  f32->bf16(SR)                                  nBlocks=   4  nElts=   4194304  avg=0.120 ms  BW=210.28 GB/s
  f32->bf16(SR)                                  nBlocks=   8  nElts=   4194304  avg=0.061 ms  BW=409.27 GB/s
  f32->bf16(SR)                                  nBlocks=  16  nElts=   4194304  avg=0.033 ms  BW=767.15 GB/s
  f32->bf16(SR)                                  nBlocks=  32  nElts=   4194304  avg=0.018 ms  BW=1364.39 GB/s
  f32->bf16(SR)                                  nBlocks=  64  nElts=   4194304  avg=0.011 ms  BW=2223.13 GB/s
  f32->bf16(SR)                                  nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=3065.18 GB/s
  f32->bf16(SR)                                  nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=3066.25 GB/s
  f32->bf16(SR)                                  nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=3066.97 GB/s
  f32->bf16(SR)                                  nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=3067.45 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_FloatToBf16 (145 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_Bf16ToFloat

--- reduceCopyMixed block sweep: BF16 -> Float (4M elts) ---
  bf16->f32                                      nBlocks=   1  nElts=   4194304  avg=0.452 ms  BW=55.72 GB/s
  bf16->f32                                      nBlocks=   2  nElts=   4194304  avg=0.234 ms  BW=107.45 GB/s
  bf16->f32                                      nBlocks=   4  nElts=   4194304  avg=0.121 ms  BW=207.56 GB/s
  bf16->f32                                      nBlocks=   8  nElts=   4194304  avg=0.063 ms  BW=397.70 GB/s
  bf16->f32                                      nBlocks=  16  nElts=   4194304  avg=0.033 ms  BW=760.90 GB/s
  bf16->f32                                      nBlocks=  32  nElts=   4194304  avg=0.018 ms  BW=1364.22 GB/s
  bf16->f32                                      nBlocks=  64  nElts=   4194304  avg=0.012 ms  BW=2105.86 GB/s
  bf16->f32                                      nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=3031.85 GB/s
  bf16->f32                                      nBlocks= 256  nElts=   4194304  avg=0.006 ms  BW=4076.04 GB/s
  bf16->f32                                      nBlocks= 512  nElts=   4194304  avg=0.006 ms  BW=4083.24 GB/s
  bf16->f32                                      nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=4085.57 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_Bf16ToFloat (142 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_TwoFloatToFloat

--- reduceCopyMixed block sweep: 2xFloat -> Float (4M elts) ---
  2xf32->f32                                     nBlocks=   1  nElts=   4194304  avg=0.531 ms  BW=94.79 GB/s
  2xf32->f32                                     nBlocks=   2  nElts=   4194304  avg=0.267 ms  BW=188.49 GB/s
  2xf32->f32                                     nBlocks=   4  nElts=   4194304  avg=0.138 ms  BW=365.64 GB/s
  2xf32->f32                                     nBlocks=   8  nElts=   4194304  avg=0.072 ms  BW=702.61 GB/s
  2xf32->f32                                     nBlocks=  16  nElts=   4194304  avg=0.038 ms  BW=1319.13 GB/s
  2xf32->f32                                     nBlocks=  32  nElts=   4194304  avg=0.021 ms  BW=2443.78 GB/s
  2xf32->f32                                     nBlocks=  64  nElts=   4194304  avg=0.012 ms  BW=4086.95 GB/s
  2xf32->f32                                     nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=6133.70 GB/s
  2xf32->f32                                     nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=6129.63 GB/s
  2xf32->f32                                     nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=6132.50 GB/s
  2xf32->f32                                     nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=6131.07 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_TwoFloatToFloat (159 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_TwoFloatToBf16

--- reduceCopyMixed block sweep: 2xFloat -> BF16 SR (4M elts) ---
  2xf32->bf16(SR)                                nBlocks=   1  nElts=   4194304  avg=0.735 ms  BW=57.06 GB/s
  2xf32->bf16(SR)                                nBlocks=   2  nElts=   4194304  avg=0.372 ms  BW=112.85 GB/s
  2xf32->bf16(SR)                                nBlocks=   4  nElts=   4194304  avg=0.189 ms  BW=221.86 GB/s
  2xf32->bf16(SR)                                nBlocks=   8  nElts=   4194304  avg=0.096 ms  BW=436.14 GB/s
  2xf32->bf16(SR)                                nBlocks=  16  nElts=   4194304  avg=0.050 ms  BW=844.34 GB/s
  2xf32->bf16(SR)                                nBlocks=  32  nElts=   4194304  avg=0.027 ms  BW=1574.15 GB/s
  2xf32->bf16(SR)                                nBlocks=  64  nElts=   4194304  avg=0.016 ms  BW=2557.40 GB/s
  2xf32->bf16(SR)                                nBlocks= 128  nElts=   4194304  avg=0.010 ms  BW=4090.12 GB/s
  2xf32->bf16(SR)                                nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=5109.82 GB/s
  2xf32->bf16(SR)                                nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=5105.44 GB/s
  2xf32->bf16(SR)                                nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=5110.42 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_TwoFloatToBf16 (203 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_MixedBf16FloatToFloat

--- reduceCopyMixed block sweep: BF16+Float -> Float (4M elts) ---
  bf16+f32->f32                                  nBlocks=   1  nElts=   4194304  avg=0.780 ms  BW=53.78 GB/s
  bf16+f32->f32                                  nBlocks=   2  nElts=   4194304  avg=0.395 ms  BW=106.18 GB/s
  bf16+f32->f32                                  nBlocks=   4  nElts=   4194304  avg=0.203 ms  BW=206.24 GB/s
  bf16+f32->f32                                  nBlocks=   8  nElts=   4194304  avg=0.104 ms  BW=404.07 GB/s
  bf16+f32->f32                                  nBlocks=  16  nElts=   4194304  avg=0.054 ms  BW=778.27 GB/s
  bf16+f32->f32                                  nBlocks=  32  nElts=   4194304  avg=0.029 ms  BW=1458.84 GB/s
  bf16+f32->f32                                  nBlocks=  64  nElts=   4194304  avg=0.016 ms  BW=2557.00 GB/s
  bf16+f32->f32                                  nBlocks= 128  nElts=   4194304  avg=0.010 ms  BW=4084.39 GB/s
  bf16+f32->f32                                  nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=5112.01 GB/s
  bf16+f32->f32                                  nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=5111.61 GB/s
  bf16+f32->f32                                  nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=5110.22 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_MixedBf16FloatToFloat (214 ms)
[ RUN      ] ReduceCopyMixedBench.PatStep_Initial_FloatToBf16

--- PAT Step: Initial (no recv) - FP32 -> BF16 with SR ---
Type config case 6: First step quantizes local data
  PAT-Initial: f32->bf16(SR)                     nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.49 GB/s
  PAT-Initial: f32->bf16(SR)                     nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=11.97 GB/s
  PAT-Initial: f32->bf16(SR)                     nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=95.27 GB/s
  PAT-Initial: f32->bf16(SR)                     nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=764.00 GB/s
  PAT-Initial: f32->bf16(SR)                     nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=3068.05 GB/s
  PAT-Initial: f32->bf16(SR)                     nBlocks=1024  nElts=  16777216  avg=0.017 ms  BW=5847.51 GB/s
[       OK ] ReduceCopyMixedBench.PatStep_Initial_FloatToBf16 (40 ms)
[ RUN      ] ReduceCopyMixedBench.PatStep_Intermediate_Bf16FloatToBf16

--- PAT Step: Intermediate - BF16 + FP32 -> BF16 with SR ---
Type config case 2: Most common step - recv + reduce + send
  PAT-Intermediate: bf16+f32->bf16(SR)           nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.99 GB/s
  PAT-Intermediate: bf16+f32->bf16(SR)           nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=15.90 GB/s
  PAT-Intermediate: bf16+f32->bf16(SR)           nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=127.02 GB/s
  PAT-Intermediate: bf16+f32->bf16(SR)           nBlocks=1024  nElts=    524288  avg=0.005 ms  BW=816.60 GB/s
  PAT-Intermediate: bf16+f32->bf16(SR)           nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=4087.06 GB/s
  PAT-Intermediate: bf16+f32->bf16(SR)           nBlocks=1024  nElts=  16777216  avg=0.031 ms  BW=4360.12 GB/s
[       OK ] ReduceCopyMixedBench.PatStep_Intermediate_Bf16FloatToBf16 (41 ms)
[ RUN      ] ReduceCopyMixedBench.PatStep_Reaccumulate_Bf16ToBf16

--- PAT Step: Re-accumulate - BF16 + BF16 -> BF16 with SR ---
Type config case 0: Re-accumulation from send buffer
  PAT-Reaccumulate: bf16+bf16->bf16(SR)          nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=1.49 GB/s
  PAT-Reaccumulate: bf16+bf16->bf16(SR)          nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=11.92 GB/s
  PAT-Reaccumulate: bf16+bf16->bf16(SR)          nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=95.49 GB/s
  PAT-Reaccumulate: bf16+bf16->bf16(SR)          nBlocks=1024  nElts=    524288  avg=0.005 ms  BW=605.77 GB/s
  PAT-Reaccumulate: bf16+bf16->bf16(SR)          nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=3063.62 GB/s
  PAT-Reaccumulate: bf16+bf16->bf16(SR)          nBlocks=1024  nElts=  16777216  avg=0.025 ms  BW=3962.37 GB/s
[       OK ] ReduceCopyMixedBench.PatStep_Reaccumulate_Bf16ToBf16 (41 ms)
[ RUN      ] ReduceCopyMixedBench.PatStep_Final_Bf16FloatToFloat

--- PAT Step: Final output - BF16 + FP32 -> FP32 ---
Type config case 3: Final step writes FP32 result
  PAT-Final: bf16+f32->f32                       nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=2.49 GB/s
  PAT-Final: bf16+f32->f32                       nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=19.93 GB/s
  PAT-Final: bf16+f32->f32                       nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=158.86 GB/s
  PAT-Final: bf16+f32->f32                       nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=1269.88 GB/s
  PAT-Final: bf16+f32->f32                       nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=5110.62 GB/s
  PAT-Final: bf16+f32->f32                       nBlocks=1024  nElts=  16777216  avg=0.036 ms  BW=4659.26 GB/s
[       OK ] ReduceCopyMixedBench.PatStep_Final_Bf16FloatToFloat (42 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_PatStep_Initial

--- PAT Step block sweep: Initial f32->bf16(SR) (4M elts) ---
  PAT-Initial                                    nBlocks=   1  nElts=   4194304  avg=0.465 ms  BW=54.09 GB/s
  PAT-Initial                                    nBlocks=   2  nElts=   4194304  avg=0.235 ms  BW=107.17 GB/s
  PAT-Initial                                    nBlocks=   4  nElts=   4194304  avg=0.120 ms  BW=210.47 GB/s
  PAT-Initial                                    nBlocks=   8  nElts=   4194304  avg=0.061 ms  BW=409.35 GB/s
  PAT-Initial                                    nBlocks=  16  nElts=   4194304  avg=0.033 ms  BW=767.53 GB/s
  PAT-Initial                                    nBlocks=  32  nElts=   4194304  avg=0.018 ms  BW=1363.44 GB/s
  PAT-Initial                                    nBlocks=  64  nElts=   4194304  avg=0.011 ms  BW=2223.13 GB/s
  PAT-Initial                                    nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=3064.70 GB/s
  PAT-Initial                                    nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=3066.13 GB/s
  PAT-Initial                                    nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=3069.48 GB/s
  PAT-Initial                                    nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=3066.85 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_PatStep_Initial (144 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_PatStep_Intermediate

--- PAT Step block sweep: Intermediate bf16+f32->bf16(SR) (4M elts) ---
  PAT-Intermediate                               nBlocks=   1  nElts=   4194304  avg=0.922 ms  BW=36.39 GB/s
  PAT-Intermediate                               nBlocks=   2  nElts=   4194304  avg=0.463 ms  BW=72.42 GB/s
  PAT-Intermediate                               nBlocks=   4  nElts=   4194304  avg=0.235 ms  BW=142.52 GB/s
  PAT-Intermediate                               nBlocks=   8  nElts=   4194304  avg=0.120 ms  BW=279.49 GB/s
  PAT-Intermediate                               nBlocks=  16  nElts=   4194304  avg=0.061 ms  BW=545.89 GB/s
  PAT-Intermediate                               nBlocks=  32  nElts=   4194304  avg=0.033 ms  BW=1023.32 GB/s
  PAT-Intermediate                               nBlocks=  64  nElts=   4194304  avg=0.018 ms  BW=1818.83 GB/s
  PAT-Intermediate                               nBlocks= 128  nElts=   4194304  avg=0.011 ms  BW=2974.18 GB/s
  PAT-Intermediate                               nBlocks= 256  nElts=   4194304  avg=0.010 ms  BW=3458.71 GB/s
  PAT-Intermediate                               nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=4082.44 GB/s
  PAT-Intermediate                               nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=4085.63 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_PatStep_Intermediate (245 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_PatStep_Reaccumulate

--- PAT Step block sweep: Reaccumulate bf16+bf16->bf16(SR) (4M elts) ---
  PAT-Reaccumulate                               nBlocks=   1  nElts=   4194304  avg=0.946 ms  BW=26.60 GB/s
  PAT-Reaccumulate                               nBlocks=   2  nElts=   4194304  avg=0.478 ms  BW=52.65 GB/s
  PAT-Reaccumulate                               nBlocks=   4  nElts=   4194304  avg=0.243 ms  BW=103.42 GB/s
  PAT-Reaccumulate                               nBlocks=   8  nElts=   4194304  avg=0.124 ms  BW=202.83 GB/s
  PAT-Reaccumulate                               nBlocks=  16  nElts=   4194304  avg=0.063 ms  BW=396.52 GB/s
  PAT-Reaccumulate                               nBlocks=  32  nElts=   4194304  avg=0.033 ms  BW=754.52 GB/s
  PAT-Reaccumulate                               nBlocks=  64  nElts=   4194304  avg=0.018 ms  BW=1363.44 GB/s
  PAT-Reaccumulate                               nBlocks= 128  nElts=   4194304  avg=0.012 ms  BW=2045.76 GB/s
  PAT-Reaccumulate                               nBlocks= 256  nElts=   4194304  avg=0.010 ms  BW=2454.46 GB/s
  PAT-Reaccumulate                               nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=3060.64 GB/s
  PAT-Reaccumulate                               nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=3044.17 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_PatStep_Reaccumulate (249 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_PatStep_Final

--- PAT Step block sweep: Final bf16+f32->f32 (4M elts) ---
  PAT-Final                                      nBlocks=   1  nElts=   4194304  avg=0.780 ms  BW=53.78 GB/s
  PAT-Final                                      nBlocks=   2  nElts=   4194304  avg=0.395 ms  BW=106.23 GB/s
  PAT-Final                                      nBlocks=   4  nElts=   4194304  avg=0.203 ms  BW=206.33 GB/s
  PAT-Final                                      nBlocks=   8  nElts=   4194304  avg=0.104 ms  BW=402.93 GB/s
  PAT-Final                                      nBlocks=  16  nElts=   4194304  avg=0.054 ms  BW=775.62 GB/s
  PAT-Final                                      nBlocks=  32  nElts=   4194304  avg=0.029 ms  BW=1457.72 GB/s
  PAT-Final                                      nBlocks=  64  nElts=   4194304  avg=0.016 ms  BW=2557.70 GB/s
  PAT-Final                                      nBlocks= 128  nElts=   4194304  avg=0.010 ms  BW=4088.46 GB/s
  PAT-Final                                      nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=5108.82 GB/s
  PAT-Final                                      nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=5111.02 GB/s
  PAT-Final                                      nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=5108.23 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_PatStep_Final (214 ms)
[ RUN      ] ReduceCopyMixedBench.AlignedBaseline_2xFloatToFloat

--- reduceCopyMixed: 2x Float -> Float, ALIGNED baseline ---
  2xf32->f32 (aligned baseline)                  nBlocks=   4  nElts=      1024  avg=0.004 ms  BW=2.99 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=23.87 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=190.97 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=1527.41 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=6136.09 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=1024  nElts=  16777216  avg=0.033 ms  BW=6141.00 GB/s
[       OK ] ReduceCopyMixedBench.AlignedBaseline_2xFloatToFloat (41 ms)
[ RUN      ] ReduceCopyMixedBench.MisalignedSrc1_2xFloatToFloat

--- reduceCopyMixed: 2x Float -> Float, src1 misaligned by 1 ---
  2xf32->f32 (src1+1)                            nBlocks=   4  nElts=      1023  avg=0.004 ms  BW=2.98 GB/s
  2xf32->f32 (src1+1)                            nBlocks=  32  nElts=      8191  avg=0.004 ms  BW=23.90 GB/s
  2xf32->f32 (src1+1)                            nBlocks= 256  nElts=     65535  avg=0.004 ms  BW=191.32 GB/s
  2xf32->f32 (src1+1)                            nBlocks=1024  nElts=    524287  avg=0.006 ms  BW=1054.31 GB/s
  2xf32->f32 (src1+1)                            nBlocks=1024  nElts=   4194303  avg=0.010 ms  BW=4907.84 GB/s
  2xf32->f32 (src1+1)                            nBlocks=1024  nElts=  16777215  avg=0.033 ms  BW=6071.60 GB/s
[       OK ] ReduceCopyMixedBench.MisalignedSrc1_2xFloatToFloat (42 ms)
[ RUN      ] ReduceCopyMixedBench.MisalignedSrc1_Bf16FloatToFloat

--- reduceCopyMixed: BF16 + Float -> Float, src1(float) misaligned by 1 ---
  bf16+f32->f32 (src1+1)                         nBlocks=   4  nElts=      1023  avg=0.004 ms  BW=2.48 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=  32  nElts=      8191  avg=0.004 ms  BW=18.26 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks= 256  nElts=     65535  avg=0.005 ms  BW=126.74 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=1024  nElts=    524287  avg=0.006 ms  BW=851.07 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=1024  nElts=   4194303  avg=0.010 ms  BW=4088.97 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=1024  nElts=  16777215  avg=0.034 ms  BW=4935.17 GB/s
[       OK ] ReduceCopyMixedBench.MisalignedSrc1_Bf16FloatToFloat (42 ms)
[ RUN      ] ReduceCopyMixedBench.MisalignedSrc1_Bf16FloatToBf16SR

--- reduceCopyMixed: BF16 + Float -> BF16 SR, src1(float) misaligned by 1 ---
  bf16+f32->bf16SR (src1+1)                      nBlocks=   4  nElts=      1023  avg=0.004 ms  BW=1.98 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=  32  nElts=      8191  avg=0.005 ms  BW=12.72 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks= 256  nElts=     65535  avg=0.006 ms  BW=86.19 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=1024  nElts=    524287  avg=0.006 ms  BW=681.67 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=1024  nElts=   4194303  avg=0.010 ms  BW=3269.65 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=1024  nElts=  16777215  avg=0.033 ms  BW=4096.16 GB/s
[       OK ] ReduceCopyMixedBench.MisalignedSrc1_Bf16FloatToBf16SR (42 ms)
[ RUN      ] ReduceCopyMixedBench.MisalignedAll_2xFloatToFloat

--- reduceCopyMixed: 2x Float -> Float, all misaligned (src0+1, src1+2, dst+3) ---
  2xf32->f32 (all misaligned)                    nBlocks=   4  nElts=      1021  avg=0.004 ms  BW=2.98 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=  32  nElts=      8189  avg=0.004 ms  BW=23.78 GB/s
  2xf32->f32 (all misaligned)                    nBlocks= 256  nElts=     65533  avg=0.005 ms  BW=152.37 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=1024  nElts=    524285  avg=0.006 ms  BW=1019.11 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=1024  nElts=   4194301  avg=0.014 ms  BW=3533.73 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=1024  nElts=  16777213  avg=0.042 ms  BW=4839.84 GB/s
[       OK ] ReduceCopyMixedBench.MisalignedAll_2xFloatToFloat (43 ms)
[ RUN      ] ReduceCopyMixedBench.MisalignedDst0_FloatToBf16SR

--- reduceCopyMixed: 2x Float -> BF16 SR, dst misaligned by 1 ---
  2xf32->bf16SR (dst+1)                          nBlocks=   4  nElts=      1023  avg=0.004 ms  BW=2.48 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=  32  nElts=      8191  avg=0.005 ms  BW=15.90 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks= 256  nElts=     65535  avg=0.005 ms  BW=128.14 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=1024  nElts=    524287  avg=0.006 ms  BW=852.49 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=1024  nElts=   4194303  avg=0.010 ms  BW=4089.74 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=1024  nElts=  16777215  avg=0.031 ms  BW=5407.71 GB/s
[       OK ] ReduceCopyMixedBench.MisalignedDst0_FloatToBf16SR (42 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_AlignedBaseline_2xFloatToFloat

--- block sweep: 2x Float -> Float, ALIGNED baseline (4M elts) ---
  2xf32->f32 (aligned baseline)                  nBlocks=   1  nElts=   4194304  avg=0.531 ms  BW=94.85 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=   2  nElts=   4194304  avg=0.267 ms  BW=188.64 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=   4  nElts=   4194304  avg=0.138 ms  BW=366.01 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=   8  nElts=   4194304  avg=0.072 ms  BW=702.91 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=  16  nElts=   4194304  avg=0.038 ms  BW=1322.62 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=  32  nElts=   4194304  avg=0.021 ms  BW=2440.97 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=  64  nElts=   4194304  avg=0.012 ms  BW=4086.74 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=6128.92 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=6132.50 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=6133.46 GB/s
  2xf32->f32 (aligned baseline)                  nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=6129.16 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_AlignedBaseline_2xFloatToFloat (157 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_MisalignedSrc1_2xFloatToFloat

--- block sweep: 2x Float -> Float, src1 misaligned by 1 (4M elts) ---
  2xf32->f32 (src1+1)                            nBlocks=   1  nElts=   4194303  avg=0.689 ms  BW=73.09 GB/s
  2xf32->f32 (src1+1)                            nBlocks=   2  nElts=   4194303  avg=0.352 ms  BW=143.19 GB/s
  2xf32->f32 (src1+1)                            nBlocks=   4  nElts=   4194303  avg=0.179 ms  BW=280.44 GB/s
  2xf32->f32 (src1+1)                            nBlocks=   8  nElts=   4194303  avg=0.091 ms  BW=554.44 GB/s
  2xf32->f32 (src1+1)                            nBlocks=  16  nElts=   4194303  avg=0.048 ms  BW=1053.09 GB/s
  2xf32->f32 (src1+1)                            nBlocks=  32  nElts=   4194303  avg=0.027 ms  BW=1888.67 GB/s
  2xf32->f32 (src1+1)                            nBlocks=  64  nElts=   4194303  avg=0.016 ms  BW=3069.18 GB/s
  2xf32->f32 (src1+1)                            nBlocks= 128  nElts=   4194303  avg=0.010 ms  BW=4905.39 GB/s
  2xf32->f32 (src1+1)                            nBlocks= 256  nElts=   4194303  avg=0.010 ms  BW=4908.14 GB/s
  2xf32->f32 (src1+1)                            nBlocks= 512  nElts=   4194303  avg=0.010 ms  BW=4906.15 GB/s
  2xf32->f32 (src1+1)                            nBlocks=1024  nElts=   4194303  avg=0.010 ms  BW=4907.84 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_MisalignedSrc1_2xFloatToFloat (194 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_MisalignedSrc1_Bf16FloatToFloat

--- block sweep: BF16 + Float -> Float, src1 misaligned by 1 (4M elts) ---
  bf16+f32->f32 (src1+1)                         nBlocks=   1  nElts=   4194303  avg=0.729 ms  BW=57.50 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=   2  nElts=   4194303  avg=0.374 ms  BW=112.05 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=   4  nElts=   4194303  avg=0.195 ms  BW=215.20 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=   8  nElts=   4194303  avg=0.100 ms  BW=418.35 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=  16  nElts=   4194303  avg=0.053 ms  BW=789.99 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=  32  nElts=   4194303  avg=0.029 ms  BW=1438.47 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=  64  nElts=   4194303  avg=0.018 ms  BW=2338.69 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks= 128  nElts=   4194303  avg=0.012 ms  BW=3549.78 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks= 256  nElts=   4194303  avg=0.010 ms  BW=4090.37 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks= 512  nElts=   4194303  avg=0.010 ms  BW=4083.88 GB/s
  bf16+f32->f32 (src1+1)                         nBlocks=1024  nElts=   4194303  avg=0.010 ms  BW=4090.12 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_MisalignedSrc1_Bf16FloatToFloat (207 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_MisalignedSrc1_Bf16FloatToBf16SR

--- block sweep: BF16 + Float -> BF16 SR, src1 misaligned by 1 (4M elts) ---
  bf16+f32->bf16SR (src1+1)                      nBlocks=   1  nElts=   4194303  avg=0.985 ms  BW=34.06 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=   2  nElts=   4194303  avg=0.497 ms  BW=67.54 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=   4  nElts=   4194303  avg=0.252 ms  BW=132.97 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=   8  nElts=   4194303  avg=0.129 ms  BW=259.99 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=  16  nElts=   4194303  avg=0.067 ms  BW=497.12 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=  32  nElts=   4194303  avg=0.036 ms  BW=925.01 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=  64  nElts=   4194303  avg=0.021 ms  BW=1636.28 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks= 128  nElts=   4194303  avg=0.012 ms  BW=2718.21 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks= 256  nElts=   4194303  avg=0.010 ms  BW=3268.42 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks= 512  nElts=   4194303  avg=0.010 ms  BW=3267.00 GB/s
  bf16+f32->bf16SR (src1+1)                      nBlocks=1024  nElts=   4194303  avg=0.010 ms  BW=3271.59 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_MisalignedSrc1_Bf16FloatToBf16SR (260 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_MisalignedAll_2xFloatToFloat

--- block sweep: 2x Float -> Float, all misaligned (4M elts) ---
  2xf32->f32 (all misaligned)                    nBlocks=   1  nElts=   4194301  avg=1.224 ms  BW=41.14 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=   2  nElts=   4194301  avg=0.617 ms  BW=81.60 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=   4  nElts=   4194301  avg=0.311 ms  BW=161.90 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=   8  nElts=   4194301  avg=0.159 ms  BW=317.51 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=  16  nElts=   4194301  avg=0.082 ms  BW=614.40 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=  32  nElts=   4194301  avg=0.044 ms  BW=1140.68 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=  64  nElts=   4194301  avg=0.025 ms  BW=2047.89 GB/s
  2xf32->f32 (all misaligned)                    nBlocks= 128  nElts=   4194301  avg=0.014 ms  BW=3501.32 GB/s
  2xf32->f32 (all misaligned)                    nBlocks= 256  nElts=   4194301  avg=0.014 ms  BW=3504.52 GB/s
  2xf32->f32 (all misaligned)                    nBlocks= 512  nElts=   4194301  avg=0.014 ms  BW=3507.02 GB/s
  2xf32->f32 (all misaligned)                    nBlocks=1024  nElts=   4194301  avg=0.014 ms  BW=3609.97 GB/s
[       OK ] ReduceCopyMixedBench.BlockSweep_MisalignedAll_2xFloatToFloat (312 ms)
[ RUN      ] ReduceCopyMixedBench.BlockSweep_MisalignedDst0_FloatToBf16SR

--- block sweep: 2x Float -> BF16 SR, dst misaligned by 1 (4M elts) ---
  2xf32->bf16SR (dst+1)                          nBlocks=   1  nElts=   4194303  avg=1.095 ms  BW=38.32 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=   2  nElts=   4194303  avg=0.548 ms  BW=76.47 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=   4  nElts=   4194303  avg=0.278 ms  BW=150.64 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=   8  nElts=   4194303  avg=0.141 ms  BW=296.63 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=  16  nElts=   4194303  avg=0.073 ms  BW=575.43 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=  32  nElts=   4194303  avg=0.039 ms  BW=1077.81 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=  64  nElts=   4194303  avg=0.022 ms  BW=1921.06 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks= 128  nElts=   4194303  avg=0.013 ms  BW=3322.57 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks= 256  nElts=   4194303  avg=0.010 ms  BW=4076.89 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks= 512  nElts=   4194303  avg=0.010 ms  BW=4091.14 GB/s
  2xf32->bf16SR (dst+1)                          nBlocks=1024  nElts=   4194303  avg=0.010 ms  BW=4089.35 GB/s
*/
