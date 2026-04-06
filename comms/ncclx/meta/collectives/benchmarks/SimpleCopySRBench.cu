// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
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

// Direct reduceCopyPacksSR wrapper — bypasses the multi-pass alignment
// logic in reduceCopySR and calls the inner loop directly.
// Generic over AccType, SrcType, DstType.
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename SrcType>
__global__ __launch_bounds__(256, 1) void reduce_copy_packs_sr_kernel(
    DstType* dst,
    const SrcType* src,
    ssize_t nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  ssize_t nEltsBehind = 0;
  ssize_t nEltsAhead = nElts;
  meta::comms::ncclx::kernels::simplecopy::
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

// Direct reduceCopyPacks wrapper (baseline, RTN) — same shape as above.
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename SrcType>
__global__ __launch_bounds__(256, 1) void reduce_copy_packs_kernel(
    DstType* dst,
    const SrcType* src,
    ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  ssize_t nEltsBehind = 0;
  ssize_t nEltsAhead = nElts;
  meta::comms::ncclx::kernels::simplecopy::
      reduceCopyPacks<Unroll, EltPerPack, AccType, 1>(
          nThreads, thread, nEltsBehind, nEltsAhead, src, dst);
}

// 2-src direct reduceCopyPacksSR wrapper.
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(256, 1) void reduce_copy_packs_sr_2src_kernel(
    DstType* dst,
    const Src0Type* src0,
    const Src1Type* src1,
    ssize_t nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  ssize_t nEltsBehind = 0;
  ssize_t nEltsAhead = nElts;
  meta::comms::ncclx::kernels::simplecopy::
      reduceCopyPacksSR<Unroll, EltPerPack, AccType, 1>(
          nThreads,
          thread,
          nEltsBehind,
          nEltsAhead,
          randomSeed,
          randomBaseOffset,
          src0,
          src1,
          dst);
}

// 2-src direct reduceCopyPacks wrapper (baseline, RTN).
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(256, 1) void reduce_copy_packs_2src_kernel(
    DstType* dst,
    const Src0Type* src0,
    const Src1Type* src1,
    ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  ssize_t nEltsBehind = 0;
  ssize_t nEltsAhead = nElts;
  meta::comms::ncclx::kernels::simplecopy::
      reduceCopyPacks<Unroll, EltPerPack, AccType, 1>(
          nThreads, thread, nEltsBehind, nEltsAhead, src0, src1, dst);
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

// 2-src reduceCopyMixed wrapper (for baseline comparison)
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

class SimpleCopySRBench : public ::testing::Test {
 public:
  static constexpr int64_t kN = 4L * 1024L * 1024L; // 4M elements
  static constexpr int kBlockSize = 256;
  static constexpr int kDefaultBlocks = 32;
  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;
  static constexpr uint64_t kSeed = 42;
  static constexpr uint64_t kBaseOffset = 0;

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
// 1-src: FP32 -> BF16 with SR vs baseline
// =============================================================================

TEST_F(SimpleCopySRBench, OneSrc_FloatToBf16_SR) {
  printf("\n--- reduceCopySR 1-src: FP32 -> BF16 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = kN * sizeof(float) + kN * sizeof(__nv_bfloat16);

  // Baseline: reduceCopyMixed (round-to-nearest)
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_1src_kernel<4, float, __nv_bfloat16, float>
            <<<nBlk, blockSize>>>(d_dstBf16, d_srcFloat0, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "baseline: 1-src f32 -> bf16 (RTN)",
      totalBytes);

  // SR: reduceCopySR (stochastic rounding)
  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, float>
            <<<nBlk, blockSize>>>(
                d_dstBf16, d_srcFloat0, (ssize_t)nElts, kSeed, kBaseOffset);
        CUDACHECK(cudaGetLastError());
      },
      "SR:       1-src f32 -> bf16",
      totalBytes);
}

// =============================================================================
// 1-src: BF16 -> BF16 with SR (AccType=FP32)
// =============================================================================

TEST_F(SimpleCopySRBench, OneSrc_Bf16ToBf16_SR) {
  printf("\n--- reduceCopySR 1-src: BF16 -> BF16 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = kN * sizeof(__nv_bfloat16) + kN * sizeof(__nv_bfloat16);

  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_1src_kernel<4, float, __nv_bfloat16, __nv_bfloat16>
            <<<nBlk, blockSize>>>(d_dstBf16, d_srcBf16_0, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "baseline: 1-src bf16 -> bf16 (RTN)",
      totalBytes);

  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_sr_1src_kernel<4, float, __nv_bfloat16, __nv_bfloat16>
            <<<nBlk, blockSize>>>(
                d_dstBf16, d_srcBf16_0, (ssize_t)nElts, kSeed, kBaseOffset);
        CUDACHECK(cudaGetLastError());
      },
      "SR:       1-src bf16 -> bf16",
      totalBytes);
}

// =============================================================================
// 2-src: FP32+FP32 -> BF16 with SR
// =============================================================================

TEST_F(SimpleCopySRBench, TwoSrc_FloatFloat_Bf16Dst_SR) {
  printf(
      "\n--- reduceCopySR 2-src: FP32+FP32 -> BF16 (32 blocks, 4M elts) ---\n");
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
      "baseline: 2-src f32+f32 -> bf16 (RTN)",
      totalBytes);

  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_sr_2src_kernel<4, float, __nv_bfloat16, float, float>
            <<<nBlk, blockSize>>>(
                d_dstBf16,
                d_srcFloat0,
                d_srcFloat1,
                (ssize_t)nElts,
                kSeed,
                kBaseOffset);
        CUDACHECK(cudaGetLastError());
      },
      "SR:       2-src f32+f32 -> bf16",
      totalBytes);
}

// =============================================================================
// 2-src: BF16+BF16 -> BF16 with SR
// =============================================================================

TEST_F(SimpleCopySRBench, TwoSrc_Bf16Bf16_Bf16Dst_SR) {
  printf(
      "\n--- reduceCopySR 2-src: BF16+BF16 -> BF16 (32 blocks, 4M elts) ---\n");
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
      "baseline: 2-src bf16+bf16 -> bf16 (RTN)",
      totalBytes);

  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_sr_2src_kernel<
            4,
            float,
            __nv_bfloat16,
            __nv_bfloat16,
            __nv_bfloat16><<<nBlk, blockSize>>>(
            d_dstBf16,
            d_srcBf16_0,
            d_srcBf16_1,
            (ssize_t)nElts,
            kSeed,
            kBaseOffset);
        CUDACHECK(cudaGetLastError());
      },
      "SR:       2-src bf16+bf16 -> bf16",
      totalBytes);
}

// =============================================================================
// 2-src: BF16+FP32 -> BF16 with SR
// =============================================================================

TEST_F(SimpleCopySRBench, TwoSrc_Bf16Float_Bf16Dst_SR) {
  printf(
      "\n--- reduceCopySR 2-src: BF16+FP32 -> BF16 (32 blocks, 4M elts) ---\n");
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
      "baseline: 2-src bf16+f32 -> bf16 (RTN)",
      totalBytes);

  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_sr_2src_kernel<
            4,
            float,
            __nv_bfloat16,
            __nv_bfloat16,
            float><<<nBlk, blockSize>>>(
            d_dstBf16,
            d_srcBf16_0,
            d_srcFloat1,
            (ssize_t)nElts,
            kSeed,
            kBaseOffset);
        CUDACHECK(cudaGetLastError());
      },
      "SR:       2-src bf16+f32 -> bf16",
      totalBytes);
}

// =============================================================================
// 1-src: FP32 -> FP32 with SR (no-op SR path, verify no overhead)
// =============================================================================

TEST_F(SimpleCopySRBench, OneSrc_FloatToFloat_SR) {
  printf("\n--- reduceCopySR 1-src: FP32 -> FP32 (32 blocks, 4M elts) ---\n");
  size_t totalBytes = kN * sizeof(float) + kN * sizeof(float);

  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_mixed_1src_kernel<4, float, float, float>
            <<<nBlk, blockSize>>>(d_dstFloat, d_srcFloat0, (ssize_t)nElts);
        CUDACHECK(cudaGetLastError());
      },
      "baseline: 1-src f32 -> f32 (RTN)",
      totalBytes);

  runBenchCore(
      kN,
      kDefaultBlocks,
      [&](int nBlk, int blockSize, int64_t nElts) {
        reduce_copy_sr_1src_kernel<4, float, float, float><<<nBlk, blockSize>>>(
            d_dstFloat, d_srcFloat0, (ssize_t)nElts, kSeed, kBaseOffset);
        CUDACHECK(cudaGetLastError());
      },
      "SR:       1-src f32 -> f32 (no-op SR)",
      totalBytes);
}

// =============================================================================
// Direct reduceCopyPacksSR benchmarks: Unroll x EltPerPack matrix.
// Calls the inner loop directly (no multi-pass alignment logic).
// Each test covers a different src→dst type combination.
// =============================================================================

// Helper: run the full Unroll={1,2,4,8} x EltPerPack={1,2,4} matrix for a
// given AccType/SrcType/DstType combination.
template <typename AccType, typename DstType, typename SrcType>
void runPacksMatrix(
    SimpleCopySRBench* self,
    SrcType* d_src,
    DstType* d_dst,
    const char* tag) {
  size_t totalBytes = self->kN * sizeof(SrcType) + self->kN * sizeof(DstType);

  auto benchOne = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;

    char labelRTN[80];
    char labelSR[80];
    snprintf(
        labelRTN,
        sizeof(labelRTN),
        "baseline: Packs RTN  U=%d EPP=%d  %s",
        U,
        EPP,
        tag);
    snprintf(
        labelSR,
        sizeof(labelSR),
        "SR:       PacksSR    U=%d EPP=%d  %s",
        U,
        EPP,
        tag);

    self->runBenchCore(
        self->kN,
        self->kDefaultBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          reduce_copy_packs_kernel<U, EPP, AccType>
              <<<nBlk, blockSize>>>(d_dst, d_src, (ssize_t)nElts);
          auto err = cudaGetLastError();
          if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
          }
        },
        labelRTN,
        totalBytes);

    self->runBenchCore(
        self->kN,
        self->kDefaultBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          reduce_copy_packs_sr_kernel<U, EPP, AccType><<<nBlk, blockSize>>>(
              d_dst, d_src, (ssize_t)nElts, self->kSeed, self->kBaseOffset);
          auto err = cudaGetLastError();
          if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
          }
        },
        labelSR,
        totalBytes);
  };

  // Unroll=1
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 4>{});

  // Unroll=2
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{});

  // Unroll=4
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});

  // Unroll=8
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{});
}

// --- FP32 src -> BF16 dst (AccType=float, SR active) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_FloatToBf16) {
  printf(
      "\n--- reduceCopyPacksSR matrix: FP32 -> BF16 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix<float>(this, d_srcFloat0, d_dstBf16, "f32->bf16");
}

// --- BF16 src -> BF16 dst (AccType=float, SR active) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_Bf16ToBf16) {
  printf(
      "\n--- reduceCopyPacksSR matrix: BF16 -> BF16 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix<float>(this, d_srcBf16_0, d_dstBf16, "bf16->bf16");
}

// --- FP32 src -> FP32 dst (AccType=float, SR no-op) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_FloatToFloat) {
  printf(
      "\n--- reduceCopyPacksSR matrix: FP32 -> FP32 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix<float>(this, d_srcFloat0, d_dstFloat, "f32->f32");
}

// =============================================================================
// 2-src direct reduceCopyPacksSR benchmarks: Unroll x EltPerPack matrix.
// =============================================================================

// Helper: run the full Unroll={1,2,4,8} x EltPerPack={1,2,4} matrix for a
// given 2-src AccType/Src0Type/Src1Type/DstType combination.
template <
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
void runPacksMatrix2Src(
    SimpleCopySRBench* self,
    Src0Type* d_src0,
    Src1Type* d_src1,
    DstType* d_dst,
    const char* tag) {
  size_t totalBytes = self->kN * sizeof(Src0Type) +
      self->kN * sizeof(Src1Type) + self->kN * sizeof(DstType);

  auto benchOne = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;

    char labelRTN[96];
    char labelSR[96];
    snprintf(
        labelRTN,
        sizeof(labelRTN),
        "baseline: Packs RTN  U=%d EPP=%d  %s",
        U,
        EPP,
        tag);
    snprintf(
        labelSR,
        sizeof(labelSR),
        "SR:       PacksSR    U=%d EPP=%d  %s",
        U,
        EPP,
        tag);

    self->runBenchCore(
        self->kN,
        self->kDefaultBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          reduce_copy_packs_2src_kernel<U, EPP, AccType>
              <<<nBlk, blockSize>>>(d_dst, d_src0, d_src1, (ssize_t)nElts);
          auto err = cudaGetLastError();
          if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
          }
        },
        labelRTN,
        totalBytes);

    self->runBenchCore(
        self->kN,
        self->kDefaultBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          reduce_copy_packs_sr_2src_kernel<U, EPP, AccType>
              <<<nBlk, blockSize>>>(
                  d_dst,
                  d_src0,
                  d_src1,
                  (ssize_t)nElts,
                  self->kSeed,
                  self->kBaseOffset);
          auto err = cudaGetLastError();
          if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
          }
        },
        labelSR,
        totalBytes);
  };

  // Unroll=1
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 4>{});

  // Unroll=2
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{});

  // Unroll=4
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});

  // Unroll=8
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{});
}

// --- 2-src: FP32+FP32 -> BF16 (SR active) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_2src_FloatFloat_Bf16) {
  printf(
      "\n--- reduceCopyPacksSR 2-src matrix: FP32+FP32 -> BF16 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix2Src<float>(
      this, d_srcFloat0, d_srcFloat1, d_dstBf16, "f32+f32->bf16");
}

// --- 2-src: FP32+FP32 -> FP32 (SR no-op) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_2src_FloatFloat_Float) {
  printf(
      "\n--- reduceCopyPacksSR 2-src matrix: FP32+FP32 -> FP32 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix2Src<float>(
      this, d_srcFloat0, d_srcFloat1, d_dstFloat, "f32+f32->f32");
}

// --- 2-src: BF16+BF16 -> BF16 (SR active) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_2src_Bf16Bf16_Bf16) {
  printf(
      "\n--- reduceCopyPacksSR 2-src matrix: BF16+BF16 -> BF16 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix2Src<float>(
      this, d_srcBf16_0, d_srcBf16_1, d_dstBf16, "bf16+bf16->bf16");
}

// --- 2-src: BF16+BF16 -> FP32 (SR no-op) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_2src_Bf16Bf16_Float) {
  printf(
      "\n--- reduceCopyPacksSR 2-src matrix: BF16+BF16 -> FP32 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix2Src<float>(
      this, d_srcBf16_0, d_srcBf16_1, d_dstFloat, "bf16+bf16->f32");
}

// --- 2-src: BF16+FP32 -> BF16 (SR active) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_2src_Bf16Float_Bf16) {
  printf(
      "\n--- reduceCopyPacksSR 2-src matrix: BF16+FP32 -> BF16 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix2Src<float>(
      this, d_srcBf16_0, d_srcFloat1, d_dstBf16, "bf16+f32->bf16");
}

// --- 2-src: BF16+FP32 -> FP32 (SR no-op) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_2src_Bf16Float_Float) {
  printf(
      "\n--- reduceCopyPacksSR 2-src matrix: BF16+FP32 -> FP32 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix2Src<float>(
      this, d_srcBf16_0, d_srcFloat1, d_dstFloat, "bf16+f32->f32");
}

// --- 2-src: FP32+BF16 -> BF16 (SR active) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_2src_FloatBf16_Bf16) {
  printf(
      "\n--- reduceCopyPacksSR 2-src matrix: FP32+BF16 -> BF16 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix2Src<float>(
      this, d_srcFloat0, d_srcBf16_1, d_dstBf16, "f32+bf16->bf16");
}

// --- 2-src: FP32+BF16 -> FP32 (SR no-op) ---
TEST_F(SimpleCopySRBench, PacksSR_Matrix_2src_FloatBf16_Float) {
  printf(
      "\n--- reduceCopyPacksSR 2-src matrix: FP32+BF16 -> FP32 "
      "(32 blocks, 4M elts) ---\n");
  runPacksMatrix2Src<float>(
      this, d_srcFloat0, d_srcBf16_1, d_dstFloat, "f32+bf16->f32");
}
