// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "meta/collectives/kernels/reduce_copy.cuh"
#include "meta/collectives/kernels/reduce_copy_sr.cuh"
#include "meta/collectives/kernels/reduce_copy_sr_v2.cuh"

// =============================================================================
// Wrapper Kernels
// =============================================================================

// V1 direct reduceCopyPacksSR
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename SrcType>
__global__ __launch_bounds__(256, 1) void v1_packs_sr_kernel(
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

// V2 direct reduceCopyPacksSR
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename SrcType>
__global__ __launch_bounds__(256, 1) void v2_packs_sr_kernel(
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

// V1 top-level reduceCopySR
template <int Unroll, typename AccType, typename DstType, typename SrcType>
__global__ __launch_bounds__(256, 1) void v1_sr_kernel(
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

// V2 top-level reduceCopySR
template <int Unroll, typename AccType, typename DstType, typename SrcType>
__global__ __launch_bounds__(256, 1) void v2_sr_kernel(
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

// V1 2-src top-level reduceCopySR
template <
    int Unroll,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(256, 1) void v1_sr_2src_kernel(
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

// V2 2-src top-level reduceCopySR
template <
    int Unroll,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(256, 1) void v2_sr_2src_kernel(
    DstType* dst,
    const Src0Type* src0,
    const Src1Type* src1,
    ssize_t nElts,
    uint64_t randomSeed,
    uint64_t randomBaseOffset) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy_v2::reduceCopySR<Unroll, AccType>(
      thread, nThreads, dst, nElts, randomSeed, randomBaseOffset, src0, src1);
}

// V1 2-src direct reduceCopyPacksSR
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(256, 1) void v1_packs_sr_2src_kernel(
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

// V2 2-src direct reduceCopyPacksSR
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(256, 1) void v2_packs_sr_2src_kernel(
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
  meta::comms::ncclx::kernels::simplecopy_v2::
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

// 1-src baseline (RTN, no SR)
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename SrcType>
__global__ __launch_bounds__(256, 1) void baseline_packs_kernel(
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

// 2-src baseline (RTN, no SR)
template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(256, 1) void baseline_packs_2src_kernel(
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

// ---- 640-thread kernel wrappers for block-count sweep benchmarks ----

template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename SrcType>
__global__ __launch_bounds__(640, 1) void v1_packs_sr_kernel_640(
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

template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename SrcType>
__global__ __launch_bounds__(640, 1) void v2_packs_sr_kernel_640(
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

template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename SrcType>
__global__ __launch_bounds__(640, 1) void baseline_packs_kernel_640(
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

template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(640, 1) void v1_packs_sr_2src_kernel_640(
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

template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(640, 1) void v2_packs_sr_2src_kernel_640(
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
  meta::comms::ncclx::kernels::simplecopy_v2::
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

template <
    int Unroll,
    int EltPerPack,
    typename AccType,
    typename DstType,
    typename Src0Type,
    typename Src1Type>
__global__ __launch_bounds__(640, 1) void baseline_packs_2src_kernel_640(
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

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Benchmark Fixture
// =============================================================================

class SimpleCopySRV2Bench : public ::testing::Test {
 public:
  static constexpr int64_t kN = 64L * 1024L * 1024L;
  static constexpr int kBlockSize = 256;
  static constexpr int kDefaultBlocks = 32;
  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;
  static constexpr uint64_t kSeed = 42;
  static constexpr uint64_t kBaseOffset = 0;

  // L2 flush buffer: write 64 MB (> H100's 50 MB L2) to evict all cached lines
  static constexpr size_t kL2FlushSize = 64ULL * 1024 * 1024;

  float* d_srcFloat0 = nullptr;
  float* d_srcFloat1 = nullptr;
  __nv_bfloat16* d_srcBf16 = nullptr;
  __nv_bfloat16* d_dstBf16 = nullptr;
  float* d_dstFloat = nullptr;
  uint8_t* d_l2Flush = nullptr;
  cudaEvent_t startEvent, stopEvent;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_srcFloat0, kN * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcFloat1, kN * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_srcBf16, kN * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_dstBf16, kN * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_dstFloat, kN * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_l2Flush, kL2FlushSize));

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
        d_srcBf16,
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
    CUDACHECK(cudaFree(d_srcBf16));
    CUDACHECK(cudaFree(d_dstBf16));
    CUDACHECK(cudaFree(d_dstFloat));
    CUDACHECK(cudaFree(d_l2Flush));
  }

  // Evict all L2 cache lines by writing a buffer larger than L2.
  void flushL2() {
    CUDACHECK(cudaMemsetAsync(d_l2Flush, 0, kL2FlushSize));
    CUDACHECK(cudaDeviceSynchronize());
  }

  template <typename LaunchFn>
  void runBenchCore(
      int64_t nElts,
      int nBlocks,
      LaunchFn launchFn,
      const char* label,
      size_t totalBytes,
      int blockSize = kBlockSize) {
    // Warmup
    for (int i = 0; i < kWarmupIters; i++) {
      flushL2();
      launchFn(nBlocks, blockSize, nElts);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark: per-iteration timing with L2 flush between launches
    float totalMs = 0.0f;
    for (int i = 0; i < kBenchIters; i++) {
      flushL2();
      CUDACHECK(cudaEventRecord(startEvent));
      launchFn(nBlocks, blockSize, nElts);
      CUDACHECK(cudaEventRecord(stopEvent));
      CUDACHECK(cudaDeviceSynchronize());
      float ms = 0.0f;
      CUDACHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
      totalMs += ms;
    }
    float avgMs = totalMs / kBenchIters;

    double gbPerSec = (double)totalBytes / (avgMs * 1e6);
    printf(
        "  %-62s  nBlocks=%4d  avg=%.3f ms  BW=%.2f GB/s\n",
        label,
        nBlocks,
        avgMs,
        gbPerSec);
  }
};

// =============================================================================
// Top-level reduceCopySR: V1 vs V2, FP32 → BF16
// =============================================================================

TEST_F(SimpleCopySRV2Bench, TopLevel_FloatToBf16) {
  printf("\n--- reduceCopySR V1 vs V2: FP32 -> BF16 (64M elts) ---\n");
  // Inner passes for AccType=float: BigPack(U,EPP=4), SmallPack(U*2,EPP=1)
  // Exchange: BigPack at U>=2, SmallPack at U*2>=8 i.e. U>=4
  size_t totalBytes = kN * sizeof(float) + kN * sizeof(__nv_bfloat16);

  auto benchUnroll = [&](auto unrollTag) {
    constexpr int U = decltype(unrollTag)::value;
    // BigPack: exch when U>=2; SmallPack: exch when U*2>=8
    constexpr bool bigExch = (U >= 2);
    constexpr bool smallExch = (U * 2 >= 8);
    char v1Label[96], v2Label[96];
    snprintf(v1Label, sizeof(v1Label), "V1 reduceCopySR U=%d f32->bf16", U);
    snprintf(
        v2Label,
        sizeof(v2Label),
        "V2 reduceCopySR U=%d f32->bf16 [big:%s small:%s]",
        U,
        bigExch ? "exch" : "simple",
        smallExch ? "exch" : "simple");

    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v1_sr_kernel<U, float, __nv_bfloat16, float><<<nBlk, bs>>>(
              d_dstBf16, d_srcFloat0, (ssize_t)n, kSeed, kBaseOffset);
        },
        v1Label,
        totalBytes);
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v2_sr_kernel<U, float, __nv_bfloat16, float><<<nBlk, bs>>>(
              d_dstBf16, d_srcFloat0, (ssize_t)n, kSeed, kBaseOffset);
        },
        v2Label,
        totalBytes);
  };

  benchUnroll(std::integral_constant<int, 1>{});
  benchUnroll(std::integral_constant<int, 2>{});
  benchUnroll(std::integral_constant<int, 4>{});
  benchUnroll(std::integral_constant<int, 8>{});
}

// =============================================================================
// Direct reduceCopyPacksSR: V1 vs V2, full (Unroll, EltPerPack) matrix
// =============================================================================

TEST_F(SimpleCopySRV2Bench, PacksMatrix_FloatToBf16) {
  printf(
      "\n--- reduceCopyPacksSR V1 vs V2: FP32 -> BF16 "
      "(Unroll x EltPerPack, 64M elts) ---\n");
  size_t totalBytes = kN * sizeof(float) + kN * sizeof(__nv_bfloat16);

  auto benchOne = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;
    constexpr int G = 8 / EPP;
    constexpr bool exch = (U >= G);
    char rtnLabel[96], v1Label[96], v2Label[96];
    snprintf(rtnLabel, sizeof(rtnLabel), "RTN   U=%d EPP=%d f32->bf16", U, EPP);
    snprintf(v1Label, sizeof(v1Label), "V1 SR U=%d EPP=%d f32->bf16", U, EPP);
    snprintf(
        v2Label,
        sizeof(v2Label),
        "V2 SR U=%d EPP=%d f32->bf16 [%s]",
        U,
        EPP,
        exch ? "exch" : "simple");

    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          baseline_packs_kernel<U, EPP, float, __nv_bfloat16, float>
              <<<nBlk, bs>>>(d_dstBf16, d_srcFloat0, (ssize_t)n);
        },
        rtnLabel,
        totalBytes);
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v1_packs_sr_kernel<U, EPP, float, __nv_bfloat16, float><<<nBlk, bs>>>(
              d_dstBf16, d_srcFloat0, (ssize_t)n, kSeed, kBaseOffset);
        },
        v1Label,
        totalBytes);
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v2_packs_sr_kernel<U, EPP, float, __nv_bfloat16, float><<<nBlk, bs>>>(
              d_dstBf16, d_srcFloat0, (ssize_t)n, kSeed, kBaseOffset);
        },
        v2Label,
        totalBytes);
    printf("\n");
  };

  // EPP=4
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 4>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{});

  // EPP=2
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 2>{});

  // EPP=1
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
}

// =============================================================================
// Direct reduceCopyPacksSR 2-src: V1 vs V2, FP32+FP32 -> BF16
// =============================================================================

TEST_F(SimpleCopySRV2Bench, PacksMatrix_2src_FloatFloat_Bf16) {
  printf(
      "\n--- reduceCopyPacksSR 2-src V1 vs V2: FP32+FP32 -> BF16 "
      "(Unroll x EltPerPack, 64M elts) ---\n");
  size_t totalBytes = 2 * kN * sizeof(float) + kN * sizeof(__nv_bfloat16);

  auto benchOne = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;
    constexpr int G = 8 / EPP;
    constexpr bool exch = (U >= G);
    char rtnLabel[96], v1Label[96], v2Label[96];
    snprintf(
        rtnLabel,
        sizeof(rtnLabel),
        "RTN   U=%d EPP=%d 2-src f32+f32->bf16",
        U,
        EPP);
    snprintf(
        v1Label,
        sizeof(v1Label),
        "V1 SR U=%d EPP=%d 2-src f32+f32->bf16",
        U,
        EPP);
    snprintf(
        v2Label,
        sizeof(v2Label),
        "V2 SR U=%d EPP=%d 2-src f32+f32->bf16 [%s]",
        U,
        EPP,
        exch ? "exch" : "simple");

    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          baseline_packs_2src_kernel<U, EPP, float, __nv_bfloat16, float, float>
              <<<nBlk, bs>>>(d_dstBf16, d_srcFloat0, d_srcFloat1, (ssize_t)n);
        },
        rtnLabel,
        totalBytes);
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v1_packs_sr_2src_kernel<U, EPP, float, __nv_bfloat16, float, float>
              <<<nBlk, bs>>>(
                  d_dstBf16,
                  d_srcFloat0,
                  d_srcFloat1,
                  (ssize_t)n,
                  kSeed,
                  kBaseOffset);
        },
        v1Label,
        totalBytes);
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v2_packs_sr_2src_kernel<U, EPP, float, __nv_bfloat16, float, float>
              <<<nBlk, bs>>>(
                  d_dstBf16,
                  d_srcFloat0,
                  d_srcFloat1,
                  (ssize_t)n,
                  kSeed,
                  kBaseOffset);
        },
        v2Label,
        totalBytes);
    printf("\n");
  };

  // EPP=4
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 4>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{});

  // EPP=2
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 2>{});

  // EPP=1
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
}

// =============================================================================
// Direct reduceCopyPacksSR 2-src: V1 vs V2, FP32+BF16 -> BF16
// =============================================================================

TEST_F(SimpleCopySRV2Bench, PacksMatrix_2src_FloatBf16_Bf16) {
  printf(
      "\n--- reduceCopyPacksSR 2-src V1 vs V2: FP32+BF16 -> BF16 "
      "(Unroll x EltPerPack, 64M elts) ---\n");
  size_t totalBytes = kN * sizeof(float) + 2 * kN * sizeof(__nv_bfloat16);

  auto benchOne = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;
    constexpr int G = 8 / EPP;
    constexpr bool exch = (U >= G);
    char rtnLabel[96], v1Label[96], v2Label[96];
    snprintf(
        rtnLabel,
        sizeof(rtnLabel),
        "RTN   U=%d EPP=%d 2-src f32+bf16->bf16",
        U,
        EPP);
    snprintf(
        v1Label,
        sizeof(v1Label),
        "V1 SR U=%d EPP=%d 2-src f32+bf16->bf16",
        U,
        EPP);
    snprintf(
        v2Label,
        sizeof(v2Label),
        "V2 SR U=%d EPP=%d 2-src f32+bf16->bf16 [%s]",
        U,
        EPP,
        exch ? "exch" : "simple");

    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          baseline_packs_2src_kernel<
              U,
              EPP,
              float,
              __nv_bfloat16,
              float,
              __nv_bfloat16>
              <<<nBlk, bs>>>(d_dstBf16, d_srcFloat0, d_srcBf16, (ssize_t)n);
        },
        rtnLabel,
        totalBytes);
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v1_packs_sr_2src_kernel<
              U,
              EPP,
              float,
              __nv_bfloat16,
              float,
              __nv_bfloat16><<<nBlk, bs>>>(
              d_dstBf16,
              d_srcFloat0,
              d_srcBf16,
              (ssize_t)n,
              kSeed,
              kBaseOffset);
        },
        v1Label,
        totalBytes);
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v2_packs_sr_2src_kernel<
              U,
              EPP,
              float,
              __nv_bfloat16,
              float,
              __nv_bfloat16><<<nBlk, bs>>>(
              d_dstBf16,
              d_srcFloat0,
              d_srcBf16,
              (ssize_t)n,
              kSeed,
              kBaseOffset);
        },
        v2Label,
        totalBytes);
    printf("\n");
  };

  // EPP=4
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 4>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{});

  // EPP=2
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 2>{});

  // EPP=1
  benchOne(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{});
  benchOne(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
}

// =============================================================================
// FP32 -> FP32 (no-op SR path): verify no overhead from V2 changes
// =============================================================================

TEST_F(SimpleCopySRV2Bench, NoopSR_FloatToFloat) {
  printf(
      "\n--- No-op SR path (f32->f32, 64M elts): V1 vs V2 overhead check ---\n");
  size_t totalBytes = 2 * kN * sizeof(float);

  auto benchUnroll = [&](auto unrollTag) {
    constexpr int U = decltype(unrollTag)::value;
    char v1Label[80], v2Label[80];
    snprintf(v1Label, sizeof(v1Label), "V1 SR U=%d f32->f32 (no-op)", U);
    snprintf(v2Label, sizeof(v2Label), "V2 SR U=%d f32->f32 (no-op)", U);

    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v1_sr_kernel<U, float, float, float><<<nBlk, bs>>>(
              d_dstFloat, d_srcFloat0, (ssize_t)n, kSeed, kBaseOffset);
        },
        v1Label,
        totalBytes);
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v2_sr_kernel<U, float, float, float><<<nBlk, bs>>>(
              d_dstFloat, d_srcFloat0, (ssize_t)n, kSeed, kBaseOffset);
        },
        v2Label,
        totalBytes);
  };

  benchUnroll(std::integral_constant<int, 4>{});
  benchUnroll(std::integral_constant<int, 8>{});
}

// =============================================================================
// 2-src: FP32 + FP32 -> BF16 with SR
// =============================================================================

TEST_F(SimpleCopySRV2Bench, TwoSrc_FloatFloat_Bf16) {
  printf(
      "\n--- reduceCopySR 2-src V1 vs V2: FP32+FP32 -> BF16 (64M elts) ---\n");
  size_t totalBytes = 2 * kN * sizeof(float) + kN * sizeof(__nv_bfloat16);

  auto benchUnroll = [&](auto unrollTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr bool bigExch = (U >= 2);
    constexpr bool smallExch = (U * 2 >= 8);
    char v1Label[96], v2Label[96];
    snprintf(v1Label, sizeof(v1Label), "V1 SR U=%d 2-src f32+f32->bf16", U);
    snprintf(
        v2Label,
        sizeof(v2Label),
        "V2 SR U=%d 2-src f32+f32->bf16 [big:%s small:%s]",
        U,
        bigExch ? "exch" : "simple",
        smallExch ? "exch" : "simple");

    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v1_sr_2src_kernel<U, float, __nv_bfloat16, float, float>
              <<<nBlk, bs>>>(
                  d_dstBf16,
                  d_srcFloat0,
                  d_srcFloat1,
                  (ssize_t)n,
                  kSeed,
                  kBaseOffset);
        },
        v1Label,
        totalBytes);
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v2_sr_2src_kernel<U, float, __nv_bfloat16, float, float>
              <<<nBlk, bs>>>(
                  d_dstBf16,
                  d_srcFloat0,
                  d_srcFloat1,
                  (ssize_t)n,
                  kSeed,
                  kBaseOffset);
        },
        v2Label,
        totalBytes);
  };

  benchUnroll(std::integral_constant<int, 1>{});
  benchUnroll(std::integral_constant<int, 2>{});
  benchUnroll(std::integral_constant<int, 4>{});
  benchUnroll(std::integral_constant<int, 8>{});
}

// =============================================================================
// 2-src: FP32 + BF16 -> BF16 with SR
// =============================================================================

TEST_F(SimpleCopySRV2Bench, TwoSrc_FloatBf16_Bf16) {
  printf(
      "\n--- reduceCopySR 2-src V1 vs V2: FP32+BF16 -> BF16 (64M elts) ---\n");
  size_t totalBytes = kN * sizeof(float) + 2 * kN * sizeof(__nv_bfloat16);

  auto benchIt = [&](auto unrollTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr bool bigExch = (U >= 2);
    constexpr bool smallExch = (U * 2 >= 8);
    char v1Label[96], v2Label[96];
    snprintf(v1Label, sizeof(v1Label), "V1 SR U=%d 2-src f32+bf16->bf16", U);
    snprintf(
        v2Label,
        sizeof(v2Label),
        "V2 SR U=%d 2-src f32+bf16->bf16 [big:%s small:%s]",
        U,
        bigExch ? "exch" : "simple",
        smallExch ? "exch" : "simple");

    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v1_sr_2src_kernel<U, float, __nv_bfloat16, float, __nv_bfloat16>
              <<<nBlk, bs>>>(
                  d_dstBf16,
                  d_srcFloat0,
                  d_srcBf16,
                  (ssize_t)n,
                  kSeed,
                  kBaseOffset);
        },
        v1Label,
        totalBytes);
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int bs, int64_t n) {
          v2_sr_2src_kernel<U, float, __nv_bfloat16, float, __nv_bfloat16>
              <<<nBlk, bs>>>(
                  d_dstBf16,
                  d_srcFloat0,
                  d_srcBf16,
                  (ssize_t)n,
                  kSeed,
                  kBaseOffset);
        },
        v2Label,
        totalBytes);
  };

  benchIt(std::integral_constant<int, 1>{});
  benchIt(std::integral_constant<int, 2>{});
  benchIt(std::integral_constant<int, 4>{});
  benchIt(std::integral_constant<int, 8>{});
}

// =============================================================================
// Block-count sweep: blockSize=640, numBlocks=4..32
// Two configs: (U=4,EPP=4) [exch] and (U=8,EPP=1) [exch]
// =============================================================================

TEST_F(SimpleCopySRV2Bench, BlockSweep_1src_FloatToBf16) {
  printf("\n--- Block sweep (bs=640): 1-src FP32 -> BF16 (64M elts) ---\n");
  constexpr int kBS = 640;
  size_t totalBytes = kN * sizeof(float) + kN * sizeof(__nv_bfloat16);

  auto sweep = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;
    constexpr int G = 8 / EPP;
    constexpr bool exch = (U >= G);

    for (int nBlk = 4; nBlk <= 32; nBlk *= 2) {
      char rtnLabel[96], v1Label[96], v2Label[96];
      snprintf(
          rtnLabel, sizeof(rtnLabel), "RTN   U=%d EPP=%d f32->bf16", U, EPP);
      snprintf(v1Label, sizeof(v1Label), "V1 SR U=%d EPP=%d f32->bf16", U, EPP);
      snprintf(
          v2Label,
          sizeof(v2Label),
          "V2 SR U=%d EPP=%d f32->bf16 [%s]",
          U,
          EPP,
          exch ? "exch" : "simple");

      runBenchCore(
          kN,
          nBlk,
          [&](int nb, int bs, int64_t n) {
            baseline_packs_kernel_640<U, EPP, float, __nv_bfloat16, float>
                <<<nb, bs>>>(d_dstBf16, d_srcFloat0, (ssize_t)n);
          },
          rtnLabel,
          totalBytes,
          kBS);
      runBenchCore(
          kN,
          nBlk,
          [&](int nb, int bs, int64_t n) {
            v1_packs_sr_kernel_640<U, EPP, float, __nv_bfloat16, float>
                <<<nb, bs>>>(
                    d_dstBf16, d_srcFloat0, (ssize_t)n, kSeed, kBaseOffset);
          },
          v1Label,
          totalBytes,
          kBS);
      runBenchCore(
          kN,
          nBlk,
          [&](int nb, int bs, int64_t n) {
            v2_packs_sr_kernel_640<U, EPP, float, __nv_bfloat16, float>
                <<<nb, bs>>>(
                    d_dstBf16, d_srcFloat0, (ssize_t)n, kSeed, kBaseOffset);
          },
          v2Label,
          totalBytes,
          kBS);
      printf("\n");
    }
  };

  printf("  -- U=4, EPP=4 --\n");
  sweep(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  printf("  -- U=8, EPP=1 --\n");
  sweep(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
}

TEST_F(SimpleCopySRV2Bench, BlockSweep_2src_FloatFloat_Bf16) {
  printf(
      "\n--- Block sweep (bs=640): 2-src FP32+FP32 -> BF16 (64M elts) ---\n");
  constexpr int kBS = 640;
  size_t totalBytes = 2 * kN * sizeof(float) + kN * sizeof(__nv_bfloat16);

  auto sweep = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;
    constexpr int G = 8 / EPP;
    constexpr bool exch = (U >= G);

    for (int nBlk = 4; nBlk <= 32; nBlk *= 2) {
      char rtnLabel[96], v1Label[96], v2Label[96];
      snprintf(
          rtnLabel,
          sizeof(rtnLabel),
          "RTN   U=%d EPP=%d 2-src f32+f32->bf16",
          U,
          EPP);
      snprintf(
          v1Label,
          sizeof(v1Label),
          "V1 SR U=%d EPP=%d 2-src f32+f32->bf16",
          U,
          EPP);
      snprintf(
          v2Label,
          sizeof(v2Label),
          "V2 SR U=%d EPP=%d 2-src f32+f32->bf16 [%s]",
          U,
          EPP,
          exch ? "exch" : "simple");

      runBenchCore(
          kN,
          nBlk,
          [&](int nb, int bs, int64_t n) {
            baseline_packs_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                float>
                <<<nb, bs>>>(d_dstBf16, d_srcFloat0, d_srcFloat1, (ssize_t)n);
          },
          rtnLabel,
          totalBytes,
          kBS);
      runBenchCore(
          kN,
          nBlk,
          [&](int nb, int bs, int64_t n) {
            v1_packs_sr_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                float><<<nb, bs>>>(
                d_dstBf16,
                d_srcFloat0,
                d_srcFloat1,
                (ssize_t)n,
                kSeed,
                kBaseOffset);
          },
          v1Label,
          totalBytes,
          kBS);
      runBenchCore(
          kN,
          nBlk,
          [&](int nb, int bs, int64_t n) {
            v2_packs_sr_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                float><<<nb, bs>>>(
                d_dstBf16,
                d_srcFloat0,
                d_srcFloat1,
                (ssize_t)n,
                kSeed,
                kBaseOffset);
          },
          v2Label,
          totalBytes,
          kBS);
      printf("\n");
    }
  };

  printf("  -- U=4, EPP=4 --\n");
  sweep(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  printf("  -- U=8, EPP=1 --\n");
  sweep(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
}

TEST_F(SimpleCopySRV2Bench, BlockSweep_2src_FloatBf16_Bf16) {
  printf(
      "\n--- Block sweep (bs=640): 2-src FP32+BF16 -> BF16 (64M elts) ---\n");
  constexpr int kBS = 640;
  size_t totalBytes = kN * sizeof(float) + 2 * kN * sizeof(__nv_bfloat16);

  auto sweep = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;
    constexpr int G = 8 / EPP;
    constexpr bool exch = (U >= G);

    for (int nBlk = 4; nBlk <= 32; nBlk *= 2) {
      char rtnLabel[96], v1Label[96], v2Label[96];
      snprintf(
          rtnLabel,
          sizeof(rtnLabel),
          "RTN   U=%d EPP=%d 2-src f32+bf16->bf16",
          U,
          EPP);
      snprintf(
          v1Label,
          sizeof(v1Label),
          "V1 SR U=%d EPP=%d 2-src f32+bf16->bf16",
          U,
          EPP);
      snprintf(
          v2Label,
          sizeof(v2Label),
          "V2 SR U=%d EPP=%d 2-src f32+bf16->bf16 [%s]",
          U,
          EPP,
          exch ? "exch" : "simple");

      runBenchCore(
          kN,
          nBlk,
          [&](int nb, int bs, int64_t n) {
            baseline_packs_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                __nv_bfloat16>
                <<<nb, bs>>>(d_dstBf16, d_srcFloat0, d_srcBf16, (ssize_t)n);
          },
          rtnLabel,
          totalBytes,
          kBS);
      runBenchCore(
          kN,
          nBlk,
          [&](int nb, int bs, int64_t n) {
            v1_packs_sr_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                __nv_bfloat16><<<nb, bs>>>(
                d_dstBf16,
                d_srcFloat0,
                d_srcBf16,
                (ssize_t)n,
                kSeed,
                kBaseOffset);
          },
          v1Label,
          totalBytes,
          kBS);
      runBenchCore(
          kN,
          nBlk,
          [&](int nb, int bs, int64_t n) {
            v2_packs_sr_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                __nv_bfloat16><<<nb, bs>>>(
                d_dstBf16,
                d_srcFloat0,
                d_srcBf16,
                (ssize_t)n,
                kSeed,
                kBaseOffset);
          },
          v2Label,
          totalBytes,
          kBS);
      printf("\n");
    }
  };

  printf("  -- U=4, EPP=4 --\n");
  sweep(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  printf("  -- U=8, EPP=1 --\n");
  sweep(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
}

// =============================================================================
// Block-size sweep: numBlocks=16, blockSize={128,256,384,512,640}
// Two configs: (U=4,EPP=4) [exch] and (U=8,EPP=1) [exch]
// =============================================================================

TEST_F(SimpleCopySRV2Bench, BlockSizeSweep_1src_FloatToBf16) {
  printf(
      "\n--- BlockSize sweep (nBlk=16): 1-src FP32 -> BF16 (64M elts) ---\n");
  constexpr int kNBlk = 16;
  constexpr int kBlockSizes[] = {128, 256, 384, 512, 640};
  size_t totalBytes = kN * sizeof(float) + kN * sizeof(__nv_bfloat16);

  auto sweep = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;
    constexpr int G = 8 / EPP;
    constexpr bool exch = (U >= G);

    for (int bs : kBlockSizes) {
      char rtnLabel[96], v1Label[96], v2Label[96];
      snprintf(
          rtnLabel,
          sizeof(rtnLabel),
          "RTN   U=%d EPP=%d f32->bf16 bs=%d",
          U,
          EPP,
          bs);
      snprintf(
          v1Label,
          sizeof(v1Label),
          "V1 SR U=%d EPP=%d f32->bf16 bs=%d",
          U,
          EPP,
          bs);
      snprintf(
          v2Label,
          sizeof(v2Label),
          "V2 SR U=%d EPP=%d f32->bf16 bs=%d [%s]",
          U,
          EPP,
          bs,
          exch ? "exch" : "simple");

      runBenchCore(
          kN,
          kNBlk,
          [&](int nb, int blockSize, int64_t n) {
            baseline_packs_kernel_640<U, EPP, float, __nv_bfloat16, float>
                <<<nb, blockSize>>>(d_dstBf16, d_srcFloat0, (ssize_t)n);
          },
          rtnLabel,
          totalBytes,
          bs);
      runBenchCore(
          kN,
          kNBlk,
          [&](int nb, int blockSize, int64_t n) {
            v1_packs_sr_kernel_640<U, EPP, float, __nv_bfloat16, float>
                <<<nb, blockSize>>>(
                    d_dstBf16, d_srcFloat0, (ssize_t)n, kSeed, kBaseOffset);
          },
          v1Label,
          totalBytes,
          bs);
      runBenchCore(
          kN,
          kNBlk,
          [&](int nb, int blockSize, int64_t n) {
            v2_packs_sr_kernel_640<U, EPP, float, __nv_bfloat16, float>
                <<<nb, blockSize>>>(
                    d_dstBf16, d_srcFloat0, (ssize_t)n, kSeed, kBaseOffset);
          },
          v2Label,
          totalBytes,
          bs);
      printf("\n");
    }
  };

  printf("  -- U=4, EPP=4 --\n");
  sweep(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  printf("  -- U=8, EPP=1 --\n");
  sweep(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
}

TEST_F(SimpleCopySRV2Bench, BlockSizeSweep_2src_FloatFloat_Bf16) {
  printf(
      "\n--- BlockSize sweep (nBlk=16): 2-src FP32+FP32 -> BF16 "
      "(64M elts) ---\n");
  constexpr int kNBlk = 16;
  constexpr int kBlockSizes[] = {128, 256, 384, 512, 640};
  size_t totalBytes = 2 * kN * sizeof(float) + kN * sizeof(__nv_bfloat16);

  auto sweep = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;
    constexpr int G = 8 / EPP;
    constexpr bool exch = (U >= G);

    for (int bs : kBlockSizes) {
      char rtnLabel[96], v1Label[96], v2Label[96];
      snprintf(
          rtnLabel,
          sizeof(rtnLabel),
          "RTN   U=%d EPP=%d 2-src f32+f32->bf16 bs=%d",
          U,
          EPP,
          bs);
      snprintf(
          v1Label,
          sizeof(v1Label),
          "V1 SR U=%d EPP=%d 2-src f32+f32->bf16 bs=%d",
          U,
          EPP,
          bs);
      snprintf(
          v2Label,
          sizeof(v2Label),
          "V2 SR U=%d EPP=%d 2-src f32+f32->bf16 bs=%d [%s]",
          U,
          EPP,
          bs,
          exch ? "exch" : "simple");

      runBenchCore(
          kN,
          kNBlk,
          [&](int nb, int blockSize, int64_t n) {
            baseline_packs_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                float><<<nb, blockSize>>>(
                d_dstBf16, d_srcFloat0, d_srcFloat1, (ssize_t)n);
          },
          rtnLabel,
          totalBytes,
          bs);
      runBenchCore(
          kN,
          kNBlk,
          [&](int nb, int blockSize, int64_t n) {
            v1_packs_sr_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                float><<<nb, blockSize>>>(
                d_dstBf16,
                d_srcFloat0,
                d_srcFloat1,
                (ssize_t)n,
                kSeed,
                kBaseOffset);
          },
          v1Label,
          totalBytes,
          bs);
      runBenchCore(
          kN,
          kNBlk,
          [&](int nb, int blockSize, int64_t n) {
            v2_packs_sr_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                float><<<nb, blockSize>>>(
                d_dstBf16,
                d_srcFloat0,
                d_srcFloat1,
                (ssize_t)n,
                kSeed,
                kBaseOffset);
          },
          v2Label,
          totalBytes,
          bs);
      printf("\n");
    }
  };

  printf("  -- U=4, EPP=4 --\n");
  sweep(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  printf("  -- U=8, EPP=1 --\n");
  sweep(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
}

TEST_F(SimpleCopySRV2Bench, BlockSizeSweep_2src_FloatBf16_Bf16) {
  printf(
      "\n--- BlockSize sweep (nBlk=16): 2-src FP32+BF16 -> BF16 "
      "(64M elts) ---\n");
  constexpr int kNBlk = 16;
  constexpr int kBlockSizes[] = {128, 256, 384, 512, 640};
  size_t totalBytes = kN * sizeof(float) + 2 * kN * sizeof(__nv_bfloat16);

  auto sweep = [&](auto unrollTag, auto eppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int EPP = decltype(eppTag)::value;
    constexpr int G = 8 / EPP;
    constexpr bool exch = (U >= G);

    for (int bs : kBlockSizes) {
      char rtnLabel[96], v1Label[96], v2Label[96];
      snprintf(
          rtnLabel,
          sizeof(rtnLabel),
          "RTN   U=%d EPP=%d 2-src f32+bf16->bf16 bs=%d",
          U,
          EPP,
          bs);
      snprintf(
          v1Label,
          sizeof(v1Label),
          "V1 SR U=%d EPP=%d 2-src f32+bf16->bf16 bs=%d",
          U,
          EPP,
          bs);
      snprintf(
          v2Label,
          sizeof(v2Label),
          "V2 SR U=%d EPP=%d 2-src f32+bf16->bf16 bs=%d [%s]",
          U,
          EPP,
          bs,
          exch ? "exch" : "simple");

      runBenchCore(
          kN,
          kNBlk,
          [&](int nb, int blockSize, int64_t n) {
            baseline_packs_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                __nv_bfloat16><<<nb, blockSize>>>(
                d_dstBf16, d_srcFloat0, d_srcBf16, (ssize_t)n);
          },
          rtnLabel,
          totalBytes,
          bs);
      runBenchCore(
          kN,
          kNBlk,
          [&](int nb, int blockSize, int64_t n) {
            v1_packs_sr_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                __nv_bfloat16><<<nb, blockSize>>>(
                d_dstBf16,
                d_srcFloat0,
                d_srcBf16,
                (ssize_t)n,
                kSeed,
                kBaseOffset);
          },
          v1Label,
          totalBytes,
          bs);
      runBenchCore(
          kN,
          kNBlk,
          [&](int nb, int blockSize, int64_t n) {
            v2_packs_sr_2src_kernel_640<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                __nv_bfloat16><<<nb, blockSize>>>(
                d_dstBf16,
                d_srcFloat0,
                d_srcBf16,
                (ssize_t)n,
                kSeed,
                kBaseOffset);
          },
          v2Label,
          totalBytes,
          bs);
      printf("\n");
    }
  };

  printf("  -- U=4, EPP=4 --\n");
  sweep(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  printf("  -- U=8, EPP=1 --\n");
  sweep(std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
}

// =============================================================================
// Size sweep: test Hypothesis 1 (L2 cache makes kernel compute-bound)
// Sweep kN from 4M to 256M elements to move from L2-resident to HBM-dominated.
// If SR/baseline BW ratio improves at large sizes, L2 caching is the cause.
// =============================================================================

TEST_F(SimpleCopySRV2Bench, SizeSweep_1src_FloatToBf16) {
  printf(
      "\n--- Size sweep (Hypothesis 1: L2 vs HBM): 1-src FP32 -> BF16 ---\n");
  printf(
      "  H100 L2 = 50 MB. Working set = N*4 (src) + N*2 (dst) = 6*N bytes.\n");
  printf(
      "  If SR/baseline gap shrinks at large N, kernel is compute-bound only "
      "at L2 speeds.\n\n");

  constexpr int64_t kSizes[] = {
      4L * 1024L * 1024L, // 24 MB total — fits L2
      16L * 1024L * 1024L, // 96 MB total — exceeds L2
      64L * 1024L * 1024L, // 384 MB total — HBM dominated
      256L * 1024L * 1024L, // 1.5 GB total — fully HBM
  };

  for (int64_t N : kSizes) {
    float* src = nullptr;
    __nv_bfloat16* dst = nullptr;
    CUDACHECK(cudaMalloc(&src, N * sizeof(float)));
    CUDACHECK(cudaMalloc(&dst, N * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMemset(src, 0x3f, N * sizeof(float)));

    size_t totalBytes = N * sizeof(float) + N * sizeof(__nv_bfloat16);
    double totalMB = (double)totalBytes / (1024.0 * 1024.0);
    printf(
        "  -- N=%ldM elts (%.0f MB working set) --\n",
        N / (1024 * 1024),
        totalMB);

    auto benchConfig = [&](auto unrollTag, auto eppTag) {
      constexpr int U = decltype(unrollTag)::value;
      constexpr int EPP = decltype(eppTag)::value;
      constexpr int G = 8 / EPP;
      constexpr bool exch = (U >= G);
      char rtnLabel[96], v1Label[96], v2Label[96];
      snprintf(
          rtnLabel, sizeof(rtnLabel), "RTN   U=%d EPP=%d f32->bf16", U, EPP);
      snprintf(v1Label, sizeof(v1Label), "V1 SR U=%d EPP=%d f32->bf16", U, EPP);
      snprintf(
          v2Label,
          sizeof(v2Label),
          "V2 SR U=%d EPP=%d f32->bf16 [%s]",
          U,
          EPP,
          exch ? "exch" : "simple");

      runBenchCore(
          N,
          kDefaultBlocks,
          [&](int nBlk, int bs, int64_t n) {
            baseline_packs_kernel<U, EPP, float, __nv_bfloat16, float>
                <<<nBlk, bs>>>(dst, src, (ssize_t)n);
          },
          rtnLabel,
          totalBytes);
      runBenchCore(
          N,
          kDefaultBlocks,
          [&](int nBlk, int bs, int64_t n) {
            v1_packs_sr_kernel<U, EPP, float, __nv_bfloat16, float>
                <<<nBlk, bs>>>(dst, src, (ssize_t)n, kSeed, kBaseOffset);
          },
          v1Label,
          totalBytes);
      runBenchCore(
          N,
          kDefaultBlocks,
          [&](int nBlk, int bs, int64_t n) {
            v2_packs_sr_kernel<U, EPP, float, __nv_bfloat16, float>
                <<<nBlk, bs>>>(dst, src, (ssize_t)n, kSeed, kBaseOffset);
          },
          v2Label,
          totalBytes);
      printf("\n");
    };

    // Test most relevant configs: EPP=4 (BigPack path) at U=4 and U=8
    benchConfig(
        std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
    benchConfig(
        std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{});

    CUDACHECK(cudaFree(src));
    CUDACHECK(cudaFree(dst));
  }
}

TEST_F(SimpleCopySRV2Bench, SizeSweep_2src_FloatFloat_Bf16) {
  printf(
      "\n--- Size sweep (Hypothesis 1: L2 vs HBM): 2-src FP32+FP32 -> BF16 "
      "---\n");
  printf(
      "  Working set = N*4 (src0) + N*4 (src1) + N*2 (dst) = 10*N bytes.\n\n");

  constexpr int64_t kSizes[] = {
      4L * 1024L * 1024L, // 40 MB total — fits L2
      16L * 1024L * 1024L, // 160 MB total — exceeds L2
      64L * 1024L * 1024L, // 640 MB total — HBM dominated
      128L * 1024L * 1024L, // 1.28 GB total — fully HBM
  };

  for (int64_t N : kSizes) {
    float* src0 = nullptr;
    float* src1 = nullptr;
    __nv_bfloat16* dst = nullptr;
    CUDACHECK(cudaMalloc(&src0, N * sizeof(float)));
    CUDACHECK(cudaMalloc(&src1, N * sizeof(float)));
    CUDACHECK(cudaMalloc(&dst, N * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMemset(src0, 0x3f, N * sizeof(float)));
    CUDACHECK(cudaMemset(src1, 0x3f, N * sizeof(float)));

    size_t totalBytes = 2 * N * sizeof(float) + N * sizeof(__nv_bfloat16);
    double totalMB = (double)totalBytes / (1024.0 * 1024.0);
    printf(
        "  -- N=%ldM elts (%.0f MB working set) --\n",
        N / (1024 * 1024),
        totalMB);

    auto benchConfig = [&](auto unrollTag, auto eppTag) {
      constexpr int U = decltype(unrollTag)::value;
      constexpr int EPP = decltype(eppTag)::value;
      constexpr int G = 8 / EPP;
      constexpr bool exch = (U >= G);
      char rtnLabel[96], v1Label[96], v2Label[96];
      snprintf(
          rtnLabel,
          sizeof(rtnLabel),
          "RTN   U=%d EPP=%d 2-src f32+f32->bf16",
          U,
          EPP);
      snprintf(
          v1Label,
          sizeof(v1Label),
          "V1 SR U=%d EPP=%d 2-src f32+f32->bf16",
          U,
          EPP);
      snprintf(
          v2Label,
          sizeof(v2Label),
          "V2 SR U=%d EPP=%d 2-src f32+f32->bf16 [%s]",
          U,
          EPP,
          exch ? "exch" : "simple");

      runBenchCore(
          N,
          kDefaultBlocks,
          [&](int nBlk, int bs, int64_t n) {
            baseline_packs_2src_kernel<
                U,
                EPP,
                float,
                __nv_bfloat16,
                float,
                float><<<nBlk, bs>>>(dst, src0, src1, (ssize_t)n);
          },
          rtnLabel,
          totalBytes);
      runBenchCore(
          N,
          kDefaultBlocks,
          [&](int nBlk, int bs, int64_t n) {
            v1_packs_sr_2src_kernel<U, EPP, float, __nv_bfloat16, float, float>
                <<<nBlk, bs>>>(dst, src0, src1, (ssize_t)n, kSeed, kBaseOffset);
          },
          v1Label,
          totalBytes);
      runBenchCore(
          N,
          kDefaultBlocks,
          [&](int nBlk, int bs, int64_t n) {
            v2_packs_sr_2src_kernel<U, EPP, float, __nv_bfloat16, float, float>
                <<<nBlk, bs>>>(dst, src0, src1, (ssize_t)n, kSeed, kBaseOffset);
          },
          v2Label,
          totalBytes);
      printf("\n");
    };

    benchConfig(
        std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
    benchConfig(
        std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{});

    CUDACHECK(cudaFree(src0));
    CUDACHECK(cudaFree(src1));
    CUDACHECK(cudaFree(dst));
  }
}

// clang-format off

/*
--- reduceCopySR V1 vs V2: FP32 -> BF16 (4M elts) ---
  V1 reduceCopySR U=1 f32->bf16                                   nBlocks=  32  avg=0.027 ms  BW=943.37 GB/s
  V2 reduceCopySR U=1 f32->bf16 [big:simple small:simple]         nBlocks=  32  avg=0.027 ms  BW=943.56 GB/s
  V1 reduceCopySR U=2 f32->bf16                                   nBlocks=  32  avg=0.016 ms  BW=1534.71 GB/s
  V2 reduceCopySR U=2 f32->bf16 [big:exch small:simple]           nBlocks=  32  avg=0.016 ms  BW=1532.68 GB/s
  V1 reduceCopySR U=4 f32->bf16                                   nBlocks=  32  avg=0.015 ms  BW=1633.47 GB/s
  V2 reduceCopySR U=4 f32->bf16 [big:exch small:exch]             nBlocks=  32  avg=0.012 ms  BW=2043.05 GB/s
  V1 reduceCopySR U=8 f32->bf16                                   nBlocks=  32  avg=0.014 ms  BW=1753.08 GB/s
  V2 reduceCopySR U=8 f32->bf16 [big:exch small:exch]             nBlocks=  32  avg=0.012 ms  BW=2046.88 GB/s
[       OK ] SimpleCopySRV2Bench.TopLevel_FloatToBf16 (218 ms)
[ RUN      ] SimpleCopySRV2Bench.PacksMatrix_FloatToBf16

--- reduceCopyPacksSR V1 vs V2: FP32 -> BF16 (Unroll x EltPerPack, 4M elts) ---
  RTN   U=1 EPP=4 f32->bf16                                       nBlocks=  32  avg=0.027 ms  BW=943.56 GB/s
  V1 SR U=1 EPP=4 f32->bf16                                       nBlocks=  32  avg=0.027 ms  BW=943.55 GB/s
  V2 SR U=1 EPP=4 f32->bf16 [simple]                              nBlocks=  32  avg=0.027 ms  BW=944.35 GB/s

  RTN   U=2 EPP=4 f32->bf16                                       nBlocks=  32  avg=0.016 ms  BW=1533.36 GB/s
  V1 SR U=2 EPP=4 f32->bf16                                       nBlocks=  32  avg=0.016 ms  BW=1528.27 GB/s
  V2 SR U=2 EPP=4 f32->bf16 [exch]                                nBlocks=  32  avg=0.016 ms  BW=1533.16 GB/s

  RTN   U=4 EPP=4 f32->bf16                                       nBlocks=  32  avg=0.011 ms  BW=2239.59 GB/s
  V1 SR U=4 EPP=4 f32->bf16                                       nBlocks=  32  avg=0.015 ms  BW=1636.76 GB/s
  V2 SR U=4 EPP=4 f32->bf16 [exch]                                nBlocks=  32  avg=0.012 ms  BW=2044.80 GB/s

  RTN   U=8 EPP=4 f32->bf16                                       nBlocks=  32  avg=0.010 ms  BW=2455.37 GB/s
  V1 SR U=8 EPP=4 f32->bf16                                       nBlocks=  32  avg=0.016 ms  BW=1607.33 GB/s
  V2 SR U=8 EPP=4 f32->bf16 [exch]                                nBlocks=  32  avg=0.012 ms  BW=2060.39 GB/s

  RTN   U=1 EPP=2 f32->bf16                                       nBlocks=  32  avg=0.047 ms  BW=534.11 GB/s
  V1 SR U=1 EPP=2 f32->bf16                                       nBlocks=  32  avg=0.047 ms  BW=534.18 GB/s
  V2 SR U=1 EPP=2 f32->bf16 [simple]                              nBlocks=  32  avg=0.047 ms  BW=533.91 GB/s

  RTN   U=2 EPP=2 f32->bf16                                       nBlocks=  32  avg=0.027 ms  BW=944.64 GB/s
  V1 SR U=2 EPP=2 f32->bf16                                       nBlocks=  32  avg=0.027 ms  BW=944.58 GB/s
  V2 SR U=2 EPP=2 f32->bf16 [simple]                              nBlocks=  32  avg=0.029 ms  BW=875.65 GB/s

  RTN   U=4 EPP=2 f32->bf16                                       nBlocks=  32  avg=0.016 ms  BW=1534.68 GB/s
  V1 SR U=4 EPP=2 f32->bf16                                       nBlocks=  32  avg=0.023 ms  BW=1117.03 GB/s
  V2 SR U=4 EPP=2 f32->bf16 [exch]                                nBlocks=  32  avg=0.017 ms  BW=1487.14 GB/s

  RTN   U=8 EPP=2 f32->bf16                                       nBlocks=  32  avg=0.012 ms  BW=2043.80 GB/s
  V1 SR U=8 EPP=2 f32->bf16                                       nBlocks=  32  avg=0.021 ms  BW=1215.73 GB/s
  V2 SR U=8 EPP=2 f32->bf16 [exch]                                nBlocks=  32  avg=0.014 ms  BW=1753.63 GB/s

  RTN   U=1 EPP=1 f32->bf16                                       nBlocks=  32  avg=0.092 ms  BW=272.86 GB/s
  V1 SR U=1 EPP=1 f32->bf16                                       nBlocks=  32  avg=0.145 ms  BW=173.66 GB/s
  V2 SR U=1 EPP=1 f32->bf16 [simple]                              nBlocks=  32  avg=0.154 ms  BW=163.93 GB/s

  RTN   U=2 EPP=1 f32->bf16                                       nBlocks=  32  avg=0.049 ms  BW=511.90 GB/s
  V1 SR U=2 EPP=1 f32->bf16                                       nBlocks=  32  avg=0.059 ms  BW=423.78 GB/s
  V2 SR U=2 EPP=1 f32->bf16 [simple]                              nBlocks=  32  avg=0.063 ms  BW=396.40 GB/s

  RTN   U=4 EPP=1 f32->bf16                                       nBlocks=  32  avg=0.029 ms  BW=877.05 GB/s
  V1 SR U=4 EPP=1 f32->bf16                                       nBlocks=  32  avg=0.065 ms  BW=386.00 GB/s
  V2 SR U=4 EPP=1 f32->bf16 [simple]                              nBlocks=  32  avg=0.053 ms  BW=472.53 GB/s

  RTN   U=8 EPP=1 f32->bf16                                       nBlocks=  32  avg=0.017 ms  BW=1446.68 GB/s
  V1 SR U=8 EPP=1 f32->bf16                                       nBlocks=  32  avg=0.053 ms  BW=472.08 GB/s
  V2 SR U=8 EPP=1 f32->bf16 [exch]                                nBlocks=  32  avg=0.035 ms  BW=722.54 GB/s

[       OK ] SimpleCopySRV2Bench.PacksMatrix_FloatToBf16 (162 ms)
[ RUN      ] SimpleCopySRV2Bench.PacksMatrix_2src_FloatFloat_Bf16

--- reduceCopyPacksSR 2-src V1 vs V2: FP32+FP32 -> BF16 (Unroll x EltPerPack, 4M elts) ---
  RTN   U=1 EPP=4 2-src f32+f32->bf16                             nBlocks=  32  avg=0.029 ms  BW=1459.11 GB/s
  V1 SR U=1 EPP=4 2-src f32+f32->bf16                             nBlocks=  32  avg=0.030 ms  BW=1387.43 GB/s
  V2 SR U=1 EPP=4 2-src f32+f32->bf16 [simple]                    nBlocks=  32  avg=0.030 ms  BW=1412.25 GB/s

  RTN   U=2 EPP=4 2-src f32+f32->bf16                             nBlocks=  32  avg=0.018 ms  BW=2273.98 GB/s
  V1 SR U=2 EPP=4 2-src f32+f32->bf16                             nBlocks=  32  avg=0.018 ms  BW=2271.65 GB/s
  V2 SR U=2 EPP=4 2-src f32+f32->bf16 [exch]                      nBlocks=  32  avg=0.026 ms  BW=1639.59 GB/s

  RTN   U=4 EPP=4 2-src f32+f32->bf16                             nBlocks=  32  avg=0.012 ms  BW=3386.35 GB/s
  V1 SR U=4 EPP=4 2-src f32+f32->bf16                             nBlocks=  32  avg=0.016 ms  BW=2558.05 GB/s
  V2 SR U=4 EPP=4 2-src f32+f32->bf16 [exch]                      nBlocks=  32  avg=0.020 ms  BW=2047.20 GB/s

  RTN   U=8 EPP=4 2-src f32+f32->bf16                             nBlocks=  32  avg=0.012 ms  BW=3405.00 GB/s
  V1 SR U=8 EPP=4 2-src f32+f32->bf16                             nBlocks=  32  avg=0.016 ms  BW=2557.85 GB/s
  V2 SR U=8 EPP=4 2-src f32+f32->bf16 [exch]                      nBlocks=  32  avg=0.018 ms  BW=2283.25 GB/s

  RTN   U=1 EPP=2 2-src f32+f32->bf16                             nBlocks=  32  avg=0.052 ms  BW=810.89 GB/s
  V1 SR U=1 EPP=2 2-src f32+f32->bf16                             nBlocks=  32  avg=0.051 ms  BW=819.97 GB/s
  V2 SR U=1 EPP=2 2-src f32+f32->bf16 [simple]                    nBlocks=  32  avg=0.051 ms  BW=819.46 GB/s

  RTN   U=2 EPP=2 2-src f32+f32->bf16                             nBlocks=  32  avg=0.029 ms  BW=1460.67 GB/s
  V1 SR U=2 EPP=2 2-src f32+f32->bf16                             nBlocks=  32  avg=0.029 ms  BW=1460.66 GB/s
  V2 SR U=2 EPP=2 2-src f32+f32->bf16 [simple]                    nBlocks=  32  avg=0.031 ms  BW=1364.34 GB/s

  RTN   U=4 EPP=2 2-src f32+f32->bf16                             nBlocks=  32  avg=0.018 ms  BW=2299.87 GB/s
  V1 SR U=4 EPP=2 2-src f32+f32->bf16                             nBlocks=  32  avg=0.025 ms  BW=1706.91 GB/s
  V2 SR U=4 EPP=2 2-src f32+f32->bf16 [exch]                      nBlocks=  32  avg=0.027 ms  BW=1575.27 GB/s

  RTN   U=8 EPP=2 2-src f32+f32->bf16                             nBlocks=  32  avg=0.014 ms  BW=2957.20 GB/s
  V1 SR U=8 EPP=2 2-src f32+f32->bf16                             nBlocks=  32  avg=0.023 ms  BW=1829.82 GB/s
  V2 SR U=8 EPP=2 2-src f32+f32->bf16 [exch]                      nBlocks=  32  avg=0.021 ms  BW=2045.41 GB/s

  RTN   U=1 EPP=1 2-src f32+f32->bf16                             nBlocks=  32  avg=0.105 ms  BW=399.74 GB/s
  V1 SR U=1 EPP=1 2-src f32+f32->bf16                             nBlocks=  32  avg=0.153 ms  BW=273.42 GB/s
  V2 SR U=1 EPP=1 2-src f32+f32->bf16 [simple]                    nBlocks=  32  avg=0.161 ms  BW=261.20 GB/s

  RTN   U=2 EPP=1 2-src f32+f32->bf16                             nBlocks=  32  avg=0.053 ms  BW=787.38 GB/s
  V1 SR U=2 EPP=1 2-src f32+f32->bf16                             nBlocks=  32  avg=0.064 ms  BW=659.95 GB/s
  V2 SR U=2 EPP=1 2-src f32+f32->bf16 [simple]                    nBlocks=  32  avg=0.066 ms  BW=639.64 GB/s

  RTN   U=4 EPP=1 2-src f32+f32->bf16                             nBlocks=  32  avg=0.031 ms  BW=1364.18 GB/s
  V1 SR U=4 EPP=1 2-src f32+f32->bf16                             nBlocks=  32  avg=0.068 ms  BW=620.35 GB/s
  V2 SR U=4 EPP=1 2-src f32+f32->bf16 [simple]                    nBlocks=  32  avg=0.055 ms  BW=758.29 GB/s

  RTN   U=8 EPP=1 2-src f32+f32->bf16                             nBlocks=  32  avg=0.020 ms  BW=2046.62 GB/s
  V1 SR U=8 EPP=1 2-src f32+f32->bf16                             nBlocks=  32  avg=0.057 ms  BW=731.08 GB/s
  V2 SR U=8 EPP=1 2-src f32+f32->bf16 [exch]                      nBlocks=  32  avg=0.047 ms  BW=890.50 GB/s

[       OK ] SimpleCopySRV2Bench.PacksMatrix_2src_FloatFloat_Bf16 (176 ms)
[ RUN      ] SimpleCopySRV2Bench.PacksMatrix_2src_FloatBf16_Bf16

--- reduceCopyPacksSR 2-src V1 vs V2: FP32+BF16 -> BF16 (Unroll x EltPerPack, 4M elts) ---
  RTN   U=1 EPP=4 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.029 ms  BW=1149.55 GB/s
  V1 SR U=1 EPP=4 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.030 ms  BW=1125.90 GB/s
  V2 SR U=1 EPP=4 2-src f32+bf16->bf16 [simple]                   nBlocks=  32  avg=0.031 ms  BW=1092.21 GB/s

  RTN   U=2 EPP=4 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.018 ms  BW=1819.21 GB/s
  V1 SR U=2 EPP=4 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.018 ms  BW=1817.26 GB/s
  V2 SR U=2 EPP=4 2-src f32+bf16->bf16 [exch]                     nBlocks=  32  avg=0.027 ms  BW=1259.55 GB/s

  RTN   U=4 EPP=4 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.013 ms  BW=2591.96 GB/s
  V1 SR U=4 EPP=4 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.017 ms  BW=2015.72 GB/s
  V2 SR U=4 EPP=4 2-src f32+bf16->bf16 [exch]                     nBlocks=  32  avg=0.020 ms  BW=1636.97 GB/s

  RTN   U=8 EPP=4 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.012 ms  BW=2727.26 GB/s
  V1 SR U=8 EPP=4 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.016 ms  BW=2044.29 GB/s
  V2 SR U=8 EPP=4 2-src f32+bf16->bf16 [exch]                     nBlocks=  32  avg=0.018 ms  BW=1819.02 GB/s

  RTN   U=1 EPP=2 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.051 ms  BW=654.91 GB/s
  V1 SR U=1 EPP=2 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.051 ms  BW=655.41 GB/s
  V2 SR U=1 EPP=2 2-src f32+bf16->bf16 [simple]                   nBlocks=  32  avg=0.051 ms  BW=655.16 GB/s

  RTN   U=2 EPP=2 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.029 ms  BW=1169.48 GB/s
  V1 SR U=2 EPP=2 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.031 ms  BW=1092.24 GB/s
  V2 SR U=2 EPP=2 2-src f32+bf16->bf16 [simple]                   nBlocks=  32  avg=0.032 ms  BW=1055.60 GB/s

  RTN   U=4 EPP=2 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.018 ms  BW=1819.21 GB/s
  V1 SR U=4 EPP=2 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.025 ms  BW=1363.27 GB/s
  V2 SR U=4 EPP=2 2-src f32+bf16->bf16 [exch]                     nBlocks=  32  avg=0.027 ms  BW=1258.16 GB/s

  RTN   U=8 EPP=2 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.014 ms  BW=2341.88 GB/s
  V1 SR U=8 EPP=2 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.026 ms  BW=1284.81 GB/s
  V2 SR U=8 EPP=2 2-src f32+bf16->bf16 [exch]                     nBlocks=  32  avg=0.021 ms  BW=1585.75 GB/s

  RTN   U=1 EPP=1 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.101 ms  BW=333.23 GB/s
  V1 SR U=1 EPP=1 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.145 ms  BW=231.73 GB/s
  V2 SR U=1 EPP=1 2-src f32+bf16->bf16 [simple]                   nBlocks=  32  avg=0.150 ms  BW=223.83 GB/s

  RTN   U=2 EPP=1 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.053 ms  BW=629.51 GB/s
  V1 SR U=2 EPP=1 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.064 ms  BW=520.62 GB/s
  V2 SR U=2 EPP=1 2-src f32+bf16->bf16 [simple]                   nBlocks=  32  avg=0.068 ms  BW=496.02 GB/s

  RTN   U=4 EPP=1 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.031 ms  BW=1091.60 GB/s
  V1 SR U=4 EPP=1 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.068 ms  BW=492.73 GB/s
  V2 SR U=4 EPP=1 2-src f32+bf16->bf16 [simple]                   nBlocks=  32  avg=0.057 ms  BW=584.80 GB/s

  RTN   U=8 EPP=1 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.020 ms  BW=1637.61 GB/s
  V1 SR U=8 EPP=1 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.057 ms  BW=585.24 GB/s
  V2 SR U=8 EPP=1 2-src f32+bf16->bf16 [exch]                     nBlocks=  32  avg=0.047 ms  BW=712.52 GB/s

[       OK ] SimpleCopySRV2Bench.PacksMatrix_2src_FloatBf16_Bf16 (175 ms)
[ RUN      ] SimpleCopySRV2Bench.NoopSR_FloatToFloat

--- No-op SR path (f32->f32): V1 vs V2 overhead check ---
  V1 SR U=4 f32->f32 (no-op)                                      nBlocks=  32  avg=0.014 ms  BW=2337.13 GB/s
  V2 SR U=4 f32->f32 (no-op)                                      nBlocks=  32  avg=0.014 ms  BW=2338.33 GB/s
  V1 SR U=8 f32->f32 (no-op)                                      nBlocks=  32  avg=0.014 ms  BW=2338.33 GB/s
  V2 SR U=8 f32->f32 (no-op)                                      nBlocks=  32  avg=0.014 ms  BW=2337.44 GB/s
[       OK ] SimpleCopySRV2Bench.NoopSR_FloatToFloat (16 ms)
[ RUN      ] SimpleCopySRV2Bench.TwoSrc_FloatFloat_Bf16

--- reduceCopySR 2-src V1 vs V2: FP32+FP32 -> BF16 (4M elts) ---
  V1 SR U=1 2-src f32+f32->bf16                                   nBlocks=  32  avg=0.029 ms  BW=1461.68 GB/s
  V2 SR U=1 2-src f32+f32->bf16 [big:simple small:simple]         nBlocks=  32  avg=0.029 ms  BW=1459.83 GB/s
  V1 SR U=2 2-src f32+f32->bf16                                   nBlocks=  32  avg=0.018 ms  BW=2272.16 GB/s
  V2 SR U=2 2-src f32+f32->bf16 [big:exch small:simple]           nBlocks=  32  avg=0.027 ms  BW=1575.25 GB/s
  V1 SR U=4 2-src f32+f32->bf16                                   nBlocks=  32  avg=0.016 ms  BW=2557.60 GB/s
  V2 SR U=4 2-src f32+f32->bf16 [big:exch small:exch]             nBlocks=  32  avg=0.020 ms  BW=2047.42 GB/s
  V1 SR U=8 2-src f32+f32->bf16                                   nBlocks=  32  avg=0.016 ms  BW=2557.25 GB/s
  V2 SR U=8 2-src f32+f32->bf16 [big:exch small:exch]             nBlocks=  32  avg=0.018 ms  BW=2273.94 GB/s
[       OK ] SimpleCopySRV2Bench.TwoSrc_FloatFloat_Bf16 (29 ms)
[ RUN      ] SimpleCopySRV2Bench.TwoSrc_FloatBf16_Bf16

--- reduceCopySR 2-src V1 vs V2: FP32+BF16 -> BF16 (4M elts) ---
  V1 SR U=1 2-src f32+bf16->bf16                                  nBlocks=  32  avg=0.029 ms  BW=1168.82 GB/s
  V2 SR U=1 2-src f32+bf16->bf16 [big:simple small:simple]        nBlocks=  32  avg=0.029 ms  BW=1169.55 GB/s
  V1 SR U=2 2-src f32+bf16->bf16                                  nBlocks=  32  avg=0.018 ms  BW=1817.67 GB/s
  V2 SR U=2 2-src f32+bf16->bf16 [big:exch small:simple]          nBlocks=  32  avg=0.027 ms  BW=1258.43 GB/s
  V1 SR U=4 2-src f32+bf16->bf16                                  nBlocks=  32  avg=0.017 ms  BW=1945.77 GB/s
  V2 SR U=4 2-src f32+bf16->bf16 [big:exch small:exch]            nBlocks=  32  avg=0.021 ms  BW=1636.12 GB/s
  V1 SR U=8 2-src f32+bf16->bf16                                  nBlocks=  32  avg=0.016 ms  BW=2045.24 GB/s
  V2 SR U=8 2-src f32+bf16->bf16 [big:exch small:exch]            nBlocks=  32  avg=0.018 ms  BW=1818.99 GB/s
[       OK ] SimpleCopySRV2Bench.TwoSrc_FloatBf16_Bf16 (29 ms)
[ RUN      ] SimpleCopySRV2Bench.BlockSweep_1src_FloatToBf16

--- Block sweep (bs=640): 1-src FP32 -> BF16 (4M elts) ---
  -- U=4, EPP=4 --
  RTN   U=4 EPP=4 f32->bf16                                       nBlocks=   4  avg=0.055 ms  BW=455.93 GB/s
  V1 SR U=4 EPP=4 f32->bf16                                       nBlocks=   4  avg=0.088 ms  BW=285.39 GB/s
  V2 SR U=4 EPP=4 f32->bf16 [exch]                                nBlocks=   4  avg=0.063 ms  BW=402.54 GB/s

  RTN   U=4 EPP=4 f32->bf16                                       nBlocks=   8  avg=0.029 ms  BW=876.29 GB/s
  V1 SR U=4 EPP=4 f32->bf16                                       nBlocks=   8  avg=0.047 ms  BW=534.41 GB/s
  V2 SR U=4 EPP=4 f32->bf16 [exch]                                nBlocks=   8  avg=0.033 ms  BW=766.55 GB/s

  RTN   U=4 EPP=4 f32->bf16                                       nBlocks=  16  avg=0.016 ms  BW=1534.83 GB/s
  V1 SR U=4 EPP=4 f32->bf16                                       nBlocks=  16  avg=0.025 ms  BW=1019.45 GB/s
  V2 SR U=4 EPP=4 f32->bf16 [exch]                                nBlocks=  16  avg=0.018 ms  BW=1363.51 GB/s

  RTN   U=4 EPP=4 f32->bf16                                       nBlocks=  32  avg=0.010 ms  BW=2452.16 GB/s
  V1 SR U=4 EPP=4 f32->bf16                                       nBlocks=  32  avg=0.014 ms  BW=1752.49 GB/s
  V2 SR U=4 EPP=4 f32->bf16 [exch]                                nBlocks=  32  avg=0.011 ms  BW=2264.09 GB/s

  -- U=8, EPP=1 --
  RTN   U=8 EPP=1 f32->bf16                                       nBlocks=   4  avg=0.059 ms  BW=423.80 GB/s
  V1 SR U=8 EPP=1 f32->bf16                                       nBlocks=   4  avg=0.305 ms  BW=82.48 GB/s
  V2 SR U=8 EPP=1 f32->bf16 [exch]                                nBlocks=   4  avg=0.235 ms  BW=107.25 GB/s

  RTN   U=8 EPP=1 f32->bf16                                       nBlocks=   8  avg=0.032 ms  BW=790.19 GB/s
  V1 SR U=8 EPP=1 f32->bf16                                       nBlocks=   8  avg=0.155 ms  BW=161.85 GB/s
  V2 SR U=8 EPP=1 f32->bf16 [exch]                                nBlocks=   8  avg=0.120 ms  BW=210.49 GB/s

  RTN   U=8 EPP=1 f32->bf16                                       nBlocks=  16  avg=0.018 ms  BW=1364.17 GB/s
  V1 SR U=8 EPP=1 f32->bf16                                       nBlocks=  16  avg=0.080 ms  BW=314.98 GB/s
  V2 SR U=8 EPP=1 f32->bf16 [exch]                                nBlocks=  16  avg=0.062 ms  BW=407.64 GB/s

  RTN   U=8 EPP=1 f32->bf16                                       nBlocks=  32  avg=0.011 ms  BW=2228.99 GB/s
  V1 SR U=8 EPP=1 f32->bf16                                       nBlocks=  32  avg=0.042 ms  BW=602.85 GB/s
  V2 SR U=8 EPP=1 f32->bf16 [exch]                                nBlocks=  32  avg=0.033 ms  BW=766.89 GB/s

[       OK ] SimpleCopySRV2Bench.BlockSweep_1src_FloatToBf16 (182 ms)
[ RUN      ] SimpleCopySRV2Bench.BlockSweep_2src_FloatFloat_Bf16

--- Block sweep (bs=640): 2-src FP32+FP32 -> BF16 (4M elts) ---
  -- U=4, EPP=4 --
  RTN   U=4 EPP=4 2-src f32+f32->bf16                             nBlocks=   4  avg=0.074 ms  BW=568.92 GB/s
  V1 SR U=4 EPP=4 2-src f32+f32->bf16                             nBlocks=   4  avg=0.094 ms  BW=444.86 GB/s
  V2 SR U=4 EPP=4 2-src f32+f32->bf16 [exch]                      nBlocks=   4  avg=0.082 ms  BW=509.22 GB/s

  RTN   U=4 EPP=4 2-src f32+f32->bf16                             nBlocks=   8  avg=0.039 ms  BW=1077.29 GB/s
  V1 SR U=4 EPP=4 2-src f32+f32->bf16                             nBlocks=   8  avg=0.049 ms  BW=852.75 GB/s
  V2 SR U=4 EPP=4 2-src f32+f32->bf16 [exch]                      nBlocks=   8  avg=0.043 ms  BW=968.59 GB/s

  RTN   U=4 EPP=4 2-src f32+f32->bf16                             nBlocks=  16  avg=0.021 ms  BW=2015.75 GB/s
  V1 SR U=4 EPP=4 2-src f32+f32->bf16                             nBlocks=  16  avg=0.027 ms  BW=1572.85 GB/s
  V2 SR U=4 EPP=4 2-src f32+f32->bf16 [exch]                      nBlocks=  16  avg=0.024 ms  BW=1728.29 GB/s

  RTN   U=4 EPP=4 2-src f32+f32->bf16                             nBlocks=  32  avg=0.012 ms  BW=3408.01 GB/s
  V1 SR U=4 EPP=4 2-src f32+f32->bf16                             nBlocks=  32  avg=0.015 ms  BW=2740.03 GB/s
  V2 SR U=4 EPP=4 2-src f32+f32->bf16 [exch]                      nBlocks=  32  avg=0.014 ms  BW=2923.17 GB/s

  -- U=8, EPP=1 --
  RTN   U=8 EPP=1 2-src f32+f32->bf16                             nBlocks=   4  avg=0.078 ms  BW=538.82 GB/s
  V1 SR U=8 EPP=1 2-src f32+f32->bf16                             nBlocks=   4  avg=0.315 ms  BW=133.02 GB/s
  V2 SR U=8 EPP=1 2-src f32+f32->bf16 [exch]                      nBlocks=   4  avg=0.242 ms  BW=173.49 GB/s

  RTN   U=8 EPP=1 2-src f32+f32->bf16                             nBlocks=   8  avg=0.041 ms  BW=1016.88 GB/s
  V1 SR U=8 EPP=1 2-src f32+f32->bf16                             nBlocks=   8  avg=0.160 ms  BW=262.33 GB/s
  V2 SR U=8 EPP=1 2-src f32+f32->bf16 [exch]                      nBlocks=   8  avg=0.123 ms  BW=339.69 GB/s

  RTN   U=8 EPP=1 2-src f32+f32->bf16                             nBlocks=  16  avg=0.024 ms  BW=1746.18 GB/s
  V1 SR U=8 EPP=1 2-src f32+f32->bf16                             nBlocks=  16  avg=0.083 ms  BW=507.56 GB/s
  V2 SR U=8 EPP=1 2-src f32+f32->bf16 [exch]                      nBlocks=  16  avg=0.065 ms  BW=649.51 GB/s

  RTN   U=8 EPP=1 2-src f32+f32->bf16                             nBlocks=  32  avg=0.014 ms  BW=2923.17 GB/s
  V1 SR U=8 EPP=1 2-src f32+f32->bf16                             nBlocks=  32  avg=0.043 ms  BW=974.11 GB/s
  V2 SR U=8 EPP=1 2-src f32+f32->bf16 [exch]                      nBlocks=  32  avg=0.035 ms  BW=1204.61 GB/s

[       OK ] SimpleCopySRV2Bench.BlockSweep_2src_FloatFloat_Bf16 (199 ms)
[ RUN      ] SimpleCopySRV2Bench.BlockSweep_2src_FloatBf16_Bf16

--- Block sweep (bs=640): 2-src FP32+BF16 -> BF16 (4M elts) ---
  -- U=4, EPP=4 --
  RTN   U=4 EPP=4 2-src f32+bf16->bf16                            nBlocks=   4  avg=0.064 ms  BW=528.22 GB/s
  V1 SR U=4 EPP=4 2-src f32+bf16->bf16                            nBlocks=   4  avg=0.100 ms  BW=334.73 GB/s
  V2 SR U=4 EPP=4 2-src f32+bf16->bf16 [exch]                     nBlocks=   4  avg=0.092 ms  BW=366.08 GB/s

  RTN   U=4 EPP=4 2-src f32+bf16->bf16                            nBlocks=   8  avg=0.034 ms  BW=994.63 GB/s
  V1 SR U=4 EPP=4 2-src f32+bf16->bf16                            nBlocks=   8  avg=0.052 ms  BW=644.35 GB/s
  V2 SR U=4 EPP=4 2-src f32+bf16->bf16 [exch]                     nBlocks=   8  avg=0.048 ms  BW=692.63 GB/s

  RTN   U=4 EPP=4 2-src f32+bf16->bf16                            nBlocks=  16  avg=0.018 ms  BW=1816.79 GB/s
  V1 SR U=4 EPP=4 2-src f32+bf16->bf16                            nBlocks=  16  avg=0.029 ms  BW=1169.59 GB/s
  V2 SR U=4 EPP=4 2-src f32+bf16->bf16 [exch]                     nBlocks=  16  avg=0.027 ms  BW=1258.95 GB/s

  RTN   U=4 EPP=4 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.011 ms  BW=2976.63 GB/s
  V1 SR U=4 EPP=4 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.016 ms  BW=2045.60 GB/s
  V2 SR U=4 EPP=4 2-src f32+bf16->bf16 [exch]                     nBlocks=  32  avg=0.016 ms  BW=2061.93 GB/s

  -- U=8, EPP=1 --
  RTN   U=8 EPP=1 2-src f32+bf16->bf16                            nBlocks=   4  avg=0.082 ms  BW=409.83 GB/s
  V1 SR U=8 EPP=1 2-src f32+bf16->bf16                            nBlocks=   4  avg=0.319 ms  BW=105.14 GB/s
  V2 SR U=8 EPP=1 2-src f32+bf16->bf16 [exch]                     nBlocks=   4  avg=0.242 ms  BW=138.67 GB/s

  RTN   U=8 EPP=1 2-src f32+bf16->bf16                            nBlocks=   8  avg=0.044 ms  BW=771.22 GB/s
  V1 SR U=8 EPP=1 2-src f32+bf16->bf16                            nBlocks=   8  avg=0.162 ms  BW=207.23 GB/s
  V2 SR U=8 EPP=1 2-src f32+bf16->bf16 [exch]                     nBlocks=   8  avg=0.124 ms  BW=270.67 GB/s

  RTN   U=8 EPP=1 2-src f32+bf16->bf16                            nBlocks=  16  avg=0.025 ms  BW=1364.98 GB/s
  V1 SR U=8 EPP=1 2-src f32+bf16->bf16                            nBlocks=  16  avg=0.084 ms  BW=399.66 GB/s
  V2 SR U=8 EPP=1 2-src f32+bf16->bf16 [exch]                     nBlocks=  16  avg=0.065 ms  BW=516.83 GB/s

  RTN   U=8 EPP=1 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.014 ms  BW=2337.86 GB/s
  V1 SR U=8 EPP=1 2-src f32+bf16->bf16                            nBlocks=  32  avg=0.043 ms  BW=778.46 GB/s
  V2 SR U=8 EPP=1 2-src f32+bf16->bf16 [exch]                     nBlocks=  32  avg=0.035 ms  BW=963.71 GB/s

[       OK ] SimpleCopySRV2Bench.BlockSweep_2src_FloatBf16_Bf16 (202 ms)
[ RUN      ] SimpleCopySRV2Bench.BlockSizeSweep_1src_FloatToBf16

--- BlockSize sweep (nBlk=16): 1-src FP32 -> BF16 (4M elts) ---
  -- U=4, EPP=4 --
  RTN   U=4 EPP=4 f32->bf16 bs=128                                nBlocks=  16  avg=0.031 ms  BW=809.14 GB/s
  V1 SR U=4 EPP=4 f32->bf16 bs=128                                nBlocks=  16  avg=0.035 ms  BW=720.89 GB/s
  V2 SR U=4 EPP=4 f32->bf16 bs=128 [exch]                         nBlocks=  16  avg=0.033 ms  BW=767.12 GB/s

  RTN   U=4 EPP=4 f32->bf16 bs=256                                nBlocks=  16  avg=0.018 ms  BW=1362.45 GB/s
  V1 SR U=4 EPP=4 f32->bf16 bs=256                                nBlocks=  16  avg=0.027 ms  BW=943.85 GB/s
  V2 SR U=4 EPP=4 f32->bf16 bs=256 [exch]                         nBlocks=  16  avg=0.022 ms  BW=1147.37 GB/s

  RTN   U=4 EPP=4 f32->bf16 bs=384                                nBlocks=  16  avg=0.016 ms  BW=1535.04 GB/s
  V1 SR U=4 EPP=4 f32->bf16 bs=384                                nBlocks=  16  avg=0.026 ms  BW=950.96 GB/s
  V2 SR U=4 EPP=4 f32->bf16 bs=384 [exch]                         nBlocks=  16  avg=0.020 ms  BW=1276.45 GB/s

  RTN   U=4 EPP=4 f32->bf16 bs=512                                nBlocks=  16  avg=0.016 ms  BW=1533.60 GB/s
  V1 SR U=4 EPP=4 f32->bf16 bs=512                                nBlocks=  16  avg=0.025 ms  BW=1020.69 GB/s
  V2 SR U=4 EPP=4 f32->bf16 bs=512 [exch]                         nBlocks=  16  avg=0.018 ms  BW=1363.06 GB/s

  RTN   U=4 EPP=4 f32->bf16 bs=640                                nBlocks=  16  avg=0.016 ms  BW=1534.80 GB/s
  V1 SR U=4 EPP=4 f32->bf16 bs=640                                nBlocks=  16  avg=0.025 ms  BW=1018.55 GB/s
  V2 SR U=4 EPP=4 f32->bf16 bs=640 [exch]                         nBlocks=  16  avg=0.018 ms  BW=1363.32 GB/s

  -- U=8, EPP=1 --
  RTN   U=8 EPP=1 f32->bf16 bs=128                                nBlocks=  16  avg=0.055 ms  BW=454.67 GB/s
  V1 SR U=8 EPP=1 f32->bf16 bs=128                                nBlocks=  16  avg=0.178 ms  BW=141.28 GB/s
  V2 SR U=8 EPP=1 f32->bf16 bs=128 [exch]                         nBlocks=  16  avg=0.081 ms  BW=311.01 GB/s

  RTN   U=8 EPP=1 f32->bf16 bs=256                                nBlocks=  16  avg=0.031 ms  BW=812.05 GB/s
  V1 SR U=8 EPP=1 f32->bf16 bs=256                                nBlocks=  16  avg=0.104 ms  BW=241.01 GB/s
  V2 SR U=8 EPP=1 f32->bf16 bs=256 [exch]                         nBlocks=  16  avg=0.066 ms  BW=383.94 GB/s

  RTN   U=8 EPP=1 f32->bf16 bs=384                                nBlocks=  16  avg=0.024 ms  BW=1055.87 GB/s
  V1 SR U=8 EPP=1 f32->bf16 bs=384                                nBlocks=  16  avg=0.088 ms  BW=285.84 GB/s
  V2 SR U=8 EPP=1 f32->bf16 bs=384 [exch]                         nBlocks=  16  avg=0.063 ms  BW=396.87 GB/s

  RTN   U=8 EPP=1 f32->bf16 bs=512                                nBlocks=  16  avg=0.020 ms  BW=1231.40 GB/s
  V1 SR U=8 EPP=1 f32->bf16 bs=512                                nBlocks=  16  avg=0.082 ms  BW=307.26 GB/s
  V2 SR U=8 EPP=1 f32->bf16 bs=512 [exch]                         nBlocks=  16  avg=0.061 ms  BW=409.50 GB/s

  RTN   U=8 EPP=1 f32->bf16 bs=640                                nBlocks=  16  avg=0.018 ms  BW=1364.24 GB/s
  V1 SR U=8 EPP=1 f32->bf16 bs=640                                nBlocks=  16  avg=0.080 ms  BW=314.93 GB/s
  V2 SR U=8 EPP=1 f32->bf16 bs=640 [exch]                         nBlocks=  16  avg=0.062 ms  BW=407.31 GB/s

[       OK ] SimpleCopySRV2Bench.BlockSizeSweep_1src_FloatToBf16 (160 ms)
[ RUN      ] SimpleCopySRV2Bench.BlockSizeSweep_2src_FloatFloat_Bf16

--- BlockSize sweep (nBlk=16): 2-src FP32+FP32 -> BF16 (4M elts) ---
  -- U=4, EPP=4 --
  RTN   U=4 EPP=4 2-src f32+f32->bf16 bs=128                      nBlocks=  16  avg=0.037 ms  BW=1136.90 GB/s
  V1 SR U=4 EPP=4 2-src f32+f32->bf16 bs=128                      nBlocks=  16  avg=0.039 ms  BW=1076.81 GB/s
  V2 SR U=4 EPP=4 2-src f32+f32->bf16 bs=128 [exch]               nBlocks=  16  avg=0.055 ms  BW=757.58 GB/s

  RTN   U=4 EPP=4 2-src f32+f32->bf16 bs=256                      nBlocks=  16  avg=0.023 ms  BW=1861.45 GB/s
  V1 SR U=4 EPP=4 2-src f32+f32->bf16 bs=256                      nBlocks=  16  avg=0.029 ms  BW=1461.73 GB/s
  V2 SR U=4 EPP=4 2-src f32+f32->bf16 bs=256 [exch]               nBlocks=  16  avg=0.037 ms  BW=1140.38 GB/s

  RTN   U=4 EPP=4 2-src f32+f32->bf16 bs=384                      nBlocks=  16  avg=0.021 ms  BW=2034.27 GB/s
  V1 SR U=4 EPP=4 2-src f32+f32->bf16 bs=384                      nBlocks=  16  avg=0.027 ms  BW=1573.93 GB/s
  V2 SR U=4 EPP=4 2-src f32+f32->bf16 bs=384 [exch]               nBlocks=  16  avg=0.029 ms  BW=1429.01 GB/s

  RTN   U=4 EPP=4 2-src f32+f32->bf16 bs=512                      nBlocks=  16  avg=0.021 ms  BW=2035.03 GB/s
  V1 SR U=4 EPP=4 2-src f32+f32->bf16 bs=512                      nBlocks=  16  avg=0.027 ms  BW=1575.42 GB/s
  V2 SR U=4 EPP=4 2-src f32+f32->bf16 bs=512 [exch]               nBlocks=  16  avg=0.026 ms  BW=1607.44 GB/s

  RTN   U=4 EPP=4 2-src f32+f32->bf16 bs=640                      nBlocks=  16  avg=0.021 ms  BW=2023.06 GB/s
  V1 SR U=4 EPP=4 2-src f32+f32->bf16 bs=640                      nBlocks=  16  avg=0.027 ms  BW=1573.87 GB/s
  V2 SR U=4 EPP=4 2-src f32+f32->bf16 bs=640 [exch]               nBlocks=  16  avg=0.024 ms  BW=1724.00 GB/s

  -- U=8, EPP=1 --
  RTN   U=8 EPP=1 2-src f32+f32->bf16 bs=128                      nBlocks=  16  avg=0.065 ms  BW=641.39 GB/s
  V1 SR U=8 EPP=1 2-src f32+f32->bf16 bs=128                      nBlocks=  16  avg=0.188 ms  BW=223.35 GB/s
  V2 SR U=8 EPP=1 2-src f32+f32->bf16 bs=128 [exch]               nBlocks=  16  avg=0.123 ms  BW=341.45 GB/s

  RTN   U=8 EPP=1 2-src f32+f32->bf16 bs=256                      nBlocks=  16  avg=0.037 ms  BW=1140.34 GB/s
  V1 SR U=8 EPP=1 2-src f32+f32->bf16 bs=256                      nBlocks=  16  avg=0.111 ms  BW=378.33 GB/s
  V2 SR U=8 EPP=1 2-src f32+f32->bf16 bs=256 [exch]               nBlocks=  16  avg=0.088 ms  BW=476.88 GB/s

  RTN   U=8 EPP=1 2-src f32+f32->bf16 bs=384                      nBlocks=  16  avg=0.027 ms  BW=1533.92 GB/s
  V1 SR U=8 EPP=1 2-src f32+f32->bf16 bs=384                      nBlocks=  16  avg=0.092 ms  BW=455.05 GB/s
  V2 SR U=8 EPP=1 2-src f32+f32->bf16 bs=384 [exch]               nBlocks=  16  avg=0.072 ms  BW=583.61 GB/s

  RTN   U=8 EPP=1 2-src f32+f32->bf16 bs=512                      nBlocks=  16  avg=0.025 ms  BW=1706.44 GB/s
  V1 SR U=8 EPP=1 2-src f32+f32->bf16 bs=512                      nBlocks=  16  avg=0.084 ms  BW=499.30 GB/s
  V2 SR U=8 EPP=1 2-src f32+f32->bf16 bs=512 [exch]               nBlocks=  16  avg=0.066 ms  BW=638.40 GB/s

  RTN   U=8 EPP=1 2-src f32+f32->bf16 bs=640                      nBlocks=  16  avg=0.024 ms  BW=1732.79 GB/s
  V1 SR U=8 EPP=1 2-src f32+f32->bf16 bs=640                      nBlocks=  16  avg=0.083 ms  BW=507.44 GB/s
  V2 SR U=8 EPP=1 2-src f32+f32->bf16 bs=640 [exch]               nBlocks=  16  avg=0.065 ms  BW=649.07 GB/s

[       OK ] SimpleCopySRV2Bench.BlockSizeSweep_2src_FloatFloat_Bf16 (185 ms)
[ RUN      ] SimpleCopySRV2Bench.BlockSizeSweep_2src_FloatBf16_Bf16

--- BlockSize sweep (nBlk=16): 2-src FP32+BF16 -> BF16 (4M elts) ---
  -- U=4, EPP=4 --
  RTN   U=4 EPP=4 2-src f32+bf16->bf16 bs=128                     nBlocks=  16  avg=0.037 ms  BW=916.84 GB/s
  V1 SR U=4 EPP=4 2-src f32+bf16->bf16 bs=128                     nBlocks=  16  avg=0.040 ms  BW=838.51 GB/s
  V2 SR U=4 EPP=4 2-src f32+bf16->bf16 bs=128 [exch]              nBlocks=  16  avg=0.056 ms  BW=597.59 GB/s

  RTN   U=4 EPP=4 2-src f32+bf16->bf16 bs=256                     nBlocks=  16  avg=0.023 ms  BW=1488.21 GB/s
  V1 SR U=4 EPP=4 2-src f32+bf16->bf16 bs=256                     nBlocks=  16  avg=0.031 ms  BW=1089.15 GB/s
  V2 SR U=4 EPP=4 2-src f32+bf16->bf16 bs=256 [exch]              nBlocks=  16  avg=0.037 ms  BW=909.71 GB/s

  RTN   U=4 EPP=4 2-src f32+bf16->bf16 bs=384                     nBlocks=  16  avg=0.018 ms  BW=1818.64 GB/s
  V1 SR U=4 EPP=4 2-src f32+bf16->bf16 bs=384                     nBlocks=  16  avg=0.029 ms  BW=1169.40 GB/s
  V2 SR U=4 EPP=4 2-src f32+bf16->bf16 bs=384 [exch]              nBlocks=  16  avg=0.031 ms  BW=1091.49 GB/s

  RTN   U=4 EPP=4 2-src f32+bf16->bf16 bs=512                     nBlocks=  16  avg=0.018 ms  BW=1818.68 GB/s
  V1 SR U=4 EPP=4 2-src f32+bf16->bf16 bs=512                     nBlocks=  16  avg=0.029 ms  BW=1171.29 GB/s
  V2 SR U=4 EPP=4 2-src f32+bf16->bf16 bs=512 [exch]              nBlocks=  16  avg=0.029 ms  BW=1170.01 GB/s

  RTN   U=4 EPP=4 2-src f32+bf16->bf16 bs=640                     nBlocks=  16  avg=0.018 ms  BW=1818.46 GB/s
  V1 SR U=4 EPP=4 2-src f32+bf16->bf16 bs=640                     nBlocks=  16  avg=0.029 ms  BW=1170.21 GB/s
  V2 SR U=4 EPP=4 2-src f32+bf16->bf16 bs=640 [exch]              nBlocks=  16  avg=0.027 ms  BW=1258.40 GB/s

  -- U=8, EPP=1 --
  RTN   U=8 EPP=1 2-src f32+bf16->bf16 bs=128                     nBlocks=  16  avg=0.066 ms  BW=511.65 GB/s
  V1 SR U=8 EPP=1 2-src f32+bf16->bf16 bs=128                     nBlocks=  16  avg=0.188 ms  BW=178.15 GB/s
  V2 SR U=8 EPP=1 2-src f32+bf16->bf16 bs=128 [exch]              nBlocks=  16  avg=0.125 ms  BW=268.49 GB/s

  RTN   U=8 EPP=1 2-src f32+bf16->bf16 bs=256                     nBlocks=  16  avg=0.036 ms  BW=929.88 GB/s
  V1 SR U=8 EPP=1 2-src f32+bf16->bf16 bs=256                     nBlocks=  16  avg=0.111 ms  BW=303.55 GB/s
  V2 SR U=8 EPP=1 2-src f32+bf16->bf16 bs=256 [exch]              nBlocks=  16  avg=0.088 ms  BW=380.43 GB/s

  RTN   U=8 EPP=1 2-src f32+bf16->bf16 bs=384                     nBlocks=  16  avg=0.027 ms  BW=1256.61 GB/s
  V1 SR U=8 EPP=1 2-src f32+bf16->bf16 bs=384                     nBlocks=  16  avg=0.092 ms  BW=364.36 GB/s
  V2 SR U=8 EPP=1 2-src f32+bf16->bf16 bs=384 [exch]              nBlocks=  16  avg=0.072 ms  BW=467.02 GB/s

  RTN   U=8 EPP=1 2-src f32+bf16->bf16 bs=512                     nBlocks=  16  avg=0.025 ms  BW=1342.54 GB/s
  V1 SR U=8 EPP=1 2-src f32+bf16->bf16 bs=512                     nBlocks=  16  avg=0.084 ms  BW=399.28 GB/s
  V2 SR U=8 EPP=1 2-src f32+bf16->bf16 bs=512 [exch]              nBlocks=  16  avg=0.066 ms  BW=509.93 GB/s

  RTN   U=8 EPP=1 2-src f32+bf16->bf16 bs=640                     nBlocks=  16  avg=0.025 ms  BW=1365.37 GB/s
  V1 SR U=8 EPP=1 2-src f32+bf16->bf16 bs=640                     nBlocks=  16  avg=0.084 ms  BW=399.65 GB/s
  V2 SR U=8 EPP=1 2-src f32+bf16->bf16 bs=640 [exch]              nBlocks=  16  avg=0.065 ms  BW=518.30 GB/s
*/
