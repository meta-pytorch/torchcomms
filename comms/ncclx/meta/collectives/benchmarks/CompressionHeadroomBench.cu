// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "comms/utils/kernels/rng/philox_rng.cuh"
#include "meta/collectives/kernels/reduce_copy.cuh"

// =============================================================================
// Dummy Compute Functions
//
// These are injected between reduce and store to measure how much compute
// can be performed "for free" while the kernel remains memory-bandwidth bound.
// =============================================================================

// Variant 1: FMA identity ops (acc-dependent, FP32 pipeline)
// Each "op" adds 1 serial FMA per element per unroll slot.
// fma(x, 1.0f, 0.0f) = x, so the result is unchanged.
template <int NumOps, int Unroll, int EltPerPack>
__device__ __forceinline__ void dummyComputeFMA(
    BytePack<EltPerPack * sizeof(float)> acc[Unroll]) {
#pragma unroll
  for (int u = 0; u < Unroll; u++) {
    float* vals = reinterpret_cast<float*>(&acc[u]);
#pragma unroll
    for (int e = 0; e < EltPerPack; e++) {
#pragma unroll
      for (int op = 0; op < NumOps; op++) {
        asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f00000000;"
                     : "+f"(vals[e]));
      }
    }
  }
}

// Variant 2: Integer XOR pairs (acc-dependent, INT32 pipeline)
// Two XORs with the same constant cancel out, preserving the value.
template <int NumOps, int Unroll, int EltPerPack>
__device__ __forceinline__ void dummyComputeINT(
    BytePack<EltPerPack * sizeof(float)> acc[Unroll]) {
#pragma unroll
  for (int u = 0; u < Unroll; u++) {
    uint32_t* vals = reinterpret_cast<uint32_t*>(&acc[u]);
#pragma unroll
    for (int e = 0; e < EltPerPack; e++) {
#pragma unroll
      for (int op = 0; op < NumOps; op++) {
        asm volatile(
            "xor.b32 %0, %0, 0x12345678;\n"
            "xor.b32 %0, %0, 0x12345678;"
            : "+r"(vals[e]));
      }
    }
  }
}

// Variant 3: FMA + extra registers (acc-dependent, register pressure test)
// Simulates compression state (histograms, LUTs) that must be kept in
// registers alongside acc[].
template <int NumOps, int NumExtraRegs, int Unroll, int EltPerPack>
__device__ __forceinline__ void dummyComputeWithRegs(
    BytePack<EltPerPack * sizeof(float)> acc[Unroll]) {
  float extra[NumExtraRegs];
#pragma unroll
  for (int r = 0; r < NumExtraRegs; r++) {
    extra[r] = 0.0f;
  }

#pragma unroll
  for (int u = 0; u < Unroll; u++) {
    float* vals = reinterpret_cast<float*>(&acc[u]);
#pragma unroll
    for (int e = 0; e < EltPerPack; e++) {
#pragma unroll
      for (int op = 0; op < NumOps; op++) {
        asm volatile("fma.rn.f32 %0, %1, 0f3F800000, %0;"
                     : "+f"(extra[op % NumExtraRegs])
                     : "f"(vals[e]));
      }
    }
  }

  // Prevent DCE: fold extra[] into acc with zero impact
  float sink = 0.0f;
#pragma unroll
  for (int r = 0; r < NumExtraRegs; r++) {
    asm volatile("fma.rn.f32 %0, %1, 0f00000000, %0;"
                 : "+f"(sink)
                 : "f"(extra[r]));
  }
  float* vals = reinterpret_cast<float*>(&acc[0]);
  asm volatile("add.f32 %0, %0, %1;" : "+f"(vals[0]) : "f"(sink));
}

// Variant 4: Acc-independent compute (simulates RNG / compression prep)
// The compute chain does NOT depend on acc[] values, so the GPU's instruction
// scheduler can potentially issue these operations in parallel with loads.
template <int NumOps, int Unroll, int EltPerPack>
__device__ __forceinline__ void dummyComputeIndependent(
    BytePack<EltPerPack * sizeof(float)> acc[Unroll],
    uint64_t seed,
    uint64_t threadOffset) {
  // Philox-style RNG: integer multiply-add chains.
  uint32_t state0 = static_cast<uint32_t>(seed);
  uint32_t state1 = static_cast<uint32_t>(seed >> 32);
  uint32_t ctr0 = static_cast<uint32_t>(threadOffset);
  uint32_t ctr1 = static_cast<uint32_t>(threadOffset >> 32);

#pragma unroll
  for (int op = 0; op < NumOps; op++) {
    asm volatile(
        "mad.lo.u32 %0, %0, 0xD2511F53, %2;\n"
        "mad.lo.u32 %1, %1, 0xCD9E8D57, %3;\n"
        "xor.b32 %0, %0, %1;\n"
        "add.u32 %2, %2, 1;"
        : "+r"(state0), "+r"(state1), "+r"(ctr0), "+r"(ctr1));
  }

  // Prevent DCE: fold RNG output into acc[0] with zero impact
  float rng_sink = __uint_as_float((state0 & 0x007FFFFF) | 0x3F800000) - 1.0f;
  asm volatile("fma.rn.f32 %0, %1, 0f00000000, %0;"
               : "+f"(*reinterpret_cast<float*>(&acc[0]))
               : "f"(rng_sink));
}

// Variant 5: Real Philox RNG (acc-independent, uses philox_rng.cuh)
// Calls philox_randint4x NumRounds times, each generating 4 random uint32s.
// Independent of acc[] — models stochastic rounding RNG generation.
template <int NumRounds, int Unroll, int EltPerPack>
__device__ __forceinline__ void dummyComputePhilox(
    BytePack<EltPerPack * sizeof(float)> acc[Unroll],
    uint64_t seed,
    uint64_t baseOffset) {
  uint32_t r0, r1, r2, r3;
  uint32_t sink = 0;

#pragma unroll
  for (int round = 0; round < NumRounds; round++) {
    philox_randint4x(seed, baseOffset + round, r0, r1, r2, r3);
    // Accumulate into sink to prevent DCE across rounds
    sink ^= r0 ^ r1 ^ r2 ^ r3;
  }

  // Prevent DCE: fold sink into acc[0] with zero impact
  float fsink = __uint_as_float((sink & 0x007FFFFF) | 0x3F800000) - 1.0f;
  asm volatile("fma.rn.f32 %0, %1, 0f00000000, %0;"
               : "+f"(*reinterpret_cast<float*>(&acc[0]))
               : "f"(fsink));
}

// =============================================================================
// Compute function wrappers (for use as template parameters)
// =============================================================================

struct ComputeFMA {
  template <int NumOps, int Unroll, int EltPerPack>
  static __device__ __forceinline__ void apply(
      BytePack<EltPerPack * sizeof(float)> acc[Unroll]) {
    dummyComputeFMA<NumOps, Unroll, EltPerPack>(acc);
  }
};

struct ComputeINT {
  template <int NumOps, int Unroll, int EltPerPack>
  static __device__ __forceinline__ void apply(
      BytePack<EltPerPack * sizeof(float)> acc[Unroll]) {
    dummyComputeINT<NumOps, Unroll, EltPerPack>(acc);
  }
};

template <int NumExtraRegs>
struct ComputeWithRegs {
  template <int NumOps, int Unroll, int EltPerPack>
  static __device__ __forceinline__ void apply(
      BytePack<EltPerPack * sizeof(float)> acc[Unroll]) {
    dummyComputeWithRegs<NumOps, NumExtraRegs, Unroll, EltPerPack>(acc);
  }
};

struct ComputeIndependent {
  template <int NumOps, int Unroll, int EltPerPack>
  static __device__ __forceinline__ void apply(
      BytePack<EltPerPack * sizeof(float)> acc[Unroll]) {
    // Use a fixed seed and thread-derived offset
    uint64_t seed = 0xDEADBEEFCAFEBABEULL;
    uint64_t threadOffset =
        static_cast<uint64_t>(threadIdx.x + blockIdx.x * blockDim.x);
    dummyComputeIndependent<NumOps, Unroll, EltPerPack>(
        acc, seed, threadOffset);
  }
};

struct ComputePhilox {
  template <int NumRounds, int Unroll, int EltPerPack>
  static __device__ __forceinline__ void apply(
      BytePack<EltPerPack * sizeof(float)> acc[Unroll]) {
    uint64_t seed = 0xDEADBEEFCAFEBABEULL;
    uint64_t baseOffset =
        static_cast<uint64_t>(threadIdx.x + blockIdx.x * blockDim.x) * 1024;
    dummyComputePhilox<NumRounds, Unroll, EltPerPack>(acc, seed, baseOffset);
  }
};

// =============================================================================
// Benchmark kernels
//
// These replicate the reduceCopyPacks inner loop with dummyCompute injected
// between reduce and store. We use the CopyIterator and helper functions
// from copy_kernel.cuh directly.
// =============================================================================

// 2-source reduce-copy with injected compute
template <
    int Unroll,
    int NumOps,
    typename ComputeFn,
    typename T,
    int EltPerPack = 16 / sizeof(T)>
__global__ __launch_bounds__(256, 1) void reduce_copy_headroom_2src_kernel(
    T* dst,
    const T* src0,
    const T* src1,
    ssize_t nElts) {
  auto nThreads = blockDim.x * gridDim.x;
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  ssize_t nEltsBehind = 0;
  ssize_t nEltsAhead = nElts;

  meta::comms::ncclx::kernels::CopyIterator<
      Unroll,
      EltPerPack,
      ssize_t,
      decltype(nThreads),
      const T,
      const T,
      T>
      iter(nThreads, thread, nEltsBehind, nEltsAhead, src0, src1, dst);

  while (iter.hasWork()) {
    BytePack<EltPerPack * sizeof(T)> acc[Unroll];

    meta::comms::ncclx::kernels::loadFirstSource<Unroll, EltPerPack, T>(
        acc, iter);

    meta::comms::ncclx::kernels::ReduceSources<1, 2>::
        template apply<Unroll, EltPerPack, T>(acc, iter);

    // >>> INJECTED COMPUTE <<<
    ComputeFn::template apply<NumOps, Unroll, EltPerPack>(acc);

    meta::comms::ncclx::kernels::simplecopy::
        storeFirstDestination<Unroll, EltPerPack, T, 2>(acc, iter);

    iter.advance();
  }
}

// 1-source copy (no reduce) with injected compute
template <
    int Unroll,
    int NumOps,
    typename ComputeFn,
    typename T,
    int EltPerPack = 16 / sizeof(T)>
__global__ __launch_bounds__(
    256,
    1) void copy_headroom_kernel(T* dst, const T* src, ssize_t nElts) {
  auto nThreads = blockDim.x * gridDim.x;
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  ssize_t nEltsBehind = 0;
  ssize_t nEltsAhead = nElts;

  meta::comms::ncclx::kernels::
      CopyIterator<Unroll, EltPerPack, ssize_t, decltype(nThreads), const T, T>
          iter(nThreads, thread, nEltsBehind, nEltsAhead, src, dst);

  while (iter.hasWork()) {
    BytePack<EltPerPack * sizeof(T)> acc[Unroll];

    meta::comms::ncclx::kernels::loadFirstSource<Unroll, EltPerPack, T>(
        acc, iter);

    // No reduce for single-source copy

    // >>> INJECTED COMPUTE <<<
    ComputeFn::template apply<NumOps, Unroll, EltPerPack>(acc);

    meta::comms::ncclx::kernels::simplecopy::
        storeFirstDestination<Unroll, EltPerPack, T, 1>(acc, iter);

    iter.advance();
  }
}

// =============================================================================
// CUDA error checking
// =============================================================================

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Benchmark Fixture
// =============================================================================

class CompressionHeadroomBench : public ::testing::Test {
 protected:
  using T = float;
  static constexpr int64_t kN = 4L * 1024L * 1024L; // 4M elements
  static constexpr int64_t kNLarge = 64L * 1024L * 1024L; // 64M elements
  static constexpr int kBlockSize = 256;
  static constexpr int kDefaultBlocks = 32;
  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;

  T* d_input = nullptr;
  T* d_output = nullptr;
  T* d_input2 = nullptr;
  cudaEvent_t startEvent, stopEvent;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_input, kNLarge * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_output, kNLarge * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_input2, kNLarge * sizeof(T)));
    CUDACHECK(cudaMemset(d_input, 1, kNLarge * sizeof(T)));
    CUDACHECK(cudaMemset(d_input2, 2, kNLarge * sizeof(T)));
    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&stopEvent));
  }

  void TearDown() override {
    CUDACHECK(cudaEventDestroy(startEvent));
    CUDACHECK(cudaEventDestroy(stopEvent));
    CUDACHECK(cudaFree(d_input));
    CUDACHECK(cudaFree(d_output));
    CUDACHECK(cudaFree(d_input2));
  }

  template <typename LaunchFn>
  float runBench(int64_t nElts, int nBlocks, LaunchFn launchFn) {
    // Warmup
    for (int i = 0; i < kWarmupIters; i++) {
      launchFn(nBlocks, kBlockSize, nElts);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Timed iterations
    EXPECT_EQ(cudaEventRecord(startEvent), cudaSuccess);
    for (int i = 0; i < kBenchIters; i++) {
      launchFn(nBlocks, kBlockSize, nElts);
    }
    EXPECT_EQ(cudaEventRecord(stopEvent), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float elapsedMs = 0.0f;
    EXPECT_EQ(
        cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), cudaSuccess);
    return elapsedMs / kBenchIters;
  }

  static double computeBW(float avgMs, int64_t nElts, int nMemAccesses) {
    size_t totalBytes = (size_t)nMemAccesses * nElts * sizeof(T);
    return (double)totalBytes / (avgMs * 1e6);
  }

  // Run a NumOps sweep for copy (1 src) kernels
  template <typename ComputeFn>
  void runCopyNumOpsSweep(const char* computeName, int nBlocks, int64_t nElts) {
    printf(
        "\n--- Copy %s: NumOps sweep (%d blocks, %ldM elements) ---\n",
        computeName,
        nBlocks,
        (long)(nElts / (1024L * 1024L)));

    // Baseline (NumOps=0)
    float baseMs = runBench(nElts, nBlocks, [&](int nBlk, int bs, int64_t n) {
      copy_headroom_kernel<4, 0, ComputeFn, T>
          <<<nBlk, bs>>>(d_output, d_input, (ssize_t)n);
      CUDACHECK(cudaGetLastError());
    });
    double baseBW = computeBW(baseMs, nElts, 2);
    printf(
        "  NumOps=%-5d  avg=%.3f ms  BW=%.2f GB/s  (baseline)\n",
        0,
        baseMs,
        baseBW);

    auto runOne = [&](auto numOpsTag) {
      constexpr int N = decltype(numOpsTag)::value;
      float ms = runBench(nElts, nBlocks, [&](int nBlk, int bs, int64_t n) {
        copy_headroom_kernel<4, N, ComputeFn, T>
            <<<nBlk, bs>>>(d_output, d_input, (ssize_t)n);
        CUDACHECK(cudaGetLastError());
      });
      double bw = computeBW(ms, nElts, 2);
      printf(
          "  NumOps=%-5d  avg=%.3f ms  BW=%.2f GB/s  (%.1f%% of baseline)\n",
          N,
          ms,
          bw,
          100.0 * bw / baseBW);
    };

    runOne(std::integral_constant<int, 1>{});
    runOne(std::integral_constant<int, 2>{});
    runOne(std::integral_constant<int, 4>{});
    runOne(std::integral_constant<int, 8>{});
    runOne(std::integral_constant<int, 16>{});
    runOne(std::integral_constant<int, 32>{});
    runOne(std::integral_constant<int, 64>{});
    runOne(std::integral_constant<int, 128>{});
    runOne(std::integral_constant<int, 256>{});
    runOne(std::integral_constant<int, 512>{});
    runOne(std::integral_constant<int, 1024>{});
  }

  // Run a NumOps sweep for reduce-copy (2 src) kernels
  template <typename ComputeFn>
  void runReduceCopy2SrcNumOpsSweep(
      const char* computeName,
      int nBlocks,
      int64_t nElts) {
    printf(
        "\n--- ReduceCopy 2-src %s: NumOps sweep (%d blocks, %ldM elements) ---\n",
        computeName,
        nBlocks,
        (long)(nElts / (1024L * 1024L)));

    // Baseline (NumOps=0)
    float baseMs = runBench(nElts, nBlocks, [&](int nBlk, int bs, int64_t n) {
      reduce_copy_headroom_2src_kernel<4, 0, ComputeFn, T>
          <<<nBlk, bs>>>(d_output, d_input, d_input2, (ssize_t)n);
      CUDACHECK(cudaGetLastError());
    });
    double baseBW = computeBW(baseMs, nElts, 3);
    printf(
        "  NumOps=%-5d  avg=%.3f ms  BW=%.2f GB/s  (baseline)\n",
        0,
        baseMs,
        baseBW);

    auto runOne = [&](auto numOpsTag) {
      constexpr int N = decltype(numOpsTag)::value;
      float ms = runBench(nElts, nBlocks, [&](int nBlk, int bs, int64_t n) {
        reduce_copy_headroom_2src_kernel<4, N, ComputeFn, T>
            <<<nBlk, bs>>>(d_output, d_input, d_input2, (ssize_t)n);
        CUDACHECK(cudaGetLastError());
      });
      double bw = computeBW(ms, nElts, 3);
      printf(
          "  NumOps=%-5d  avg=%.3f ms  BW=%.2f GB/s  (%.1f%% of baseline)\n",
          N,
          ms,
          bw,
          100.0 * bw / baseBW);
    };

    runOne(std::integral_constant<int, 1>{});
    runOne(std::integral_constant<int, 2>{});
    runOne(std::integral_constant<int, 4>{});
    runOne(std::integral_constant<int, 8>{});
    runOne(std::integral_constant<int, 16>{});
    runOne(std::integral_constant<int, 32>{});
    runOne(std::integral_constant<int, 64>{});
    runOne(std::integral_constant<int, 128>{});
    runOne(std::integral_constant<int, 256>{});
    runOne(std::integral_constant<int, 512>{});
    runOne(std::integral_constant<int, 1024>{});
  }
};

// =============================================================================
// Pure compute headroom: FMA (acc-dependent, FP32 pipeline)
// =============================================================================

TEST_F(CompressionHeadroomBench, CopyFMA_NumOpsSweep) {
  runCopyNumOpsSweep<ComputeFMA>("FMA", kDefaultBlocks, kN);
}

TEST_F(CompressionHeadroomBench, ReduceCopy2SrcFMA_NumOpsSweep) {
  runReduceCopy2SrcNumOpsSweep<ComputeFMA>("FMA", kDefaultBlocks, kN);
}

// =============================================================================
// Pure compute headroom: INT (acc-dependent, INT32 pipeline)
// =============================================================================

TEST_F(CompressionHeadroomBench, CopyINT_NumOpsSweep) {
  runCopyNumOpsSweep<ComputeINT>("INT", kDefaultBlocks, kN);
}

TEST_F(CompressionHeadroomBench, ReduceCopy2SrcINT_NumOpsSweep) {
  runReduceCopy2SrcNumOpsSweep<ComputeINT>("INT", kDefaultBlocks, kN);
}

// =============================================================================
// Acc-independent compute (RNG / compression prep that can overlap with loads)
// =============================================================================

TEST_F(CompressionHeadroomBench, CopyIndependent_NumOpsSweep) {
  runCopyNumOpsSweep<ComputeIndependent>("Independent", kDefaultBlocks, kN);
}

TEST_F(CompressionHeadroomBench, ReduceCopy2SrcIndependent_NumOpsSweep) {
  runReduceCopy2SrcNumOpsSweep<ComputeIndependent>(
      "Independent", kDefaultBlocks, kN);
}

// =============================================================================
// Real Philox RNG headroom (acc-independent, uses philox_rng.cuh)
// Sweeps number of philox_randint4x calls per loop iteration.
// Each call generates 4 random uint32s using 7-round Philox.
// =============================================================================

TEST_F(CompressionHeadroomBench, CopyPhilox_NumRoundsSweep) {
  runCopyNumOpsSweep<ComputePhilox>("Philox", kDefaultBlocks, kN);
}

TEST_F(CompressionHeadroomBench, ReduceCopy2SrcPhilox_NumRoundsSweep) {
  runReduceCopy2SrcNumOpsSweep<ComputePhilox>("Philox", kDefaultBlocks, kN);
}

// =============================================================================
// Register pressure tests: sweep NumExtraRegs at fixed NumOps
// =============================================================================

TEST_F(CompressionHeadroomBench, ReduceCopy2SrcFMA_RegPressureSweep) {
  printf(
      "\n--- ReduceCopy 2-src FMA: Register pressure sweep (NumOps=8, %d blocks, 4M elements) ---\n",
      kDefaultBlocks);

  // Baseline (no extra regs, NumOps=8)
  float baseMs = runBench(kN, kDefaultBlocks, [&](int nBlk, int bs, int64_t n) {
    reduce_copy_headroom_2src_kernel<4, 8, ComputeFMA, T>
        <<<nBlk, bs>>>(d_output, d_input, d_input2, (ssize_t)n);
    CUDACHECK(cudaGetLastError());
  });
  double baseBW = computeBW(baseMs, kN, 3);
  printf(
      "  NumExtraRegs=%-3d  avg=%.3f ms  BW=%.2f GB/s  (baseline, no extra regs)\n",
      0,
      baseMs,
      baseBW);

  auto runRegs = [&](auto regsTag) {
    constexpr int R = decltype(regsTag)::value;
    float ms = runBench(kN, kDefaultBlocks, [&](int nBlk, int bs, int64_t n) {
      reduce_copy_headroom_2src_kernel<4, 8, ComputeWithRegs<R>, T>
          <<<nBlk, bs>>>(d_output, d_input, d_input2, (ssize_t)n);
      CUDACHECK(cudaGetLastError());
    });
    double bw = computeBW(ms, kN, 3);
    printf(
        "  NumExtraRegs=%-3d  avg=%.3f ms  BW=%.2f GB/s  (%.1f%% of baseline)\n",
        R,
        ms,
        bw,
        100.0 * bw / baseBW);
  };

  runRegs(std::integral_constant<int, 4>{});
  runRegs(std::integral_constant<int, 8>{});
  runRegs(std::integral_constant<int, 16>{});
  runRegs(std::integral_constant<int, 32>{});
}

// =============================================================================
// Register pressure + ops 2D sweep: NumOps x NumExtraRegs
// =============================================================================

TEST_F(CompressionHeadroomBench, ReduceCopy2SrcFMA_RegsAndOpsSweep) {
  printf(
      "\n--- ReduceCopy 2-src FMA: NumOps x NumExtraRegs sweep (%d blocks, 4M elements) ---\n",
      kDefaultBlocks);

  // Baseline (NumOps=0, no extra regs)
  float baseMs = runBench(kN, kDefaultBlocks, [&](int nBlk, int bs, int64_t n) {
    reduce_copy_headroom_2src_kernel<4, 0, ComputeFMA, T>
        <<<nBlk, bs>>>(d_output, d_input, d_input2, (ssize_t)n);
    CUDACHECK(cudaGetLastError());
  });
  double baseBW = computeBW(baseMs, kN, 3);
  printf(
      "  NumOps=0, NumExtraRegs=0   avg=%.3f ms  BW=%.2f GB/s  (baseline)\n",
      baseMs,
      baseBW);

  auto runConfig = [&](auto opsTag, auto regsTag) {
    constexpr int O = decltype(opsTag)::value;
    constexpr int R = decltype(regsTag)::value;
    float ms = runBench(kN, kDefaultBlocks, [&](int nBlk, int bs, int64_t n) {
      reduce_copy_headroom_2src_kernel<4, O, ComputeWithRegs<R>, T>
          <<<nBlk, bs>>>(d_output, d_input, d_input2, (ssize_t)n);
      CUDACHECK(cudaGetLastError());
    });
    double bw = computeBW(ms, kN, 3);
    printf(
        "  NumOps=%-3d, NumExtraRegs=%-3d  avg=%.3f ms  BW=%.2f GB/s  (%.1f%% of baseline)\n",
        O,
        R,
        ms,
        bw,
        100.0 * bw / baseBW);
  };

  // Sweep: NumOps = {4, 8, 16, 32} x NumExtraRegs = {4, 8, 16, 32}
  auto sweep = [&](auto opsTag) {
    runConfig(opsTag, std::integral_constant<int, 4>{});
    runConfig(opsTag, std::integral_constant<int, 8>{});
    runConfig(opsTag, std::integral_constant<int, 16>{});
    runConfig(opsTag, std::integral_constant<int, 32>{});
  };

  sweep(std::integral_constant<int, 4>{});
  sweep(std::integral_constant<int, 8>{});
  sweep(std::integral_constant<int, 16>{});
  sweep(std::integral_constant<int, 32>{});
}

// =============================================================================
// Size sweep: 64M elements
// =============================================================================

TEST_F(CompressionHeadroomBench, ReduceCopy2SrcFMA_SizeSweep) {
  printf(
      "\n--- ReduceCopy 2-src FMA: Size sweep (%d blocks, 64M elements) ---\n",
      kDefaultBlocks);

  // Baseline (NumOps=0)
  float baseMs =
      runBench(kNLarge, kDefaultBlocks, [&](int nBlk, int bs, int64_t n) {
        reduce_copy_headroom_2src_kernel<4, 0, ComputeFMA, T>
            <<<nBlk, bs>>>(d_output, d_input, d_input2, (ssize_t)n);
        CUDACHECK(cudaGetLastError());
      });
  double baseBW = computeBW(baseMs, kNLarge, 3);
  printf(
      "  NumOps=%-5d  avg=%.3f ms  BW=%.2f GB/s  (baseline)\n",
      0,
      baseMs,
      baseBW);

  auto runOne = [&](auto numOpsTag) {
    constexpr int N = decltype(numOpsTag)::value;
    float ms =
        runBench(kNLarge, kDefaultBlocks, [&](int nBlk, int bs, int64_t n) {
          reduce_copy_headroom_2src_kernel<4, N, ComputeFMA, T>
              <<<nBlk, bs>>>(d_output, d_input, d_input2, (ssize_t)n);
          CUDACHECK(cudaGetLastError());
        });
    double bw = computeBW(ms, kNLarge, 3);
    printf(
        "  NumOps=%-5d  avg=%.3f ms  BW=%.2f GB/s  (%.1f%% of baseline)\n",
        N,
        ms,
        bw,
        100.0 * bw / baseBW);
  };

  runOne(std::integral_constant<int, 1>{});
  runOne(std::integral_constant<int, 2>{});
  runOne(std::integral_constant<int, 4>{});
  runOne(std::integral_constant<int, 8>{});
  runOne(std::integral_constant<int, 16>{});
  runOne(std::integral_constant<int, 32>{});
  runOne(std::integral_constant<int, 64>{});
  runOne(std::integral_constant<int, 128>{});
  runOne(std::integral_constant<int, 256>{});
  runOne(std::integral_constant<int, 512>{});
  runOne(std::integral_constant<int, 1024>{});
}

// clang-format off

/*
--- Copy FMA: NumOps sweep (32 blocks, 4M elements) ---
  NumOps=0      avg=0.014 ms  BW=2337.96 GB/s  (baseline)
  NumOps=1      avg=0.014 ms  BW=2338.33 GB/s  (100.0% of baseline)
  NumOps=2      avg=0.014 ms  BW=2337.96 GB/s  (100.0% of baseline)
  NumOps=4      avg=0.014 ms  BW=2337.02 GB/s  (100.0% of baseline)
  NumOps=8      avg=0.016 ms  BW=2046.64 GB/s  (87.5% of baseline)
  NumOps=16     avg=0.020 ms  BW=1637.58 GB/s  (70.0% of baseline)
  NumOps=32     avg=0.029 ms  BW=1166.96 GB/s  (49.9% of baseline)
  NumOps=64     avg=0.047 ms  BW=718.14 GB/s  (30.7% of baseline)
  NumOps=128    avg=0.080 ms  BW=419.57 GB/s  (17.9% of baseline)
  NumOps=256    avg=0.186 ms  BW=180.10 GB/s  (7.7% of baseline)
  NumOps=512    avg=0.379 ms  BW=88.59 GB/s  (3.8% of baseline)
  NumOps=1024   avg=1.604 ms  BW=20.93 GB/s  (0.9% of baseline)
[       OK ] CompressionHeadroomBench.CopyFMA_NumOpsSweep (463 ms)
[ RUN      ] CompressionHeadroomBench.ReduceCopy2SrcFMA_NumOpsSweep

--- ReduceCopy 2-src FMA: NumOps sweep (32 blocks, 4M elements) ---
  NumOps=0      avg=0.016 ms  BW=3066.97 GB/s  (baseline)
  NumOps=1      avg=0.016 ms  BW=3068.40 GB/s  (100.0% of baseline)
  NumOps=2      avg=0.016 ms  BW=3070.08 GB/s  (100.1% of baseline)
  NumOps=4      avg=0.016 ms  BW=3066.31 GB/s  (100.0% of baseline)
  NumOps=8      avg=0.018 ms  BW=2728.44 GB/s  (89.0% of baseline)
  NumOps=16     avg=0.021 ms  BW=2354.27 GB/s  (76.8% of baseline)
  NumOps=32     avg=0.029 ms  BW=1753.65 GB/s  (57.2% of baseline)
  NumOps=64     avg=0.045 ms  BW=1115.24 GB/s  (36.4% of baseline)
  NumOps=128    avg=0.080 ms  BW=626.44 GB/s  (20.4% of baseline)
  NumOps=256    avg=0.189 ms  BW=266.26 GB/s  (8.7% of baseline)
  NumOps=512    avg=0.377 ms  BW=133.64 GB/s  (4.4% of baseline)
  NumOps=1024   avg=1.601 ms  BW=31.44 GB/s  (1.0% of baseline)
[       OK ] CompressionHeadroomBench.ReduceCopy2SrcFMA_NumOpsSweep (268 ms)
[ RUN      ] CompressionHeadroomBench.CopyINT_NumOpsSweep

--- Copy INT: NumOps sweep (32 blocks, 4M elements) ---
  NumOps=0      avg=0.014 ms  BW=2338.07 GB/s  (baseline)
  NumOps=1      avg=0.014 ms  BW=2337.28 GB/s  (100.0% of baseline)
  NumOps=2      avg=0.014 ms  BW=2338.27 GB/s  (100.0% of baseline)
  NumOps=4      avg=0.014 ms  BW=2339.68 GB/s  (100.1% of baseline)
  NumOps=8      avg=0.014 ms  BW=2339.32 GB/s  (100.1% of baseline)
  NumOps=16     avg=0.014 ms  BW=2338.01 GB/s  (100.0% of baseline)
  NumOps=32     avg=0.014 ms  BW=2338.85 GB/s  (100.0% of baseline)
  NumOps=64     avg=0.014 ms  BW=2337.96 GB/s  (100.0% of baseline)
  NumOps=128    avg=0.014 ms  BW=2337.75 GB/s  (100.0% of baseline)
  NumOps=256    avg=0.014 ms  BW=2338.69 GB/s  (100.0% of baseline)
  NumOps=512    avg=0.014 ms  BW=2338.48 GB/s  (100.0% of baseline)
  NumOps=1024   avg=0.014 ms  BW=2338.54 GB/s  (100.0% of baseline)
[       OK ] CompressionHeadroomBench.CopyINT_NumOpsSweep (20 ms)
[ RUN      ] CompressionHeadroomBench.ReduceCopy2SrcINT_NumOpsSweep

--- ReduceCopy 2-src INT: NumOps sweep (32 blocks, 4M elements) ---
  NumOps=0      avg=0.016 ms  BW=3066.73 GB/s  (baseline)
  NumOps=1      avg=0.016 ms  BW=3069.48 GB/s  (100.1% of baseline)
  NumOps=2      avg=0.016 ms  BW=3070.38 GB/s  (100.1% of baseline)
  NumOps=4      avg=0.016 ms  BW=3069.72 GB/s  (100.1% of baseline)
  NumOps=8      avg=0.016 ms  BW=3065.65 GB/s  (100.0% of baseline)
  NumOps=16     avg=0.016 ms  BW=3069.60 GB/s  (100.1% of baseline)
  NumOps=32     avg=0.016 ms  BW=3069.36 GB/s  (100.1% of baseline)
  NumOps=64     avg=0.016 ms  BW=3069.30 GB/s  (100.1% of baseline)
  NumOps=128    avg=0.016 ms  BW=3069.00 GB/s  (100.1% of baseline)
  NumOps=256    avg=0.016 ms  BW=3070.44 GB/s  (100.1% of baseline)
  NumOps=512    avg=0.016 ms  BW=3068.64 GB/s  (100.1% of baseline)
  NumOps=1024   avg=0.016 ms  BW=3069.12 GB/s  (100.1% of baseline)
[       OK ] CompressionHeadroomBench.ReduceCopy2SrcINT_NumOpsSweep (22 ms)
[ RUN      ] CompressionHeadroomBench.CopyIndependent_NumOpsSweep

--- Copy Independent: NumOps sweep (32 blocks, 4M elements) ---
  NumOps=0      avg=0.014 ms  BW=2337.86 GB/s  (baseline)
  NumOps=1      avg=0.014 ms  BW=2339.68 GB/s  (100.1% of baseline)
  NumOps=2      avg=0.014 ms  BW=2337.81 GB/s  (100.0% of baseline)
  NumOps=4      avg=0.014 ms  BW=2338.38 GB/s  (100.0% of baseline)
  NumOps=8      avg=0.014 ms  BW=2338.54 GB/s  (100.0% of baseline)
  NumOps=16     avg=0.014 ms  BW=2337.91 GB/s  (100.0% of baseline)
  NumOps=32     avg=0.014 ms  BW=2337.44 GB/s  (100.0% of baseline)
  NumOps=64     avg=0.014 ms  BW=2337.08 GB/s  (100.0% of baseline)
  NumOps=128    avg=0.014 ms  BW=2338.22 GB/s  (100.0% of baseline)
  NumOps=256    avg=0.015 ms  BW=2239.59 GB/s  (95.8% of baseline)
  NumOps=512    avg=0.016 ms  BW=2044.61 GB/s  (87.5% of baseline)
  NumOps=1024   avg=0.019 ms  BW=1763.65 GB/s  (75.4% of baseline)
[       OK ] CompressionHeadroomBench.CopyIndependent_NumOpsSweep (21 ms)
[ RUN      ] CompressionHeadroomBench.ReduceCopy2SrcIndependent_NumOpsSweep

--- ReduceCopy 2-src Independent: NumOps sweep (32 blocks, 4M elements) ---
  NumOps=0      avg=0.016 ms  BW=3065.53 GB/s  (baseline)
  NumOps=1      avg=0.016 ms  BW=3067.81 GB/s  (100.1% of baseline)
  NumOps=2      avg=0.016 ms  BW=3069.06 GB/s  (100.1% of baseline)
  NumOps=4      avg=0.016 ms  BW=3069.30 GB/s  (100.1% of baseline)
  NumOps=8      avg=0.016 ms  BW=3069.96 GB/s  (100.1% of baseline)
  NumOps=16     avg=0.016 ms  BW=3066.49 GB/s  (100.0% of baseline)
  NumOps=32     avg=0.016 ms  BW=3070.02 GB/s  (100.1% of baseline)
  NumOps=64     avg=0.016 ms  BW=3068.64 GB/s  (100.1% of baseline)
  NumOps=128    avg=0.016 ms  BW=3068.58 GB/s  (100.1% of baseline)
  NumOps=256    avg=0.018 ms  BW=2765.09 GB/s  (90.2% of baseline)
  NumOps=512    avg=0.018 ms  BW=2726.31 GB/s  (88.9% of baseline)
  NumOps=1024   avg=0.022 ms  BW=2244.61 GB/s  (73.2% of baseline)
[       OK ] CompressionHeadroomBench.ReduceCopy2SrcIndependent_NumOpsSweep (24 ms)
[ RUN      ] CompressionHeadroomBench.ReduceCopy2SrcFMA_RegPressureSweep

--- ReduceCopy 2-src FMA: Register pressure sweep (NumOps=8, 32 blocks, 4M elements) ---
  NumExtraRegs=0    avg=0.018 ms  BW=2727.73 GB/s  (baseline, no extra regs)
  NumExtraRegs=4    avg=0.016 ms  BW=3069.48 GB/s  (112.5% of baseline)
  NumExtraRegs=8    avg=0.016 ms  BW=3069.06 GB/s  (112.5% of baseline)
  NumExtraRegs=16   avg=0.016 ms  BW=3069.48 GB/s  (112.5% of baseline)
  NumExtraRegs=32   avg=0.016 ms  BW=3070.14 GB/s  (112.6% of baseline)
[       OK ] CompressionHeadroomBench.ReduceCopy2SrcFMA_RegPressureSweep (10 ms)
[ RUN      ] CompressionHeadroomBench.ReduceCopy2SrcFMA_RegsAndOpsSweep

--- ReduceCopy 2-src FMA: NumOps x NumExtraRegs sweep (32 blocks, 4M elements) ---
  NumOps=0, NumExtraRegs=0   avg=0.016 ms  BW=3067.57 GB/s  (baseline)
  NumOps=4  , NumExtraRegs=4    avg=0.016 ms  BW=3069.72 GB/s  (100.1% of baseline)
  NumOps=4  , NumExtraRegs=8    avg=0.016 ms  BW=3068.05 GB/s  (100.0% of baseline)
  NumOps=4  , NumExtraRegs=16   avg=0.016 ms  BW=3069.12 GB/s  (100.1% of baseline)
  NumOps=4  , NumExtraRegs=32   avg=0.016 ms  BW=3069.54 GB/s  (100.1% of baseline)
  NumOps=8  , NumExtraRegs=4    avg=0.016 ms  BW=3066.61 GB/s  (100.0% of baseline)
  NumOps=8  , NumExtraRegs=8    avg=0.016 ms  BW=3069.42 GB/s  (100.1% of baseline)
  NumOps=8  , NumExtraRegs=16   avg=0.016 ms  BW=3067.15 GB/s  (100.0% of baseline)
  NumOps=8  , NumExtraRegs=32   avg=0.016 ms  BW=3068.16 GB/s  (100.0% of baseline)
  NumOps=16 , NumExtraRegs=4    avg=0.017 ms  BW=2886.62 GB/s  (94.1% of baseline)
  NumOps=16 , NumExtraRegs=8    avg=0.016 ms  BW=3069.42 GB/s  (100.1% of baseline)
  NumOps=16 , NumExtraRegs=16   avg=0.016 ms  BW=3068.28 GB/s  (100.0% of baseline)
  NumOps=16 , NumExtraRegs=32   avg=0.016 ms  BW=3068.52 GB/s  (100.0% of baseline)
  NumOps=32 , NumExtraRegs=4    avg=0.023 ms  BW=2232.18 GB/s  (72.8% of baseline)
  NumOps=32 , NumExtraRegs=8    avg=0.018 ms  BW=2840.23 GB/s  (92.6% of baseline)
  NumOps=32 , NumExtraRegs=16   avg=0.016 ms  BW=3067.63 GB/s  (100.0% of baseline)
  NumOps=32 , NumExtraRegs=32   avg=0.016 ms  BW=3066.61 GB/s  (100.0% of baseline)
[       OK ] CompressionHeadroomBench.ReduceCopy2SrcFMA_RegsAndOpsSweep (33 ms)
[ RUN      ] CompressionHeadroomBench.ReduceCopy2SrcFMA_SizeSweep

--- ReduceCopy 2-src FMA: Size sweep (32 blocks, 64M elements) ---
  NumOps=0      avg=0.392 ms  BW=2052.62 GB/s  (baseline)
  NumOps=1      avg=0.398 ms  BW=2023.29 GB/s  (98.6% of baseline)
  NumOps=2      avg=0.398 ms  BW=2022.17 GB/s  (98.5% of baseline)
  NumOps=4      avg=0.407 ms  BW=1977.66 GB/s  (96.3% of baseline)
  NumOps=8      avg=0.430 ms  BW=1874.53 GB/s  (91.3% of baseline)
  NumOps=16     avg=0.485 ms  BW=1658.79 GB/s  (80.8% of baseline)
  NumOps=32     avg=0.611 ms  BW=1318.01 GB/s  (64.2% of baseline)
  NumOps=64     avg=0.855 ms  BW=941.84 GB/s  (45.9% of baseline)
  NumOps=128    avg=1.399 ms  BW=575.47 GB/s  (28.0% of baseline)
  NumOps=256    avg=3.168 ms  BW=254.20 GB/s  (12.4% of baseline)
  NumOps=512    avg=6.027 ms  BW=133.61 GB/s  (6.5% of baseline)
  NumOps=1024   avg=25.999 ms  BW=30.97 GB/s  (1.5% of baseline)
*/
