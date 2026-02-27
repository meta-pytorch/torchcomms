// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "comms/utils/kernels/rng/philox_rng.cuh"

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Benchmark Kernels
// =============================================================================

// Benchmark kernel: Generate random numbers using Philox RNG
// Each thread generates 4 random uint32 values and writes them to output
template <int Unroll>
__global__ void benchPhiloxKernel(
    uint64_t seed,
    uint64_t baseOffset,
    uint32_t* output,
    int64_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  // Process 4 elements at a time (Philox generates 4 random numbers per call)
  for (int64_t i = thread * 4; i < nElts; i += nThreads * 4 * Unroll) {
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      int64_t idx = i + u * nThreads * 4;
      if (idx + 3 < nElts) {
        uint32_t r0, r1, r2, r3;
        philox_randint4x(seed, baseOffset + idx, r0, r1, r2, r3);
        output[idx + 0] = r0;
        output[idx + 1] = r1;
        output[idx + 2] = r2;
        output[idx + 3] = r3;
      }
    }
  }
}

// =============================================================================
// Benchmark Fixture
// =============================================================================

class PhiloxRngBench : public ::testing::Test {
 protected:
  static constexpr int64_t kMaxElts =
      1024L * 8L * 8L * 8L * 8L * 8L * 8L * 8L; // 2G elements
  static constexpr int kBlockSize = 256;
  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;

  uint32_t* d_output = nullptr;
  cudaEvent_t startEvent, stopEvent;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_output, kMaxElts * sizeof(uint32_t)));
    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&stopEvent));
  }

  void TearDown() override {
    CUDACHECK(cudaEventDestroy(startEvent));
    CUDACHECK(cudaEventDestroy(stopEvent));
    CUDACHECK(cudaFree(d_output));
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

    // Calculate throughput: GB/s of random bytes produced
    size_t totalBytes = nElts * sizeof(uint32_t);
    double gbPerSec = (double)totalBytes / (avgMs * 1e6);
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
  void runBench(int64_t nElts, LaunchFn launchFn, const char* label) {
    runBenchCore(nElts, maxBlocks(nElts), launchFn, label);
  }
};

// =============================================================================
// Benchmarks: Throughput / Block count sweep
// =============================================================================

TEST_F(PhiloxRngBench, Throughput) {
  printf("\n--- Philox RNG: Throughput (varying sizes) ---\n");
  // 1K to 2G in powers of 8
  int64_t sizes[] = {
      1024L, // 1K
      1024L * 8L, // 8K
      1024L * 8L * 8L, // 64K
      1024L * 8L * 8L * 8L, // 512K
      1024L * 8L * 8L * 8L * 8L, // 4M
      1024L * 8L * 8L * 8L * 8L * 8L, // 32M
      1024L * 8L * 8L * 8L * 8L * 8L * 8L, // 256M
      1024L * 8L * 8L * 8L * 8L * 8L * 8L * 8L, // 2G
  };
  uint64_t seed = 12345ULL;
  uint64_t baseOffset = 0;

  for (int64_t n : sizes) {
    runBench(
        n,
        [&](int nBlocks, int blockSize, int64_t nElts) {
          benchPhiloxKernel<4>
              <<<nBlocks, blockSize>>>(seed, baseOffset, d_output, nElts);
          CUDACHECK(cudaGetLastError());
        },
        "Philox RNG (comms, 7 rounds)");
    printf("\n");
  }
}

// =============================================================================
// Benchmarks: Combined Block count sweep x Unroll factor
// =============================================================================

TEST_F(PhiloxRngBench, BlockSweepUnrollComparison) {
  constexpr int64_t N = 4 * 1024 * 1024; // 4M elements
  printf(
      "\n--- Philox RNG: Block sweep x Unroll comparison (4M elements) ---\n");
  uint64_t seed = 12345ULL;
  uint64_t baseOffset = 0;

  // Helper to run a specific unroll factor with a given block count
  auto runUnrollWithBlocks = [&](auto unrollTag, int nBlocks) {
    constexpr int U = decltype(unrollTag)::value;
    char label[128];
    snprintf(
        label,
        sizeof(label),
        "comms (7 rounds): Unroll=%d, Blocks=%d",
        U,
        nBlocks);
    runBenchCore(
        N,
        nBlocks,
        [&](int /*ignored*/, int blockSize, int64_t nElts) {
          benchPhiloxKernel<U>
              <<<nBlocks, blockSize>>>(seed, baseOffset, d_output, nElts);
          CUDACHECK(cudaGetLastError());
        },
        label);
  };

  // Sweep blocks for each unroll factor
  for (int b : blockCountSweep(maxBlocks(N))) {
    runUnrollWithBlocks(std::integral_constant<int, 1>{}, b);
    runUnrollWithBlocks(std::integral_constant<int, 2>{}, b);
    runUnrollWithBlocks(std::integral_constant<int, 4>{}, b);
    runUnrollWithBlocks(std::integral_constant<int, 8>{}, b);
    runUnrollWithBlocks(std::integral_constant<int, 16>{}, b);
    runUnrollWithBlocks(std::integral_constant<int, 32>{}, b);
    runUnrollWithBlocks(std::integral_constant<int, 64>{}, b);

    // Line break between block sizes
    printf("\n");
  }
}

// clang-format off
// We want to keep the format for the result below

/*
-----------------------------------------------------------------------------------------------
H100 Results
-----------------------------------------------------------------------------------------------

--- Philox RNG: Throughput (varying sizes) ---
  Philox RNG                                     nBlocks=   4  nElts=      1024  avg=0.003 ms  BW=1.54 GB/s
  Philox RNG                                     nBlocks=  32  nElts=      8192  avg=0.003 ms  BW=12.41 GB/s
  Philox RNG                                     nBlocks= 256  nElts=     65536  avg=0.003 ms  BW=104.50 GB/s
  Philox RNG                                     nBlocks=1024  nElts=    524288  avg=0.003 ms  BW=608.62 GB/s
  Philox RNG                                     nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=2200.39 GB/s
  Philox RNG                                     nBlocks=1024  nElts=  33554432  avg=0.054 ms  BW=2493.97 GB/s
  Philox RNG                                     nBlocks=1024  nElts= 268435456  avg=0.432 ms  BW=2483.83 GB/s
  Philox RNG                                     nBlocks=1024  nElts=2147483648  avg=3.468 ms  BW=2476.81 GB/s

--- Philox RNG: Block sweep x Unroll comparison (4M elements) ---
  Philox RNG: Unroll=1, Blocks=1                 nBlocks=   1  nElts=   4194304  avg=0.487 ms  BW=34.43 GB/s
  Philox RNG: Unroll=2, Blocks=1                 nBlocks=   1  nElts=   4194304  avg=0.493 ms  BW=34.02 GB/s
  Philox RNG: Unroll=4, Blocks=1                 nBlocks=   1  nElts=   4194304  avg=0.476 ms  BW=35.22 GB/s
  Philox RNG: Unroll=8, Blocks=1                 nBlocks=   1  nElts=   4194304  avg=0.492 ms  BW=34.08 GB/s
  Philox RNG: Unroll=16, Blocks=1                nBlocks=   1  nElts=   4194304  avg=0.463 ms  BW=36.23 GB/s
  Philox RNG: Unroll=32, Blocks=1                nBlocks=   1  nElts=   4194304  avg=0.476 ms  BW=35.23 GB/s
  Philox RNG: Unroll=64, Blocks=1                nBlocks=   1  nElts=   4194304  avg=0.476 ms  BW=35.24 GB/s

  Philox RNG: Unroll=1, Blocks=2                 nBlocks=   2  nElts=   4194304  avg=0.245 ms  BW=68.59 GB/s
  Philox RNG: Unroll=2, Blocks=2                 nBlocks=   2  nElts=   4194304  avg=0.248 ms  BW=67.74 GB/s
  Philox RNG: Unroll=4, Blocks=2                 nBlocks=   2  nElts=   4194304  avg=0.239 ms  BW=70.12 GB/s
  Philox RNG: Unroll=8, Blocks=2                 nBlocks=   2  nElts=   4194304  avg=0.248 ms  BW=67.55 GB/s
  Philox RNG: Unroll=16, Blocks=2                nBlocks=   2  nElts=   4194304  avg=0.233 ms  BW=72.12 GB/s
  Philox RNG: Unroll=32, Blocks=2                nBlocks=   2  nElts=   4194304  avg=0.239 ms  BW=70.16 GB/s
  Philox RNG: Unroll=64, Blocks=2                nBlocks=   2  nElts=   4194304  avg=0.239 ms  BW=70.07 GB/s

  Philox RNG: Unroll=1, Blocks=4                 nBlocks=   4  nElts=   4194304  avg=0.124 ms  BW=135.13 GB/s
  Philox RNG: Unroll=2, Blocks=4                 nBlocks=   4  nElts=   4194304  avg=0.125 ms  BW=133.96 GB/s
  Philox RNG: Unroll=4, Blocks=4                 nBlocks=   4  nElts=   4194304  avg=0.121 ms  BW=138.82 GB/s
  Philox RNG: Unroll=8, Blocks=4                 nBlocks=   4  nElts=   4194304  avg=0.125 ms  BW=133.77 GB/s
  Philox RNG: Unroll=16, Blocks=4                nBlocks=   4  nElts=   4194304  avg=0.118 ms  BW=142.78 GB/s
  Philox RNG: Unroll=32, Blocks=4                nBlocks=   4  nElts=   4194304  avg=0.121 ms  BW=139.00 GB/s
  Philox RNG: Unroll=64, Blocks=4                nBlocks=   4  nElts=   4194304  avg=0.121 ms  BW=138.96 GB/s

  Philox RNG: Unroll=1, Blocks=8                 nBlocks=   8  nElts=   4194304  avg=0.063 ms  BW=267.25 GB/s
  Philox RNG: Unroll=2, Blocks=8                 nBlocks=   8  nElts=   4194304  avg=0.064 ms  BW=263.57 GB/s
  Philox RNG: Unroll=4, Blocks=8                 nBlocks=   8  nElts=   4194304  avg=0.062 ms  BW=272.59 GB/s
  Philox RNG: Unroll=8, Blocks=8                 nBlocks=   8  nElts=   4194304  avg=0.064 ms  BW=262.09 GB/s
  Philox RNG: Unroll=16, Blocks=8                nBlocks=   8  nElts=   4194304  avg=0.060 ms  BW=280.00 GB/s
  Philox RNG: Unroll=32, Blocks=8                nBlocks=   8  nElts=   4194304  avg=0.061 ms  BW=273.18 GB/s
  Philox RNG: Unroll=64, Blocks=8                nBlocks=   8  nElts=   4194304  avg=0.061 ms  BW=273.16 GB/s

  Philox RNG: Unroll=1, Blocks=16                nBlocks=  16  nElts=   4194304  avg=0.033 ms  BW=504.11 GB/s
  Philox RNG: Unroll=2, Blocks=16                nBlocks=  16  nElts=   4194304  avg=0.033 ms  BW=509.80 GB/s
  Philox RNG: Unroll=4, Blocks=16                nBlocks=  16  nElts=   4194304  avg=0.032 ms  BW=527.17 GB/s
  Philox RNG: Unroll=8, Blocks=16                nBlocks=  16  nElts=   4194304  avg=0.033 ms  BW=507.47 GB/s
  Philox RNG: Unroll=16, Blocks=16               nBlocks=  16  nElts=   4194304  avg=0.031 ms  BW=539.65 GB/s
  Philox RNG: Unroll=32, Blocks=16               nBlocks=  16  nElts=   4194304  avg=0.032 ms  BW=528.86 GB/s
  Philox RNG: Unroll=64, Blocks=16               nBlocks=  16  nElts=   4194304  avg=0.032 ms  BW=528.57 GB/s

  Philox RNG: Unroll=1, Blocks=32                nBlocks=  32  nElts=   4194304  avg=0.018 ms  BW=946.40 GB/s
  Philox RNG: Unroll=2, Blocks=32                nBlocks=  32  nElts=   4194304  avg=0.018 ms  BW=951.07 GB/s
  Philox RNG: Unroll=4, Blocks=32                nBlocks=  32  nElts=   4194304  avg=0.017 ms  BW=991.02 GB/s
  Philox RNG: Unroll=8, Blocks=32                nBlocks=  32  nElts=   4194304  avg=0.018 ms  BW=953.15 GB/s
  Philox RNG: Unroll=16, Blocks=32               nBlocks=  32  nElts=   4194304  avg=0.017 ms  BW=1006.20 GB/s
  Philox RNG: Unroll=32, Blocks=32               nBlocks=  32  nElts=   4194304  avg=0.017 ms  BW=993.67 GB/s
  Philox RNG: Unroll=64, Blocks=32               nBlocks=  32  nElts=   4194304  avg=0.017 ms  BW=991.81 GB/s

  Philox RNG: Unroll=1, Blocks=64                nBlocks=  64  nElts=   4194304  avg=0.010 ms  BW=1688.26 GB/s
  Philox RNG: Unroll=2, Blocks=64                nBlocks=  64  nElts=   4194304  avg=0.010 ms  BW=1701.63 GB/s
  Philox RNG: Unroll=4, Blocks=64                nBlocks=  64  nElts=   4194304  avg=0.010 ms  BW=1762.19 GB/s
  Philox RNG: Unroll=8, Blocks=64                nBlocks=  64  nElts=   4194304  avg=0.010 ms  BW=1698.81 GB/s
  Philox RNG: Unroll=16, Blocks=64               nBlocks=  64  nElts=   4194304  avg=0.009 ms  BW=1771.18 GB/s
  Philox RNG: Unroll=32, Blocks=64               nBlocks=  64  nElts=   4194304  avg=0.009 ms  BW=1771.06 GB/s
  Philox RNG: Unroll=64, Blocks=64               nBlocks=  64  nElts=   4194304  avg=0.010 ms  BW=1764.09 GB/s

  Philox RNG: Unroll=1, Blocks=128               nBlocks= 128  nElts=   4194304  avg=0.006 ms  BW=2647.52 GB/s
  Philox RNG: Unroll=2, Blocks=128               nBlocks= 128  nElts=   4194304  avg=0.006 ms  BW=2753.18 GB/s
  Philox RNG: Unroll=4, Blocks=128               nBlocks= 128  nElts=   4194304  avg=0.006 ms  BW=2720.89 GB/s
  Philox RNG: Unroll=8, Blocks=128               nBlocks= 128  nElts=   4194304  avg=0.006 ms  BW=2659.47 GB/s
  Philox RNG: Unroll=16, Blocks=128              nBlocks= 128  nElts=   4194304  avg=0.006 ms  BW=2652.88 GB/s
  Philox RNG: Unroll=32, Blocks=128              nBlocks= 128  nElts=   4194304  avg=0.006 ms  BW=2742.38 GB/s
  Philox RNG: Unroll=64, Blocks=128              nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=2071.06 GB/s

  Philox RNG: Unroll=1, Blocks=256               nBlocks= 256  nElts=   4194304  avg=0.007 ms  BW=2412.29 GB/s
  Philox RNG: Unroll=2, Blocks=256               nBlocks= 256  nElts=   4194304  avg=0.006 ms  BW=2615.16 GB/s
  Philox RNG: Unroll=4, Blocks=256               nBlocks= 256  nElts=   4194304  avg=0.007 ms  BW=2531.45 GB/s
  Philox RNG: Unroll=8, Blocks=256               nBlocks= 256  nElts=   4194304  avg=0.007 ms  BW=2564.76 GB/s
  Philox RNG: Unroll=16, Blocks=256              nBlocks= 256  nElts=   4194304  avg=0.007 ms  BW=2316.99 GB/s
  Philox RNG: Unroll=32, Blocks=256              nBlocks= 256  nElts=   4194304  avg=0.007 ms  BW=2381.61 GB/s
  Philox RNG: Unroll=64, Blocks=256              nBlocks= 256  nElts=   4194304  avg=0.009 ms  BW=1825.90 GB/s

  Philox RNG: Unroll=1, Blocks=512               nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=2204.47 GB/s
  Philox RNG: Unroll=2, Blocks=512               nBlocks= 512  nElts=   4194304  avg=0.007 ms  BW=2373.63 GB/s
  Philox RNG: Unroll=4, Blocks=512               nBlocks= 512  nElts=   4194304  avg=0.007 ms  BW=2261.03 GB/s
  Philox RNG: Unroll=8, Blocks=512               nBlocks= 512  nElts=   4194304  avg=0.007 ms  BW=2316.68 GB/s
  Philox RNG: Unroll=16, Blocks=512              nBlocks= 512  nElts=   4194304  avg=0.007 ms  BW=2239.49 GB/s
  Philox RNG: Unroll=32, Blocks=512              nBlocks= 512  nElts=   4194304  avg=0.009 ms  BW=1924.28 GB/s
  Philox RNG: Unroll=64, Blocks=512              nBlocks= 512  nElts=   4194304  avg=0.012 ms  BW=1371.01 GB/s

  Philox RNG: Unroll=1, Blocks=1024              nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=2197.07 GB/s
  Philox RNG: Unroll=2, Blocks=1024              nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=2167.73 GB/s
  Philox RNG: Unroll=4, Blocks=1024              nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=2209.48 GB/s
  Philox RNG: Unroll=8, Blocks=1024              nBlocks=1024  nElts=   4194304  avg=0.007 ms  BW=2314.23 GB/s
  Philox RNG: Unroll=16, Blocks=1024             nBlocks=1024  nElts=   4194304  avg=0.009 ms  BW=1924.06 GB/s
  Philox RNG: Unroll=32, Blocks=1024             nBlocks=1024  nElts=   4194304  avg=0.009 ms  BW=1940.80 GB/s
  Philox RNG: Unroll=64, Blocks=1024             nBlocks=1024  nElts=   4194304  avg=0.016 ms  BW=1072.58 GB/s

-----------------------------------------------------------------------------------------------
GB200 Results
-----------------------------------------------------------------------------------------------

--- Philox RNG: Throughput (varying sizes) ---
  Philox RNG                                     nBlocks=   4  nElts=      1024  avg=0.003 ms  BW=1.25 GB/s
  Philox RNG                                     nBlocks=  32  nElts=      8192  avg=0.004 ms  BW=8.08 GB/s
  Philox RNG                                     nBlocks= 256  nElts=     65536  avg=0.004 ms  BW=64.95 GB/s
  Philox RNG                                     nBlocks=1024  nElts=    524288  avg=0.004 ms  BW=507.48 GB/s
  Philox RNG                                     nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=2674.94 GB/s
  Philox RNG                                     nBlocks=1024  nElts=  33554432  avg=0.029 ms  BW=4673.58 GB/s
  Philox RNG                                     nBlocks=1024  nElts= 268435456  avg=0.198 ms  BW=5415.49 GB/s
  Philox RNG                                     nBlocks=1024  nElts=2147483648  avg=1.555 ms  BW=5525.12 GB/s

--- Philox RNG: Block sweep x Unroll comparison (4M elements) ---
  Philox RNG: Unroll=1, Blocks=1                 nBlocks=   1  nElts=   4194304  avg=0.467 ms  BW=35.94 GB/s
  Philox RNG: Unroll=2, Blocks=1                 nBlocks=   1  nElts=   4194304  avg=0.524 ms  BW=31.99 GB/s
  Philox RNG: Unroll=4, Blocks=1                 nBlocks=   1  nElts=   4194304  avg=0.562 ms  BW=29.83 GB/s
  Philox RNG: Unroll=8, Blocks=1                 nBlocks=   1  nElts=   4194304  avg=0.566 ms  BW=29.64 GB/s
  Philox RNG: Unroll=16, Blocks=1                nBlocks=   1  nElts=   4194304  avg=0.480 ms  BW=34.93 GB/s
  Philox RNG: Unroll=32, Blocks=1                nBlocks=   1  nElts=   4194304  avg=0.456 ms  BW=36.75 GB/s
  Philox RNG: Unroll=64, Blocks=1                nBlocks=   1  nElts=   4194304  avg=0.469 ms  BW=35.74 GB/s

  Philox RNG: Unroll=1, Blocks=2                 nBlocks=   2  nElts=   4194304  avg=0.227 ms  BW=73.77 GB/s
  Philox RNG: Unroll=2, Blocks=2                 nBlocks=   2  nElts=   4194304  avg=0.262 ms  BW=64.09 GB/s
  Philox RNG: Unroll=4, Blocks=2                 nBlocks=   2  nElts=   4194304  avg=0.283 ms  BW=59.32 GB/s
  Philox RNG: Unroll=8, Blocks=2                 nBlocks=   2  nElts=   4194304  avg=0.285 ms  BW=58.92 GB/s
  Philox RNG: Unroll=16, Blocks=2                nBlocks=   2  nElts=   4194304  avg=0.243 ms  BW=68.95 GB/s
  Philox RNG: Unroll=32, Blocks=2                nBlocks=   2  nElts=   4194304  avg=0.230 ms  BW=72.83 GB/s
  Philox RNG: Unroll=64, Blocks=2                nBlocks=   2  nElts=   4194304  avg=0.237 ms  BW=70.65 GB/s

  Philox RNG: Unroll=1, Blocks=4                 nBlocks=   4  nElts=   4194304  avg=0.115 ms  BW=145.35 GB/s
  Philox RNG: Unroll=2, Blocks=4                 nBlocks=   4  nElts=   4194304  avg=0.133 ms  BW=125.83 GB/s
  Philox RNG: Unroll=4, Blocks=4                 nBlocks=   4  nElts=   4194304  avg=0.143 ms  BW=117.01 GB/s
  Philox RNG: Unroll=8, Blocks=4                 nBlocks=   4  nElts=   4194304  avg=0.144 ms  BW=116.22 GB/s
  Philox RNG: Unroll=16, Blocks=4                nBlocks=   4  nElts=   4194304  avg=0.126 ms  BW=133.62 GB/s
  Philox RNG: Unroll=32, Blocks=4                nBlocks=   4  nElts=   4194304  avg=0.118 ms  BW=142.32 GB/s
  Philox RNG: Unroll=64, Blocks=4                nBlocks=   4  nElts=   4194304  avg=0.121 ms  BW=138.89 GB/s

  Philox RNG: Unroll=1, Blocks=8                 nBlocks=   8  nElts=   4194304  avg=0.059 ms  BW=282.19 GB/s
  Philox RNG: Unroll=2, Blocks=8                 nBlocks=   8  nElts=   4194304  avg=0.069 ms  BW=244.34 GB/s
  Philox RNG: Unroll=4, Blocks=8                 nBlocks=   8  nElts=   4194304  avg=0.074 ms  BW=227.62 GB/s
  Philox RNG: Unroll=8, Blocks=8                 nBlocks=   8  nElts=   4194304  avg=0.074 ms  BW=227.31 GB/s
  Philox RNG: Unroll=16, Blocks=8                nBlocks=   8  nElts=   4194304  avg=0.066 ms  BW=255.93 GB/s
  Philox RNG: Unroll=32, Blocks=8                nBlocks=   8  nElts=   4194304  avg=0.061 ms  BW=275.03 GB/s
  Philox RNG: Unroll=64, Blocks=8                nBlocks=   8  nElts=   4194304  avg=0.061 ms  BW=272.84 GB/s

  Philox RNG: Unroll=1, Blocks=16                nBlocks=  16  nElts=   4194304  avg=0.032 ms  BW=528.47 GB/s
  Philox RNG: Unroll=2, Blocks=16                nBlocks=  16  nElts=   4194304  avg=0.037 ms  BW=454.87 GB/s
  Philox RNG: Unroll=4, Blocks=16                nBlocks=  16  nElts=   4194304  avg=0.039 ms  BW=430.74 GB/s
  Philox RNG: Unroll=8, Blocks=16                nBlocks=  16  nElts=   4194304  avg=0.039 ms  BW=430.95 GB/s
  Philox RNG: Unroll=16, Blocks=16               nBlocks=  16  nElts=   4194304  avg=0.035 ms  BW=481.66 GB/s
  Philox RNG: Unroll=32, Blocks=16               nBlocks=  16  nElts=   4194304  avg=0.033 ms  BW=511.43 GB/s
  Philox RNG: Unroll=64, Blocks=16               nBlocks=  16  nElts=   4194304  avg=0.033 ms  BW=511.72 GB/s

  Philox RNG: Unroll=1, Blocks=32                nBlocks=  32  nElts=   4194304  avg=0.018 ms  BW=909.59 GB/s
  Philox RNG: Unroll=2, Blocks=32                nBlocks=  32  nElts=   4194304  avg=0.021 ms  BW=818.34 GB/s
  Philox RNG: Unroll=4, Blocks=32                nBlocks=  32  nElts=   4194304  avg=0.021 ms  BW=817.71 GB/s
  Philox RNG: Unroll=8, Blocks=32                nBlocks=  32  nElts=   4194304  avg=0.021 ms  BW=817.28 GB/s
  Philox RNG: Unroll=16, Blocks=32               nBlocks=  32  nElts=   4194304  avg=0.018 ms  BW=908.90 GB/s
  Philox RNG: Unroll=32, Blocks=32               nBlocks=  32  nElts=   4194304  avg=0.018 ms  BW=909.45 GB/s
  Philox RNG: Unroll=64, Blocks=32               nBlocks=  32  nElts=   4194304  avg=0.018 ms  BW=908.39 GB/s

  Philox RNG: Unroll=1, Blocks=64                nBlocks=  64  nElts=   4194304  avg=0.010 ms  BW=1631.47 GB/s
  Philox RNG: Unroll=2, Blocks=64                nBlocks=  64  nElts=   4194304  avg=0.012 ms  BW=1363.56 GB/s
  Philox RNG: Unroll=4, Blocks=64                nBlocks=  64  nElts=   4194304  avg=0.012 ms  BW=1358.93 GB/s
  Philox RNG: Unroll=8, Blocks=64                nBlocks=  64  nElts=   4194304  avg=0.012 ms  BW=1359.46 GB/s
  Philox RNG: Unroll=16, Blocks=64               nBlocks=  64  nElts=   4194304  avg=0.012 ms  BW=1365.12 GB/s
  Philox RNG: Unroll=32, Blocks=64               nBlocks=  64  nElts=   4194304  avg=0.010 ms  BW=1632.84 GB/s
  Philox RNG: Unroll=64, Blocks=64               nBlocks=  64  nElts=   4194304  avg=0.010 ms  BW=1628.12 GB/s

  Philox RNG: Unroll=1, Blocks=128               nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=2042.26 GB/s
  Philox RNG: Unroll=2, Blocks=128               nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=2044.01 GB/s
  Philox RNG: Unroll=4, Blocks=128               nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=2039.95 GB/s
  Philox RNG: Unroll=8, Blocks=128               nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=2038.84 GB/s
  Philox RNG: Unroll=16, Blocks=128              nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=2043.93 GB/s
  Philox RNG: Unroll=32, Blocks=128              nBlocks= 128  nElts=   4194304  avg=0.008 ms  BW=2101.44 GB/s
  Philox RNG: Unroll=64, Blocks=128              nBlocks= 128  nElts=   4194304  avg=0.010 ms  BW=1636.25 GB/s

  Philox RNG: Unroll=1, Blocks=256               nBlocks= 256  nElts=   4194304  avg=0.006 ms  BW=2713.00 GB/s
  Philox RNG: Unroll=2, Blocks=256               nBlocks= 256  nElts=   4194304  avg=0.007 ms  BW=2328.10 GB/s
  Philox RNG: Unroll=4, Blocks=256               nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=2043.37 GB/s
  Philox RNG: Unroll=8, Blocks=256               nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=2041.22 GB/s
  Philox RNG: Unroll=16, Blocks=256              nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=2192.57 GB/s
  Philox RNG: Unroll=32, Blocks=256              nBlocks= 256  nElts=   4194304  avg=0.008 ms  BW=2036.54 GB/s
  Philox RNG: Unroll=64, Blocks=256              nBlocks= 256  nElts=   4194304  avg=0.010 ms  BW=1633.80 GB/s

  Philox RNG: Unroll=1, Blocks=512               nBlocks= 512  nElts=   4194304  avg=0.006 ms  BW=2719.20 GB/s
  Philox RNG: Unroll=2, Blocks=512               nBlocks= 512  nElts=   4194304  avg=0.007 ms  BW=2341.51 GB/s
  Philox RNG: Unroll=4, Blocks=512               nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=2041.62 GB/s
  Philox RNG: Unroll=8, Blocks=512               nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=2063.64 GB/s
  Philox RNG: Unroll=16, Blocks=512              nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=2041.54 GB/s
  Philox RNG: Unroll=32, Blocks=512              nBlocks= 512  nElts=   4194304  avg=0.008 ms  BW=2039.24 GB/s
  Philox RNG: Unroll=64, Blocks=512              nBlocks= 512  nElts=   4194304  avg=0.012 ms  BW=1364.48 GB/s

  Philox RNG: Unroll=1, Blocks=1024              nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=2713.14 GB/s
  Philox RNG: Unroll=2, Blocks=1024              nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=2713.71 GB/s
  Philox RNG: Unroll=4, Blocks=1024              nBlocks=1024  nElts=   4194304  avg=0.006 ms  BW=2683.02 GB/s
  Philox RNG: Unroll=8, Blocks=1024              nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=2047.04 GB/s
  Philox RNG: Unroll=16, Blocks=1024             nBlocks=1024  nElts=   4194304  avg=0.008 ms  BW=2043.53 GB/s
  Philox RNG: Unroll=32, Blocks=1024             nBlocks=1024  nElts=   4194304  avg=0.010 ms  BW=1632.63 GB/s
  Philox RNG: Unroll=64, Blocks=1024             nBlocks=1024  nElts=   4194304  avg=0.015 ms  BW=1117.41 GB/s
  */
