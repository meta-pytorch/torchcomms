// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "meta/collectives/kernels/reduce_copy.cuh"

// Wrapper kernel for the vectorized copy __device__ function.
template <int Unroll, typename T>
__global__ __launch_bounds__(
    256,
    1) void copy_kernel(T* dst, const T* src, ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::copy<Unroll, T>(
      thread, nThreads, (void*)src, (void*)dst, nElts);
}

// Wrapper kernel for reduceCopy with 2 sources.
template <int Unroll, typename T>
__global__ __launch_bounds__(256, 1) void reduce_copy_2src_kernel(
    T* dst,
    const T* src0,
    const T* src1,
    ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopy<Unroll, T>(
      thread, nThreads, (void*)dst, nElts, src0, src1);
}

// Wrapper kernel for reduceCopy with 3 sources.
template <int Unroll, typename T>
__global__ __launch_bounds__(256, 1) void reduce_copy_3src_kernel(
    T* dst,
    const T* src0,
    const T* src1,
    const T* src2,
    ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopy<Unroll, T>(
      thread, nThreads, (void*)dst, nElts, src0, src1, src2);
}

// Wrapper kernel for reduceCopy with 1 source (degenerates to plain copy).
template <int Unroll, typename T>
__global__ __launch_bounds__(
    256,
    1) void reduce_copy_1src_kernel(T* dst, const T* src, ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  meta::comms::ncclx::kernels::simplecopy::reduceCopy<Unroll, T>(
      thread, nThreads, (void*)dst, nElts, src);
}

// Wrapper kernel for reduceCopyPacks with 2 sources.
template <int Unroll, int EltPerPack, typename T>
__global__ __launch_bounds__(256, 1) void reduce_copy_packs_2src_kernel(
    T* dst,
    const T* src0,
    const T* src1,
    ssize_t nElts) {
  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;
  ssize_t nEltsBehind = 0;
  ssize_t nEltsAhead = nElts;
  meta::comms::ncclx::kernels::simplecopy::
      reduceCopyPacks<Unroll, EltPerPack, T, 1>(
          nThreads, thread, nEltsBehind, nEltsAhead, src0, src1, dst);

  if (nEltsBehind < nElts) {
    __trap();
  }

  if (nEltsAhead > 0) {
    __trap();
  }
}

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// Benchmark Fixture
// =============================================================================

class SimpleCopyBench : public ::testing::Test {
 protected:
  using T = float;
  static constexpr int64_t kN = 4L * 1024L * 1024L; // 4M elements
  static constexpr int kBlockSize = 256;
  static constexpr int kDefaultBlocks = 32;
  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;

  T* d_input = nullptr;
  T* d_output = nullptr;
  T* d_input2 = nullptr;
  T* d_input3 = nullptr;
  cudaEvent_t startEvent, stopEvent;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_input, kN * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_output, kN * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_input2, kN * sizeof(T)));
    CUDACHECK(cudaMalloc(&d_input3, kN * sizeof(T)));
    CUDACHECK(cudaMemset(d_input, 1, kN * sizeof(T)));
    CUDACHECK(cudaMemset(d_input2, 2, kN * sizeof(T)));
    CUDACHECK(cudaMemset(d_input3, 3, kN * sizeof(T)));
    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&stopEvent));
  }

  void TearDown() override {
    CUDACHECK(cudaEventDestroy(startEvent));
    CUDACHECK(cudaEventDestroy(stopEvent));
    CUDACHECK(cudaFree(d_input));
    CUDACHECK(cudaFree(d_output));
    CUDACHECK(cudaFree(d_input2));
    CUDACHECK(cudaFree(d_input3));
  }

  // Generate power-of-2 block count sweep up to maxBlk
  static std::vector<int> blockCountSweep(int maxBlk) {
    std::vector<int> counts;
    for (int b = 1; b <= maxBlk; b *= 2) {
      counts.push_back(b);
    }
    if (counts.empty() || counts.back() != maxBlk) {
      counts.push_back(maxBlk);
    }
    return counts;
  }

  template <typename LaunchFn>
  void runBenchCore(
      int64_t nElts,
      int nBlocks,
      LaunchFn launchFn,
      const char* label,
      int nMemAccesses = 2) {
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

    // Bandwidth: nMemAccesses * size bytes transferred
    // (e.g. 2 for plain copy: 1 read + 1 write,
    //        3 for 2-src reduce: 2 reads + 1 write)
    size_t totalBytes = (size_t)nMemAccesses * nElts * sizeof(T);
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
// Benchmark: Vectorized copy with different Unroll values (32 blocks, 4M elts)
// =============================================================================

TEST_F(SimpleCopyBench, CopyUnrollComparison) {
  printf(
      "\n--- Vectorized Copy: Unroll comparison (32 blocks, 4M elements) ---\n");

  auto runUnroll = [&](auto unrollTag) {
    constexpr int U = decltype(unrollTag)::value;

    char label[128];
    snprintf(label, sizeof(label), "Vectorized copy (Unroll=%d)", U);

    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          copy_kernel<U, T>
              <<<nBlk, blockSize>>>(d_output, d_input, (ssize_t)nElts);
          CUDACHECK(cudaGetLastError());
        },
        label);
  };

  runUnroll(std::integral_constant<int, 1>{});
  runUnroll(std::integral_constant<int, 2>{});
  runUnroll(std::integral_constant<int, 4>{});
  runUnroll(std::integral_constant<int, 8>{});
}

// =============================================================================
// Benchmark: Block count sweep for vectorized copy (4M elts)
// =============================================================================

TEST_F(SimpleCopyBench, CopyBlockSweep) {
  printf("\n--- Vectorized Copy: Block count sweep (4M elements) ---\n");

  constexpr int kUnroll = 4;
  constexpr int kMaxBlocks = 1024;

  for (int nBlocks : blockCountSweep(kMaxBlocks)) {
    char label[128];
    snprintf(label, sizeof(label), "Vectorized copy (Unroll=%d)", kUnroll);

    runBenchCore(
        kN,
        nBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          copy_kernel<kUnroll, T>
              <<<nBlk, blockSize>>>(d_output, d_input, (ssize_t)nElts);
          CUDACHECK(cudaGetLastError());
        },
        label);
  }
}

// =============================================================================
// Benchmark: Misaligned buffers + non-power-of-2 element counts
// =============================================================================

TEST_F(SimpleCopyBench, CopyMisaligned) {
  printf(
      "\n--- Vectorized Copy: Misaligned buffers + non-power-of-2 sizes ---\n");

  struct Config {
    int srcOffset;
    int dstOffset;
    int64_t nElts;
  };

  constexpr Config configs[] = {
      {1, 1, 4000000}, // both 4B-misaligned
      {3, 3, 3000007}, // both 12B-misaligned, prime-ish count
      {1, 3, 4100000}, // different misalignment
      {0, 1, 3999999}, // only dst misaligned
  };

  constexpr int kBlocks[] = {16, 32};

  for (int nBlocks : kBlocks) {
    for (const auto& cfg : configs) {
      auto runUnroll = [&](auto unrollTag) {
        constexpr int U = decltype(unrollTag)::value;
        T* src = d_input + cfg.srcOffset;
        T* dst = d_output + cfg.dstOffset;

        char label[128];
        snprintf(
            label,
            sizeof(label),
            "Misaligned copy (srcOff=%d, dstOff=%d, Unroll=%d)",
            cfg.srcOffset,
            cfg.dstOffset,
            U);

        runBenchCore(
            cfg.nElts,
            nBlocks,
            [&](int nBlk, int blockSize, int64_t nElts) {
              copy_kernel<U, T><<<nBlk, blockSize>>>(dst, src, (ssize_t)nElts);
              CUDACHECK(cudaGetLastError());
            },
            label);
      };

      runUnroll(std::integral_constant<int, 4>{});
      runUnroll(std::integral_constant<int, 8>{});
    }
  }
}

// =============================================================================
// Benchmark: reduceCopy with different Unroll values (2-src, 32 blocks, 4M)
// =============================================================================

TEST_F(SimpleCopyBench, ReduceCopyUnrollComparison) {
  printf(
      "\n--- ReduceCopy 2-src: Unroll comparison (32 blocks, 4M elements) ---\n");

  auto runUnroll = [&](auto unrollTag) {
    constexpr int U = decltype(unrollTag)::value;

    char label[128];
    snprintf(label, sizeof(label), "reduceCopy 2-src (Unroll=%d)", U);

    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          reduce_copy_2src_kernel<U, T><<<nBlk, blockSize>>>(
              d_output, d_input, d_input2, (ssize_t)nElts);
          CUDACHECK(cudaGetLastError());
        },
        label,
        3); // 2 reads + 1 write
  };

  runUnroll(std::integral_constant<int, 1>{});
  runUnroll(std::integral_constant<int, 2>{});
  runUnroll(std::integral_constant<int, 4>{});
  runUnroll(std::integral_constant<int, 8>{});
}

// =============================================================================
// Benchmark: reduceCopy with different source counts (32 blocks, 4M elts)
// =============================================================================

TEST_F(SimpleCopyBench, ReduceCopySourceCountComparison) {
  printf(
      "\n--- ReduceCopy: Source count comparison (Unroll=4, 32 blocks, 4M elements) ---\n");

  {
    char label[] = "reduceCopy 1-src (Unroll=4)";
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          reduce_copy_1src_kernel<4, T>
              <<<nBlk, blockSize>>>(d_output, d_input, (ssize_t)nElts);
          CUDACHECK(cudaGetLastError());
        },
        label,
        2); // 1 read + 1 write
  }
  {
    char label[] = "reduceCopy 2-src (Unroll=4)";
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          reduce_copy_2src_kernel<4, T><<<nBlk, blockSize>>>(
              d_output, d_input, d_input2, (ssize_t)nElts);
          CUDACHECK(cudaGetLastError());
        },
        label,
        3); // 2 reads + 1 write
  }
  {
    char label[] = "reduceCopy 3-src (Unroll=4)";
    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          reduce_copy_3src_kernel<4, T><<<nBlk, blockSize>>>(
              d_output, d_input, d_input2, d_input3, (ssize_t)nElts);
          CUDACHECK(cudaGetLastError());
        },
        label,
        4); // 3 reads + 1 write
  }
}

// =============================================================================
// Benchmark: reduceCopy block count sweep (2-src, Unroll=4, 4M elts)
// =============================================================================

TEST_F(SimpleCopyBench, ReduceCopyBlockSweep) {
  printf(
      "\n--- ReduceCopy 2-src: Block count sweep (Unroll=4, 4M elements) ---\n");

  constexpr int kUnroll = 4;
  constexpr int kMaxBlocks = 1024;

  for (int nBlocks : blockCountSweep(kMaxBlocks)) {
    char label[128];
    snprintf(label, sizeof(label), "reduceCopy 2-src (Unroll=%d)", kUnroll);

    runBenchCore(
        kN,
        nBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          reduce_copy_2src_kernel<kUnroll, T><<<nBlk, blockSize>>>(
              d_output, d_input, d_input2, (ssize_t)nElts);
          CUDACHECK(cudaGetLastError());
        },
        label,
        3);
  }
}

// =============================================================================
// Benchmark: reduceCopy 2-src with misaligned buffers + non-power-of-2 sizes
// =============================================================================

TEST_F(SimpleCopyBench, ReduceCopyMisaligned) {
  printf(
      "\n--- ReduceCopy 2-src: Misaligned buffers + non-power-of-2 sizes ---\n");

  struct Config {
    int src0Offset;
    int src1Offset;
    int dstOffset;
    int64_t nElts;
  };

  constexpr Config configs[] = {
      {1, 1, 1, 4000000}, // all 4B-misaligned
      {3, 3, 3, 3000007}, // all 12B-misaligned, prime-ish count
      {1, 3, 0, 4100000}, // mixed misalignment, dst aligned
      {0, 1, 3, 3999999}, // src0 aligned, others misaligned
  };

  constexpr int kBlocks[] = {16, 32};

  for (int nBlocks : kBlocks) {
    for (const auto& cfg : configs) {
      auto runUnroll = [&](auto unrollTag) {
        constexpr int U = decltype(unrollTag)::value;
        T* src0 = d_input + cfg.src0Offset;
        T* src1 = d_input2 + cfg.src1Offset;
        T* dst = d_output + cfg.dstOffset;

        char label[128];
        snprintf(
            label,
            sizeof(label),
            "Misaligned reduceCopy 2-src (s0Off=%d, s1Off=%d, dOff=%d, U=%d)",
            cfg.src0Offset,
            cfg.src1Offset,
            cfg.dstOffset,
            U);

        runBenchCore(
            cfg.nElts,
            nBlocks,
            [&](int nBlk, int blockSize, int64_t nElts) {
              reduce_copy_2src_kernel<U, T>
                  <<<nBlk, blockSize>>>(dst, src0, src1, (ssize_t)nElts);
              CUDACHECK(cudaGetLastError());
            },
            label,
            3);
      };

      runUnroll(std::integral_constant<int, 4>{});
      runUnroll(std::integral_constant<int, 8>{});
    }
  }
}

// =============================================================================
// Benchmark: reduceCopyPacks with different Unroll/EltPerPack (2-src, 32 blks)
// =============================================================================

TEST_F(SimpleCopyBench, ReduceCopyPacksGrid) {
  printf(
      "\n--- reduceCopyPacks 2-src: Unroll x EltPerPack grid (32 blocks, 4M elements) ---\n");

  auto runConfig = [&](auto unrollTag, auto bppTag) {
    constexpr int U = decltype(unrollTag)::value;
    constexpr int B = decltype(bppTag)::value / sizeof(T);

    char label[128];
    snprintf(
        label,
        sizeof(label),
        "reduceCopyPacks 2-src (Unroll=%d, EltPerPack=%d)",
        U,
        B);

    runBenchCore(
        kN,
        kDefaultBlocks,
        [&](int nBlk, int blockSize, int64_t nElts) {
          reduce_copy_packs_2src_kernel<U, B, T><<<nBlk, blockSize>>>(
              d_output, d_input, d_input2, (ssize_t)nElts);
          CUDACHECK(cudaGetLastError());
        },
        label,
        3);
  };

  // EltPerPack=1 (sizeof(float) bytes per pack), varying Unroll
  runConfig(std::integral_constant<int, 1>{}, std::integral_constant<int, 4>{});
  runConfig(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{});
  runConfig(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
  runConfig(std::integral_constant<int, 8>{}, std::integral_constant<int, 4>{});

  // EltPerPack=2 (8 bytes per pack), varying Unroll
  runConfig(std::integral_constant<int, 1>{}, std::integral_constant<int, 8>{});
  runConfig(std::integral_constant<int, 2>{}, std::integral_constant<int, 8>{});
  runConfig(std::integral_constant<int, 4>{}, std::integral_constant<int, 8>{});
  runConfig(std::integral_constant<int, 8>{}, std::integral_constant<int, 8>{});

  // EltPerPack=4 (16 bytes per pack), varying Unroll
  runConfig(
      std::integral_constant<int, 1>{}, std::integral_constant<int, 16>{});
  runConfig(
      std::integral_constant<int, 2>{}, std::integral_constant<int, 16>{});
  runConfig(
      std::integral_constant<int, 4>{}, std::integral_constant<int, 16>{});
  runConfig(
      std::integral_constant<int, 8>{}, std::integral_constant<int, 16>{});
}
