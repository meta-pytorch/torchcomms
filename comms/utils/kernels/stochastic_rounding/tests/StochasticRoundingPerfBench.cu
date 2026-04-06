// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "comms/utils/kernels/rng/philox_rng.cuh"
#include "comms/utils/kernels/stochastic_rounding/stochastic_rounding.cuh"

#define CUDACHECK(cmd)                                                    \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e); \
  } while (0)

// =============================================================================
// SR Variant Enum
// =============================================================================

enum class SRVariant {
  // Original variants (waste randomness)
  kOriginal_1elem, // 1 Philox call per 1 element, uses r0 only
  kEfficient_1elem, // 1 Philox call per 1 element, uses uint16_t(r0)
  kOriginal_2elem, // 1 Philox call per 2 elements, uses r0 and r1
  kEfficient_2elem, // 1 Philox call per 2 elements, uses only r0 (split 2x16)
  kOriginal_4elem, // 1 Philox call per 4 elements, uses all 4 outputs
  kEfficient_4elem, // 1 Philox call per 4 elements, each split 2x16 = 8 SR ops
  kEfficient_8elem, // 1 Philox call per 8 elements (natural efficient pack)
};

// =============================================================================
// Benchmark Kernels
// =============================================================================

// Each kernel reads FP32 values, chains RepeatRounds iterations of Philox+SR
// in registers, and writes the final BF16 result. The chaining ensures we hit
// the compute limit rather than memory bandwidth.

// --- Original 1-element: 1 Philox per element, uses r0 only ---
template <int Unroll, int RepeatRounds>
__global__ void benchSR_Original_1elem(
    const float* input,
    __nv_bfloat16* output,
    int64_t nElts,
    uint64_t seed,
    uint64_t baseOffset) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  for (int64_t i = tid; i < nElts; i += nThreads * Unroll) {
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      int64_t idx = i + u * nThreads;
      if (idx < nElts) {
        float val = input[idx];
        __nv_bfloat16 result;
        for (int rep = 0; rep < RepeatRounds; rep++) {
          uint32_t r0, r1, r2, r3;
          philox_randint4x(
              seed, baseOffset + idx + rep * nElts, r0, r1, r2, r3);
          result = stochastic_round_bf16_software(val, r0);
          // Feed result back as input for next round to prevent optimization
          val = __bfloat162float(result);
        }
        output[idx] = result;
      }
    }
  }
}

// --- Efficient 1-element: 1 Philox per element, uses uint16_t(r0) ---
template <int Unroll, int RepeatRounds>
__global__ void benchSR_Efficient_1elem(
    const float* input,
    __nv_bfloat16* output,
    int64_t nElts,
    uint64_t seed,
    uint64_t baseOffset) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  for (int64_t i = tid; i < nElts; i += nThreads * Unroll) {
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      int64_t idx = i + u * nThreads;
      if (idx < nElts) {
        float val = input[idx];
        __nv_bfloat16 result;
        for (int rep = 0; rep < RepeatRounds; rep++) {
          uint32_t r0, r1, r2, r3;
          philox_randint4x(
              seed, baseOffset + idx + rep * nElts, r0, r1, r2, r3);
          result = stochastic_round_bf16_software_16bit(
              val, static_cast<uint16_t>(r0));
          val = __bfloat162float(result);
        }
        output[idx] = result;
      }
    }
  }
}

// --- Original 2-element: 1 Philox per 2 elements, uses r0 and r1 ---
template <int Unroll, int RepeatRounds>
__global__ void benchSR_Original_2elem(
    const float* input,
    __nv_bfloat16* output,
    int64_t nElts,
    uint64_t seed,
    uint64_t baseOffset) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  for (int64_t i = tid * 2; i < nElts; i += nThreads * 2 * Unroll) {
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      int64_t idx = i + u * nThreads * 2;
      if (idx + 1 < nElts) {
        float2 vals = make_float2(input[idx], input[idx + 1]);
        __nv_bfloat162 result;
        for (int rep = 0; rep < RepeatRounds; rep++) {
          uint32_t r0, r1, r2, r3;
          philox_randint4x(
              seed, baseOffset + idx + rep * nElts, r0, r1, r2, r3);
          result = stochastic_round_bf16x2_software(vals, r0, r1);
          vals.x = __bfloat162float(__low2bfloat16(result));
          vals.y = __bfloat162float(__high2bfloat16(result));
        }
        output[idx] = __low2bfloat16(result);
        output[idx + 1] = __high2bfloat16(result);
      }
    }
  }
}

// --- Efficient 2-element: 1 Philox per 2 elements, uses only r0 (split) ---
template <int Unroll, int RepeatRounds>
__global__ void benchSR_Efficient_2elem(
    const float* input,
    __nv_bfloat16* output,
    int64_t nElts,
    uint64_t seed,
    uint64_t baseOffset) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  for (int64_t i = tid * 2; i < nElts; i += nThreads * 2 * Unroll) {
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      int64_t idx = i + u * nThreads * 2;
      if (idx + 1 < nElts) {
        float2 vals = make_float2(input[idx], input[idx + 1]);
        __nv_bfloat162 result;
        for (int rep = 0; rep < RepeatRounds; rep++) {
          uint32_t r0, r1, r2, r3;
          philox_randint4x(
              seed, baseOffset + idx + rep * nElts, r0, r1, r2, r3);
          result = stochastic_round_bf16x2_software_32bit(vals, r0);
          vals.x = __bfloat162float(__low2bfloat16(result));
          vals.y = __bfloat162float(__high2bfloat16(result));
        }
        output[idx] = __low2bfloat16(result);
        output[idx + 1] = __high2bfloat16(result);
      }
    }
  }
}

// --- Original 4-element: 1 Philox per 4 elements, uses all 4 outputs ---
template <int Unroll, int RepeatRounds>
__global__ void benchSR_Original_4elem(
    const float* input,
    __nv_bfloat16* output,
    int64_t nElts,
    uint64_t seed,
    uint64_t baseOffset) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  for (int64_t i = tid * 4; i < nElts; i += nThreads * 4 * Unroll) {
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      int64_t idx = i + u * nThreads * 4;
      if (idx + 3 < nElts) {
        float v0 = input[idx], v1 = input[idx + 1];
        float v2 = input[idx + 2], v3 = input[idx + 3];
        __nv_bfloat16 b0, b1, b2, b3;
        for (int rep = 0; rep < RepeatRounds; rep++) {
          uint32_t r0, r1, r2, r3;
          philox_randint4x(
              seed, baseOffset + idx + rep * nElts, r0, r1, r2, r3);
          b0 = stochastic_round_bf16_software(v0, r0);
          b1 = stochastic_round_bf16_software(v1, r1);
          b2 = stochastic_round_bf16_software(v2, r2);
          b3 = stochastic_round_bf16_software(v3, r3);
          v0 = __bfloat162float(b0);
          v1 = __bfloat162float(b1);
          v2 = __bfloat162float(b2);
          v3 = __bfloat162float(b3);
        }
        output[idx] = b0;
        output[idx + 1] = b1;
        output[idx + 2] = b2;
        output[idx + 3] = b3;
      }
    }
  }
}

// --- Efficient 4-element: 1 Philox per 4 elements, split into 8x16-bit ---
// Each 32-bit Philox output provides 2 SR operations via the 32-bit bf16x2
// variant. So 4 Philox outputs = 8 SR operations, but we only do 4 here
// to match the same element count as Original_4elem (uses 2 of 4 outputs).
// This shows the savings when not wasting bits within each 32-bit word.
template <int Unroll, int RepeatRounds>
__global__ void benchSR_Efficient_4elem(
    const float* input,
    __nv_bfloat16* output,
    int64_t nElts,
    uint64_t seed,
    uint64_t baseOffset) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  for (int64_t i = tid * 4; i < nElts; i += nThreads * 4 * Unroll) {
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      int64_t idx = i + u * nThreads * 4;
      if (idx + 3 < nElts) {
        float2 vals01 = make_float2(input[idx], input[idx + 1]);
        float2 vals23 = make_float2(input[idx + 2], input[idx + 3]);
        __nv_bfloat162 res01, res23;
        for (int rep = 0; rep < RepeatRounds; rep++) {
          uint32_t r0, r1, r2, r3;
          philox_randint4x(
              seed, baseOffset + idx + rep * nElts, r0, r1, r2, r3);
          // Each 32-bit r provides 2 SR ops via the efficient 32-bit variant
          res01 = stochastic_round_bf16x2_software_32bit(vals01, r0);
          res23 = stochastic_round_bf16x2_software_32bit(vals23, r1);
          // r2, r3 unused — same Philox call count as Original_4elem
          vals01.x = __bfloat162float(__low2bfloat16(res01));
          vals01.y = __bfloat162float(__high2bfloat16(res01));
          vals23.x = __bfloat162float(__low2bfloat16(res23));
          vals23.y = __bfloat162float(__high2bfloat16(res23));
        }
        output[idx] = __low2bfloat16(res01);
        output[idx + 1] = __high2bfloat16(res01);
        output[idx + 2] = __low2bfloat16(res23);
        output[idx + 3] = __high2bfloat16(res23);
      }
    }
  }
}

// --- Efficient 8-element: 1 Philox per 8 elements (natural efficient pack) ---
// 4 Philox outputs × 2 SR ops each = 8 elements per Philox call.
// This is the key comparison: Original_4elem does 1 Philox per 4 elements,
// while Efficient_8elem does 1 Philox per 8 elements — half the Philox compute.
template <int Unroll, int RepeatRounds>
__global__ void benchSR_Efficient_8elem(
    const float* input,
    __nv_bfloat16* output,
    int64_t nElts,
    uint64_t seed,
    uint64_t baseOffset) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto nThreads = blockDim.x * gridDim.x;

  for (int64_t i = tid * 8; i < nElts; i += nThreads * 8 * Unroll) {
#pragma unroll
    for (int u = 0; u < Unroll; u++) {
      int64_t idx = i + u * nThreads * 8;
      if (idx + 7 < nElts) {
        float2 vals01 = make_float2(input[idx], input[idx + 1]);
        float2 vals23 = make_float2(input[idx + 2], input[idx + 3]);
        float2 vals45 = make_float2(input[idx + 4], input[idx + 5]);
        float2 vals67 = make_float2(input[idx + 6], input[idx + 7]);
        __nv_bfloat162 res01, res23, res45, res67;
        for (int rep = 0; rep < RepeatRounds; rep++) {
          uint32_t r0, r1, r2, r3;
          philox_randint4x(
              seed, baseOffset + idx + rep * nElts, r0, r1, r2, r3);
          res01 = stochastic_round_bf16x2_software_32bit(vals01, r0);
          res23 = stochastic_round_bf16x2_software_32bit(vals23, r1);
          res45 = stochastic_round_bf16x2_software_32bit(vals45, r2);
          res67 = stochastic_round_bf16x2_software_32bit(vals67, r3);
          vals01.x = __bfloat162float(__low2bfloat16(res01));
          vals01.y = __bfloat162float(__high2bfloat16(res01));
          vals23.x = __bfloat162float(__low2bfloat16(res23));
          vals23.y = __bfloat162float(__high2bfloat16(res23));
          vals45.x = __bfloat162float(__low2bfloat16(res45));
          vals45.y = __bfloat162float(__high2bfloat16(res45));
          vals67.x = __bfloat162float(__low2bfloat16(res67));
          vals67.y = __bfloat162float(__high2bfloat16(res67));
        }
        output[idx] = __low2bfloat16(res01);
        output[idx + 1] = __high2bfloat16(res01);
        output[idx + 2] = __low2bfloat16(res23);
        output[idx + 3] = __high2bfloat16(res23);
        output[idx + 4] = __low2bfloat16(res45);
        output[idx + 5] = __high2bfloat16(res45);
        output[idx + 6] = __low2bfloat16(res67);
        output[idx + 7] = __high2bfloat16(res67);
      }
    }
  }
}

// =============================================================================
// Benchmark Fixture
// =============================================================================

class StochasticRoundingPerfBench : public ::testing::Test {
 protected:
  static constexpr int64_t kNumElts = 256L * 1024L * 1024L; // 256M elements
  static constexpr int kBlockSize = 256;
  static constexpr int kNumBlocks = 1024;
  static constexpr int kUnroll = 4;
  static constexpr int kWarmupIters = 10;
  static constexpr int kBenchIters = 100;

  float* d_input = nullptr;
  __nv_bfloat16* d_output = nullptr;
  cudaEvent_t startEvent, stopEvent;

  void SetUp() override {
    CUDACHECK(cudaMalloc(&d_input, kNumElts * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_output, kNumElts * sizeof(__nv_bfloat16)));
    CUDACHECK(cudaEventCreate(&startEvent));
    CUDACHECK(cudaEventCreate(&stopEvent));
    // Fill input with a non-exact bf16 value
    float fillVal = 1.004f;
    std::vector<float> h_input(1024, fillVal);
    // Fill first 1024 and let the rest be whatever — perf bench doesn't care
    CUDACHECK(cudaMemcpy(
        d_input, h_input.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice));
  }

  void TearDown() override {
    CUDACHECK(cudaEventDestroy(startEvent));
    CUDACHECK(cudaEventDestroy(stopEvent));
    CUDACHECK(cudaFree(d_output));
    CUDACHECK(cudaFree(d_input));
  }

  template <typename LaunchFn>
  void runBench(
      int64_t nElts,
      int repeatRounds,
      int elemsPerPhilox,
      LaunchFn launchFn,
      const char* label) {
    // Warmup
    for (int i = 0; i < kWarmupIters; i++) {
      launchFn();
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Timed iterations
    CUDACHECK(cudaEventRecord(startEvent));
    for (int i = 0; i < kBenchIters; i++) {
      launchFn();
    }
    CUDACHECK(cudaEventRecord(stopEvent));
    CUDACHECK(cudaDeviceSynchronize());

    float elapsedMs = 0.0f;
    CUDACHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    float avgMs = elapsedMs / kBenchIters;

    // ActualBW: physical memory read + write
    size_t readBytes = nElts * sizeof(float);
    size_t writeBytes = nElts * sizeof(__nv_bfloat16);
    size_t totalBytes = readBytes + writeBytes;
    double actualBW = (double)totalBytes / (avgMs * 1e6);

    // EffectiveBW: accounting for RepeatRounds (effective data processed)
    double effectiveBW = (double)totalBytes * repeatRounds / (avgMs * 1e6);

    // Philox calls per iteration
    int64_t philoxCalls = (nElts / elemsPerPhilox) * (int64_t)repeatRounds;
    // Each Philox call = 70 integer ops (7 rounds × 10 ops/round)
    double totalOps = (double)philoxCalls * 70.0;
    double effectiveTOPs = totalOps / (avgMs * 1e9);

    printf(
        "  %-42s  R=%3d  avg=%.3f ms  ActualBW=%7.2f GB/s  "
        "EffBW=%9.2f GB/s  EffTOPs=%6.2f\n",
        label,
        repeatRounds,
        avgMs,
        actualBW,
        effectiveBW,
        effectiveTOPs);
  }
};

// =============================================================================
// Benchmark: 1-element variants (Original vs Efficient)
// =============================================================================

TEST_F(StochasticRoundingPerfBench, Compare_1elem) {
  printf(
      "\n--- SR Perf: 1-element variants (256M elts, Unroll=%d, Blocks=%d) ---\n",
      kUnroll,
      kNumBlocks);
  uint64_t seed = 12345ULL;
  uint64_t baseOffset = 0;

  auto runRepeatRounds = [&](auto repeatTag) {
    constexpr int R = decltype(repeatTag)::value;

    char label1[128], label2[128];
    snprintf(label1, sizeof(label1), "Original_1elem (32-bit rand)");
    snprintf(label2, sizeof(label2), "Efficient_1elem (16-bit rand)");

    runBench(
        kNumElts,
        R,
        1,
        [&]() {
          benchSR_Original_1elem<kUnroll, R><<<kNumBlocks, kBlockSize>>>(
              d_input, d_output, kNumElts, seed, baseOffset);
          CUDACHECK(cudaGetLastError());
        },
        label1);
    runBench(
        kNumElts,
        R,
        1,
        [&]() {
          benchSR_Efficient_1elem<kUnroll, R><<<kNumBlocks, kBlockSize>>>(
              d_input, d_output, kNumElts, seed, baseOffset);
          CUDACHECK(cudaGetLastError());
        },
        label2);
    printf("\n");
  };

  runRepeatRounds(std::integral_constant<int, 1>{});
  runRepeatRounds(std::integral_constant<int, 4>{});
  runRepeatRounds(std::integral_constant<int, 16>{});
  runRepeatRounds(std::integral_constant<int, 64>{});
  runRepeatRounds(std::integral_constant<int, 256>{});
}

// =============================================================================
// Benchmark: 2-element variants (Original vs Efficient)
// =============================================================================

TEST_F(StochasticRoundingPerfBench, Compare_2elem) {
  printf(
      "\n--- SR Perf: 2-element variants (256M elts, Unroll=%d, Blocks=%d) ---\n",
      kUnroll,
      kNumBlocks);
  uint64_t seed = 12345ULL;
  uint64_t baseOffset = 0;

  auto runRepeatRounds = [&](auto repeatTag) {
    constexpr int R = decltype(repeatTag)::value;

    char label1[128], label2[128];
    snprintf(label1, sizeof(label1), "Original_2elem (2x32-bit rand)");
    snprintf(label2, sizeof(label2), "Efficient_2elem (1x32-bit rand)");

    runBench(
        kNumElts,
        R,
        2,
        [&]() {
          benchSR_Original_2elem<kUnroll, R><<<kNumBlocks, kBlockSize>>>(
              d_input, d_output, kNumElts, seed, baseOffset);
          CUDACHECK(cudaGetLastError());
        },
        label1);
    runBench(
        kNumElts,
        R,
        2,
        [&]() {
          benchSR_Efficient_2elem<kUnroll, R><<<kNumBlocks, kBlockSize>>>(
              d_input, d_output, kNumElts, seed, baseOffset);
          CUDACHECK(cudaGetLastError());
        },
        label2);
    printf("\n");
  };

  runRepeatRounds(std::integral_constant<int, 1>{});
  runRepeatRounds(std::integral_constant<int, 4>{});
  runRepeatRounds(std::integral_constant<int, 16>{});
  runRepeatRounds(std::integral_constant<int, 64>{});
  runRepeatRounds(std::integral_constant<int, 256>{});
}

// =============================================================================
// Benchmark: 4-element vs 8-element (the most important comparison)
// =============================================================================

TEST_F(StochasticRoundingPerfBench, Compare_4elem_vs_8elem) {
  printf(
      "\n--- SR Perf: 4-elem vs 8-elem (256M elts, Unroll=%d, Blocks=%d) ---\n",
      kUnroll,
      kNumBlocks);
  printf(
      "  Key: Original_4elem = 1 Philox/4 elems; Efficient_4elem = 1 Philox/4 elems (no waste);\n"
      "       Efficient_8elem = 1 Philox/8 elems (2x amortization)\n\n");
  uint64_t seed = 12345ULL;
  uint64_t baseOffset = 0;

  auto runRepeatRounds = [&](auto repeatTag) {
    constexpr int R = decltype(repeatTag)::value;

    char label1[128], label2[128], label3[128];
    snprintf(label1, sizeof(label1), "Original_4elem (4x32-bit rand)");
    snprintf(label2, sizeof(label2), "Efficient_4elem (2x32-bit rand)");
    snprintf(label3, sizeof(label3), "Efficient_8elem (4x32-bit for 8 elems)");

    runBench(
        kNumElts,
        R,
        4,
        [&]() {
          benchSR_Original_4elem<kUnroll, R><<<kNumBlocks, kBlockSize>>>(
              d_input, d_output, kNumElts, seed, baseOffset);
          CUDACHECK(cudaGetLastError());
        },
        label1);
    runBench(
        kNumElts,
        R,
        4,
        [&]() {
          benchSR_Efficient_4elem<kUnroll, R><<<kNumBlocks, kBlockSize>>>(
              d_input, d_output, kNumElts, seed, baseOffset);
          CUDACHECK(cudaGetLastError());
        },
        label2);
    runBench(
        kNumElts,
        R,
        8,
        [&]() {
          benchSR_Efficient_8elem<kUnroll, R><<<kNumBlocks, kBlockSize>>>(
              d_input, d_output, kNumElts, seed, baseOffset);
          CUDACHECK(cudaGetLastError());
        },
        label3);
    printf("\n");
  };

  runRepeatRounds(std::integral_constant<int, 1>{});
  runRepeatRounds(std::integral_constant<int, 4>{});
  runRepeatRounds(std::integral_constant<int, 16>{});
  runRepeatRounds(std::integral_constant<int, 64>{});
  runRepeatRounds(std::integral_constant<int, 256>{});
}
