// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Numerical accuracy benchmark: measures the error introduced by BF16
// conversion (RNE vs stochastic rounding) in a cancellation scenario.
//
// Setup: one large negative float + N identical small positive floats that
// sum to zero in FP32. Each value is independently converted to BF16, then
// the BF16 values are summed in FP64 to measure the deviation from zero.
//
// We compare three strategies:
//   1. FP32 baseline (no conversion — should give ~0)
//   2. BF16 RNE (round-to-nearest-even, deterministic)
//   3. BF16 SR  (stochastic rounding, averaged over many independent trials)
//
// Reported counters per benchmark iteration:
//   - rne_abs_error : |sum_rne - fp32_exact_sum|
//   - sr_abs_error  : |avg_sum_sr - fp32_exact_sum|
//   - rne_rel_error : rne_abs_error / |largeVal|
//   - sr_rel_error  : sr_abs_error  / |largeVal|
//   - error_ratio   : rne_abs_error / sr_abs_error  (>1 means SR wins)

#include <benchmark/benchmark.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include "comms/utils/kernels/rng/philox_rng.cuh"
#include "comms/utils/kernels/stochastic_rounding/stochastic_rounding.cuh"

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(cmd)             \
  do {                              \
    cudaError_t e = (cmd);          \
    if (e != cudaSuccess) {         \
      fprintf(                      \
          stderr,                   \
          "CUDA error %s:%d: %s\n", \
          __FILE__,                 \
          __LINE__,                 \
          cudaGetErrorString(e));   \
      abort();                      \
    }                               \
  } while (0)

// ---------------------------------------------------------------------------
// Device kernel: stochastically round an array of floats to bf16
// ---------------------------------------------------------------------------

__global__ void srRoundArrayKernel(
    const float* __restrict__ inputs,
    int n,
    uint64_t seed,
    __nv_bfloat16* __restrict__ outputs) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  uint32_t r0, r1, r2, r3;
  philox_randint4x(seed, (uint64_t)idx, r0, r1, r2, r3);
  outputs[idx] = stochastic_round_bf16_software(inputs[idx], r0);
}

// ---------------------------------------------------------------------------
// Benchmark body
// ---------------------------------------------------------------------------

static void BM_CancellationError(benchmark::State& state) {
  const int smallCount = static_cast<int>(state.range(0));
  const int srTrials = static_cast<int>(state.range(1));
  const float smallVal = 1.33f; // not exactly representable in bf16
  const int n = 1 + smallCount;

  // Compute large value so that double-precision sum is exactly 0
  double largeDbl = -((double)smallVal * smallCount);
  float largeVal = (float)largeDbl;

  // FP32 "exact" sum (may have tiny residual from the float cast)
  double fp32ExactSum = (double)largeVal;
  for (int i = 0; i < smallCount; i++)
    fp32ExactSum += (double)smallVal;

  // RNE sum (deterministic, computed once)
  double rneSum = (double)__bfloat162float(__float2bfloat16(largeVal));
  for (int i = 0; i < smallCount; i++)
    rneSum += (double)__bfloat162float(__float2bfloat16(smallVal));
  double rneAbsError = std::abs(rneSum - fp32ExactSum);

  // Prepare device memory
  std::vector<float> h_inputs(n);
  h_inputs[0] = largeVal;
  for (int i = 1; i < n; i++)
    h_inputs[i] = smallVal;

  float* d_inputs = nullptr;
  __nv_bfloat16* d_outputs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_inputs, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_outputs, n * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemcpy(
      d_inputs, h_inputs.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<__nv_bfloat16> h_outputs(n);
  int blocks = (n + 255) / 256;

  double srAbsError = 0.0;

  for (auto _ : state) {
    // Run srTrials independent stochastic-rounding trials and average the sum
    double totalSum = 0.0;
    for (int t = 0; t < srTrials; t++) {
      uint64_t seed = 22 + t;
      srRoundArrayKernel<<<blocks, 256>>>(d_inputs, n, seed, d_outputs);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(
          h_outputs.data(),
          d_outputs,
          n * sizeof(__nv_bfloat16),
          cudaMemcpyDeviceToHost));

      double trialSum = 0.0;
      for (int i = 0; i < n; i++)
        trialSum += (double)__bfloat162float(h_outputs[i]);
      totalSum += trialSum;
    }
    double avgSum = totalSum / srTrials;
    srAbsError = std::abs(avgSum - fp32ExactSum);
  }

  // Report counters
  double absLarge = std::abs((double)largeVal);
  state.counters["rne_abs_error"] =
      benchmark::Counter(rneAbsError, benchmark::Counter::kAvgThreads);
  state.counters["sr_abs_error"] =
      benchmark::Counter(srAbsError, benchmark::Counter::kAvgThreads);
  state.counters["rne_rel_error"] = benchmark::Counter(
      absLarge > 0 ? rneAbsError / absLarge : 0,
      benchmark::Counter::kAvgThreads);
  state.counters["sr_rel_error"] = benchmark::Counter(
      absLarge > 0 ? srAbsError / absLarge : 0,
      benchmark::Counter::kAvgThreads);
  state.counters["error_ratio"] = benchmark::Counter(
      srAbsError > 0 ? rneAbsError / srAbsError : 0,
      benchmark::Counter::kAvgThreads);
  state.counters["n_elements"] =
      benchmark::Counter(n, benchmark::Counter::kAvgThreads);
  state.counters["sr_trials"] =
      benchmark::Counter(srTrials, benchmark::Counter::kAvgThreads);

  CUDA_CHECK(cudaFree(d_inputs));
  CUDA_CHECK(cudaFree(d_outputs));
}

// Register benchmarks:  Args(smallCount, srTrials)
//   smallCount from 7 (8 elements) to 1023 (1024 elements)
//   srTrials = 256 for all (enough to average out noise)
BENCHMARK(BM_CancellationError)
    ->Args({7, 256})
    ->Args({15, 256})
    ->Args({31, 256})
    ->Args({63, 256})
    ->Args({127, 256})
    ->Args({255, 256})
    ->Args({511, 256})
    ->Args({1023, 256})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
