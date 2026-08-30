// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>

// NCCL Device headers
#include "device.h" // @manual Without device.h, op128.h would cause a compile error
#include "op128.h" // @manual

#include "comms/testinfra/TestXPlatUtils.h"
#include "meta/collectives/kernels/stochastic_cast.cuh" // @manual

// Kernel: test Apply_StochasticCast<float, __nv_bfloat16, 1>
__global__ void stochasticCast1Kernel(
    float input,
    uint64_t seed,
    uint64_t offset,
    __nv_bfloat16* output) {
  BytePack<4> pack = toPack(input);
  BytePack<2> result =
      Apply_StochasticCast<float, __nv_bfloat16, 1>::cast(pack, seed, offset);
  *output = fromPack<__nv_bfloat16>(result);
}

// Kernel: test Apply_StochasticCast<float, __nv_bfloat16, 2>
__global__ void stochasticCast2Kernel(
    float x,
    float y,
    uint64_t seed,
    uint64_t offset,
    __nv_bfloat16* output) {
  float2 vals = make_float2(x, y);
  BytePack<8> pack = toPack(vals);
  BytePack<4> result =
      Apply_StochasticCast<float, __nv_bfloat16, 2>::cast(pack, seed, offset);
  __nv_bfloat162 bf16_pair = fromPack<__nv_bfloat162>(result);
  output[0] = __low2bfloat16(bf16_pair);
  output[1] = __high2bfloat16(bf16_pair);
}

// Kernel: test Apply_StochasticCast<float, __nv_bfloat16, 4>
__global__ void stochasticCast4Kernel(
    float x,
    float y,
    float z,
    float w,
    uint64_t seed,
    uint64_t offset,
    __nv_bfloat16* output) {
  float4 vals = make_float4(x, y, z, w);
  BytePack<16> pack = toPack(vals);
  BytePack<8> result =
      Apply_StochasticCast<float, __nv_bfloat16, 4>::cast(pack, seed, offset);

  __nv_bfloat162 lo = fromPack<__nv_bfloat162>(result.half[0]);
  __nv_bfloat162 hi = fromPack<__nv_bfloat162>(result.half[1]);
  output[0] = __low2bfloat16(lo);
  output[1] = __high2bfloat16(lo);
  output[2] = __low2bfloat16(hi);
  output[3] = __high2bfloat16(hi);
}

// Kernel: test applyStochasticCast public API (1 element)
__global__ void applyStochasticCastApiKernel(
    float input,
    uint64_t seed,
    uint64_t offset,
    __nv_bfloat16* output) {
  BytePack<4> pack = toPack(input);
  auto result = applyStochasticCast<float, __nv_bfloat16>(pack, seed, offset);
  *output = fromPack<__nv_bfloat16>(result);
}

// Kernel: repeat applyStochasticCast with different offsets to test
// unbiasedness
__global__ void applyStochasticCastRepeatKernel(
    float input,
    int n,
    uint64_t seed,
    __nv_bfloat16* outputs) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  BytePack<4> pack = toPack(input);
  auto result =
      applyStochasticCast<float, __nv_bfloat16>(pack, seed, (uint64_t)idx);
  outputs[idx] = fromPack<__nv_bfloat16>(result);
}

class StochasticCastTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  // Helper: check if running on Blackwell (SM >= 100) GPU
  static bool isBlackwell() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major >= 10;
  }

  // Helper: convert bf16 back to float on host
  static float bf16ToFloat(__nv_bfloat16 val) {
    return __bfloat162float(val);
  }

  // Helper: get the two nearest bf16 values bracketing a float
  static void getBracketingBf16(float val, float& lower, float& upper) {
    __nv_bfloat16 rounded_down = __float2bfloat16_rd(val);
    float rd = __bfloat162float(rounded_down);
    ASSERT_TRUE(rd <= val) << "Rounding down should not increase value";

    if (rd == val) {
      lower = upper = val;
      return;
    }

    __nv_bfloat16 rounded_up = __float2bfloat16_ru(val);
    float ru = __bfloat162float(rounded_up);
    ASSERT_TRUE(ru >= val) << "Rounding up should not decrease value";

    lower = rd;
    upper = ru;
  }
};

// =============================================================================
// Tests: Apply_StochasticCast specializations
// =============================================================================

TEST_F(StochasticCastTest, StochasticCast1Element) {
  __nv_bfloat16* d_out;
  CUDACHECK_TEST(cudaMalloc(&d_out, sizeof(__nv_bfloat16)));

  float input = 3.0f;
  stochasticCast1Kernel<<<1, 1>>>(input, 42, 0, d_out);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  __nv_bfloat16 h_out;
  CUDACHECK_TEST(
      cudaMemcpy(&h_out, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  EXPECT_EQ(bf16ToFloat(h_out), 3.0f);

  cudaFree(d_out);
}

TEST_F(StochasticCastTest, StochasticCast2Elements) {
  __nv_bfloat16* d_out;
  CUDACHECK_TEST(cudaMalloc(&d_out, 2 * sizeof(__nv_bfloat16)));

  stochasticCast2Kernel<<<1, 1>>>(1.0f, 2.0f, 42, 0, d_out);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  __nv_bfloat16 h_out[2];
  CUDACHECK_TEST(cudaMemcpy(
      h_out, d_out, 2 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  EXPECT_EQ(bf16ToFloat(h_out[0]), 1.0f);
  EXPECT_EQ(bf16ToFloat(h_out[1]), 2.0f);

  cudaFree(d_out);
}

TEST_F(StochasticCastTest, StochasticCast4Elements) {
  __nv_bfloat16* d_out;
  CUDACHECK_TEST(cudaMalloc(&d_out, 4 * sizeof(__nv_bfloat16)));

  stochasticCast4Kernel<<<1, 1>>>(1.0f, 2.0f, 3.0f, 4.0f, 42, 0, d_out);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  __nv_bfloat16 h_out[4];
  CUDACHECK_TEST(cudaMemcpy(
      h_out, d_out, 4 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  EXPECT_EQ(bf16ToFloat(h_out[0]), 1.0f);
  EXPECT_EQ(bf16ToFloat(h_out[1]), 2.0f);
  EXPECT_EQ(bf16ToFloat(h_out[2]), 3.0f);
  EXPECT_EQ(bf16ToFloat(h_out[3]), 4.0f);

  cudaFree(d_out);
}

// Test the public applyStochasticCast API
TEST_F(StochasticCastTest, ApplyStochasticCastApi) {
  __nv_bfloat16* d_out;
  CUDACHECK_TEST(cudaMalloc(&d_out, sizeof(__nv_bfloat16)));

  applyStochasticCastApiKernel<<<1, 1>>>(5.0f, 42, 0, d_out);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  __nv_bfloat16 h_out;
  CUDACHECK_TEST(
      cudaMemcpy(&h_out, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  EXPECT_EQ(bf16ToFloat(h_out), 5.0f);

  cudaFree(d_out);
}

// =============================================================================
// Tests: Statistical properties of Apply_StochasticCast
// =============================================================================

TEST_F(StochasticCastTest, ApplyStochasticCastUnbiased) {
  constexpr int N = 8192;
  __nv_bfloat16* d_out;
  CUDACHECK_TEST(cudaMalloc(&d_out, N * sizeof(__nv_bfloat16)));

  float testValue = 1.004f;

  applyStochasticCastRepeatKernel<<<(N + 255) / 256, 256>>>(
      testValue, N, 42, d_out);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> h_out(N);
  CUDACHECK_TEST(cudaMemcpy(
      h_out.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  double sum = 0.0;
  for (int i = 0; i < N; i++) {
    sum += bf16ToFloat(h_out[i]);
  }
  double avg = sum / N;

  float lower, upper;
  getBracketingBf16(testValue, lower, upper);
  double gap = upper - lower;

  EXPECT_NEAR(avg, (double)testValue, gap * 0.15)
      << "applyStochasticCast should be unbiased. "
      << "Expected: " << testValue << " Got avg: " << avg;

  cudaFree(d_out);
}

// =============================================================================
// Tests: Determinism
// =============================================================================

TEST_F(StochasticCastTest, DeterministicWithSameSeedOffset) {
  __nv_bfloat16* d_out;
  CUDACHECK_TEST(cudaMalloc(&d_out, sizeof(__nv_bfloat16)));

  float input = 1.337f;
  uint64_t seed = 42, offset = 17;

  stochasticCast1Kernel<<<1, 1>>>(input, seed, offset, d_out);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  __nv_bfloat16 result1;
  CUDACHECK_TEST(cudaMemcpy(
      &result1, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  stochasticCast1Kernel<<<1, 1>>>(input, seed, offset, d_out);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  __nv_bfloat16 result2;
  CUDACHECK_TEST(cudaMemcpy(
      &result2, d_out, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  EXPECT_EQ(bf16ToFloat(result1), bf16ToFloat(result2))
      << "Same seed+offset should produce same result";

  cudaFree(d_out);
}
