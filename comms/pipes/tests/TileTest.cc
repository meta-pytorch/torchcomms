// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/pipes/tests/TileTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes {

using comms::pipes::test::kTileTest2DCols;
using comms::pipes::test::kTileTest2DRows;
using comms::pipes::test::kTileTestBlockSize;
using comms::pipes::test::kTileTestTileElems;
using comms::pipes::test::test_tile_accumulate_max_float;
using comms::pipes::test::test_tile_accumulate_max_half;
using comms::pipes::test::test_tile_accumulate_min_bf16;
using comms::pipes::test::test_tile_accumulate_min_float;
using comms::pipes::test::test_tile_accumulate_sum_bf16;
using comms::pipes::test::test_tile_accumulate_sum_float;
using comms::pipes::test::test_tile_load_accumulate_sum_float;
using comms::pipes::test::test_tile_load_store_2d_float;
using comms::pipes::test::test_tile_load_store_bf16;
using comms::pipes::test::test_tile_load_store_float;
using comms::pipes::test::test_tile_load_store_half;
using comms::pipes::test::test_tile_partial_load_accumulate_max_float;
using comms::pipes::test::test_tile_partial_load_accumulate_min_float;
using comms::pipes::test::test_tile_partial_load_accumulate_sum_float;
using comms::pipes::test::test_tile_partial_load_float;
using comms::pipes::test::test_tile_partial_load_store_float;
using comms::pipes::test::test_tile_partial_load_tile_idx1_float;
using comms::pipes::test::test_tile_partial_store_float;
using comms::pipes::test::test_tile_zero_float;

constexpr int kTileElems = kTileTestTileElems;
constexpr int kNumTiles = 4;
constexpr int kNumElems = kTileElems * kNumTiles;

class TileTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }
  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

// =============================================================================
// Load/Store roundtrip tests
// =============================================================================

template <typename T>
void run_load_store_roundtrip(
    void (*kernel)(const T*, T*, std::size_t),
    float (*to_float)(T),
    T (*from_float)(float),
    int mod) {
  std::vector<T> input_h(kNumElems);
  for (int i = 0; i < kNumElems; i++) {
    input_h[i] = from_float(static_cast<float>(i % mod));
  }

  DeviceBuffer inputBuf(kNumElems * sizeof(T));
  DeviceBuffer outputBuf(kNumElems * sizeof(T));
  auto* input_d = static_cast<T*>(inputBuf.get());
  auto* output_d = static_cast<T*>(outputBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      input_d, input_h.data(), kNumElems * sizeof(T), cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(output_d, 0, kNumElems * sizeof(T)));

  kernel(input_d, output_d, kNumTiles);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<T> output_h(kNumElems);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      output_d,
      kNumElems * sizeof(T),
      cudaMemcpyDeviceToHost));

  for (int i = 0; i < kNumElems; i++) {
    EXPECT_EQ(to_float(input_h[i]), to_float(output_h[i]))
        << "Mismatch at index " << i;
  }
}

TEST_F(TileTestFixture, LoadStoreFloat) {
  run_load_store_roundtrip<float>(
      test_tile_load_store_float,
      [](float v) { return v; },
      [](float v) { return v; },
      kNumElems);
}

TEST_F(TileTestFixture, LoadStoreBF16) {
  run_load_store_roundtrip<__nv_bfloat16>(
      test_tile_load_store_bf16, __bfloat162float, __float2bfloat16, 256);
}

TEST_F(TileTestFixture, LoadStoreHalf) {
  run_load_store_roundtrip<__half>(
      test_tile_load_store_half, __half2float, __float2half, 256);
}

// =============================================================================
// Zero test
// =============================================================================

TEST_F(TileTestFixture, TileZero) {
  DeviceBuffer outputBuf(kNumElems * sizeof(float));
  auto* output_d = static_cast<float*>(outputBuf.get());
  CUDACHECK_TEST(cudaMemset(output_d, 0xFF, kNumElems * sizeof(float)));

  test_tile_zero_float(output_d, kNumTiles);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> output_h(kNumElems);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      output_d,
      kNumElems * sizeof(float),
      cudaMemcpyDeviceToHost));

  std::vector<float> expected(kNumElems, 0.0f);
  EXPECT_EQ(output_h, expected);
}

// =============================================================================
// Accumulate tests (float)
// =============================================================================

using AccumKernelFn = void (*)(const float*, const float*, float*, std::size_t);
using AccumExpectFn = float (*)(float, float);

void run_accumulate_float_test(
    AccumKernelFn kernel,
    AccumExpectFn expect,
    std::vector<float>& a_h,
    std::vector<float>& b_h) {
  const int n = static_cast<int>(a_h.size());
  DeviceBuffer aBuf(n * sizeof(float));
  DeviceBuffer bBuf(n * sizeof(float));
  DeviceBuffer outBuf(n * sizeof(float));
  auto* a_d = static_cast<float*>(aBuf.get());
  auto* b_d = static_cast<float*>(bBuf.get());
  auto* out_d = static_cast<float*>(outBuf.get());

  CUDACHECK_TEST(
      cudaMemcpy(a_d, a_h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK_TEST(
      cudaMemcpy(b_d, b_h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  kernel(a_d, b_d, out_d, kNumTiles);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> output_h(n);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(), out_d, n * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(output_h[i], expect(a_h[i], b_h[i])) << "mismatch at index " << i;
  }
}

TEST_F(TileTestFixture, AccumulateSum) {
  std::vector<float> a_h(kNumElems), b_h(kNumElems);
  for (int i = 0; i < kNumElems; i++) {
    a_h[i] = static_cast<float>(i);
    b_h[i] = static_cast<float>(i * 2);
  }
  run_accumulate_float_test(
      test_tile_accumulate_sum_float,
      [](float a, float b) { return a + b; },
      a_h,
      b_h);
}

TEST_F(TileTestFixture, AccumulateSumBF16) {
  std::vector<__nv_bfloat16> a_h(kNumElems), b_h(kNumElems);
  for (int i = 0; i < kNumElems; i++) {
    a_h[i] = __float2bfloat16(static_cast<float>(i % 64));
    b_h[i] = __float2bfloat16(static_cast<float>((i + 1) % 64));
  }

  DeviceBuffer aBuf(kNumElems * sizeof(__nv_bfloat16));
  DeviceBuffer bBuf(kNumElems * sizeof(__nv_bfloat16));
  DeviceBuffer outBuf(kNumElems * sizeof(__nv_bfloat16));
  auto* a_d = static_cast<__nv_bfloat16*>(aBuf.get());
  auto* b_d = static_cast<__nv_bfloat16*>(bBuf.get());
  auto* out_d = static_cast<__nv_bfloat16*>(outBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      a_d,
      a_h.data(),
      kNumElems * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      b_d,
      b_h.data(),
      kNumElems * sizeof(__nv_bfloat16),
      cudaMemcpyHostToDevice));

  test_tile_accumulate_sum_bf16(a_d, b_d, out_d, kNumTiles);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> output_h(kNumElems);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      out_d,
      kNumElems * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost));

  for (int i = 0; i < kNumElems; i++) {
    float a_val = __bfloat162float(a_h[i]);
    float b_val = __bfloat162float(b_h[i]);
    float got = __bfloat162float(output_h[i]);
    EXPECT_FLOAT_EQ(got, a_val + b_val) << "Mismatch at index " << i;
  }
}

TEST_F(TileTestFixture, AccumulateMax) {
  std::vector<float> a_h(kNumElems), b_h(kNumElems);
  for (int i = 0; i < kNumElems; i++) {
    a_h[i] = static_cast<float>(i);
    b_h[i] = static_cast<float>(kNumElems - i);
  }
  run_accumulate_float_test(
      test_tile_accumulate_max_float,
      [](float a, float b) { return std::max(a, b); },
      a_h,
      b_h);
}

TEST_F(TileTestFixture, AccumulateMin) {
  std::vector<float> a_h(kNumElems), b_h(kNumElems);
  for (int i = 0; i < kNumElems; i++) {
    a_h[i] = static_cast<float>(i);
    b_h[i] = static_cast<float>(kNumElems - i);
  }
  run_accumulate_float_test(
      test_tile_accumulate_min_float,
      [](float a, float b) { return std::min(a, b); },
      a_h,
      b_h);
}

// =============================================================================
// Fused load+accumulate test
// =============================================================================

TEST_F(TileTestFixture, LoadAccumulateSum) {
  std::vector<float> base_h(kNumElems), add_h(kNumElems), expected(kNumElems);
  for (int i = 0; i < kNumElems; i++) {
    base_h[i] = static_cast<float>(i);
    add_h[i] = static_cast<float>(i * 3);
    expected[i] = base_h[i] + add_h[i];
  }

  DeviceBuffer baseBuf(kNumElems * sizeof(float));
  DeviceBuffer addBuf(kNumElems * sizeof(float));
  DeviceBuffer outBuf(kNumElems * sizeof(float));
  auto* base_d = static_cast<float*>(baseBuf.get());
  auto* add_d = static_cast<float*>(addBuf.get());
  auto* out_d = static_cast<float*>(outBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      base_d,
      base_h.data(),
      kNumElems * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      add_d, add_h.data(), kNumElems * sizeof(float), cudaMemcpyHostToDevice));

  test_tile_load_accumulate_sum_float(base_d, add_d, out_d, kNumTiles);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> output_h(kNumElems);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      out_d,
      kNumElems * sizeof(float),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(output_h, expected);
}

// =============================================================================
// Partial tile masking tests (parameterized)
// =============================================================================

struct MaskParams {
  std::size_t valid_elems;
  std::string name;
};

std::string mask_param_name(const ::testing::TestParamInfo<MaskParams>& info) {
  return info.param.name;
}

const auto kMaskValues = ::testing::Values(
    MaskParams{0, "zero_elems"},
    MaskParams{1, "single_elem"},
    MaskParams{3, "sub_vector"},
    MaskParams{4, "one_vector"},
    MaskParams{5, "one_vec_plus_one"},
    MaskParams{1024, "aligned_half"},
    MaskParams{1025, "unaligned"},
    MaskParams{kTileTestTileElems - 1, "one_less_than_full"},
    MaskParams{kTileTestTileElems, "full_tile"});

// ---------------------------------------------------------------------------
// tile_load masking
// ---------------------------------------------------------------------------

class TilePartialLoadTest : public TileTestFixture,
                            public ::testing::WithParamInterface<MaskParams> {};

TEST_P(TilePartialLoadTest, LoadMask) {
  const std::size_t valid = GetParam().valid_elems;

  std::vector<float> input_h(kTileElems);
  for (int i = 0; i < kTileElems; i++) {
    input_h[i] = static_cast<float>(i + 1);
  }

  DeviceBuffer inputBuf(kTileElems * sizeof(float));
  DeviceBuffer outputBuf(kTileElems * sizeof(float));
  auto* input_d = static_cast<float*>(inputBuf.get());
  auto* output_d = static_cast<float*>(outputBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      input_d,
      input_h.data(),
      kTileElems * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(output_d, 0, kTileElems * sizeof(float)));

  test_tile_partial_load_float(input_d, output_d, valid);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> output_h(kTileElems);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      output_d,
      kTileElems * sizeof(float),
      cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < valid; i++) {
    EXPECT_EQ(output_h[i], input_h[i]) << "mismatch at index " << i;
  }
  for (std::size_t i = valid; i < kTileElems; i++) {
    EXPECT_EQ(output_h[i], 0.0f) << "should be zero at index " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    PartialLoad,
    TilePartialLoadTest,
    kMaskValues,
    mask_param_name);

// ---------------------------------------------------------------------------
// tile_store masking
// ---------------------------------------------------------------------------

class TilePartialStoreTest : public TileTestFixture,
                             public ::testing::WithParamInterface<MaskParams> {
};

TEST_P(TilePartialStoreTest, StoreMask) {
  const std::size_t valid = GetParam().valid_elems;
  const float kSentinel = -999.0f;

  std::vector<float> input_h(kTileElems);
  for (int i = 0; i < kTileElems; i++) {
    input_h[i] = static_cast<float>(i + 1);
  }

  DeviceBuffer inputBuf(kTileElems * sizeof(float));
  DeviceBuffer outputBuf(kTileElems * sizeof(float));
  auto* input_d = static_cast<float*>(inputBuf.get());
  auto* output_d = static_cast<float*>(outputBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      input_d,
      input_h.data(),
      kTileElems * sizeof(float),
      cudaMemcpyHostToDevice));

  std::vector<float> sentinel_h(kTileElems, kSentinel);
  CUDACHECK_TEST(cudaMemcpy(
      output_d,
      sentinel_h.data(),
      kTileElems * sizeof(float),
      cudaMemcpyHostToDevice));

  test_tile_partial_store_float(input_d, output_d, valid);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> output_h(kTileElems);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      output_d,
      kTileElems * sizeof(float),
      cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < valid; i++) {
    EXPECT_EQ(output_h[i], input_h[i]) << "mismatch at index " << i;
  }
  for (std::size_t i = valid; i < kTileElems; i++) {
    EXPECT_EQ(output_h[i], kSentinel)
        << "should be untouched sentinel at index " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    PartialStore,
    TilePartialStoreTest,
    kMaskValues,
    mask_param_name);

// ---------------------------------------------------------------------------
// tile_load_accumulate masking (parameterized by op)
//
// The MaxOp/MinOp variants are regression tests: the old zero-padding
// approach computed max(dst, 0) / min(dst, 0) on padded lanes, clamping
// signed values toward zero. With scalar reduction on only the valid
// elements, values are preserved.
// ---------------------------------------------------------------------------

using PartialAccumKernelFn =
    void (*)(const float*, const float*, float*, std::size_t);

struct PartialAccumParams {
  MaskParams mask;
  PartialAccumKernelFn kernel;
  AccumExpectFn expect;
  float base_offset;
  float base_scale;
  float add_scale;
  std::string op_name;
};

std::string partial_accum_param_name(
    const ::testing::TestParamInfo<PartialAccumParams>& info) {
  return info.param.op_name + "_" + info.param.mask.name;
}

class TilePartialLoadAccumulateTest
    : public TileTestFixture,
      public ::testing::WithParamInterface<PartialAccumParams> {};

TEST_P(TilePartialLoadAccumulateTest, LoadAccumulateMask) {
  const auto& p = GetParam();
  const std::size_t valid = p.mask.valid_elems;

  std::vector<float> base_h(kTileElems), addend_h(kTileElems);
  for (int i = 0; i < kTileElems; i++) {
    base_h[i] = p.base_offset + p.base_scale * static_cast<float>(i + 1);
    addend_h[i] = p.add_scale * static_cast<float>(i + 1);
  }

  DeviceBuffer baseBuf(kTileElems * sizeof(float));
  DeviceBuffer addBuf(kTileElems * sizeof(float));
  DeviceBuffer outBuf(kTileElems * sizeof(float));
  auto* base_d = static_cast<float*>(baseBuf.get());
  auto* add_d = static_cast<float*>(addBuf.get());
  auto* out_d = static_cast<float*>(outBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      base_d,
      base_h.data(),
      kTileElems * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      add_d,
      addend_h.data(),
      kTileElems * sizeof(float),
      cudaMemcpyHostToDevice));

  p.kernel(base_d, add_d, out_d, valid);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> output_h(kTileElems);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      out_d,
      kTileElems * sizeof(float),
      cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < valid; i++) {
    EXPECT_EQ(output_h[i], p.expect(base_h[i], addend_h[i]))
        << "mismatch at index " << i;
  }
  for (std::size_t i = valid; i < kTileElems; i++) {
    EXPECT_EQ(output_h[i], base_h[i])
        << "should be base value (not accumulated) at index " << i;
  }
}

static auto make_partial_accum_values(
    PartialAccumKernelFn kernel,
    AccumExpectFn expect,
    float base_offset,
    float base_scale,
    float add_scale,
    const std::string& op_name) {
  std::vector<PartialAccumParams> params;
  for (auto& m :
       {MaskParams{0, "zero_elems"},
        MaskParams{1, "single_elem"},
        MaskParams{3, "sub_vector"},
        MaskParams{4, "one_vector"},
        MaskParams{5, "one_vec_plus_one"},
        MaskParams{1024, "aligned_half"},
        MaskParams{1025, "unaligned"},
        MaskParams{kTileTestTileElems - 1, "one_less_than_full"},
        MaskParams{kTileTestTileElems, "full_tile"}}) {
    params.push_back(
        {m, kernel, expect, base_offset, base_scale, add_scale, op_name});
  }
  return ::testing::ValuesIn(params);
}

INSTANTIATE_TEST_SUITE_P(
    Sum,
    TilePartialLoadAccumulateTest,
    make_partial_accum_values(
        test_tile_partial_load_accumulate_sum_float,
        [](float a, float b) { return a + b; },
        0.0f,
        1.0f,
        10.0f,
        "Sum"),
    partial_accum_param_name);

INSTANTIATE_TEST_SUITE_P(
    Max,
    TilePartialLoadAccumulateTest,
    make_partial_accum_values(
        test_tile_partial_load_accumulate_max_float,
        [](float a, float b) { return std::max(a, b); },
        0.0f,
        -1.0f,
        -2.0f,
        "Max"),
    partial_accum_param_name);

INSTANTIATE_TEST_SUITE_P(
    Min,
    TilePartialLoadAccumulateTest,
    make_partial_accum_values(
        test_tile_partial_load_accumulate_min_float,
        [](float a, float b) { return std::min(a, b); },
        100.0f,
        1.0f,
        3.0f,
        "Min"),
    partial_accum_param_name);

// ---------------------------------------------------------------------------
// Combined load-masked + store-masked roundtrip
// ---------------------------------------------------------------------------

class TilePartialLoadStoreTest
    : public TileTestFixture,
      public ::testing::WithParamInterface<MaskParams> {};

TEST_P(TilePartialLoadStoreTest, LoadAndStoreMask) {
  const std::size_t valid = GetParam().valid_elems;
  const float kSentinel = -999.0f;

  std::vector<float> input_h(kTileElems);
  for (int i = 0; i < kTileElems; i++) {
    input_h[i] = static_cast<float>(i + 1);
  }

  DeviceBuffer inputBuf(kTileElems * sizeof(float));
  DeviceBuffer outputBuf(kTileElems * sizeof(float));
  auto* input_d = static_cast<float*>(inputBuf.get());
  auto* output_d = static_cast<float*>(outputBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      input_d,
      input_h.data(),
      kTileElems * sizeof(float),
      cudaMemcpyHostToDevice));
  std::vector<float> sentinel_h(kTileElems, kSentinel);
  CUDACHECK_TEST(cudaMemcpy(
      output_d,
      sentinel_h.data(),
      kTileElems * sizeof(float),
      cudaMemcpyHostToDevice));

  test_tile_partial_load_store_float(input_d, output_d, valid);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> output_h(kTileElems);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      output_d,
      kTileElems * sizeof(float),
      cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < valid; i++) {
    EXPECT_EQ(output_h[i], input_h[i]) << "mismatch at index " << i;
  }
  for (std::size_t i = valid; i < kTileElems; i++) {
    EXPECT_EQ(output_h[i], kSentinel)
        << "should be untouched sentinel at index " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    PartialLoadStore,
    TilePartialLoadStoreTest,
    kMaskValues,
    mask_param_name);

// ---------------------------------------------------------------------------
// Partial load at tile_idx > 0
// ---------------------------------------------------------------------------

TEST_F(TileTestFixture, PartialLoadTileIdx1) {
  constexpr int kTotalElems = kTileElems * 2;
  constexpr std::size_t kValidElems = 1025;

  std::vector<float> input_h(kTotalElems);
  for (int i = 0; i < kTotalElems; i++) {
    input_h[i] = static_cast<float>(i + 1);
  }

  DeviceBuffer inputBuf(kTotalElems * sizeof(float));
  DeviceBuffer outputBuf(kTileElems * sizeof(float));
  auto* input_d = static_cast<float*>(inputBuf.get());
  auto* output_d = static_cast<float*>(outputBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      input_d,
      input_h.data(),
      kTotalElems * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(output_d, 0, kTileElems * sizeof(float)));

  test_tile_partial_load_tile_idx1_float(input_d, output_d, kValidElems);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> output_h(kTileElems);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      output_d,
      kTileElems * sizeof(float),
      cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < kValidElems; i++) {
    EXPECT_EQ(output_h[i], input_h[kTileElems + i])
        << "mismatch at index " << i;
  }
  for (std::size_t i = kValidElems; i < kTileElems; i++) {
    EXPECT_EQ(output_h[i], 0.0f) << "should be zero at index " << i;
  }
}

// =============================================================================
// Half-precision accumulate tests (exercises __hmax2 / __hmin2 / __hadd2)
// =============================================================================

template <typename T>
void run_accumulate_half_test(
    void (*kernel)(const T*, const T*, T*, std::size_t),
    float (*to_float)(T),
    T (*from_float)(float),
    AccumExpectFn expect) {
  constexpr int kN = kTileElems;
  std::vector<T> a_h(kN), b_h(kN);
  for (int i = 0; i < kN; i++) {
    a_h[i] = from_float(static_cast<float>(i % 64));
    b_h[i] = from_float(static_cast<float>(63 - (i % 64)));
  }

  DeviceBuffer aBuf(kN * sizeof(T));
  DeviceBuffer bBuf(kN * sizeof(T));
  DeviceBuffer outBuf(kN * sizeof(T));
  auto* a_d = static_cast<T*>(aBuf.get());
  auto* b_d = static_cast<T*>(bBuf.get());
  auto* out_d = static_cast<T*>(outBuf.get());

  CUDACHECK_TEST(
      cudaMemcpy(a_d, a_h.data(), kN * sizeof(T), cudaMemcpyHostToDevice));
  CUDACHECK_TEST(
      cudaMemcpy(b_d, b_h.data(), kN * sizeof(T), cudaMemcpyHostToDevice));

  kernel(a_d, b_d, out_d, 1);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<T> output_h(kN);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(), out_d, kN * sizeof(T), cudaMemcpyDeviceToHost));

  for (int i = 0; i < kN; i++) {
    float expected = expect(to_float(a_h[i]), to_float(b_h[i]));
    EXPECT_EQ(to_float(output_h[i]), expected) << "mismatch at index " << i;
  }
}

TEST_F(TileTestFixture, AccumulateMaxHalf) {
  run_accumulate_half_test<__half>(
      test_tile_accumulate_max_half,
      __half2float,
      __float2half,
      [](float a, float b) { return std::max(a, b); });
}

TEST_F(TileTestFixture, AccumulateMinBF16) {
  run_accumulate_half_test<__nv_bfloat16>(
      test_tile_accumulate_min_bf16,
      __bfloat162float,
      __float2bfloat16,
      [](float a, float b) { return std::min(a, b); });
}

// =============================================================================
// 2D tile tests
// =============================================================================

constexpr int k2DRows = kTileTest2DRows;
constexpr int k2DCols = kTileTest2DCols;

TEST_F(TileTestFixture, LoadStore2DFull) {
  constexpr int kM = k2DRows + 4;
  constexpr int kN = k2DCols + 8;

  std::vector<float> matrix_h(kM * kN, 0.0f);
  for (int r = 0; r < kM; r++) {
    for (int c = 0; c < kN; c++) {
      matrix_h[r * kN + c] = static_cast<float>(r * 1000 + c + 1);
    }
  }

  DeviceBuffer srcBuf(kM * kN * sizeof(float));
  DeviceBuffer dstBuf(kM * kN * sizeof(float));
  auto* src_d = static_cast<float*>(srcBuf.get());
  auto* dst_d = static_cast<float*>(dstBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      src_d, matrix_h.data(), kM * kN * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, kM * kN * sizeof(float)));

  test_tile_load_store_2d_float(src_d, dst_d, kN, 2, 4, k2DRows, k2DCols);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> dst_h(kM * kN, 0.0f);
  CUDACHECK_TEST(cudaMemcpy(
      dst_h.data(), dst_d, kM * kN * sizeof(float), cudaMemcpyDeviceToHost));

  for (int r = 0; r < k2DRows; r++) {
    for (int c = 0; c < k2DCols; c++) {
      int src_idx = (2 + r) * kN + (4 + c);
      int dst_idx = (2 + r) * kN + (4 + c);
      EXPECT_EQ(dst_h[dst_idx], matrix_h[src_idx])
          << "mismatch at row=" << r << " col=" << c;
    }
  }
}

TEST_F(TileTestFixture, LoadStore2DPartialRows) {
  constexpr int kM = k2DRows;
  constexpr int kN = k2DCols;
  constexpr int kValidRows = 5;
  const float kSentinel = -1.0f;

  std::vector<float> matrix_h(kM * kN);
  for (int i = 0; i < kM * kN; i++) {
    matrix_h[i] = static_cast<float>(i + 1);
  }

  DeviceBuffer srcBuf(kM * kN * sizeof(float));
  DeviceBuffer dstBuf(kM * kN * sizeof(float));
  auto* src_d = static_cast<float*>(srcBuf.get());
  auto* dst_d = static_cast<float*>(dstBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      src_d, matrix_h.data(), kM * kN * sizeof(float), cudaMemcpyHostToDevice));
  std::vector<float> sentinel_h(kM * kN, kSentinel);
  CUDACHECK_TEST(cudaMemcpy(
      dst_d,
      sentinel_h.data(),
      kM * kN * sizeof(float),
      cudaMemcpyHostToDevice));

  test_tile_load_store_2d_float(src_d, dst_d, kN, 0, 0, kValidRows, kN);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> dst_h(kM * kN);
  CUDACHECK_TEST(cudaMemcpy(
      dst_h.data(), dst_d, kM * kN * sizeof(float), cudaMemcpyDeviceToHost));

  for (int r = 0; r < kValidRows; r++) {
    for (int c = 0; c < kN; c++) {
      EXPECT_EQ(dst_h[r * kN + c], matrix_h[r * kN + c])
          << "mismatch at row=" << r << " col=" << c;
    }
  }
  for (int r = kValidRows; r < kM; r++) {
    for (int c = 0; c < kN; c++) {
      EXPECT_EQ(dst_h[r * kN + c], kSentinel)
          << "should be sentinel at row=" << r << " col=" << c;
    }
  }
}

TEST_F(TileTestFixture, LoadStore2DPartialColsUnaligned) {
  constexpr int kM = k2DRows;
  constexpr int kN = k2DCols;
  constexpr int kValidCols = 201;

  std::vector<float> matrix_h(kM * kN);
  for (int i = 0; i < kM * kN; i++) {
    matrix_h[i] = static_cast<float>(i + 1);
  }

  DeviceBuffer srcBuf(kM * kN * sizeof(float));
  DeviceBuffer dstBuf(kM * kN * sizeof(float));
  auto* src_d = static_cast<float*>(srcBuf.get());
  auto* dst_d = static_cast<float*>(dstBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      src_d, matrix_h.data(), kM * kN * sizeof(float), cudaMemcpyHostToDevice));
  const float kSentinel = -1.0f;
  std::vector<float> sentinel_h(kM * kN, kSentinel);
  CUDACHECK_TEST(cudaMemcpy(
      dst_d,
      sentinel_h.data(),
      kM * kN * sizeof(float),
      cudaMemcpyHostToDevice));

  test_tile_load_store_2d_float(src_d, dst_d, kN, 0, 0, kM, kValidCols);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> dst_h(kM * kN);
  CUDACHECK_TEST(cudaMemcpy(
      dst_h.data(), dst_d, kM * kN * sizeof(float), cudaMemcpyDeviceToHost));

  for (int r = 0; r < kM; r++) {
    for (int c = 0; c < kValidCols; c++) {
      EXPECT_EQ(dst_h[r * kN + c], matrix_h[r * kN + c])
          << "mismatch at row=" << r << " col=" << c;
    }
    for (int c = kValidCols; c < kN; c++) {
      EXPECT_EQ(dst_h[r * kN + c], kSentinel)
          << "should be sentinel at row=" << r << " col=" << c;
    }
  }
}

TEST_F(TileTestFixture, LoadStore2DPartialBoth) {
  constexpr int kM = k2DRows;
  constexpr int kN = k2DCols;
  constexpr int kValidRows = 5;
  constexpr int kValidCols = 201;

  std::vector<float> matrix_h(kM * kN);
  for (int i = 0; i < kM * kN; i++) {
    matrix_h[i] = static_cast<float>(i + 1);
  }

  DeviceBuffer srcBuf(kM * kN * sizeof(float));
  DeviceBuffer dstBuf(kM * kN * sizeof(float));
  auto* src_d = static_cast<float*>(srcBuf.get());
  auto* dst_d = static_cast<float*>(dstBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      src_d, matrix_h.data(), kM * kN * sizeof(float), cudaMemcpyHostToDevice));
  const float kSentinel = -1.0f;
  std::vector<float> sentinel_h(kM * kN, kSentinel);
  CUDACHECK_TEST(cudaMemcpy(
      dst_d,
      sentinel_h.data(),
      kM * kN * sizeof(float),
      cudaMemcpyHostToDevice));

  test_tile_load_store_2d_float(src_d, dst_d, kN, 0, 0, kValidRows, kValidCols);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<float> dst_h(kM * kN);
  CUDACHECK_TEST(cudaMemcpy(
      dst_h.data(), dst_d, kM * kN * sizeof(float), cudaMemcpyDeviceToHost));

  for (int r = 0; r < kM; r++) {
    for (int c = 0; c < kN; c++) {
      float expected =
          (r < kValidRows && c < kValidCols) ? matrix_h[r * kN + c] : kSentinel;
      EXPECT_EQ(dst_h[r * kN + c], expected) << "at row=" << r << " col=" << c;
    }
  }
}

} // namespace comms::pipes
