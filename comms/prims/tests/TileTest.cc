// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
// CUDA only, matching Tile.cuh: there is no hipify mapping for fp8, and AMD's
// MI300 formats are the FNUZ variants, which differ in exponent bias and in
// NaN/infinity encoding.
#if !defined(__HIP_PLATFORM_AMD__)
#include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#include "comms/prims/tests/TileTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::prims {

using comms::prims::test::kTileTest2DCols;
using comms::prims::test::kTileTest2DRows;
using comms::prims::test::kTileTestBlockSize;
using comms::prims::test::kTileTestByteTileElems;
using comms::prims::test::kTileTestTileElems;
#if defined(__CUDA_FP8_TYPES_EXIST__)
using comms::prims::test::test_fp8_accumulator_witness;
#endif
using comms::prims::test::test_tile_accumulate_dtype;
using comms::prims::test::test_tile_accumulate_max_float;
using comms::prims::test::test_tile_accumulate_max_half;
using comms::prims::test::test_tile_accumulate_min_bf16;
using comms::prims::test::test_tile_accumulate_min_float;
using comms::prims::test::test_tile_accumulate_sum_bf16;
using comms::prims::test::test_tile_accumulate_sum_float;
using comms::prims::test::test_tile_load_accumulate_sum_float;
using comms::prims::test::test_tile_load_store_2d_float;
using comms::prims::test::test_tile_load_store_bf16;
using comms::prims::test::test_tile_load_store_float;
using comms::prims::test::test_tile_load_store_half;
using comms::prims::test::test_tile_partial_load_accumulate_dtype;
using comms::prims::test::test_tile_partial_load_accumulate_max_float;
using comms::prims::test::test_tile_partial_load_accumulate_min_float;
using comms::prims::test::test_tile_partial_load_accumulate_sum_float;
using comms::prims::test::test_tile_partial_load_float;
using comms::prims::test::test_tile_partial_load_store_float;
using comms::prims::test::test_tile_partial_load_tile_idx1_float;
using comms::prims::test::test_tile_partial_store_float;
using comms::prims::test::test_tile_zero_float;
using comms::prims::test::TileTestDataType;
using comms::prims::test::TileTestReduceOp;

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

TEST_F(TileTestFixture, LoadStoreFloatUnalignedPointers) {
  constexpr int kInputOffset = 1;
  constexpr int kOutputOffset = 3;
  constexpr float kSentinel = -999.0f;

  std::vector<float> input_h(kNumElems + kInputOffset);
  for (int i = 0; i < static_cast<int>(input_h.size()); i++) {
    input_h[i] = static_cast<float>(i + 1);
  }

  std::vector<float> output_h(kNumElems + kOutputOffset, kSentinel);

  DeviceBuffer inputBuf(input_h.size() * sizeof(float));
  DeviceBuffer outputBuf(output_h.size() * sizeof(float));
  auto* input_d = static_cast<float*>(inputBuf.get());
  auto* output_d = static_cast<float*>(outputBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      input_d,
      input_h.data(),
      input_h.size() * sizeof(float),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      output_d,
      output_h.data(),
      output_h.size() * sizeof(float),
      cudaMemcpyHostToDevice));

  test_tile_load_store_float(
      input_d + kInputOffset, output_d + kOutputOffset, kNumTiles);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      output_d,
      output_h.size() * sizeof(float),
      cudaMemcpyDeviceToHost));

  for (int i = 0; i < kNumElems; i++) {
    EXPECT_EQ(output_h[kOutputOffset + i], input_h[kInputOffset + i])
        << "Mismatch at index " << i;
  }
  for (int i = 0; i < kOutputOffset; i++) {
    EXPECT_EQ(output_h[i], kSentinel)
        << "prefix should be untouched at index " << i;
  }
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

template <typename T>
T make_accumulate_input_a(int index) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>((index % 101 - 50) * 0.25);
  } else if constexpr (std::is_signed_v<T>) {
    return static_cast<T>(index % 101 - 50);
  } else {
    return static_cast<T>(index % 101);
  }
}

template <typename T>
T make_accumulate_input_b(int index) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(((index * 3 + 7) % 53 - 26) * 0.5);
  } else if constexpr (std::is_signed_v<T>) {
    return static_cast<T>((index * 3 + 7) % 37 - 18);
  } else {
    return static_cast<T>((index * 3 + 7) % 37);
  }
}

template <typename T>
T expected_accumulate(T a, T b, TileTestReduceOp op) {
  switch (op) {
    case TileTestReduceOp::kSum:
      return static_cast<T>(a + b);
    case TileTestReduceOp::kMax:
      return std::max(a, b);
    case TileTestReduceOp::kMin:
      return std::min(a, b);
  }
  return a;
}

template <typename T>
void run_accumulate_dtype_test(TileTestDataType dtype) {
  constexpr int kDtypeTileElems =
      sizeof(T) == 1 ? kTileTestByteTileElems : kTileElems;
  constexpr int kDtypeNumTiles = kNumElems / kDtypeTileElems;
  static_assert(kNumElems % kDtypeTileElems == 0);

  std::vector<T> a_h(kNumElems), b_h(kNumElems);
  for (int i = 0; i < kNumElems; i++) {
    a_h[i] = make_accumulate_input_a<T>(i);
    b_h[i] = make_accumulate_input_b<T>(i);
  }

  DeviceBuffer aBuf(kNumElems * sizeof(T));
  DeviceBuffer bBuf(kNumElems * sizeof(T));
  DeviceBuffer outBuf(kNumElems * sizeof(T));
  auto* a_d = static_cast<T*>(aBuf.get());
  auto* b_d = static_cast<T*>(bBuf.get());
  auto* out_d = static_cast<T*>(outBuf.get());

  CUDACHECK_TEST(cudaMemcpy(
      a_d, a_h.data(), kNumElems * sizeof(T), cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      b_d, b_h.data(), kNumElems * sizeof(T), cudaMemcpyHostToDevice));

  for (TileTestReduceOp op :
       {TileTestReduceOp::kSum,
        TileTestReduceOp::kMax,
        TileTestReduceOp::kMin}) {
    CUDACHECK_TEST(cudaMemset(out_d, 0, kNumElems * sizeof(T)));
    test_tile_accumulate_dtype(dtype, op, a_d, b_d, out_d, kDtypeNumTiles);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<T> output_h(kNumElems);
    CUDACHECK_TEST(cudaMemcpy(
        output_h.data(), out_d, kNumElems * sizeof(T), cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNumElems; i++) {
      EXPECT_EQ(output_h[i], expected_accumulate(a_h[i], b_h[i], op))
          << "mismatch at index " << i;
    }
  }
}

#if defined(__CUDA_FP8_TYPES_EXIST__)
/*
 * Drives Fp8VecOps through tile_accumulate for both formats and all three ops.
 *
 * Works in raw uint8 encodings host-side rather than fp8 types, so the
 * expectation is a bit comparison and needs no host fp8 arithmetic beyond the
 * conversions.
 *
 * The host oracle widens to FLOAT while the device widens to __half. That is
 * sound rather than sloppy: Fp8AccumulatorWidthIsNotObservable enumerates all
 * 65536 pairs and finds the two widths produce identical fp8 results, so a
 * float oracle is exact here. If that test ever fails, this one becomes invalid
 * too.
 *
 * Inputs are the finite encodings only. NaN and infinity have their own
 * semantics per format -- e4m3 has no infinities at all -- and belong in a
 * dedicated raw-bit test rather than being folded into a bulk sweep where
 * NaN != NaN would register as a spurious mismatch.
 */
template <typename Fp8HostT>
void run_accumulate_fp8_test(TileTestDataType dtype) {
  constexpr int kDtypeTileElems = kTileTestByteTileElems;
  constexpr int kDtypeNumTiles = kNumElems / kDtypeTileElems;
  static_assert(kNumElems % kDtypeTileElems == 0);

  auto isFinite = [](std::uint8_t raw) {
    Fp8HostT v{};
    std::memcpy(&v, &raw, 1);
    return std::isfinite(static_cast<float>(v));
  };

  // Cycle through the finite encodings so a full tile spans the format.
  std::vector<std::uint8_t> finite;
  for (int e = 0; e < 256; ++e) {
    const auto raw = static_cast<std::uint8_t>(e);
    if (isFinite(raw)) {
      finite.push_back(raw);
    }
  }
  ASSERT_FALSE(finite.empty());

  std::vector<std::uint8_t> a_h(kNumElems), b_h(kNumElems);
  for (int i = 0; i < kNumElems; i++) {
    a_h[i] = finite[i % finite.size()];
    b_h[i] = finite[(i * 7 + 3) % finite.size()];
  }

  DeviceBuffer aBuf(kNumElems), bBuf(kNumElems), outBuf(kNumElems);
  auto* a_d = static_cast<std::uint8_t*>(aBuf.get());
  auto* b_d = static_cast<std::uint8_t*>(bBuf.get());
  auto* out_d = static_cast<std::uint8_t*>(outBuf.get());
  CUDACHECK_TEST(
      cudaMemcpy(a_d, a_h.data(), kNumElems, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(
      cudaMemcpy(b_d, b_h.data(), kNumElems, cudaMemcpyHostToDevice));

  for (TileTestReduceOp op :
       {TileTestReduceOp::kSum,
        TileTestReduceOp::kMax,
        TileTestReduceOp::kMin}) {
    CUDACHECK_TEST(cudaMemset(out_d, 0, kNumElems));
    test_tile_accumulate_dtype(dtype, op, a_d, b_d, out_d, kDtypeNumTiles);
    CUDACHECK_TEST(cudaGetLastError());
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<std::uint8_t> output_h(kNumElems);
    CUDACHECK_TEST(
        cudaMemcpy(output_h.data(), out_d, kNumElems, cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNumElems; i++) {
      Fp8HostT av{}, bv{};
      std::memcpy(&av, &a_h[i], 1);
      std::memcpy(&bv, &b_h[i], 1);
      const float fa = static_cast<float>(av);
      const float fb = static_cast<float>(bv);
      float want = 0.0f;
      switch (op) {
        case TileTestReduceOp::kSum:
          want = fa + fb;
          break;
        case TileTestReduceOp::kMax:
          want = std::max(fa, fb);
          break;
        case TileTestReduceOp::kMin:
          want = std::min(fa, fb);
          break;
      }
      const Fp8HostT wantFp8(want);
      std::uint8_t wantRaw = 0;
      std::memcpy(&wantRaw, &wantFp8, 1);

      /*
       * Raw encodings, except where the result is zero.
       *
       * IEEE leaves the sign of max(+0, -0) unspecified and the two sides may
       * choose differently: std::max returns its first argument on a tie, the
       * device runs __hmax. Both 0x00 and 0x80 are in the operand corpus, so
       * the pairing is reachable, and a raw-bit assertion would then fail on
       * two values that are numerically equal. Compare numerically there and
       * keep bit-exactness everywhere else -- bit-exactness is what catches a
       * neighbouring-encoding error, which at 2-3 mantissa bits a tolerance
       * would not.
       */
      if (want == 0.0f) {
        Fp8HostT gotFp8{};
        std::memcpy(&gotFp8, &output_h[i], 1);
        ASSERT_EQ(static_cast<float>(gotFp8), 0.0f)
            << "op=" << static_cast<int>(op) << " index=" << i << " a=0x"
            << std::hex << static_cast<int>(a_h[i]) << " b=0x"
            << static_cast<int>(b_h[i]) << std::dec;
        continue;
      }
      ASSERT_EQ(output_h[i], wantRaw)
          << "op=" << static_cast<int>(op) << " index=" << i << " a=0x"
          << std::hex << static_cast<int>(a_h[i]) << " b=0x"
          << static_cast<int>(b_h[i]) << std::dec;
    }
  }
}

TEST_F(TileTestFixture, AccumulateFloat8E4M3) {
  run_accumulate_fp8_test<__nv_fp8_e4m3>(TileTestDataType::kFloat8E4M3);
}

TEST_F(TileTestFixture, AccumulateFloat8E5M2) {
  run_accumulate_fp8_test<__nv_fp8_e5m2>(TileTestDataType::kFloat8E5M2);
}

/*
 * The scalar tail, which the whole-tile tests above cannot reach.
 *
 * tile_load_accumulate splits into whole uint4 vectors plus a remainder, and
 * only the remainder runs VecOps<T>::reduce_scalar. fp8 packs 16 elements per
 * vector and the test tile is 4096, so every count the other tests use divides
 * evenly and the tail is never taken -- confirmed by mutation: gutting
 * reduce_scalar(SumOp) leaves them all passing.
 *
 * `kValidElems` is deliberately not a multiple of 16. Elements past it must be
 * left exactly as loaded, which is the second half of the contract: a tail loop
 * that runs off the end corrupts them, and one that never runs leaves the tail
 * un-accumulated. Both are checked below.
 */
template <typename Fp8HostT>
void run_partial_accumulate_fp8_test(TileTestDataType dtype) {
  constexpr int kTileElems = kTileTestByteTileElems;
  constexpr int kValidElems = kTileElems - 13; // 4083, remainder 3 of 16
  static_assert(kValidElems % 16 != 0, "count must leave a scalar tail");

  auto toFloat = [](std::uint8_t raw) {
    Fp8HostT v{};
    std::memcpy(&v, &raw, 1);
    return static_cast<float>(v);
  };
  auto fromFloat = [](float f) {
    const Fp8HostT v(f);
    std::uint8_t raw = 0;
    std::memcpy(&raw, &v, 1);
    return raw;
  };

  // Small same-sign magnitudes: exactly representable in both formats, and
  // their pairwise sums stay finite, so the expectation needs no tolerance.
  std::vector<std::uint8_t> base_h(kTileElems), addend_h(kTileElems);
  for (int i = 0; i < kTileElems; ++i) {
    base_h[i] = fromFloat(1.0f + static_cast<float>(i % 4) * 0.5f);
    addend_h[i] = fromFloat(0.5f + static_cast<float>(i % 3) * 0.5f);
  }

  DeviceBuffer baseBuf(kTileElems), addendBuf(kTileElems), outBuf(kTileElems);
  auto* base_d = static_cast<std::uint8_t*>(baseBuf.get());
  auto* addend_d = static_cast<std::uint8_t*>(addendBuf.get());
  auto* out_d = static_cast<std::uint8_t*>(outBuf.get());
  CUDACHECK_TEST(
      cudaMemcpy(base_d, base_h.data(), kTileElems, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      addend_d, addend_h.data(), kTileElems, cudaMemcpyHostToDevice));

  for (TileTestReduceOp op :
       {TileTestReduceOp::kSum,
        TileTestReduceOp::kMax,
        TileTestReduceOp::kMin}) {
    CUDACHECK_TEST(cudaMemset(out_d, 0, kTileElems));
    test_tile_partial_load_accumulate_dtype(
        dtype, op, base_d, addend_d, out_d, kValidElems);
    CUDACHECK_TEST(cudaGetLastError());
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<std::uint8_t> out_h(kTileElems);
    CUDACHECK_TEST(
        cudaMemcpy(out_h.data(), out_d, kTileElems, cudaMemcpyDeviceToHost));

    for (int i = 0; i < kTileElems; ++i) {
      const float a = toFloat(base_h[i]);
      const float b = toFloat(addend_h[i]);
      float want = a; // beyond kValidElems the base must survive untouched
      if (i < kValidElems) {
        switch (op) {
          case TileTestReduceOp::kSum:
            want = a + b;
            break;
          case TileTestReduceOp::kMax:
            want = std::max(a, b);
            break;
          case TileTestReduceOp::kMin:
            want = std::min(a, b);
            break;
        }
      }
      ASSERT_EQ(out_h[i], fromFloat(want))
          << "op=" << static_cast<int>(op) << " index=" << i
          << (i < kValidElems ? " (accumulated)" : " (past valid_elems)");
    }
  }
}

TEST_F(TileTestFixture, PartialAccumulateFloat8E4M3) {
  run_partial_accumulate_fp8_test<__nv_fp8_e4m3>(TileTestDataType::kFloat8E4M3);
}

TEST_F(TileTestFixture, PartialAccumulateFloat8E5M2) {
  run_partial_accumulate_fp8_test<__nv_fp8_e5m2>(TileTestDataType::kFloat8E5M2);
}

/*
 * Special values, on raw bits, for the one format that can represent them.
 *
 * e5m2 encodes +/-INF at 0x7C/0xFC and NaN above them; e4m3 is OCP and has
 * neither, so this is e5m2-only. The bulk sweeps above deliberately exclude
 * non-finite inputs -- NaN != NaN reads as a mismatch there -- which leaves the
 * behaviour untested unless it is pinned here.
 *
 * The contract is saturation, not preservation: every Fp8VecOps body narrows
 * through the SATFINITE constructors, so an INF operand returns as the format
 * maximum. That follows NCCL. This test exists so the choice is visible and
 * cannot change silently, and it covers both the vectorized body and the scalar
 * tail, which narrow independently.
 */
TEST_F(TileTestFixture, Fp8SpecialValuesSaturate) {
  constexpr int kTileElems = kTileTestByteTileElems;
  constexpr std::uint8_t kPosInf = 0x7C;
  constexpr std::uint8_t kNegInf = 0xFC;
  constexpr std::uint8_t kMaxFinite = 0x7B; // 57344
  constexpr std::uint8_t kMinFinite = 0xFB; // -57344

  auto encode = [](float f) {
    const __nv_fp8_e5m2 v(f);
    std::uint8_t raw = 0;
    std::memcpy(&raw, &v, 1);
    return raw;
  };
  const std::uint8_t kOne = encode(1.0f);

  // A whole tile of (INF, 1.0) pairs, so the vectorized body runs, plus a
  // partial count so the scalar tail runs over the same inputs.
  struct Case {
    std::uint8_t a;
    std::uint8_t b;
    TileTestReduceOp op;
    std::uint8_t want;
    const char* what;
  };
  const std::vector<Case> cases = {
      {kPosInf, kOne, TileTestReduceOp::kMax, kMaxFinite, "max(+INF, 1)"},
      {kNegInf, kOne, TileTestReduceOp::kMin, kMinFinite, "min(-INF, 1)"},
      {kPosInf, kOne, TileTestReduceOp::kSum, kMaxFinite, "sum(+INF, 1)"},
  };

  for (const Case& c : cases) {
    std::vector<std::uint8_t> a_h(kTileElems, c.a), b_h(kTileElems, c.b);
    DeviceBuffer aBuf(kTileElems), bBuf(kTileElems), outBuf(kTileElems);
    auto* a_d = static_cast<std::uint8_t*>(aBuf.get());
    auto* b_d = static_cast<std::uint8_t*>(bBuf.get());
    auto* out_d = static_cast<std::uint8_t*>(outBuf.get());
    CUDACHECK_TEST(
        cudaMemcpy(a_d, a_h.data(), kTileElems, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(
        cudaMemcpy(b_d, b_h.data(), kTileElems, cudaMemcpyHostToDevice));

    // Vectorized body: a whole tile, no remainder.
    CUDACHECK_TEST(cudaMemset(out_d, 0, kTileElems));
    test_tile_accumulate_dtype(
        TileTestDataType::kFloat8E5M2, c.op, a_d, b_d, out_d, /*ntiles=*/1);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    std::vector<std::uint8_t> out_h(kTileElems);
    CUDACHECK_TEST(
        cudaMemcpy(out_h.data(), out_d, kTileElems, cudaMemcpyDeviceToHost));
    EXPECT_EQ(out_h[0], c.want) << c.what << " (vector body)";

    // Scalar tail: a count that leaves a remainder of 3 in 16.
    CUDACHECK_TEST(cudaMemset(out_d, 0, kTileElems));
    test_tile_partial_load_accumulate_dtype(
        TileTestDataType::kFloat8E5M2, c.op, a_d, b_d, out_d, kTileElems - 13);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    CUDACHECK_TEST(
        cudaMemcpy(out_h.data(), out_d, kTileElems, cudaMemcpyDeviceToHost));
    EXPECT_EQ(out_h[kTileElems - 14], c.want) << c.what << " (scalar tail)";
  }

  // NaN is not saturated into a finite value -- it stays NaN, so the two paths
  // must agree on that too rather than quietly producing the format maximum.
  constexpr std::uint8_t kNaN = 0x7E;
  std::vector<std::uint8_t> a_h(kTileElems, kNaN), b_h(kTileElems, kOne);
  DeviceBuffer aBuf(kTileElems), bBuf(kTileElems), outBuf(kTileElems);
  auto* a_d = static_cast<std::uint8_t*>(aBuf.get());
  auto* b_d = static_cast<std::uint8_t*>(bBuf.get());
  auto* out_d = static_cast<std::uint8_t*>(outBuf.get());
  CUDACHECK_TEST(
      cudaMemcpy(a_d, a_h.data(), kTileElems, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(
      cudaMemcpy(b_d, b_h.data(), kTileElems, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(out_d, 0, kTileElems));
  test_tile_accumulate_dtype(
      TileTestDataType::kFloat8E5M2,
      TileTestReduceOp::kSum,
      a_d,
      b_d,
      out_d,
      /*ntiles=*/1);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  std::vector<std::uint8_t> out_h(kTileElems);
  CUDACHECK_TEST(
      cudaMemcpy(out_h.data(), out_d, kTileElems, cudaMemcpyDeviceToHost));
  __nv_fp8_e5m2 got{};
  std::memcpy(&got, &out_h[0], 1);
  EXPECT_TRUE(std::isnan(static_cast<float>(got)))
      << "sum(NaN, 1) must stay NaN, got raw 0x" << std::hex
      << static_cast<int>(out_h[0]) << std::dec;
}
#endif // __CUDA_FP8_TYPES_EXIST__

TEST_F(TileTestFixture, AccumulateInt8) {
  run_accumulate_dtype_test<std::int8_t>(TileTestDataType::kInt8);
}

TEST_F(TileTestFixture, AccumulateUint8) {
  run_accumulate_dtype_test<std::uint8_t>(TileTestDataType::kUint8);
}

TEST_F(TileTestFixture, AccumulateInt32) {
  run_accumulate_dtype_test<std::int32_t>(TileTestDataType::kInt32);
}

TEST_F(TileTestFixture, AccumulateUint32) {
  run_accumulate_dtype_test<std::uint32_t>(TileTestDataType::kUint32);
}

TEST_F(TileTestFixture, AccumulateInt64) {
  run_accumulate_dtype_test<std::int64_t>(TileTestDataType::kInt64);
}

TEST_F(TileTestFixture, AccumulateUint64) {
  run_accumulate_dtype_test<std::uint64_t>(TileTestDataType::kUint64);
}

TEST_F(TileTestFixture, AccumulateFloat64) {
  run_accumulate_dtype_test<double>(TileTestDataType::kFloat64);
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

/*
 * Is the fp8 accumulator width OBSERVABLE at the output?
 *
 * Prims widens fp8 to __half to reduce, following NCCL. The intuitive
 * alternative is float. Rather than argue about whether the difference can
 * escape, enumerate: both operands are 8-bit, so 256 x 256 pairs cover the
 * entire input space of a two-way reduce exhaustively.
 *
 * Reports rather than asserting a guessed count -- the measurement is the
 * point, and it decides whether the accumulator is an observable contract worth
 * pinning or implementation parity with nothing to test.
 */
#if defined(__CUDA_FP8_TYPES_EXIST__)
TEST_F(TileTestFixture, Fp8AccumulatorWidthIsNotObservable) {
  for (const bool e5m2 : {false, true}) {
    std::uint32_t* witnesses_d = nullptr;
    std::uint8_t* firstA_d = nullptr;
    std::uint8_t* firstB_d = nullptr;
    CUDACHECK_TEST(cudaMalloc(&witnesses_d, sizeof(std::uint32_t)));
    CUDACHECK_TEST(cudaMalloc(&firstA_d, 1));
    CUDACHECK_TEST(cudaMalloc(&firstB_d, 1));
    CUDACHECK_TEST(cudaMemset(witnesses_d, 0, sizeof(std::uint32_t)));
    CUDACHECK_TEST(cudaMemset(firstA_d, 0, 1));
    CUDACHECK_TEST(cudaMemset(firstB_d, 0, 1));

    test_fp8_accumulator_witness(e5m2, witnesses_d, firstA_d, firstB_d);
    CUDACHECK_TEST(cudaGetLastError());
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::uint32_t witnesses = 0;
    std::uint8_t firstA = 0, firstB = 0;
    CUDACHECK_TEST(cudaMemcpy(
        &witnesses, witnesses_d, sizeof(witnesses), cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(&firstA, firstA_d, 1, cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(&firstB, firstB_d, 1, cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaFree(witnesses_d));
    CUDACHECK_TEST(cudaFree(firstA_d));
    CUDACHECK_TEST(cudaFree(firstB_d));

    std::printf(
        "[fp8-accum-witness] format=%s pairs=65536 witnesses=%u firstA=0x%02X "
        "firstB=0x%02X\n",
        e5m2 ? "e5m2" : "e4m3",
        witnesses,
        firstA,
        firstB);

    /*
     * Zero is the measured answer, not an assumption: every pair was tried.
     *
     * A failure here does not mean the reduce is broken. It means the
     * accumulator width is observable at the output, which invalidates both the
     * choice of __half over float and the float-based host oracle in
     * run_accumulate_fp8_test, which relies on the two being equivalent.
     */
    EXPECT_EQ(witnesses, 0u)
        << "accumulator width became observable for "
        << (e5m2 ? "e5m2" : "e4m3") << "; first differing pair a=0x" << std::hex
        << static_cast<int>(firstA) << " b=0x" << static_cast<int>(firstB)
        << std::dec;
  }
}
#endif // __CUDA_FP8_TYPES_EXIST__

} // namespace comms::prims
