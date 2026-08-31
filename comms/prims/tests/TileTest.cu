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
#include <cstddef>
#include <cstdint>

#include "comms/prims/core/Tile.cuh"
#include "comms/prims/tests/Checks.h"
#include "comms/prims/tests/TileTest.cuh"

namespace comms::prims::test {

using comms::prims::make_block_group;
using comms::prims::MaxOp;
using comms::prims::MinOp;
using comms::prims::RegisterStorage;
using comms::prims::SumOp;
using comms::prims::Tile;
using comms::prims::tile_accumulate;
using comms::prims::tile_load;
using comms::prims::tile_load_2d;
using comms::prims::tile_load_accumulate;
using comms::prims::tile_store;
using comms::prims::tile_store_2d;
using comms::prims::tile_zero;

constexpr int kBS = kTileTestBlockSize;
constexpr int kTE = kTileTestTileElems;
constexpr int kByteTE = kTileTestByteTileElems;

// ============================================================================
// Load/Store roundtrip kernels
// ============================================================================

template <typename T>
__global__ void
tile_load_store_kernel(const T* input, T* output, std::size_t ntiles) {
  auto group = make_block_group();
  for (std::size_t t = 0; t < ntiles; t++) {
    auto tile = tile_load<T, kTE, kBS>(input, t, group);
    tile_store<T, kTE, kBS>(output, t, tile, group);
  }
}

void test_tile_load_store_float(
    const float* input,
    float* output,
    std::size_t ntiles) {
  tile_load_store_kernel<float><<<1, kBS>>>(input, output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void test_tile_load_store_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    std::size_t ntiles) {
  tile_load_store_kernel<__nv_bfloat16><<<1, kBS>>>(input, output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void test_tile_load_store_half(
    const __half* input,
    __half* output,
    std::size_t ntiles) {
  tile_load_store_kernel<__half><<<1, kBS>>>(input, output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Zero kernel
// ============================================================================

__global__ void tile_zero_kernel(float* output, std::size_t ntiles) {
  auto group = make_block_group();
  for (std::size_t t = 0; t < ntiles; t++) {
    Tile<float, kTE, kBS, RegisterStorage> tile;
    tile_zero<float, kTE, kBS>(tile);
    tile_store<float, kTE, kBS>(output, t, tile, group);
  }
}

void test_tile_zero_float(float* output, std::size_t ntiles) {
  tile_zero_kernel<<<1, kBS>>>(output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Accumulate kernels
// ============================================================================

template <typename T, typename Op, int TileElems = kTE>
__global__ void tile_accumulate_kernel(
    const T* a_input,
    const T* b_input,
    T* output,
    std::size_t ntiles) {
  auto group = make_block_group();
  for (std::size_t t = 0; t < ntiles; t++) {
    auto a = tile_load<T, TileElems, kBS>(a_input, t, group);
    auto b = tile_load<T, TileElems, kBS>(b_input, t, group);
    tile_accumulate<T, Op, TileElems, kBS>(a, b);
    tile_store<T, TileElems, kBS>(output, t, a, group);
  }
}

void test_tile_accumulate_sum_float(
    const float* a,
    const float* b,
    float* output,
    std::size_t ntiles) {
  tile_accumulate_kernel<float, SumOp><<<1, kBS>>>(a, b, output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void test_tile_accumulate_sum_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* output,
    std::size_t ntiles) {
  tile_accumulate_kernel<__nv_bfloat16, SumOp>
      <<<1, kBS>>>(a, b, output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void test_tile_accumulate_max_float(
    const float* a,
    const float* b,
    float* output,
    std::size_t ntiles) {
  tile_accumulate_kernel<float, MaxOp><<<1, kBS>>>(a, b, output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void test_tile_accumulate_min_float(
    const float* a,
    const float* b,
    float* output,
    std::size_t ntiles) {
  tile_accumulate_kernel<float, MinOp><<<1, kBS>>>(a, b, output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_tile_accumulate_dtype(
    TileTestReduceOp op,
    const void* a,
    const void* b,
    void* output,
    std::size_t ntiles) {
  const auto* a_t = static_cast<const T*>(a);
  const auto* b_t = static_cast<const T*>(b);
  auto* output_t = static_cast<T*>(output);
  constexpr int kTileElems = sizeof(T) == 1 ? kByteTE : kTE;
  switch (op) {
    case TileTestReduceOp::kSum:
      tile_accumulate_kernel<T, SumOp, kTileElems>
          <<<1, kBS>>>(a_t, b_t, output_t, ntiles);
      PIPES_KERNEL_LAUNCH_CHECK();
      break;
    case TileTestReduceOp::kMax:
      tile_accumulate_kernel<T, MaxOp, kTileElems>
          <<<1, kBS>>>(a_t, b_t, output_t, ntiles);
      PIPES_KERNEL_LAUNCH_CHECK();
      break;
    case TileTestReduceOp::kMin:
      tile_accumulate_kernel<T, MinOp, kTileElems>
          <<<1, kBS>>>(a_t, b_t, output_t, ntiles);
      PIPES_KERNEL_LAUNCH_CHECK();
      break;
  }
}

namespace {

/*
 * One thread per (a, b) encoding pair. Compares the shipped fp16-accumulating
 * reduce against an fp32-accumulating one, both narrowed back to fp8, and
 * counts pairs whose RESULTS differ.
 *
 * Both paths take the same inputs and produce the same type, so any difference
 * is attributable to accumulator width alone. NaN pairs are skipped: NaN != NaN
 * under any comparison and would register as spurious witnesses.
 */
#if defined(__CUDA_FP8_TYPES_EXIST__)
template <typename Fp8T>
__global__ void fp8_accumulator_witness_kernel(
    unsigned int* witnesses,
    std::uint8_t* firstA,
    std::uint8_t* firstB) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 256u * 256u) {
    return;
  }
  const std::uint8_t rawA = static_cast<std::uint8_t>(idx >> 8);
  const std::uint8_t rawB = static_cast<std::uint8_t>(idx & 0xFFu);

  Fp8T a{}, b{};
  memcpy(&a, &rawA, 1);
  memcpy(&b, &rawB, 1);

  const float fa = float(a);
  const float fb = float(b);
  if (isnan(fa) || isnan(fb)) {
    return;
  }

  // Shipped path: widen to __half, add, narrow.
  const Fp8T viaHalf = Fp8T(__hadd(__half(a), __half(b)));
  // Counterfactual: widen to float, add, narrow.
  const Fp8T viaFloat = Fp8T(fa + fb);

  std::uint8_t outHalf = 0, outFloat = 0;
  memcpy(&outHalf, &viaHalf, 1);
  memcpy(&outFloat, &viaFloat, 1);

  if (outHalf != outFloat) {
    // Record only the first, so a witness is reportable without a full list.
    if (atomicAdd(witnesses, 1u) == 0u) {
      *firstA = rawA;
      *firstB = rawB;
    }
  }
}
#endif // __CUDA_FP8_TYPES_EXIST__

} // namespace

#if defined(__CUDA_FP8_TYPES_EXIST__)
void test_fp8_accumulator_witness(
    bool e5m2,
    std::uint32_t* witnesses,
    std::uint8_t* firstA,
    std::uint8_t* firstB) {
  constexpr int kPairs = 256 * 256;
  constexpr int kThreads = 256;
  const int blocks = (kPairs + kThreads - 1) / kThreads;
  auto* counter = reinterpret_cast<unsigned int*>(witnesses);
  if (e5m2) {
    fp8_accumulator_witness_kernel<__nv_fp8_e5m2>
        <<<blocks, kThreads>>>(counter, firstA, firstB);
  } else {
    fp8_accumulator_witness_kernel<__nv_fp8_e4m3>
        <<<blocks, kThreads>>>(counter, firstA, firstB);
  }
}
#endif // __CUDA_FP8_TYPES_EXIST__

namespace {

/*
 * Loads `base` into a tile, then accumulates `addend` over only the first
 * `valid_elems` of it. A `valid_elems` that is not a whole number of vectors
 * leaves a remainder, which is what routes through VecOps<T>::reduce_scalar.
 */
template <typename T, typename Op, int TileElems>
__global__ void tile_partial_load_accumulate_dtype_kernel(
    const T* base,
    const T* addend,
    T* output,
    std::size_t valid_elems) {
  auto group = make_block_group();
  auto tile = tile_load<T, TileElems, kBS>(base, 0, group);
  tile_load_accumulate<T, Op, TileElems, kBS>(
      tile, addend, 0, group, valid_elems);
  tile_store<T, TileElems, kBS>(output, 0, tile, group);
}

template <typename T>
void launch_tile_partial_load_accumulate_dtype(
    TileTestReduceOp op,
    const void* base,
    const void* addend,
    void* output,
    std::size_t valid_elems) {
  const auto* base_t = static_cast<const T*>(base);
  const auto* addend_t = static_cast<const T*>(addend);
  auto* output_t = static_cast<T*>(output);
  constexpr int kTileElems = sizeof(T) == 1 ? kByteTE : kTE;
  switch (op) {
    case TileTestReduceOp::kSum:
      tile_partial_load_accumulate_dtype_kernel<T, SumOp, kTileElems>
          <<<1, kBS>>>(base_t, addend_t, output_t, valid_elems);
      PIPES_KERNEL_LAUNCH_CHECK();
      break;
    case TileTestReduceOp::kMax:
      tile_partial_load_accumulate_dtype_kernel<T, MaxOp, kTileElems>
          <<<1, kBS>>>(base_t, addend_t, output_t, valid_elems);
      PIPES_KERNEL_LAUNCH_CHECK();
      break;
    case TileTestReduceOp::kMin:
      tile_partial_load_accumulate_dtype_kernel<T, MinOp, kTileElems>
          <<<1, kBS>>>(base_t, addend_t, output_t, valid_elems);
      PIPES_KERNEL_LAUNCH_CHECK();
      break;
  }
}

} // namespace

void test_tile_partial_load_accumulate_dtype(
    TileTestDataType dtype,
    TileTestReduceOp op,
    const void* base,
    const void* addend,
    void* output,
    std::size_t valid_elems) {
  switch (dtype) {
    case TileTestDataType::kInt8:
      launch_tile_partial_load_accumulate_dtype<std::int8_t>(
          op, base, addend, output, valid_elems);
      break;
    case TileTestDataType::kUint8:
      launch_tile_partial_load_accumulate_dtype<std::uint8_t>(
          op, base, addend, output, valid_elems);
      break;
    case TileTestDataType::kInt32:
      launch_tile_partial_load_accumulate_dtype<std::int32_t>(
          op, base, addend, output, valid_elems);
      break;
    case TileTestDataType::kUint32:
      launch_tile_partial_load_accumulate_dtype<std::uint32_t>(
          op, base, addend, output, valid_elems);
      break;
    case TileTestDataType::kInt64:
      launch_tile_partial_load_accumulate_dtype<std::int64_t>(
          op, base, addend, output, valid_elems);
      break;
    case TileTestDataType::kUint64:
      launch_tile_partial_load_accumulate_dtype<std::uint64_t>(
          op, base, addend, output, valid_elems);
      break;
    case TileTestDataType::kFloat64:
      launch_tile_partial_load_accumulate_dtype<double>(
          op, base, addend, output, valid_elems);
      break;
    case TileTestDataType::kFloat8E4M3:
    case TileTestDataType::kFloat8E5M2:
#if defined(__CUDA_FP8_TYPES_EXIST__)
      if (dtype == TileTestDataType::kFloat8E4M3) {
        launch_tile_partial_load_accumulate_dtype<__nv_fp8_e4m3>(
            op, base, addend, output, valid_elems);
      } else {
        launch_tile_partial_load_accumulate_dtype<__nv_fp8_e5m2>(
            op, base, addend, output, valid_elems);
      }
#endif
      break;
  }
}

void test_tile_accumulate_dtype(
    TileTestDataType dtype,
    TileTestReduceOp op,
    const void* a,
    const void* b,
    void* output,
    std::size_t ntiles) {
  switch (dtype) {
    case TileTestDataType::kInt8:
      launch_tile_accumulate_dtype<std::int8_t>(op, a, b, output, ntiles);
      break;
    case TileTestDataType::kUint8:
      launch_tile_accumulate_dtype<std::uint8_t>(op, a, b, output, ntiles);
      break;
    case TileTestDataType::kInt32:
      launch_tile_accumulate_dtype<std::int32_t>(op, a, b, output, ntiles);
      break;
    case TileTestDataType::kUint32:
      launch_tile_accumulate_dtype<std::uint32_t>(op, a, b, output, ntiles);
      break;
    case TileTestDataType::kInt64:
      launch_tile_accumulate_dtype<std::int64_t>(op, a, b, output, ntiles);
      break;
    case TileTestDataType::kUint64:
      launch_tile_accumulate_dtype<std::uint64_t>(op, a, b, output, ntiles);
      break;
    case TileTestDataType::kFloat64:
      launch_tile_accumulate_dtype<double>(op, a, b, output, ntiles);
      break;
    case TileTestDataType::kFloat8E4M3:
    case TileTestDataType::kFloat8E5M2:
      // The enumerators stay declared on every platform so this switch remains
      // exhaustive; only the fp8 launches are CUDA-only.
#if defined(__CUDA_FP8_TYPES_EXIST__)
      if (dtype == TileTestDataType::kFloat8E4M3) {
        launch_tile_accumulate_dtype<__nv_fp8_e4m3>(op, a, b, output, ntiles);
      } else {
        launch_tile_accumulate_dtype<__nv_fp8_e5m2>(op, a, b, output, ntiles);
      }
#endif
      break;
  }
}

// ============================================================================
// Fused load+accumulate kernel
// ============================================================================

__global__ void tile_load_accumulate_sum_kernel(
    const float* base,
    const float* addend,
    float* output,
    std::size_t ntiles) {
  auto group = make_block_group();
  for (std::size_t t = 0; t < ntiles; t++) {
    auto tile = tile_load<float, kTE, kBS>(base, t, group);
    tile_load_accumulate<float, SumOp, kTE, kBS>(tile, addend, t, group);
    tile_store<float, kTE, kBS>(output, t, tile, group);
  }
}

void test_tile_load_accumulate_sum_float(
    const float* base,
    const float* addend,
    float* output,
    std::size_t ntiles) {
  tile_load_accumulate_sum_kernel<<<1, kBS>>>(base, addend, output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Partial tile kernel
// ============================================================================

__global__ void tile_partial_load_kernel(
    const float* input,
    float* output,
    std::size_t valid_elems) {
  auto group = make_block_group();
  auto tile = tile_load<float, kTE, kBS>(input, 0, group, valid_elems);
  tile_store<float, kTE, kBS>(output, 0, tile, group);
}

void test_tile_partial_load_float(
    const float* input,
    float* output,
    std::size_t valid_elems) {
  tile_partial_load_kernel<<<1, kBS>>>(input, output, valid_elems);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Partial store kernel
// ============================================================================

__global__ void tile_partial_store_kernel(
    const float* input,
    float* output,
    std::size_t valid_elems) {
  auto group = make_block_group();
  auto tile = tile_load<float, kTE, kBS>(input, 0, group);
  tile_store<float, kTE, kBS>(output, 0, tile, group, valid_elems);
}

void test_tile_partial_store_float(
    const float* input,
    float* output,
    std::size_t valid_elems) {
  tile_partial_store_kernel<<<1, kBS>>>(input, output, valid_elems);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Partial load-accumulate kernel
// ============================================================================

__global__ void tile_partial_load_accumulate_kernel(
    const float* base,
    const float* addend,
    float* output,
    std::size_t valid_elems) {
  auto group = make_block_group();
  auto tile = tile_load<float, kTE, kBS>(base, 0, group);
  tile_load_accumulate<float, SumOp, kTE, kBS>(
      tile, addend, 0, group, valid_elems);
  tile_store<float, kTE, kBS>(output, 0, tile, group);
}

void test_tile_partial_load_accumulate_sum_float(
    const float* base,
    const float* addend,
    float* output,
    std::size_t valid_elems) {
  tile_partial_load_accumulate_kernel<<<1, kBS>>>(
      base, addend, output, valid_elems);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Partial load-accumulate with MaxOp (regression test for zero-padding bug)
// ============================================================================

__global__ void tile_partial_load_accumulate_max_kernel(
    const float* base,
    const float* addend,
    float* output,
    std::size_t valid_elems) {
  auto group = make_block_group();
  auto tile = tile_load<float, kTE, kBS>(base, 0, group);
  tile_load_accumulate<float, MaxOp, kTE, kBS>(
      tile, addend, 0, group, valid_elems);
  tile_store<float, kTE, kBS>(output, 0, tile, group);
}

void test_tile_partial_load_accumulate_max_float(
    const float* base,
    const float* addend,
    float* output,
    std::size_t valid_elems) {
  tile_partial_load_accumulate_max_kernel<<<1, kBS>>>(
      base, addend, output, valid_elems);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Partial load-accumulate with MinOp (regression test for zero-padding bug)
// ============================================================================

__global__ void tile_partial_load_accumulate_min_kernel(
    const float* base,
    const float* addend,
    float* output,
    std::size_t valid_elems) {
  auto group = make_block_group();
  auto tile = tile_load<float, kTE, kBS>(base, 0, group);
  tile_load_accumulate<float, MinOp, kTE, kBS>(
      tile, addend, 0, group, valid_elems);
  tile_store<float, kTE, kBS>(output, 0, tile, group);
}

void test_tile_partial_load_accumulate_min_float(
    const float* base,
    const float* addend,
    float* output,
    std::size_t valid_elems) {
  tile_partial_load_accumulate_min_kernel<<<1, kBS>>>(
      base, addend, output, valid_elems);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Combined partial load + partial store kernel
// ============================================================================

__global__ void tile_partial_load_store_kernel(
    const float* input,
    float* output,
    std::size_t valid_elems) {
  auto group = make_block_group();
  auto tile = tile_load<float, kTE, kBS>(input, 0, group, valid_elems);
  tile_store<float, kTE, kBS>(output, 0, tile, group, valid_elems);
}

void test_tile_partial_load_store_float(
    const float* input,
    float* output,
    std::size_t valid_elems) {
  tile_partial_load_store_kernel<<<1, kBS>>>(input, output, valid_elems);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Partial load at tile_idx=1
// ============================================================================

__global__ void tile_partial_load_idx1_kernel(
    const float* input,
    float* output,
    std::size_t valid_elems) {
  auto group = make_block_group();
  auto tile = tile_load<float, kTE, kBS>(input, 1, group, valid_elems);
  tile_store<float, kTE, kBS>(output, 0, tile, group);
}

void test_tile_partial_load_tile_idx1_float(
    const float* input,
    float* output,
    std::size_t valid_elems) {
  tile_partial_load_idx1_kernel<<<1, kBS>>>(input, output, valid_elems);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Half-precision max / bf16 min accumulate kernels
// ============================================================================

__global__ void tile_accumulate_max_half_kernel(
    const __half* a_input,
    const __half* b_input,
    __half* output,
    std::size_t ntiles) {
  auto group = make_block_group();
  for (std::size_t t = 0; t < ntiles; t++) {
    auto a = tile_load<__half, kTE, kBS>(a_input, t, group);
    auto b = tile_load<__half, kTE, kBS>(b_input, t, group);
    tile_accumulate<__half, MaxOp, kTE, kBS>(a, b);
    tile_store<__half, kTE, kBS>(output, t, a, group);
  }
}

void test_tile_accumulate_max_half(
    const __half* a,
    const __half* b,
    __half* output,
    std::size_t ntiles) {
  tile_accumulate_max_half_kernel<<<1, kBS>>>(a, b, output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

__global__ void tile_accumulate_min_bf16_kernel(
    const __nv_bfloat16* a_input,
    const __nv_bfloat16* b_input,
    __nv_bfloat16* output,
    std::size_t ntiles) {
  auto group = make_block_group();
  for (std::size_t t = 0; t < ntiles; t++) {
    auto a = tile_load<__nv_bfloat16, kTE, kBS>(a_input, t, group);
    auto b = tile_load<__nv_bfloat16, kTE, kBS>(b_input, t, group);
    tile_accumulate<__nv_bfloat16, MinOp, kTE, kBS>(a, b);
    tile_store<__nv_bfloat16, kTE, kBS>(output, t, a, group);
  }
}

void test_tile_accumulate_min_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* output,
    std::size_t ntiles) {
  tile_accumulate_min_bf16_kernel<<<1, kBS>>>(a, b, output, ntiles);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// 2D tile load/store kernel
// ============================================================================

constexpr int k2DR = kTileTest2DRows;
constexpr int k2DC = kTileTest2DCols;

__global__ void tile_load_store_2d_kernel(
    const float* input,
    float* output,
    std::size_t stride,
    std::size_t row_offset,
    std::size_t col_offset,
    std::size_t valid_rows,
    std::size_t valid_cols) {
  auto group = make_block_group();
  auto tile = tile_load_2d<float, k2DR, k2DC, kBS>(
      input, row_offset, col_offset, stride, group, valid_rows, valid_cols);
  tile_store_2d<float, k2DR, k2DC, kBS>(
      output,
      row_offset,
      col_offset,
      stride,
      tile,
      group,
      valid_rows,
      valid_cols);
}

void test_tile_load_store_2d_float(
    const float* input,
    float* output,
    std::size_t stride,
    std::size_t row_offset,
    std::size_t col_offset,
    std::size_t valid_rows,
    std::size_t valid_cols) {
  tile_load_store_2d_kernel<<<1, kBS>>>(
      input, output, stride, row_offset, col_offset, valid_rows, valid_cols);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::prims::test
