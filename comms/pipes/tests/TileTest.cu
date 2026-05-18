// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/Tile.cuh"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/TileTest.cuh"

namespace comms::pipes::test {

using comms::pipes::make_block_group;
using comms::pipes::MaxOp;
using comms::pipes::MinOp;
using comms::pipes::RegisterStorage;
using comms::pipes::SumOp;
using comms::pipes::Tile;
using comms::pipes::tile_accumulate;
using comms::pipes::tile_load;
using comms::pipes::tile_load_2d;
using comms::pipes::tile_load_accumulate;
using comms::pipes::tile_store;
using comms::pipes::tile_store_2d;
using comms::pipes::tile_zero;

constexpr int kBS = kTileTestBlockSize;
constexpr int kTE = kTileTestTileElems;

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

template <typename T, typename Op>
__global__ void tile_accumulate_kernel(
    const T* a_input,
    const T* b_input,
    T* output,
    std::size_t ntiles) {
  auto group = make_block_group();
  for (std::size_t t = 0; t < ntiles; t++) {
    auto a = tile_load<T, kTE, kBS>(a_input, t, group);
    auto b = tile_load<T, kTE, kBS>(b_input, t, group);
    tile_accumulate<T, Op, kTE, kBS>(a, b);
    tile_store<T, kTE, kBS>(output, t, a, group);
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

} // namespace comms::pipes::test
