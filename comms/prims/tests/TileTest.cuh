// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace comms::prims::test {

constexpr int kTileTestBlockSize = 256;
constexpr int kTileTestTileElems = 2048;
constexpr int kTileTestByteTileElems = 4096;

enum class TileTestDataType {
  kInt8,
  kUint8,
  kInt32,
  kUint32,
  kInt64,
  kUint64,
  kFloat64,
  kFloat8E4M3,
  kFloat8E5M2,
};

enum class TileTestReduceOp {
  kSum,
  kMax,
  kMin,
};

void test_tile_load_store_float(
    const float* input,
    float* output,
    std::size_t ntiles);

void test_tile_load_store_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    std::size_t ntiles);

void test_tile_load_store_half(
    const __half* input,
    __half* output,
    std::size_t ntiles);

void test_tile_zero_float(float* output, std::size_t ntiles);

void test_tile_accumulate_sum_float(
    const float* a,
    const float* b,
    float* output,
    std::size_t ntiles);

void test_tile_accumulate_sum_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* output,
    std::size_t ntiles);

void test_tile_accumulate_max_float(
    const float* a,
    const float* b,
    float* output,
    std::size_t ntiles);

void test_tile_accumulate_min_float(
    const float* a,
    const float* b,
    float* output,
    std::size_t ntiles);

/*
 * Accumulator-width witness search, for the fp8 formats only.
 *
 * Prims reduces fp8 by widening to __half, operating, and narrowing back,
 * following NCCL. Whether that is OBSERVABLE at the output versus widening to
 * float instead is decidable rather than arguable: both operands are 8-bit, so
 * there are only 256 x 256 pairs. This runs every pair through both widths and
 * counts the ones whose narrowed results differ.
 *
 * `witnesses` is a single device counter. `firstA` / `firstB` receive the raw
 * encodings of one differing pair, or are left untouched when none exists.
 *
 * Takes uint8_t rather than an fp8 type so the caller need not include
 * <cuda_fp8.h>; `e5m2` selects the format.
 */
void test_fp8_accumulator_witness(
    bool e5m2,
    std::uint32_t* witnesses,
    std::uint8_t* firstA,
    std::uint8_t* firstB);

void test_tile_accumulate_dtype(
    TileTestDataType dtype,
    TileTestReduceOp op,
    const void* a,
    const void* b,
    void* output,
    std::size_t ntiles);

/*
 * Partial load-accumulate, dispatched on datatype.
 *
 * `valid_elems` below a whole number of vectors is what drives
 * tile_load_accumulate down its scalar tail, so this is the only entry point
 * that reaches VecOps<T>::reduce_scalar for anything but float. The float
 * helpers below predate it and stay as they are.
 */
void test_tile_partial_load_accumulate_dtype(
    TileTestDataType dtype,
    TileTestReduceOp op,
    const void* base,
    const void* addend,
    void* output,
    std::size_t valid_elems);

void test_tile_load_accumulate_sum_float(
    const float* base,
    const float* addend,
    float* output,
    std::size_t ntiles);

void test_tile_partial_load_float(
    const float* input,
    float* output,
    std::size_t valid_elems);

void test_tile_partial_store_float(
    const float* input,
    float* output,
    std::size_t valid_elems);

void test_tile_partial_load_accumulate_sum_float(
    const float* base,
    const float* addend,
    float* output,
    std::size_t valid_elems);

void test_tile_partial_load_accumulate_max_float(
    const float* base,
    const float* addend,
    float* output,
    std::size_t valid_elems);

void test_tile_partial_load_accumulate_min_float(
    const float* base,
    const float* addend,
    float* output,
    std::size_t valid_elems);

void test_tile_partial_load_store_float(
    const float* input,
    float* output,
    std::size_t valid_elems);

void test_tile_partial_load_tile_idx1_float(
    const float* input,
    float* output,
    std::size_t valid_elems);

void test_tile_accumulate_max_half(
    const __half* a,
    const __half* b,
    __half* output,
    std::size_t ntiles);

void test_tile_accumulate_min_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* output,
    std::size_t ntiles);

constexpr int kTileTest2DRows = 8;
constexpr int kTileTest2DCols = 256;

void test_tile_load_store_2d_float(
    const float* input,
    float* output,
    std::size_t stride,
    std::size_t row_offset,
    std::size_t col_offset,
    std::size_t valid_rows,
    std::size_t valid_cols);

} // namespace comms::prims::test
