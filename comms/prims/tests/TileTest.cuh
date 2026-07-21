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

void test_tile_accumulate_dtype(
    TileTestDataType dtype,
    TileTestReduceOp op,
    const void* a,
    const void* b,
    void* output,
    std::size_t ntiles);

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
