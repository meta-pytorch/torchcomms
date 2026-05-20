// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/Tile.cuh"
#include "comms/pipes/benchmarks/TileBench.cuh"

namespace comms::pipes::benchmark {

using namespace comms::pipes;

constexpr int kBenchBlockSize = 256;
constexpr int kBenchTileElems = 2048;

__global__ void
tile_copy_kernel(float* dst, const float* src, std::size_t nelems, int nruns) {
  auto group = make_block_group();
  const std::size_t ntiles = nelems / kBenchTileElems;
  const std::size_t tiles_per_block = (ntiles + gridDim.x - 1) / gridDim.x;
  const std::size_t tile_start = blockIdx.x * tiles_per_block;
  const std::size_t tile_end = (tile_start + tiles_per_block < ntiles)
      ? tile_start + tiles_per_block
      : ntiles;

  for (int run = 0; run < nruns; run++) {
    for (std::size_t t = tile_start; t < tile_end; t++) {
      auto tile =
          tile_load<float, kBenchTileElems, kBenchBlockSize>(src, t, group);
      tile_store<float, kBenchTileElems, kBenchBlockSize>(dst, t, tile, group);
    }
  }
}

__global__ void tile_reduce_sum_kernel(
    float* dst,
    const float* src_a,
    const float* src_b,
    std::size_t nelems,
    int nruns) {
  auto group = make_block_group();
  const std::size_t ntiles = nelems / kBenchTileElems;
  const std::size_t tiles_per_block = (ntiles + gridDim.x - 1) / gridDim.x;
  const std::size_t tile_start = blockIdx.x * tiles_per_block;
  const std::size_t tile_end = (tile_start + tiles_per_block < ntiles)
      ? tile_start + tiles_per_block
      : ntiles;

  for (int run = 0; run < nruns; run++) {
    for (std::size_t t = tile_start; t < tile_end; t++) {
      auto tile =
          tile_load<float, kBenchTileElems, kBenchBlockSize>(src_a, t, group);
      tile_load_accumulate<float, SumOp, kBenchTileElems, kBenchBlockSize>(
          tile, src_b, t, group);
      tile_store<float, kBenchTileElems, kBenchBlockSize>(dst, t, tile, group);
    }
  }
}

} // namespace comms::pipes::benchmark
