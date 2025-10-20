// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/common/algorithms/AlgoUtils.h"

#include <cmath>
#include <utility>

namespace meta {
namespace comms {

inline uint32_t divRoundUp(size_t a, size_t b) {
  return static_cast<uint32_t>((a + b - 1) / b);
}

constexpr uint32_t
calcBlockCount(size_t numThreads, size_t threadsPerBlock, size_t maxBlocks) {
  const auto uNumThreads = static_cast<uint64_t>(numThreads);
  const auto uThreadsPerBlock = static_cast<uint64_t>(threadsPerBlock);
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks =
      uNumThreads / uThreadsPerBlock + (uNumThreads % uThreadsPerBlock != 0);
  return static_cast<uint32_t>(std::min(blocks, maxBlocks));
}

std::pair<dim3, dim3>
getGridAndBlockDims(size_t count, commDataType_t datatype, size_t maxBlocks) {
#if defined(USE_ROCM)
  constexpr uint32_t kThreadsPerWarp = 64;
  constexpr uint32_t kThreadsPerBlock = 512;
#else
  constexpr uint32_t kThreadsPerWarp = 32;
  constexpr uint32_t kThreadsPerBlock = 1024;
#endif

  const uint32_t elementsPerThread =
      16 / commTypeSize(datatype); // we do 16 Byte load in kernel

  const uint32_t elementsPerWarp = elementsPerThread * kThreadsPerWarp;

  dim3 threads(0, 1, 1);
  dim3 blocks(0, 1, 1);
  if (count < elementsPerThread * kThreadsPerBlock) {
    threads.x = divRoundUp(count, elementsPerWarp) * kThreadsPerWarp;
    blocks.x = 1;
  } else {
    auto warpsRequired = divRoundUp(count, elementsPerWarp);
    blocks.x = calcBlockCount(
        divRoundUp(count, elementsPerThread), kThreadsPerBlock, maxBlocks);
    auto warpsPerBlock = divRoundUp(warpsRequired, blocks.x);
    auto threadsPerBlock =
        std::min<uint32_t>(kThreadsPerBlock, warpsPerBlock * kThreadsPerWarp);
    threads.x = threadsPerBlock;
  }

  return std::make_pair(blocks, threads);
}

} // namespace comms
} // namespace meta
