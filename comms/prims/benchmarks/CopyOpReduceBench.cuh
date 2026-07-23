// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace comms::prims::benchmark {

enum class CopyOpReducePolicy {
  TileReduceStaged,
  CpAsyncSmemReduce,
};

struct CopyOpReduceTiming {
  float timeUs;
  float payloadGBps;
  float memoryGBps;
};

CopyOpReduceTiming runCopyOpReduceBenchmark(
    CopyOpReducePolicy policy,
    std::size_t nbytes,
    int iterations);

} // namespace comms::prims::benchmark
