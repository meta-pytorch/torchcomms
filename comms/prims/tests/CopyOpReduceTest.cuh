// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace comms::prims::test {

enum class CopyOpReducePolicy {
  TileReduce,
  TileReduceStaged,
  CpAsyncSmemReduce,
};

void launchCopyOpReduce(
    CopyOpReducePolicy policy,
    float* output,
    const float* staging,
    const float* localInput,
    std::size_t byteOffset,
    std::size_t nbytes);

} // namespace comms::prims::test
