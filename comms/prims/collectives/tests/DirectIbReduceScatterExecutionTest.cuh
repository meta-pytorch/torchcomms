// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace comms::prims::test {

inline constexpr int kDirectIbTraceChannels = 2;
inline constexpr int kDirectIbTracePeers = 3;
inline constexpr int kDirectIbTraceOwnInput = 1;
inline constexpr int kDirectIbTraceOutput = 2;

struct DirectIbExecutionTrace {
  int recvCount[kDirectIbTraceChannels]{};
  int sendCount[kDirectIbTraceChannels]{};
  int recvPeers[kDirectIbTraceChannels][kDirectIbTracePeers]{};
  int sendPeers[kDirectIbTraceChannels][kDirectIbTracePeers]{};
  int recvInputKind[kDirectIbTraceChannels][kDirectIbTracePeers]{};
  float sendFirstValue[kDirectIbTraceChannels][kDirectIbTracePeers]{};
};

void launchDirectIbSingleRankRange(
    const float* input,
    std::size_t strideElements,
    float* output,
    std::size_t rangeOffsetElements,
    std::size_t rangeElements,
    bool outputAlreadyInitialized);

void launchDirectIbPeerWalkTrace(
    const float* input,
    float* output,
    DirectIbExecutionTrace* traces);

} // namespace comms::prims::test
