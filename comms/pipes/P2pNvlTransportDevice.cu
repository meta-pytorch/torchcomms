// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/P2pNvlTransportDevice.cuh"

namespace comms::pipes {

__host__ P2pNvlTransportDevice::P2pNvlTransportDevice(
    int myRank,
    int peerRank,
    const BufferState& myState,
    const BufferState& peerState)
    : myRank_(myRank),
      peerRank_(peerRank),
      myState_(myState),
      peerState_(peerState) {}

} // namespace comms::pipes
