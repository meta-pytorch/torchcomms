// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace comms::pipes {

// Device buffer pointers for data and state buffers
struct BufferState {
  void* dataBuffer_d{nullptr};
  void* stateBuffer_d{nullptr};
};

// Device-side P2P NVLink transport handle providing point-to-point operations
// like send/recv between ranks over NVLink
class P2pNvlTransportDevice {
 public:
  __host__ P2pNvlTransportDevice(
      int myRank,
      int peerRank,
      const BufferState& myState,
      const BufferState& peerState);

 private:
  const int myRank_{-1};
  const int peerRank_{-1};
  BufferState myState_;
  BufferState peerState_;

#ifdef P2pNvlTransport_TEST_FRIENDS
  P2pNvlTransport_TEST_FRIENDS
#endif
};

} // namespace comms::pipes
