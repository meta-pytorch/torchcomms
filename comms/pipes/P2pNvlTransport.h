// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/common/IpcMemHandler.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/utils/CudaRAII.h"

namespace comms::pipes {

// Configuration for P2P NVL transport buffer sizes
struct P2pNvlTransportConfig {
  std::size_t dataBufferSize{0};
  std::size_t chunkSize{0};
};

// Host-side P2P NVL transport that exchanges IPC buffer handles between ranks
// and provides API to get point-to-point NVLink transport device handle
class P2pNvlTransport {
 public:
  P2pNvlTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::MpiBootstrap> mpiBootstrap,
      const P2pNvlTransportConfig& p2pNvlTransportConfig);

  // exchange IPC buffer handles across ranks
  void exchange();

  P2pNvlTransportDevice getTransportDevice(int peerRank);

 private:
  const int myRank_{-1};
  const int nRanks_{-1};
  // TODO: make it ctran::bootstrap::IBootstrap when integrating with Ctran
  std::shared_ptr<meta::comms::MpiBootstrap> mpiBootstrap_;
  const P2pNvlTransportConfig config_;

  // data buffer: staging buffer for send/recv data
  // state buffer: buffer for signaling
  std::unique_ptr<meta::comms::DeviceBuffer> dataBuffer_d_;
  std::unique_ptr<meta::comms::DeviceBuffer> stateBuffer_d_;

  // TODO: refactor IpcMemHandler to handle multiple ipcHandles exchange and
  // merge into one IpcMemHandler
  std::unique_ptr<meta::comms::IpcMemHandler> dataBufferHandler_;
  std::unique_ptr<meta::comms::IpcMemHandler> stateBufferHandler_;
};

} // namespace comms::pipes
