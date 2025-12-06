// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/P2pNvlTransport.h"

namespace comms::pipes {

P2pNvlTransport::P2pNvlTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::MpiBootstrap> mpiBootstrap,
    const P2pNvlTransportConfig& p2pNvlTransportConfig)
    : myRank_(myRank),
      nRanks_(nRanks),
      mpiBootstrap_(mpiBootstrap),
      config_(p2pNvlTransportConfig) {
  dataBuffer_d_ =
      std::make_unique<meta::comms::DeviceBuffer>(config_.dataBufferSize);
  dataBufferHandler_ = std::make_unique<meta::comms::IpcMemHandler>(
      mpiBootstrap_, myRank, nRanks_);
  dataBufferHandler_->addSelfDeviceMemPtr(dataBuffer_d_->get());

  stateBuffer_d_ =
      std::make_unique<meta::comms::DeviceBuffer>(config_.stateBufferSize);
  stateBufferHandler_ = std::make_unique<meta::comms::IpcMemHandler>(
      mpiBootstrap_, myRank, nRanks_);
  stateBufferHandler_->addSelfDeviceMemPtr(stateBuffer_d_->get());
};

void P2pNvlTransport::exchange() {
  dataBufferHandler_->exchangeMemPtrs();
  stateBufferHandler_->exchangeMemPtrs();
}

P2pNvlTransportDevice P2pNvlTransport::getTransportDevice(int peerRank) {
  BufferState myState{
      .dataBuffer_d = dataBuffer_d_->get(),
      .stateBuffer_d = stateBuffer_d_->get()};

  BufferState peerState{
      .dataBuffer_d = dataBufferHandler_->getPeerDeviceMemPtr(peerRank),
      .stateBuffer_d = stateBufferHandler_->getPeerDeviceMemPtr(peerRank)};

  return P2pNvlTransportDevice(myRank_, peerRank, myState, peerState);
}

} // namespace comms::pipes
