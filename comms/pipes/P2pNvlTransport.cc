// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/P2pNvlTransport.h"

#include <vector>

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

  // Calculate state buffer size based on chunk size
  // stateBufferSize = dataBufferSize / chunkSize * sizeof(ChunkState<int>)
  const std::size_t numChunks =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const std::size_t stateBufferSize = numChunks * sizeof(ChunkState<int>);

  stateBuffer_d_ = std::make_unique<meta::comms::DeviceBuffer>(stateBufferSize);
  stateBufferHandler_ = std::make_unique<meta::comms::IpcMemHandler>(
      mpiBootstrap_, myRank, nRanks_);
  stateBufferHandler_->addSelfDeviceMemPtr(stateBuffer_d_->get());

  // Initialize state buffer to -1
  auto statePtr = static_cast<ChunkState<int>*>(stateBuffer_d_->get());
  std::vector<ChunkState<int>> initStates(numChunks, ChunkState<int>(-1));
  auto cudaErr = cudaMemcpy(
      statePtr, initStates.data(), stateBufferSize, cudaMemcpyDefault);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error(
        "cudaMemcpy failed in state buffer initialization");
  }
};

void P2pNvlTransport::exchange() {
  dataBufferHandler_->exchangeMemPtrs();
  stateBufferHandler_->exchangeMemPtrs();
}

P2pNvlTransportDevice P2pNvlTransport::getTransportDevice(int peerRank) {
  P2pNvlTransportOptions options{
      .dataBufferSize = config_.dataBufferSize, .chunkSize = config_.chunkSize};

  LocalState localState{
      .dataBuffer = static_cast<char*>(dataBuffer_d_->get()),
      .stateBuffer = static_cast<ChunkState<int>*>(stateBuffer_d_->get())};

  RemoteState remoteState{
      .dataBuffer =
          static_cast<char*>(dataBufferHandler_->getPeerDeviceMemPtr(peerRank)),
      .stateBuffer = static_cast<ChunkState<int>*>(
          stateBufferHandler_->getPeerDeviceMemPtr(peerRank))};

  return P2pNvlTransportDevice(
      myRank_, peerRank, options, localState, remoteState);
}

} // namespace comms::pipes
