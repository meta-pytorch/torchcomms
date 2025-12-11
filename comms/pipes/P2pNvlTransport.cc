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
  // Calculate total buffer sizes with pipelining
  const std::size_t totalDataBufferSize =
      config_.pipelineDepth * config_.dataBufferSize;

  // Calculate state buffer size based on chunk size and pipeline depth
  const std::size_t numChunksPerStep =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const std::size_t totalNumChunks = config_.pipelineDepth * numChunksPerStep;
  const std::size_t stateBufferSize = totalNumChunks * sizeof(ChunkState);

  dataBuffer_d_ =
      std::make_unique<meta::comms::DeviceBuffer>(totalDataBufferSize);
  dataBufferHandler_ = std::make_unique<meta::comms::IpcMemHandler>(
      mpiBootstrap_, myRank, nRanks_);
  dataBufferHandler_->addSelfDeviceMemPtr(dataBuffer_d_->get());

  stateBuffer_d_ = std::make_unique<meta::comms::DeviceBuffer>(stateBufferSize);
  stateBufferHandler_ = std::make_unique<meta::comms::IpcMemHandler>(
      mpiBootstrap_, myRank, nRanks_);
  stateBufferHandler_->addSelfDeviceMemPtr(stateBuffer_d_->get());

  // Initialize state buffer to READY_TO_SEND for all pipeline slots
  auto statePtr = static_cast<ChunkState*>(stateBuffer_d_->get());
  std::vector<ChunkState> initStates(totalNumChunks);
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
      .dataBufferSize = config_.dataBufferSize,
      .chunkSize = config_.chunkSize,
      .pipelineDepth = config_.pipelineDepth};

  // Calculate number of chunk states per pipeline slot
  const std::size_t numChunksPerStep =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const auto totalNumChunks =
      static_cast<uint32_t>(config_.pipelineDepth * numChunksPerStep);

  LocalState localState{
      .dataBuffer = static_cast<char*>(dataBuffer_d_->get()),
      .stateBuffer = DeviceSpan<ChunkState>(
          static_cast<ChunkState*>(stateBuffer_d_->get()), totalNumChunks)};

  RemoteState remoteState{
      .dataBuffer =
          static_cast<char*>(dataBufferHandler_->getPeerDeviceMemPtr(peerRank)),
      .stateBuffer = DeviceSpan<ChunkState>(
          static_cast<ChunkState*>(
              stateBufferHandler_->getPeerDeviceMemPtr(peerRank)),
          totalNumChunks)};

  return P2pNvlTransportDevice(
      myRank_, peerRank, options, localState, remoteState);
}

} // namespace comms::pipes
