// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiPeerNvlTransport.h"

#include <vector>

namespace comms::pipes {

MultiPeerNvlTransport::MultiPeerNvlTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
    const MultiPeerNvlTransportConfig& multiPeerNvlTransportConfig)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(multiPeerNvlTransportConfig) {
  // Calculate per-peer buffer sizes with pipelining
  perPeerDataBufferSize_ = config_.pipelineDepth * config_.dataBufferSize;

  // Calculate state buffer size based on chunk size and pipeline depth
  const std::size_t numChunksPerStep =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const std::size_t numChunksPerPeer = config_.pipelineDepth * numChunksPerStep;
  perPeerStateBufferSize_ = numChunksPerPeer * sizeof(ChunkState);

  // Allocate buffers for (nRanks - 1) peers using GpuMemHandler
  const std::size_t totalDataBufferSize =
      perPeerDataBufferSize_ * (nRanks_ - 1);
  const std::size_t totalStateBufferSize =
      perPeerStateBufferSize_ * (nRanks_ - 1);

  // Create memory handlers with automatic mode detection
  // Will use fabric handles on H100+/CUDA12.3+, otherwise cudaIpc
  dataBufferHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, totalDataBufferSize);

  stateBufferHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, totalStateBufferSize);

  // Initialize state buffer to READY_TO_SEND for all pipeline slots
  auto statePtr =
      static_cast<ChunkState*>(stateBufferHandler_->getLocalDeviceMemPtr());
  const std::size_t totalNumChunksAllPeers = numChunksPerPeer * (nRanks_ - 1);
  std::vector<ChunkState> initStates(totalNumChunksAllPeers);
  auto cudaErr = cudaMemcpy(
      statePtr, initStates.data(), totalStateBufferSize, cudaMemcpyDefault);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error(
        "cudaMemcpy failed in state buffer initialization");
  }
}

void MultiPeerNvlTransport::exchange() {
  dataBufferHandler_->exchangeMemPtrs();
  stateBufferHandler_->exchangeMemPtrs();
}

P2pNvlTransportDevice MultiPeerNvlTransport::getP2pTransportDevice(
    int peerRank) {
  // Buffer Layout Example (4 ranks, buffer size X per peer):
  //
  // Rank 0's buffer: [Slot 0: Peer1 | Slot 1: Peer2 | Slot 2: Peer3]
  // Rank 1's buffer: [Slot 0: Peer0 | Slot 1: Peer2 | Slot 2: Peer3]
  // Rank 2's buffer: [Slot 0: Peer0 | Slot 1: Peer1 | Slot 2: Peer3]
  // Rank 3's buffer: [Slot 0: Peer0 | Slot 1: Peer1 | Slot 2: Peer2]
  //
  // Communication examples:
  // Rank 0 -> Rank 1: local[0]=0*X, remote[0]=0*X
  // Rank 0 -> Rank 2: local[1]=1*X, remote[0]=0*X
  // Rank 0 -> Rank 3: local[2]=2*X, remote[0]=0*X
  // Rank 1 -> Rank 2: local[1]=1*X, remote[1]=1*X
  // Rank 1 -> Rank 3: local[2]=2*X, remote[1]=1*X
  // Rank 2 -> Rank 3: local[2]=2*X, remote[2]=2*X
  // ...

  // Calculate local peer index for buffer offset in my own buffer
  // For peerRank < myRank, they occupy slots 0, 1, 2, ...
  // For peerRank > myRank, they occupy subsequent slots
  const int localPeerIndex = (peerRank < myRank_) ? peerRank : (peerRank - 1);
  const std::size_t localDataBufferOffset =
      localPeerIndex * perPeerDataBufferSize_;
  const std::size_t localStateBufferOffset =
      localPeerIndex * perPeerStateBufferSize_;

  // Calculate remote peer index for buffer offset in peer's buffer
  // From peer's perspective, where does myRank fit in their buffer?
  const int remotePeerIndex = (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
  const std::size_t remoteDataBufferOffset =
      remotePeerIndex * perPeerDataBufferSize_;
  const std::size_t remoteStateBufferOffset =
      remotePeerIndex * perPeerStateBufferSize_;

  P2pNvlTransportOptions options{
      .dataBufferSize = config_.dataBufferSize,
      .chunkSize = config_.chunkSize,
      .pipelineDepth = config_.pipelineDepth};

  // Calculate number of chunk states per pipeline slot
  const std::size_t numChunksPerStep =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const auto numChunksPerPeer =
      static_cast<uint32_t>(config_.pipelineDepth * numChunksPerStep);

  auto* localDataPtr =
      static_cast<char*>(dataBufferHandler_->getLocalDeviceMemPtr());
  auto* localStatePtr =
      static_cast<char*>(stateBufferHandler_->getLocalDeviceMemPtr());

  LocalState localState{
      .dataBuffer = localDataPtr + localDataBufferOffset,
      .stateBuffer = DeviceSpan<ChunkState>(
          reinterpret_cast<ChunkState*>(localStatePtr + localStateBufferOffset),
          numChunksPerPeer)};

  auto* remoteDataPtr =
      static_cast<char*>(dataBufferHandler_->getPeerDeviceMemPtr(peerRank));
  auto* remoteStatePtr =
      static_cast<char*>(stateBufferHandler_->getPeerDeviceMemPtr(peerRank));

  RemoteState remoteState{
      .dataBuffer = remoteDataPtr + remoteDataBufferOffset,
      .stateBuffer = DeviceSpan<ChunkState>(
          reinterpret_cast<ChunkState*>(
              remoteStatePtr + remoteStateBufferOffset),
          numChunksPerPeer)};

  return P2pNvlTransportDevice(
      myRank_, peerRank, options, localState, remoteState);
}

} // namespace comms::pipes
