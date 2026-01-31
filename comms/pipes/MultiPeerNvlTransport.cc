// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiPeerNvlTransport.h"

#include <cuda_runtime.h>
#include <vector>

#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes {

namespace {
// Helper macro for CUDA error checking
#define CUDACHECK(cmd)                                                   \
  do {                                                                   \
    cudaError_t e = cmd;                                                 \
    if (e != cudaSuccess) {                                              \
      throw std::runtime_error(                                          \
          std::string("CUDA error: ") + cudaGetErrorString(e) + " at " + \
          __FILE__ + ":" + std::to_string(__LINE__));                    \
    }                                                                    \
  } while (0)
} // namespace

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
  // Single state mode (useDualStateBuffer=false):
  //   - 1 chunk state per chunk, stored on receiver side only
  // Dual state mode (useDualStateBuffer=true):
  //   - 2 chunk states per chunk:
  //     - First half [0, numChunksPerPeer): my chunk state (local operations)
  //     - Second half [numChunksPerPeer, 2*numChunksPerPeer): peer's chunk
  //     state
  //       (stored locally for local polling)
  const std::size_t numChunksPerStep =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const std::size_t numChunksPerPeer = config_.pipelineDepth * numChunksPerStep;
  const std::size_t chunkStateMultiplier = config_.useDualStateBuffer ? 2 : 1;
  perPeerChunkStateBufferSize_ =
      chunkStateMultiplier * numChunksPerPeer * sizeof(ChunkState);
  perPeerSignalBufferSize_ = getSignalBufferSize(config_.signalCount);

  // Allocate buffers for (nRanks - 1) peers using GpuMemHandler
  const std::size_t totalDataBufferSize =
      perPeerDataBufferSize_ * (nRanks_ - 1);
  const std::size_t totalChunkStateBufferSize =
      perPeerChunkStateBufferSize_ * (nRanks_ - 1);
  const std::size_t totalSignalBufferSize =
      perPeerSignalBufferSize_ * (nRanks_ - 1);

  // Detect memory sharing mode once and use for both handlers
  // This avoids redundant mode detection (fabric check is cached but this is
  // cleaner)
  const auto memSharingMode = GpuMemHandler::detectBestMode();

  signalBufferHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, totalSignalBufferSize, memSharingMode);

  dataBufferHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, totalDataBufferSize, memSharingMode);

  stateBufferHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, totalChunkStateBufferSize, memSharingMode);

  // Initialize state buffer to READY_TO_SEND for all pipeline slots
  // The number of states depends on useDualStateBuffer:
  // - Single mode: 1x chunk states (only myChunkStateBuffer)
  // - Dual mode: 2x chunk states (myChunkStateBuffer + peerChunkStateBuffer)
  auto statePtr =
      static_cast<ChunkState*>(stateBufferHandler_->getLocalDeviceMemPtr());
  const std::size_t totalNumChunksAllPeers =
      chunkStateMultiplier * numChunksPerPeer * (nRanks_ - 1);
  std::vector<ChunkState> initStates(totalNumChunksAllPeers);
  auto cudaErr = cudaMemcpy(
      statePtr,
      initStates.data(),
      totalChunkStateBufferSize,
      cudaMemcpyDefault);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error(
        "cudaMemcpy failed in state buffer initialization");
  }

  // Initialize signal state buffer to 0 for all ranks
  auto signalPtr =
      static_cast<SignalState*>(signalBufferHandler_->getLocalDeviceMemPtr());
  std::vector<SignalState> signalInitStates(
      config_.signalCount * (nRanks_ - 1));
  cudaErr = cudaMemcpy(
      signalPtr,
      signalInitStates.data(),
      totalSignalBufferSize,
      cudaMemcpyDefault);
  if (cudaErr != cudaSuccess) {
    throw std::runtime_error(
        "cudaMemcpy failed in signal state buffer initialization");
  }

  // Preallocate Transport array on device memory
  // We allocate nRanks_ slots including self
  // The array is populated in exchange() after peer buffer pointers are
  // available (self transport is also populated there for consistency)
  CUDACHECK(cudaMalloc(&transports_d_, nRanks_ * sizeof(Transport)));
};

MultiPeerNvlTransport::~MultiPeerNvlTransport() {
  if (transports_d_ != nullptr) {
    CUDACHECK(cudaFree(transports_d_));
    transports_d_ = nullptr;
  }
}

void MultiPeerNvlTransport::exchange() {
  dataBufferHandler_->exchangeMemPtrs();
  stateBufferHandler_->exchangeMemPtrs();
  signalBufferHandler_->exchangeMemPtrs();

  // Build and copy Transport for each rank to the preallocated device array
  // - For myRank_: construct P2pSelfTransportDevice
  // - For peers: construct P2pNvlTransportDevice using
  // buildP2pTransportDevice()
  for (int rank = 0; rank < nRanks_; ++rank) {
    if (rank == myRank_) {
      // Self transport for local copies
      // Use brace initialization to avoid most vexing parse
      P2pSelfTransportDevice selfDevice{};
      Transport hostTransport{selfDevice};
      CUDACHECK(cudaMemcpy(
          &transports_d_[rank],
          &hostTransport,
          sizeof(Transport),
          cudaMemcpyHostToDevice));
    } else {
      // P2P NVL transport for peer communication
      Transport hostTransport{buildP2pTransportDevice(rank)};
      CUDACHECK(cudaMemcpy(
          &transports_d_[rank],
          &hostTransport,
          sizeof(Transport),
          cudaMemcpyHostToDevice));
    }
  }
}

P2pNvlTransportDevice* MultiPeerNvlTransport::getP2pTransportDevice(
    int peerRank) {
  // Return pointer to the P2pNvlTransportDevice within the Transport union
  if (peerRank == myRank_) {
    throw std::runtime_error("Cannot get P2P transport for self rank");
  }
  return &transports_d_[peerRank].p2p_nvl;
}

Transport* MultiPeerNvlTransport::getTransportsArray() {
  return transports_d_;
}

P2pNvlTransportDevice MultiPeerNvlTransport::buildP2pTransportDevice(
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
  const std::size_t localChunkStateBufferOffset =
      localPeerIndex * perPeerChunkStateBufferSize_;
  const std::size_t localSignalBufferOffset =
      localPeerIndex * perPeerSignalBufferSize_;

  // Calculate remote peer index for buffer offset in peer's buffer
  // From peer's perspective, where does myRank fit in their buffer?
  const int remotePeerIndex = (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
  const std::size_t remoteDataBufferOffset =
      remotePeerIndex * perPeerDataBufferSize_;
  const std::size_t remoteChunkStateBufferOffset =
      remotePeerIndex * perPeerChunkStateBufferSize_;
  const std::size_t remoteSignalBufferOffset =
      remotePeerIndex * perPeerSignalBufferSize_;

  P2pNvlTransportOptions options{
      .dataBufferSize = config_.dataBufferSize,
      .chunkSize = config_.chunkSize,
      .pipelineDepth = config_.pipelineDepth,
      .useDualStateBuffer = config_.useDualStateBuffer};

  // Calculate number of chunk states per pipeline slot
  const std::size_t numChunksPerStep =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const auto numChunksPerPeer =
      static_cast<uint32_t>(config_.pipelineDepth * numChunksPerStep);

  auto* localSignalPtr =
      static_cast<char*>(signalBufferHandler_->getLocalDeviceMemPtr());
  auto* localDataPtr =
      static_cast<char*>(dataBufferHandler_->getLocalDeviceMemPtr());
  auto* localStatePtr =
      static_cast<char*>(stateBufferHandler_->getLocalDeviceMemPtr());

  auto* localChunkStateBase = reinterpret_cast<ChunkState*>(
      localStatePtr + localChunkStateBufferOffset);

  auto* remoteDataPtr =
      static_cast<char*>(dataBufferHandler_->getPeerDeviceMemPtr(peerRank));
  auto* remoteChunkStatePtr =
      static_cast<char*>(stateBufferHandler_->getPeerDeviceMemPtr(peerRank));
  auto* remoteSignalPtr =
      static_cast<char*>(signalBufferHandler_->getPeerDeviceMemPtr(peerRank));

  auto* remoteChunkStateBase = reinterpret_cast<ChunkState*>(
      remoteChunkStatePtr + remoteChunkStateBufferOffset);

  // Create LocalState and RemoteState based on useDualStateBuffer mode
  // Note: Using direct initialization since DeviceSpan has const members
  // that prevent copy-assignment
  if (config_.useDualStateBuffer) {
    // Dual state mode: 2x chunk states per peer
    //   Local buffer layout:
    //     [0, numChunksPerPeer): my chunk state
    //     [numChunksPerPeer, 2*numChunksPerPeer): peer's chunk state (stored
    //     locally)
    //   Remote buffer layout (on peer's memory via NVLink):
    //     [0, numChunksPerPeer): peer's chunk state
    //     [numChunksPerPeer, 2*numChunksPerPeer): my chunk state (stored on
    //     peer)
    LocalState localState{
        .dataBuffer = localDataPtr + localDataBufferOffset,
        .myChunkStateBuffer =
            DeviceSpan<ChunkState>(localChunkStateBase, numChunksPerPeer),
        .peerChunkStateBuffer = DeviceSpan<ChunkState>(
            localChunkStateBase + numChunksPerPeer, numChunksPerPeer),
        .signalBuffer = DeviceSpan<SignalState>(
            reinterpret_cast<SignalState*>(
                localSignalPtr + localSignalBufferOffset),
            config_.signalCount),
    };

    RemoteState remoteState{
        .dataBuffer = remoteDataPtr + remoteDataBufferOffset,
        .peerChunkStateBuffer =
            DeviceSpan<ChunkState>(remoteChunkStateBase, numChunksPerPeer),
        .myChunkStateBuffer = DeviceSpan<ChunkState>(
            remoteChunkStateBase + numChunksPerPeer, numChunksPerPeer),
        .signalBuffer = DeviceSpan<SignalState>(
            reinterpret_cast<SignalState*>(
                remoteSignalPtr + remoteSignalBufferOffset),
            config_.signalCount),
    };

    return P2pNvlTransportDevice(
        myRank_, peerRank, options, localState, remoteState);
  } else {
    // Single state mode: 1x chunk state per peer (on receiver side only)
    //   Local buffer: only myChunkStateBuffer is used (receiver waits here)
    //   Remote buffer: only peerChunkStateBuffer is used (points to peer's
    //   myChunkStateBuffer)
    LocalState localState{
        .dataBuffer = localDataPtr + localDataBufferOffset,
        .myChunkStateBuffer =
            DeviceSpan<ChunkState>(localChunkStateBase, numChunksPerPeer),
        .peerChunkStateBuffer = DeviceSpan<ChunkState>(), // Not used
        .signalBuffer = DeviceSpan<SignalState>(
            reinterpret_cast<SignalState*>(
                localSignalPtr + localSignalBufferOffset),
            config_.signalCount),
    };

    RemoteState remoteState{
        .dataBuffer = remoteDataPtr + remoteDataBufferOffset,
        .peerChunkStateBuffer =
            DeviceSpan<ChunkState>(remoteChunkStateBase, numChunksPerPeer),
        .myChunkStateBuffer = DeviceSpan<ChunkState>(), // Not used
        .signalBuffer = DeviceSpan<SignalState>(
            reinterpret_cast<SignalState*>(
                remoteSignalPtr + remoteSignalBufferOffset),
            config_.signalCount),
    };

    return P2pNvlTransportDevice(
        myRank_, peerRank, options, localState, remoteState);
  }
}

} // namespace comms::pipes
