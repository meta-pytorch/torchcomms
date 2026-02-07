// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiPeerNvlTransport.h"

#include <vector>

#include "comms/pipes/DeviceSignal.cuh"
#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/utils/checks.h"

namespace comms::pipes {

MultiPeerNvlTransport::MultiPeerNvlTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
    const MultiPeerNvlTransportConfig& multiPeerNvlTransportConfig)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(multiPeerNvlTransportConfig),
      memSharingMode_(GpuMemHandler::detectBestMode()) {
  // ===========================================================================
  // Buffer Allocation
  // ===========================================================================
  //
  // Memory allocation uses RAII pattern via std::unique_ptr<GpuMemHandler>.
  // If any allocation or initialization fails and throws an exception:
  // - Previously allocated GpuMemHandler objects are automatically cleaned up
  //   when the exception propagates and unique_ptr destructors run
  // - No manual cleanup is needed in the constructor
  // - The destructor only needs to handle successfully constructed objects
  //
  // Allocation order: signal -> data -> state
  // Each handler's destructor will free its GPU memory if constructed.

  // Calculate per-peer buffer sizes with pipelining
  perPeerDataBufferSize_ = config_.pipelineDepth * config_.dataBufferSize;

  // Calculate state buffer size based on chunk size and pipeline depth
  const std::size_t numChunksPerStep =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const std::size_t numChunksPerPeer = config_.pipelineDepth * numChunksPerStep;
  perPeerChunkStateBufferSize_ = numChunksPerPeer * sizeof(ChunkState);
  perPeerSignalBufferSize_ = getSignalBufferSize(config_.signalCount);

  // Allocate buffers for (nRanks - 1) peers using GpuMemHandler
  const std::size_t totalDataBufferSize =
      perPeerDataBufferSize_ * (nRanks_ - 1);
  const std::size_t totalChunkStateBufferSize =
      perPeerChunkStateBufferSize_ * (nRanks_ - 1);
  const std::size_t totalSignalBufferSize =
      perPeerSignalBufferSize_ * (nRanks_ - 1);

  signalBufferHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, totalSignalBufferSize, memSharingMode_);

  dataBufferHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, totalDataBufferSize, memSharingMode_);

  stateBufferHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, totalChunkStateBufferSize, memSharingMode_);

  // Initialize state buffer to READY_TO_SEND for all pipeline slots
  auto statePtr =
      static_cast<ChunkState*>(stateBufferHandler_->getLocalDeviceMemPtr());
  const std::size_t totalNumChunksAllPeers = numChunksPerPeer * (nRanks_ - 1);
  std::vector<ChunkState> initStates(totalNumChunksAllPeers);
  CUDA_CHECK(cudaMemcpy(
      statePtr,
      initStates.data(),
      totalChunkStateBufferSize,
      cudaMemcpyDefault));

  // Initialize signal state buffer to 0 for all ranks
  auto signalPtr =
      static_cast<SignalState*>(signalBufferHandler_->getLocalDeviceMemPtr());
  std::vector<SignalState> signalInitStates(
      config_.signalCount * (nRanks_ - 1));
  CUDA_CHECK(cudaMemcpy(
      signalPtr,
      signalInitStates.data(),
      totalSignalBufferSize,
      cudaMemcpyDefault));

  // ===========================================================================
  // Multi-peer transport buffers (inbox model)
  // ===========================================================================
  WindowMemoryConfig windowConfig{
      .signalCount = config_.signalCount, .counterCount = config_.counterCount};
  windowMemory_ = std::make_unique<WindowMemory>(
      myRank_, nRanks_, bootstrap_, windowConfig, memSharingMode_);
  // Signal inbox: signalCount slots (all peers write to same slot)
}

void MultiPeerNvlTransport::exchange() {
  // Exchange P2P transport buffer pointers
  dataBufferHandler_->exchangeMemPtrs();
  stateBufferHandler_->exchangeMemPtrs();
  signalBufferHandler_->exchangeMemPtrs();

  // Exchange multi-peer transport buffer pointers (inbox model)
  windowMemory_->exchange();
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
      .pipelineDepth = config_.pipelineDepth};

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

  LocalState localState{
      .dataBuffer = localDataPtr + localDataBufferOffset,
      .stateBuffer = DeviceSpan<ChunkState>(
          reinterpret_cast<ChunkState*>(
              localStatePtr + localChunkStateBufferOffset),
          numChunksPerPeer),
      .signalBuffer = DeviceSpan<SignalState>(
          reinterpret_cast<SignalState*>(
              localSignalPtr + localSignalBufferOffset),
          config_.signalCount),
  };

  auto* remoteDataPtr =
      static_cast<char*>(dataBufferHandler_->getPeerDeviceMemPtr(peerRank));
  auto* remoteChunkStatePtr =
      static_cast<char*>(stateBufferHandler_->getPeerDeviceMemPtr(peerRank));
  auto* remoteSignalPtr =
      static_cast<char*>(signalBufferHandler_->getPeerDeviceMemPtr(peerRank));

  RemoteState remoteState{
      .dataBuffer = remoteDataPtr + remoteDataBufferOffset,
      .stateBuffer = DeviceSpan<ChunkState>(
          reinterpret_cast<ChunkState*>(
              remoteChunkStatePtr + remoteChunkStateBufferOffset),
          numChunksPerPeer),
      .signalBuffer = DeviceSpan<SignalState>(
          reinterpret_cast<SignalState*>(
              remoteSignalPtr + remoteSignalBufferOffset),
          config_.signalCount),
  };

  return P2pNvlTransportDevice(
      myRank_, peerRank, options, localState, remoteState);
}

MultiPeerDeviceTransport MultiPeerNvlTransport::getMultiPeerDeviceTransport() {
  // Thread-safe lazy initialization of device-accessible arrays
  if (!multiPeerInitialized_) {
    initializeTransportsArray();
    multiPeerInitialized_ = true;
  }

  DeviceSignal signal = windowMemory_->getDeviceSignal();
  DeviceCounter counter = windowMemory_->getDeviceCounter();

  // Build transports span directly from the device array (no pointer
  // indirection)
  DeviceSpan<Transport> transports(
      static_cast<Transport*>(transportsDevice_->get()), nRanks_);

  return MultiPeerDeviceTransport(
      myRank_, nRanks_, transports, signal, counter);
}

void MultiPeerNvlTransport::initializeTransportsArray() {
  // Allocate device memory for Transport objects using DeviceBuffer
  transportsDevice_ =
      std::make_unique<meta::comms::DeviceBuffer>(nRanks_ * sizeof(Transport));

  // Build host-side Transport array, then batch copy to device
  // Note: We use move semantics since Transport is non-copyable
  std::vector<Transport> hostTransports;
  hostTransports.reserve(nRanks_);
  for (int rank = 0; rank < nRanks_; ++rank) {
    if (rank == myRank_) {
      hostTransports.emplace_back(P2pSelfTransportDevice());
    } else {
      hostTransports.emplace_back(getP2pTransportDevice(rank));
    }
  }

  // Single batched memcpy for all Transport objects
  // This works because Transport is designed for byte-copy (see
  // Transport.cuh)
  CUDA_CHECK(cudaMemcpy(
      transportsDevice_->get(),
      hostTransports.data(),
      nRanks_ * sizeof(Transport),
      cudaMemcpyDefault));
}

} // namespace comms::pipes
