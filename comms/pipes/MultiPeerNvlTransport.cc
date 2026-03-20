// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiPeerNvlTransport.h"

#include <vector>

#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/utils/checks.h"

namespace comms::pipes {

MultiPeerNvlTransport::MultiPeerNvlTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
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
  // Data buffers are NOT allocated here — they are either:
  // - Allocated internally in exchange() (default)
  // - Provided externally via setExternalDataBuffers() before exchange()
  //
  // Allocation order: signal -> state
  // Each handler's destructor will free its GPU memory if constructed.

  // Calculate per-peer buffer sizes with pipelining
  perPeerDataBufferSize_ = config_.pipelineDepth * config_.dataBufferSize;

  perPeerSignalBufferSize_ = getSignalBufferSize(config_.p2pSignalCount);

  // Allocate signal buffer (always needed)
  const std::size_t totalSignalBufferSize =
      perPeerSignalBufferSize_ * (nRanks_ - 1);

  signalBufferHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, totalSignalBufferSize, memSharingMode_);

  // Staging data + state buffers are only needed for send()/recv().
  // When dataBufferSize=0, skip allocation — put() works without staging.
  if (config_.dataBufferSize > 0) {
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
    const std::size_t numChunksPerPeer =
        config_.pipelineDepth * numChunksPerStep;
    const std::size_t chunkStateMultiplier = config_.useDualStateBuffer ? 2 : 1;
    perPeerChunkStateBufferSize_ =
        chunkStateMultiplier * numChunksPerPeer * sizeof(ChunkState);

    // Allocate state buffer for (nRanks - 1) peers using GpuMemHandler.
    // Data buffer allocation is deferred to exchange() to allow
    // setExternalDataBuffers() to be called first.
    const std::size_t totalChunkStateBufferSize =
        perPeerChunkStateBufferSize_ * (nRanks_ - 1);

    stateBufferHandler_ = std::make_unique<GpuMemHandler>(
        bootstrap_,
        myRank_,
        nRanks_,
        totalChunkStateBufferSize,
        memSharingMode_);

    // Initialize state buffer to READY_TO_SEND for all pipeline slots
    // The number of states depends on useDualStateBuffer:
    // - Single mode: 1x chunk states (only receiverStateBuffer)
    // - Dual mode: 2x chunk states (receiverStateBuffer + senderStateBuffer)
    auto statePtr =
        static_cast<ChunkState*>(stateBufferHandler_->getLocalDeviceMemPtr());
    const std::size_t totalNumChunksAllPeers =
        chunkStateMultiplier * numChunksPerPeer * (nRanks_ - 1);
    std::vector<ChunkState> initStates(totalNumChunksAllPeers);
    CUDA_CHECK(cudaMemcpy(
        statePtr,
        initStates.data(),
        totalChunkStateBufferSize,
        cudaMemcpyDefault));
  }

  // Initialize signal state buffer to 0 for all ranks
  auto signalPtr =
      static_cast<SignalState*>(signalBufferHandler_->getLocalDeviceMemPtr());
  std::vector<SignalState> signalInitStates(
      config_.p2pSignalCount * (nRanks_ - 1));
  CUDA_CHECK(cudaMemcpy(
      signalPtr,
      signalInitStates.data(),
      totalSignalBufferSize,
      cudaMemcpyDefault));

  // Conditionally allocate LL128 buffers
  if (config_.ll128BufferSize > 0) {
    perPeerLl128BufferSize_ = config_.ll128BufferSize;
    std::size_t totalLl128Size = perPeerLl128BufferSize_ * (nRanks_ - 1);
    ll128BufferHandler_ = std::make_unique<GpuMemHandler>(
        bootstrap_, myRank_, nRanks_, totalLl128Size, memSharingMode_);

    // Initialize all LL128 packet flags to kLl128ReadyToWrite (-1).
    // cudaMemset(0xFF) sets all bytes to 0xFF = -1 in two's complement.
    // Data payload bytes are also 0xFF but get overwritten before first read.
    auto* ll128Ptr = ll128BufferHandler_->getLocalDeviceMemPtr();
    CUDA_CHECK(cudaMemset(ll128Ptr, kLl128MemsetInitByte, totalLl128Size));
    CUDA_CHECK(
        cudaDeviceSynchronize()); // Ensure init completes before exchange
  }
}

void MultiPeerNvlTransport::setExternalDataBuffers(
    ExternalStagingBuffers externalStagingBuffers) {
  // Validate that the vectors are large enough to index by rank.
  if (static_cast<int>(externalStagingBuffers.localBuffers.size()) < nRanks_ ||
      static_cast<int>(externalStagingBuffers.remoteBuffers.size()) < nRanks_) {
    throw std::runtime_error(
        "setExternalDataBuffers: localBuffers.size()=" +
        std::to_string(externalStagingBuffers.localBuffers.size()) +
        " and remoteBuffers.size()=" +
        std::to_string(externalStagingBuffers.remoteBuffers.size()) +
        " must both be >= nRanks=" + std::to_string(nRanks_));
  }

  // Validate that every non-self peer buffer meets the minimum size.
  for (int peer = 0; peer < nRanks_; ++peer) {
    if (peer == myRank_) {
      continue;
    }
    auto localSize = externalStagingBuffers.localBuffers[peer].size();
    auto remoteSize = externalStagingBuffers.remoteBuffers[peer].size();
    if (localSize < perPeerDataBufferSize_) {
      throw std::runtime_error(
          "setExternalDataBuffers: local buffer for peer " +
          std::to_string(peer) + " has size " + std::to_string(localSize) +
          " but requires at least " + std::to_string(perPeerDataBufferSize_) +
          " (pipelineDepth * dataBufferSize)");
    }
    if (remoteSize < perPeerDataBufferSize_) {
      throw std::runtime_error(
          "setExternalDataBuffers: remote buffer for peer " +
          std::to_string(peer) + " has size " + std::to_string(remoteSize) +
          " but requires at least " + std::to_string(perPeerDataBufferSize_) +
          " (pipelineDepth * dataBufferSize)");
    }
  }
  externalStagingBuffers_ = std::move(externalStagingBuffers);
}

void MultiPeerNvlTransport::exchange() {
  // Allocate and exchange data buffers when:
  // - No external buffers were provided, AND
  // - dataBufferSize > 0 (staging buffers needed for send/recv)
  if (!externalStagingBuffers_ && config_.dataBufferSize > 0) {
    const std::size_t totalDataBufferSize =
        perPeerDataBufferSize_ * (nRanks_ - 1);
    dataBufferHandler_ = std::make_unique<GpuMemHandler>(
        bootstrap_, myRank_, nRanks_, totalDataBufferSize, memSharingMode_);
  }

  // Exchange buffer pointers across all ranks
  if (dataBufferHandler_) {
    dataBufferHandler_->exchangeMemPtrs();
  }
  if (stateBufferHandler_) {
    stateBufferHandler_->exchangeMemPtrs();
  }
  signalBufferHandler_->exchangeMemPtrs();

  if (ll128BufferHandler_) {
    ll128BufferHandler_->exchangeMemPtrs();
  }
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
      .pipelineDepth = config_.pipelineDepth,
      .useDualStateBuffer = config_.useDualStateBuffer,
      .ll128BufferNumPackets = perPeerLl128BufferSize_ / kLl128PacketSize,
  };

  auto* localSignalPtr =
      static_cast<char*>(signalBufferHandler_->getLocalDeviceMemPtr());
  auto* remoteSignalPtr =
      static_cast<char*>(signalBufferHandler_->getPeerDeviceMemPtr(peerRank));

  DeviceSpan<SignalState> localSignalSpan(
      reinterpret_cast<SignalState*>(localSignalPtr + localSignalBufferOffset),
      config_.p2pSignalCount);
  DeviceSpan<SignalState> remoteSignalSpan(
      reinterpret_cast<SignalState*>(
          remoteSignalPtr + remoteSignalBufferOffset),
      config_.p2pSignalCount);

  // When dataBufferSize=0, staging buffers are not allocated.
  // Set data/state to nullptr/empty — send()/recv() will trap.
  if (!dataBufferHandler_ && !externalStagingBuffers_) {
    LocalState localState{
        .dataBuffer = nullptr,
        .receiverStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
        .senderStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
        .signalBuffer = localSignalSpan,
    };
    RemoteState remoteState{
        .dataBuffer = nullptr,
        .receiverStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
        .senderStateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
        .signalBuffer = remoteSignalSpan,
    };
    return P2pNvlTransportDevice(
        myRank_, peerRank, options, localState, remoteState);
  }

  const std::size_t numChunksPerStep =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const auto numChunksPerPeer =
      static_cast<uint32_t>(config_.pipelineDepth * numChunksPerStep);

  auto* localStatePtr =
      static_cast<char*>(stateBufferHandler_->getLocalDeviceMemPtr());

  // Data buffer pointers: use external buffers if provided, otherwise
  // use internally allocated dataBufferHandler_.
  char* localDataBuffer = nullptr;
  char* remoteDataBuffer = nullptr;
  if (externalStagingBuffers_) {
    localDataBuffer = externalStagingBuffers_->localBuffers[peerRank].data();
    remoteDataBuffer = externalStagingBuffers_->remoteBuffers[peerRank].data();
  } else {
    localDataBuffer =
        static_cast<char*>(dataBufferHandler_->getLocalDeviceMemPtr()) +
        localDataBufferOffset;
    remoteDataBuffer =
        static_cast<char*>(dataBufferHandler_->getPeerDeviceMemPtr(peerRank)) +
        remoteDataBufferOffset;
  }

  auto* localChunkStateBase = reinterpret_cast<ChunkState*>(
      localStatePtr + localChunkStateBufferOffset);

  auto* remoteChunkStatePtr =
      static_cast<char*>(stateBufferHandler_->getPeerDeviceMemPtr(peerRank));

  // Compute LL128 buffer pointers (nullptr when LL128 is disabled)
  Ll128Packet* localLl128 = nullptr;
  Ll128Packet* remoteLl128 = nullptr;
  if (ll128BufferHandler_) {
    auto* localLl128Ptr =
        static_cast<char*>(ll128BufferHandler_->getLocalDeviceMemPtr());
    localLl128 = reinterpret_cast<Ll128Packet*>(
        localLl128Ptr + localPeerIndex * perPeerLl128BufferSize_);
    auto* remoteLl128Ptr =
        static_cast<char*>(ll128BufferHandler_->getPeerDeviceMemPtr(peerRank));
    remoteLl128 = reinterpret_cast<Ll128Packet*>(
        remoteLl128Ptr + remotePeerIndex * perPeerLl128BufferSize_);
  }

  auto* remoteChunkStateBase = reinterpret_cast<ChunkState*>(
      remoteChunkStatePtr + remoteChunkStateBufferOffset);

  // Create LocalState and RemoteState based on useDualStateBuffer mode
  // Note: Using direct initialization since DeviceSpan has const members
  // that prevent copy-assignment
  if (config_.useDualStateBuffer) {
    // Dual state mode: 2x chunk states per peer
    //   Local buffer layout:
    //     [0, numChunksPerPeer): receiver state buffer (state to poll if
    //     I am a receiver)
    //     [numChunksPerPeer, 2*numChunksPerPeer): sender state buffer
    //     (state to poll if I am a sender)
    //   Remote buffer layout (on peer's memory via NVLink):
    //     [0, numChunksPerPeer): peer's receiver state buffer (I write to
    //     signal data ready)
    //     [numChunksPerPeer, 2*numChunksPerPeer): peer's sender state buffer
    //     (I write READY_TO_SEND after reading)
    LocalState localState{
        .dataBuffer = localDataBuffer,
        .receiverStateBuffer =
            DeviceSpan<ChunkState>(localChunkStateBase, numChunksPerPeer),
        .senderStateBuffer = DeviceSpan<ChunkState>(
            localChunkStateBase + numChunksPerPeer, numChunksPerPeer),
        .signalBuffer = localSignalSpan,
        .ll128Buffer = localLl128,
    };

    RemoteState remoteState{
        .dataBuffer = remoteDataBuffer,
        .receiverStateBuffer =
            DeviceSpan<ChunkState>(remoteChunkStateBase, numChunksPerPeer),
        .senderStateBuffer = DeviceSpan<ChunkState>(
            remoteChunkStateBase + numChunksPerPeer, numChunksPerPeer),
        .signalBuffer = remoteSignalSpan,
        .ll128Buffer = remoteLl128,
    };

    return P2pNvlTransportDevice(
        myRank_, peerRank, options, localState, remoteState);
  } else {
    // Single state mode: 1x chunk state per peer (on receiver side only)
    //   Local buffer: only receiverStateBuffer is used (receiver waits here)
    //   Remote buffer: only receiverStateBuffer is used (points to peer's
    //   receiverStateBuffer, sender writes to signal data ready)
    LocalState localState{
        .dataBuffer = localDataBuffer,
        .receiverStateBuffer =
            DeviceSpan<ChunkState>(localChunkStateBase, numChunksPerPeer),
        .senderStateBuffer = DeviceSpan<ChunkState>(), // Not used
        .signalBuffer = localSignalSpan,
        .ll128Buffer = localLl128,
    };

    RemoteState remoteState{
        .dataBuffer = remoteDataBuffer,
        .receiverStateBuffer =
            DeviceSpan<ChunkState>(remoteChunkStateBase, numChunksPerPeer),
        .senderStateBuffer = DeviceSpan<ChunkState>(), // Not used
        .signalBuffer = remoteSignalSpan,
        .ll128Buffer = remoteLl128,
    };

    return P2pNvlTransportDevice(
        myRank_, peerRank, options, localState, remoteState);
  }
}

DeviceSpan<Transport> MultiPeerNvlTransport::getDeviceTransports() {
  // Thread-safe lazy initialization of device-accessible arrays
  if (!multiPeerInitialized_) {
    initializeTransportsArray();
    multiPeerInitialized_ = true;
  }

  return DeviceSpan<Transport>(
      static_cast<Transport*>(transportsDevice_->get()), nRanks_);
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

P2pNvlTransportDevice MultiPeerNvlTransport::buildP2pTransportDevice(
    int peerRank) {
  return getP2pTransportDevice(peerRank);
}

} // namespace comms::pipes
