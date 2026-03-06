// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiPeerNvlTransport.h"

#include <cuda_runtime.h>
#include <vector>

#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/window/DeviceWindowMemory.cuh"
#include "comms/pipes/window/WindowMemory.h"
#include "comms/utils/checks.h"

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
  // Allocation order: signal -> data -> state
  // Each handler's destructor will free its GPU memory if constructed.

  // Calculate per-peer buffer sizes with pipelining
  perPeerDataBufferSize_ = config_.pipelineDepth * config_.dataBufferSize;

  // Calculate state buffer size based on chunk size and pipeline depth
  const std::size_t numChunksPerStep =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const std::size_t numChunksPerPeer = config_.pipelineDepth * numChunksPerStep;
  perPeerChunkStateBufferSize_ = numChunksPerPeer * sizeof(ChunkState);
  perPeerSignalBufferSize_ = getSignalBufferSize(config_.p2pSignalCount);

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

  // Allocate device memory for Transport array (nRanks elements)
  // This array is populated in exchange() after peer buffer pointers are
  // available. Uses DeviceBuffer instead of raw cudaMalloc for RAII.
  transportsDevice_ =
      std::make_unique<meta::comms::DeviceBuffer>(nRanks_ * sizeof(Transport));

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
      config_.p2pSignalCount * (nRanks_ - 1));
  CUDA_CHECK(cudaMemcpy(
      signalPtr,
      signalInitStates.data(),
      totalSignalBufferSize,
      cudaMemcpyDefault));
}

void MultiPeerNvlTransport::exchange() {
  // Exchange P2P transport buffer pointers
  dataBufferHandler_->exchangeMemPtrs();
  stateBufferHandler_->exchangeMemPtrs();
  signalBufferHandler_->exchangeMemPtrs();

  // Build Transport array with P2pNvlTransportDevice stored by value
  auto* transports_d = static_cast<Transport*>(transportsDevice_->get());
  for (int rank = 0; rank < nRanks_; ++rank) {
    if (rank == myRank_) {
      // Self transport for local copies
      P2pSelfTransportDevice selfDevice{};
      Transport hostTransport{selfDevice};
      CUDACHECK(cudaMemcpy(
          &transports_d[rank],
          &hostTransport,
          sizeof(Transport),
          cudaMemcpyHostToDevice));
    } else {
      // P2P NVL transport - store by value in Transport
      P2pNvlTransportDevice nvlDevice = buildP2pTransportDevice(rank);
      Transport hostTransport{nvlDevice};
      CUDACHECK(cudaMemcpy(
          &transports_d[rank],
          &hostTransport,
          sizeof(Transport),
          cudaMemcpyHostToDevice));
    }
  }
}

P2pNvlTransportDevice* MultiPeerNvlTransport::getP2pTransportDevice(
    int peerRank) {
  // Return pointer to the P2pNvlTransportDevice stored in the Transport union
  if (peerRank == myRank_) {
    throw std::runtime_error("Cannot get P2P transport for self rank");
  }
  auto* transports_d = static_cast<Transport*>(transportsDevice_->get());
  return &transports_d[peerRank].p2p_nvl;
}

Transport* MultiPeerNvlTransport::getTransportsArray() {
  return static_cast<Transport*>(transportsDevice_->get());
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
          config_.p2pSignalCount),
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
          config_.p2pSignalCount),
  };

  return P2pNvlTransportDevice(
      myRank_, peerRank, options, localState, remoteState);
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

MultiPeerDeviceTransport MultiPeerNvlTransport::getMultiPeerDeviceTransport(
    const WindowMemory& wm) {
  DeviceWindowMemory dwm = wm.getDeviceWindowMemory();
  return MultiPeerDeviceTransport(myRank_, nRanks_, getDeviceTransports(), dwm);
}

void MultiPeerNvlTransport::initializeTransportsArray() {
  // NOTE: This function is for legacy lazy initialization via
  // getDeviceTransports(). The main initialization path is in exchange() which
  // populates transportsDevice_.
  //
  // If transportsDevice_ is already allocated (from constructor), this function
  // just returns as the arrays were already set up in exchange().
  if (transportsDevice_) {
    return;
  }

  // Fallback: allocate and initialize if not already done
  transportsDevice_ =
      std::make_unique<meta::comms::DeviceBuffer>(nRanks_ * sizeof(Transport));

  // Build Transport array with P2pNvlTransportDevice stored by value
  auto* transports_d = static_cast<Transport*>(transportsDevice_->get());
  for (int rank = 0; rank < nRanks_; ++rank) {
    if (rank == myRank_) {
      P2pSelfTransportDevice selfDevice{};
      Transport hostTransport{selfDevice};
      CUDACHECK(cudaMemcpy(
          &transports_d[rank],
          &hostTransport,
          sizeof(Transport),
          cudaMemcpyHostToDevice));
    } else {
      P2pNvlTransportDevice nvlDevice = buildP2pTransportDevice(rank);
      Transport hostTransport{nvlDevice};
      CUDACHECK(cudaMemcpy(
          &transports_d[rank],
          &hostTransport,
          sizeof(Transport),
          cudaMemcpyHostToDevice));
    }
  }
}

} // namespace comms::pipes
