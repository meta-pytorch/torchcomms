// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/nvl/MultiPeerNvlTransport.h"

#include <stdexcept>
#include <vector>

#include "comms/prims/transport/Transport.cuh"
#include "comms/prims/transport/ll/LlPacket.cuh"
#include "comms/prims/transport/ll128/Ll128Packet.cuh"
#include "comms/prims/transport/nvl/NvlChannelState.cuh"
#include "comms/prims/transport/self/P2pSelfTransportDevice.cuh"
#include "comms/utils/checks.h"

namespace comms::prims {

namespace {

MultiPeerNvlTransportConfig normalizeChannelConfig(
    MultiPeerNvlTransportConfig config) {
  if (config.maxNumChannels <= 0) {
    config.perChannelSize = 0;
    return config;
  }

  const auto maxChannels = static_cast<std::size_t>(config.maxNumChannels);
  if (config.perChannelSize == 0) {
    if (config.dataBufferSize % maxChannels != 0) {
      throw std::runtime_error(
          "tile send/recv requires dataBufferSize to divide evenly across maxNumChannels");
    }
    config.perChannelSize = config.dataBufferSize / maxChannels;
  } else if (
      config.dataBufferSize != 0 &&
      config.dataBufferSize != config.perChannelSize * maxChannels) {
    throw std::runtime_error(
        "tile send/recv requires dataBufferSize == perChannelSize * maxNumChannels");
  }
  if (config.perChannelSize < 16) {
    throw std::runtime_error("tile send/recv requires perChannelSize >= 16");
  }
  if (config.perChannelSize % 16 != 0) {
    throw std::runtime_error(
        "tile send/recv requires perChannelSize to be 16-byte aligned");
  }
  // Cap per-channel staging at 128MB (matches NCCL's max channel buffer). This
  // also subsumes the dataBufferSize overflow guard: 128MB * maxNumChannels
  // stays well within std::size_t for any sane channel count.
  constexpr std::size_t kMaxPerChannelSize = 128ULL * 1024 * 1024;
  if (config.perChannelSize > kMaxPerChannelSize) {
    throw std::runtime_error("tile send/recv requires perChannelSize <= 128MB");
  }
  config.dataBufferSize = config.perChannelSize * maxChannels;
  return config;
}

} // namespace

MultiPeerNvlTransport::MultiPeerNvlTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultiPeerNvlTransportConfig& multiPeerNvlTransportConfig)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(normalizeChannelConfig(multiPeerNvlTransportConfig)),
      memSharingMode_(
          config_.memSharingMode.value_or(GpuMemHandler::detectBestMode())) {
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

  // Conditionally allocate per-channel state for the tile protocol.
  // One NvlChannelState per channel per peer; the whole array is IPC-shared
  // so the remote rank's send/recv can write data_ready / slot_free into our
  // local endpoint.
  if (config_.maxNumChannels > 0) {
    if (config_.pipelineDepth < 1) {
      throw std::runtime_error("tile send/recv requires pipelineDepth >= 1");
    }
    perPeerChannelStateSize_ = config_.maxNumChannels * sizeof(NvlChannelState);
    const std::size_t totalChannelStateSize =
        perPeerChannelStateSize_ * (nRanks_ - 1);
    channelStateHandler_ = std::make_unique<GpuMemHandler>(
        bootstrap_, myRank_, nRanks_, totalChannelStateSize, memSharingMode_);
    auto* channelStatePtr = channelStateHandler_->getLocalDeviceMemPtr();
    CUDA_CHECK(cudaMemset(channelStatePtr, 0, totalChannelStateSize));
  }

  // Conditionally allocate barrier buffer
  if (config_.p2pBarrierCount > 0) {
    perPeerBarrierBufferSize_ = getBarrierBufferSize(config_.p2pBarrierCount);
    std::size_t totalBarrierSize = perPeerBarrierBufferSize_ * (nRanks_ - 1);
    barrierBufferHandler_ = std::make_unique<GpuMemHandler>(
        bootstrap_, myRank_, nRanks_, totalBarrierSize, memSharingMode_);

    // Zero-initialize barrier counters
    auto* barrierPtr = barrierBufferHandler_->getLocalDeviceMemPtr();
    CUDA_CHECK(cudaMemset(barrierPtr, 0, totalBarrierSize));
  }

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
  }

  // Conditionally allocate LL buffers
  if (config_.llBufferSize > 0) {
    perPeerLlBufferSize_ = config_.llBufferSize;
    std::size_t totalLlSize = perPeerLlBufferSize_ * (nRanks_ - 1);
    llBufferHandler_ = std::make_unique<GpuMemHandler>(
        bootstrap_, myRank_, nRanks_, totalLlSize, memSharingMode_);

    // Initialize all LL line flags to kLlReadyToWrite (0xFFFFFFFF).
    auto* llPtr = llBufferHandler_->getLocalDeviceMemPtr();
    CUDA_CHECK(cudaMemset(llPtr, kLlMemsetInitByte, totalLlSize));
  }

  // Ensure all buffer initialization completes before exchange
  if (config_.ll128BufferSize > 0 || config_.llBufferSize > 0) {
    CUDA_CHECK(cudaDeviceSynchronize());
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
  signalBufferHandler_->exchangeMemPtrs();

  if (ll128BufferHandler_) {
    ll128BufferHandler_->exchangeMemPtrs();
  }
  if (barrierBufferHandler_) {
    barrierBufferHandler_->exchangeMemPtrs();
  }
  if (channelStateHandler_) {
    channelStateHandler_->exchangeMemPtrs();
  }

  if (llBufferHandler_) {
    llBufferHandler_->exchangeMemPtrs();
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
  const std::size_t localSignalBufferOffset =
      localPeerIndex * perPeerSignalBufferSize_;

  // Calculate remote peer index for buffer offset in peer's buffer
  // From peer's perspective, where does myRank fit in their buffer?
  const int remotePeerIndex = (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
  const std::size_t remoteDataBufferOffset =
      remotePeerIndex * perPeerDataBufferSize_;
  const std::size_t remoteSignalBufferOffset =
      remotePeerIndex * perPeerSignalBufferSize_;

  const int maxChannels = config_.maxNumChannels;
  const std::size_t perChannelSlot =
      maxChannels > 0 ? config_.perChannelSize : 0;
  P2pNvlTransportOptions options{
      .dataBufferSize = config_.dataBufferSize,
      .chunkSize = config_.chunkSize,
      .pipelineDepth = config_.pipelineDepth,
      .ll128BufferNumPackets = perPeerLl128BufferSize_ / kLl128PacketSize,
      .llBufferNumLines = perPeerLlBufferSize_ / kLlLineSize,
      .per_channel_slot = perChannelSlot,
      .max_num_channels = maxChannels,
  };

  // Per-peer NvlChannelState pointers (nullptr when tile path is disabled).
  // local_channels: this rank's channel endpoint for the remote rank.
  // remote_channels: the remote rank's channel endpoint, seen via IPC.
  NvlChannelState* localChannels = nullptr;
  NvlChannelState* remoteChannels = nullptr;
  if (channelStateHandler_) {
    auto* localChBase =
        static_cast<char*>(channelStateHandler_->getLocalDeviceMemPtr());
    auto* remoteChBase =
        static_cast<char*>(channelStateHandler_->getPeerDeviceMemPtr(peerRank));
    localChannels = reinterpret_cast<NvlChannelState*>(
        localChBase + localPeerIndex * perPeerChannelStateSize_);
    remoteChannels = reinterpret_cast<NvlChannelState*>(
        remoteChBase + remotePeerIndex * perPeerChannelStateSize_);
  }

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

  // Barrier buffer spans (empty if p2pBarrierCount == 0)
  auto makeBarrierSpans =
      [&]() -> std::pair<DeviceSpan<BarrierState>, DeviceSpan<BarrierState>> {
    if (!barrierBufferHandler_) {
      return {DeviceSpan<BarrierState>(), DeviceSpan<BarrierState>()};
    }
    auto* localBarrierPtr =
        static_cast<char*>(barrierBufferHandler_->getLocalDeviceMemPtr());
    auto* remoteBarrierPtr = static_cast<char*>(
        barrierBufferHandler_->getPeerDeviceMemPtr(peerRank));
    const std::size_t localBarrierOffset =
        localPeerIndex * perPeerBarrierBufferSize_;
    const std::size_t remoteBarrierOffset =
        remotePeerIndex * perPeerBarrierBufferSize_;
    return {
        DeviceSpan<BarrierState>(
            reinterpret_cast<BarrierState*>(
                localBarrierPtr + localBarrierOffset),
            config_.p2pBarrierCount),
        DeviceSpan<BarrierState>(
            reinterpret_cast<BarrierState*>(
                remoteBarrierPtr + remoteBarrierOffset),
            config_.p2pBarrierCount)};
  };
  auto [localBarrierSpan, remoteBarrierSpan] = makeBarrierSpans();

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

  // Compute LL buffer pointers (nullptr when LL is disabled)
  LlLine* localLl = nullptr;
  LlLine* remoteLl = nullptr;
  if (llBufferHandler_) {
    auto* localLlPtr =
        static_cast<char*>(llBufferHandler_->getLocalDeviceMemPtr());
    localLl = reinterpret_cast<LlLine*>(
        localLlPtr + localPeerIndex * perPeerLlBufferSize_);
    auto* remoteLlPtr =
        static_cast<char*>(llBufferHandler_->getPeerDeviceMemPtr(peerRank));
    remoteLl = reinterpret_cast<LlLine*>(
        remoteLlPtr + remotePeerIndex * perPeerLlBufferSize_);
  }

  // Data buffer pointers: use external buffers if provided, otherwise
  // use internally allocated dataBufferHandler_. When neither is available
  // (dataBufferSize=0), pass nullptr — tile send()/recv() will trap.
  char* localDataBuffer = nullptr;
  char* remoteDataBuffer = nullptr;
  if (externalStagingBuffers_) {
    localDataBuffer = externalStagingBuffers_->localBuffers[peerRank].data();
    remoteDataBuffer = externalStagingBuffers_->remoteBuffers[peerRank].data();
  } else if (dataBufferHandler_) {
    localDataBuffer =
        static_cast<char*>(dataBufferHandler_->getLocalDeviceMemPtr()) +
        localDataBufferOffset;
    remoteDataBuffer =
        static_cast<char*>(dataBufferHandler_->getPeerDeviceMemPtr(peerRank)) +
        remoteDataBufferOffset;
  }

  LocalState localState{
      .dataBuffer = localDataBuffer,
      .signalBuffer = localSignalSpan,
      .barrierBuffer = localBarrierSpan,
      .ll128Buffer = localLl128,
      .llBuffer = localLl,
  };

  RemoteState remoteState{
      .dataBuffer = remoteDataBuffer,
      .signalBuffer = remoteSignalSpan,
      .barrierBuffer = remoteBarrierSpan,
      .ll128Buffer = remoteLl128,
      .llBuffer = remoteLl,
  };

  return P2pNvlTransportDevice(
      myRank_,
      peerRank,
      options,
      localState,
      remoteState,
      localChannels,
      remoteChannels);
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

} // namespace comms::prims
