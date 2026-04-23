#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "MultiPeerNvlTransportAmd.h"
#include "comms/pipes/amd/HipMemHandler.h"

#include <hip/hip_runtime.h>
#include <vector>

#define HIP_CHECK(call)                                               \
  do {                                                                \
    hipError_t err = (call);                                          \
    if (err != hipSuccess) {                                          \
      throw std::runtime_error(                                       \
          std::string(#call) + " failed: " + hipGetErrorString(err)); \
    }                                                                 \
  } while (0)

namespace comms::pipes {

MultiPeerNvlTransportAmd::MultiPeerNvlTransportAmd(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultiPeerNvlTransportAmdConfig& config)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(config) {
  // Enable P2P access to all peer GPUs for direct XGMI transfers.
  // Without this, IPC memory access goes through system memory (slow).
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  int myDevice = 0;
  HIP_CHECK(hipGetDevice(&myDevice));
  for (int d = 0; d < deviceCount; d++) {
    if (d == myDevice)
      continue;
    int canAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, myDevice, d));
    if (canAccess) {
      hipError_t peerErr = hipDeviceEnablePeerAccess(d, 0);
      // Ignore if already enabled
      if (peerErr != hipSuccess &&
          peerErr != hipErrorPeerAccessAlreadyEnabled) {
        HIP_CHECK(peerErr);
      }
    }
  }
  // Clear any sticky "peer access already enabled" error left by
  // hipDeviceEnablePeerAccess, so it doesn't poison later
  // hipGetLastError() checks (e.g., PIPES_KERNEL_LAUNCH_CHECK).
  (void)hipGetLastError();

  perPeerDataBufferSize_ = config_.pipelineDepth * config_.dataBufferSize;
  perPeerSignalBufferSize_ = getSignalBufferSize(config_.p2pSignalCount);

  // Allocate signal buffer
  const std::size_t totalSignalBufferSize =
      perPeerSignalBufferSize_ * (nRanks_ - 1);
  signalBufferHandler_ = std::make_unique<HipMemHandler>(
      bootstrap_, myRank_, nRanks_, totalSignalBufferSize);

  // Staging data + state buffers (only when dataBufferSize > 0)
  if (config_.dataBufferSize > 0) {
    const std::size_t numChunksPerStep =
        (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
    const std::size_t numChunksPerPeer =
        config_.pipelineDepth * numChunksPerStep;
    const std::size_t chunkStateMultiplier = config_.useDualStateBuffer ? 2 : 1;
    perPeerChunkStateBufferSize_ =
        chunkStateMultiplier * numChunksPerPeer * sizeof(ChunkState);

    const std::size_t totalDataBufferSize =
        perPeerDataBufferSize_ * (nRanks_ - 1);
    const std::size_t totalChunkStateBufferSize =
        perPeerChunkStateBufferSize_ * (nRanks_ - 1);

    dataBufferHandler_ = std::make_unique<HipMemHandler>(
        bootstrap_, myRank_, nRanks_, totalDataBufferSize);
    stateBufferHandler_ = std::make_unique<HipMemHandler>(
        bootstrap_, myRank_, nRanks_, totalChunkStateBufferSize);

    // Initialize state buffer to READY_TO_SEND for all pipeline slots.
    // ChunkState default constructor sets value_ = READY_TO_SEND (-1).
    // Without this, HipMemHandler zeroes the buffer, leaving value_ = 0,
    // which causes deadlocks on multi-step transfers (stepId > 0).
    const std::size_t totalNumChunks =
        totalChunkStateBufferSize / sizeof(ChunkState);
    std::vector<ChunkState> initStates(totalNumChunks);
    auto* statePtr =
        static_cast<ChunkState*>(stateBufferHandler_->getLocalDeviceMemPtr());
    HIP_CHECK(hipMemcpy(
        statePtr,
        initStates.data(),
        totalChunkStateBufferSize,
        hipMemcpyHostToDevice));
  }

  // Initialize signal buffer to 0
  auto signalPtr =
      static_cast<SignalState*>(signalBufferHandler_->getLocalDeviceMemPtr());
  HIP_CHECK(hipMemset(signalPtr, 0, totalSignalBufferSize));
}

MultiPeerNvlTransportAmd::~MultiPeerNvlTransportAmd() {
  // Close any IPC handles not yet freed by freeDirectCopyPtrs
  for (void* ptr : ipcPeerPtrs_) {
    if (ptr) {
      hipIpcCloseMemHandle(ptr);
    }
  }
  ipcPeerPtrs_.clear();

  if (transportsDevice_) {
    hipFree(transportsDevice_);
    transportsDevice_ = nullptr;
  }
}

MultiPeerNvlTransportAmd::MultiPeerNvlTransportAmd(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultiPeerNvlTransportAmdConfig& config,
    const std::vector<int>& localPeerRanks)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(config) {
  // Enable P2P access to all local peer GPUs for direct XGMI transfers.
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  int myDevice = 0;
  HIP_CHECK(hipGetDevice(&myDevice));
  for (int d = 0; d < deviceCount; d++) {
    if (d == myDevice)
      continue;
    int canAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, myDevice, d));
    if (canAccess) {
      hipError_t peerErr = hipDeviceEnablePeerAccess(d, 0);
      if (peerErr != hipSuccess &&
          peerErr != hipErrorPeerAccessAlreadyEnabled) {
        HIP_CHECK(peerErr);
      }
    }
  }
  (void)hipGetLastError();

  // Convert localPeerRanks to int32_t for HipMemHandler
  std::vector<int32_t> localPeers32(
      localPeerRanks.begin(), localPeerRanks.end());

  perPeerDataBufferSize_ = config_.pipelineDepth * config_.dataBufferSize;
  perPeerSignalBufferSize_ = getSignalBufferSize(config_.p2pSignalCount);

  const std::size_t totalSignalBufferSize =
      perPeerSignalBufferSize_ * (nRanks_ - 1);
  signalBufferHandler_ = std::make_unique<HipMemHandler>(
      bootstrap_, myRank_, nRanks_, totalSignalBufferSize, localPeers32);

  if (config_.dataBufferSize > 0) {
    const std::size_t numChunksPerStep =
        (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
    const std::size_t numChunksPerPeer =
        config_.pipelineDepth * numChunksPerStep;
    const std::size_t chunkStateMultiplier = config_.useDualStateBuffer ? 2 : 1;
    perPeerChunkStateBufferSize_ =
        chunkStateMultiplier * numChunksPerPeer * sizeof(ChunkState);

    const std::size_t totalDataBufferSize =
        perPeerDataBufferSize_ * (nRanks_ - 1);
    const std::size_t totalChunkStateBufferSize =
        perPeerChunkStateBufferSize_ * (nRanks_ - 1);

    dataBufferHandler_ = std::make_unique<HipMemHandler>(
        bootstrap_, myRank_, nRanks_, totalDataBufferSize, localPeers32);
    stateBufferHandler_ = std::make_unique<HipMemHandler>(
        bootstrap_, myRank_, nRanks_, totalChunkStateBufferSize, localPeers32);

    const std::size_t totalNumChunks =
        totalChunkStateBufferSize / sizeof(ChunkState);
    std::vector<ChunkState> initStates(totalNumChunks);
    auto* statePtr =
        static_cast<ChunkState*>(stateBufferHandler_->getLocalDeviceMemPtr());
    HIP_CHECK(hipMemcpy(
        statePtr,
        initStates.data(),
        totalChunkStateBufferSize,
        hipMemcpyHostToDevice));
  }

  auto signalPtr =
      static_cast<SignalState*>(signalBufferHandler_->getLocalDeviceMemPtr());
  HIP_CHECK(hipMemset(signalPtr, 0, totalSignalBufferSize));
}

void MultiPeerNvlTransportAmd::exchange() {
  // HipMemHandler auto-exchanges in constructor, so this is a no-op.
  // Kept for API compatibility with MultiPeerNvlTransport.
}

P2pNvlTransportDevice MultiPeerNvlTransportAmd::getP2pTransportDevice(
    int peerRank) {
  const int localPeerIndex = (peerRank < myRank_) ? peerRank : (peerRank - 1);
  const std::size_t localSignalBufferOffset =
      localPeerIndex * perPeerSignalBufferSize_;

  const int remotePeerIndex = (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
  const std::size_t remoteSignalBufferOffset =
      remotePeerIndex * perPeerSignalBufferSize_;

  P2pNvlTransportOptions options{
      .dataBufferSize = config_.dataBufferSize,
      .chunkSize = config_.chunkSize,
      .pipelineDepth = config_.pipelineDepth,
      .useDualStateBuffer = config_.useDualStateBuffer};

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

  if (!dataBufferHandler_ || !stateBufferHandler_) {
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

  const std::size_t localDataBufferOffset =
      localPeerIndex * perPeerDataBufferSize_;
  const std::size_t localChunkStateBufferOffset =
      localPeerIndex * perPeerChunkStateBufferSize_;
  const std::size_t remoteDataBufferOffset =
      remotePeerIndex * perPeerDataBufferSize_;
  const std::size_t remoteChunkStateBufferOffset =
      remotePeerIndex * perPeerChunkStateBufferSize_;

  const std::size_t numChunksPerStep =
      (config_.dataBufferSize + config_.chunkSize - 1) / config_.chunkSize;
  const auto numChunksPerPeer =
      static_cast<uint32_t>(config_.pipelineDepth * numChunksPerStep);

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
  auto* remoteChunkStateBase = reinterpret_cast<ChunkState*>(
      remoteChunkStatePtr + remoteChunkStateBufferOffset);

  if (config_.useDualStateBuffer) {
    LocalState localState{
        .dataBuffer = localDataPtr + localDataBufferOffset,
        .receiverStateBuffer =
            DeviceSpan<ChunkState>(localChunkStateBase, numChunksPerPeer),
        .senderStateBuffer = DeviceSpan<ChunkState>(
            localChunkStateBase + numChunksPerPeer, numChunksPerPeer),
        .signalBuffer = localSignalSpan,
    };
    RemoteState remoteState{
        .dataBuffer = remoteDataPtr + remoteDataBufferOffset,
        .receiverStateBuffer =
            DeviceSpan<ChunkState>(remoteChunkStateBase, numChunksPerPeer),
        .senderStateBuffer = DeviceSpan<ChunkState>(
            remoteChunkStateBase + numChunksPerPeer, numChunksPerPeer),
        .signalBuffer = remoteSignalSpan,
    };
    return P2pNvlTransportDevice(
        myRank_, peerRank, options, localState, remoteState);
  } else {
    LocalState localState{
        .dataBuffer = localDataPtr + localDataBufferOffset,
        .receiverStateBuffer =
            DeviceSpan<ChunkState>(localChunkStateBase, numChunksPerPeer),
        .senderStateBuffer = DeviceSpan<ChunkState>(),
        .signalBuffer = localSignalSpan,
    };
    RemoteState remoteState{
        .dataBuffer = remoteDataPtr + remoteDataBufferOffset,
        .receiverStateBuffer =
            DeviceSpan<ChunkState>(remoteChunkStateBase, numChunksPerPeer),
        .senderStateBuffer = DeviceSpan<ChunkState>(),
        .signalBuffer = remoteSignalSpan,
    };
    return P2pNvlTransportDevice(
        myRank_, peerRank, options, localState, remoteState);
  }
}

DeviceSpan<Transport> MultiPeerNvlTransportAmd::getDeviceTransports() {
  if (!multiPeerInitialized_) {
    initializeTransportsArray();
    multiPeerInitialized_ = true;
  }
  return DeviceSpan<Transport>(
      static_cast<Transport*>(transportsDevice_), nRanks_);
}

void MultiPeerNvlTransportAmd::initializeTransportsArray() {
  HIP_CHECK(hipMalloc(&transportsDevice_, nRanks_ * sizeof(Transport)));

  std::vector<Transport> hostTransports;
  hostTransports.reserve(nRanks_);
  for (int rank = 0; rank < nRanks_; ++rank) {
    if (rank == myRank_) {
      hostTransports.emplace_back(P2pSelfTransportDevice());
    } else {
      hostTransports.emplace_back(getP2pTransportDevice(rank));
    }
  }

  HIP_CHECK(hipMemcpy(
      transportsDevice_,
      hostTransports.data(),
      nRanks_ * sizeof(Transport),
      hipMemcpyDefault));
}

void* MultiPeerNvlTransportAmd::exchangeDirectCopyPtrs(
    void* recvBuf,
    std::size_t size) {
  // Get IPC handle for local recv buffer
  hipIpcMemHandle_t myHandle;
  HIP_CHECK(hipIpcGetMemHandle(&myHandle, recvBuf));

  // Exchange handles
  struct HandleInfo {
    hipIpcMemHandle_t handle;
  };
  std::vector<HandleInfo> allHandles(nRanks_);
  allHandles[myRank_].handle = myHandle;

  auto result =
      bootstrap_
          ->allGather(allHandles.data(), sizeof(HandleInfo), myRank_, nRanks_)
          .get();
  if (result != 0)
    throw std::runtime_error("exchangeDirectCopyPtrs allGather failed");

  // Open peer handles and track IPC pointers for cleanup
  std::vector<char*> peerPtrs(nRanks_, nullptr);
  ipcPeerPtrs_.clear();
  ipcPeerPtrs_.reserve(nRanks_);
  for (int r = 0; r < nRanks_; r++) {
    if (r == myRank_) {
      peerPtrs[r] = static_cast<char*>(recvBuf);
      ipcPeerPtrs_.push_back(nullptr); // no IPC handle for self
    } else {
      void* ptr = nullptr;
      HIP_CHECK(hipIpcOpenMemHandle(
          &ptr, allHandles[r].handle, hipIpcMemLazyEnablePeerAccess));
      peerPtrs[r] = static_cast<char*>(ptr);
      ipcPeerPtrs_.push_back(ptr);
    }
  }

  // Copy pointer array to device memory
  void* d_peerPtrs = nullptr;
  HIP_CHECK(hipMalloc(&d_peerPtrs, nRanks_ * sizeof(char*)));
  HIP_CHECK(hipMemcpy(
      d_peerPtrs,
      peerPtrs.data(),
      nRanks_ * sizeof(char*),
      hipMemcpyHostToDevice));

  return d_peerPtrs;
}

void MultiPeerNvlTransportAmd::freeDirectCopyPtrs(void* d_peerPtrs) {
  // Close IPC handles before freeing device pointer array
  for (void* ptr : ipcPeerPtrs_) {
    if (ptr) {
      hipIpcCloseMemHandle(ptr);
    }
  }
  ipcPeerPtrs_.clear();
  if (d_peerPtrs) {
    hipFree(d_peerPtrs);
  }
}

} // namespace comms::pipes
#endif
