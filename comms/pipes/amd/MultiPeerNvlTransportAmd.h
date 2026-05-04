// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD equivalent of MultiPeerNvlTransport.
// Same API but uses HipMemHandler (HIP IPC) instead of GpuMemHandler (CUDA
// IPC). Device-side code (P2pNvlTransportDevice, Transport) is hipified
// automatically.

#pragma once

#include <memory>
#include <vector>

#include "comms/common/bootstrap/IBootstrap.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes {

class HipMemHandler;

struct MultiPeerNvlTransportAmdConfig {
  std::size_t dataBufferSize{0};
  std::size_t chunkSize{0};
  std::size_t pipelineDepth{0};
  std::size_t p2pSignalCount{1};
  bool useDualStateBuffer{false};
};

class MultiPeerNvlTransportAmd {
 public:
  MultiPeerNvlTransportAmd(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultiPeerNvlTransportAmdConfig& config);

  // Filtered mode: only set up NVL IPC for localPeerRanks.
  // AllGather coordinates with all nRanks, but hipIpcOpenMemHandle
  // is only called for specified local peers. Used by MultiPeerTransportAmd
  // for hybrid NVL+IBGDA topology.
  MultiPeerNvlTransportAmd(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultiPeerNvlTransportAmdConfig& config,
      const std::vector<int>& localPeerRanks);

  ~MultiPeerNvlTransportAmd();

  MultiPeerNvlTransportAmd(const MultiPeerNvlTransportAmd&) = delete;
  MultiPeerNvlTransportAmd& operator=(const MultiPeerNvlTransportAmd&) = delete;

  // Collective: all ranks must call
  void exchange();

  // Get device-accessible Transport array for pipelined AllToAllv
  DeviceSpan<Transport> getDeviceTransports();

  // Register a user buffer and exchange IPC pointers for direct-copy AllToAllv.
  // Returns device-accessible array of nRanks char* pointers (peer recv bufs).
  // peerPtrs[myRank] = local recvBuf, peerPtrs[other] = IPC-mapped peer buf.
  void* exchangeDirectCopyPtrs(void* recvBuf, std::size_t size);

  // Free direct-copy pointers
  void freeDirectCopyPtrs(void* d_peerPtrs);

  int getMyRank() const {
    return myRank_;
  }
  int getNRanks() const {
    return nRanks_;
  }

  friend class MultiPeerTransportAmd;

 private:
  P2pNvlTransportDevice getP2pTransportDevice(int peerRank);
  void initializeTransportsArray();

  static std::size_t getSignalBufferSize(std::size_t signalCount) {
    return signalCount * sizeof(SignalState);
  }

  const int myRank_;
  const int nRanks_;
  std::shared_ptr<meta::comms::IBootstrap> bootstrap_;
  MultiPeerNvlTransportAmdConfig config_;

  std::unique_ptr<HipMemHandler> signalBufferHandler_;
  std::unique_ptr<HipMemHandler> dataBufferHandler_;
  std::unique_ptr<HipMemHandler> stateBufferHandler_;

  std::size_t perPeerDataBufferSize_{0};
  std::size_t perPeerChunkStateBufferSize_{0};
  std::size_t perPeerSignalBufferSize_{0};

  // Device transport array
  void* transportsDevice_{nullptr};
  bool multiPeerInitialized_{false};

  // IPC peer pointers opened via hipIpcOpenMemHandle (for direct-copy).
  // Must be closed with hipIpcCloseMemHandle in freeDirectCopyPtrs.
  std::vector<void*> ipcPeerPtrs_;
};

} // namespace comms::pipes
