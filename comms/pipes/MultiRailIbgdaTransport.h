// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/IbgdaRailConfig.h"
#include "comms/pipes/MultipeerIbgdaTransport.h"

namespace comms::pipes {

class P2pIbgdaTransportDevice;

/**
 * MultiRailIbgdaTransport — Multi-NIC wrapper around MultipeerIbgdaTransport.
 *
 * Creates one MultipeerIbgdaTransport per NIC and distributes peers across
 * them via round-robin. NIC count is determined by GPU affinity — all NICs
 * at the best PCIe topology level are used automatically.
 *
 * Exposes the same API as MultipeerIbgdaTransport so callers
 * (MultiPeerTransport, HostWindow) get multi-NIC transparently.
 */
class MultiRailIbgdaTransport {
 public:
  MultiRailIbgdaTransport(
      int myRank,
      int nRanks,
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      const MultipeerIbgdaTransportConfig& config);

  ~MultiRailIbgdaTransport() = default;

  // Collective — all ranks must call
  void exchange();

  IbgdaLocalBuffer registerBuffer(void* ptr, std::size_t size);
  void deregisterBuffer(void* ptr);
  std::vector<IbgdaRemoteBuffer> exchangeBuffer(
      const IbgdaLocalBuffer& localBuf);

  // Multi-NIC safe: routes to the correct rail based on peerRank.
  // This is the primary API used by production (MultiPeerTransport,
  // HostWindow).
  P2pIbgdaTransportDevice* getP2pTransportDevice(int peerRank) const;

  // Multi-NIC safe: routes to the correct rail based on qpIndex.
  // Used by benchmarks with numQpsPerPeer > 1.
  P2pIbgdaTransportDevice* getP2pTransportDeviceByIndex(int qpIndex) const;

  // Multi-NIC safe: returns the correct rail's lkey/rkey for the given QP.
  IbgdaLocalBuffer getLocalBufferForQp(void* ptr, int qpIndex) const;
  IbgdaRemoteBuffer getRemoteBufferForQp(void* bufferPtr, int qpIndex) const;

  // WARNING: Returns rail 0's device transport array only. Rail 1's transports
  // live in a separate GPU allocation. Do NOT use this to index across all
  // peers — use getP2pTransportDevice(peerRank) instead.
  // Kept for backward compatibility with old benchmarks and tests.
  P2pIbgdaTransportDevice* getDeviceTransportPtr() const;

  int numPeers() const;
  int totalQps() const;
  int myRank() const;
  int nRanks() const;
  int numNics() const {
    return static_cast<int>(rails_.size());
  }

  // Which NIC handles a given QP index
  int railForQpIndex(int qpIndex) const {
    auto [rail, localIdx] = mapQpIndex(qpIndex);
    return rail;
  }

  int getGidIndex() const;

 private:
  // Which rail (NIC) handles a given peer index (precomputed for balance)
  int railForPeer(int peerIndex) const {
    return peerToRail_[peerIndex];
  }

  // Map global QP index to (railIdx, localQpIndex)
  std::pair<int, int> mapQpIndex(int qpIndex) const;

  int myRank_;
  int nRanks_;
  int numPeers_;
  std::vector<std::unique_ptr<MultipeerIbgdaTransport>> rails_;
  std::vector<std::string> nicNames_;
  std::vector<int> peerToRail_; // peer index → rail assignment

  // Per-rail remote buffers: perRailRemoteBufs_[bufferPtr][railIdx] =
  // remoteBufs
  std::unordered_map<void*, std::vector<std::vector<IbgdaRemoteBuffer>>>
      perRailRemoteBufs_;
};

} // namespace comms::pipes
