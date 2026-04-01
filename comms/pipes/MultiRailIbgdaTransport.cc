// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiRailIbgdaTransport.h"

#include <glog/logging.h>

#include <fmt/core.h>

#include "comms/pipes/rdma/NicDiscovery.h"

namespace comms::pipes {

MultiRailIbgdaTransport::MultiRailIbgdaTransport(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultipeerIbgdaTransportConfig& config)
    : myRank_(myRank), nRanks_(nRanks), numPeers_(nRanks - 1) {
  // Discover all NICs at the best GPU affinity level
  GpuNicDiscovery discovery(config.cudaDevice, config.ibHca);
  auto nics = discovery.getBestAffinityNics();
  if (nics.empty()) {
    throw std::runtime_error(
        "MultiRailIbgdaTransport: no NIC candidates found for GPU " +
        std::to_string(config.cudaDevice));
  }
  if (nics.size() > kMaxIbgdaRails) {
    throw std::runtime_error(
        fmt::format(
            "MultiRailIbgdaTransport: found {} NICs at best affinity but "
            "kMaxIbgdaRails={}",
            nics.size(),
            kMaxIbgdaRails));
  }
  // Create one transport per NIC
  for (const auto& nic : nics) {
    MultipeerIbgdaTransportConfig railConfig = config;
    railConfig.gpuNicMap[config.cudaDevice] = {nic.name};
    rails_.push_back(
        std::make_unique<MultipeerIbgdaTransport>(
            myRank, nRanks, bootstrap, railConfig));
    nicNames_.push_back(nic.name);
  }

  // Round-robin peer-to-rail assignment
  auto numNics = nicNames_.size();
  peerToRail_.resize(numPeers_);
  for (size_t p = 0; p < numPeers_; p++) {
    peerToRail_[p] = p % numNics;
  }

  LOG(INFO) << fmt::format(
      "MultiRailIbgdaTransport: {} NIC(s), numQpsPerPeer={}",
      numNics,
      config.numQpsPerPeer);
  for (size_t n = 0; n < numNics; n++) {
    int peerCount =
        std::count(peerToRail_.begin(), peerToRail_.end(), static_cast<int>(n));
    LOG(INFO) << fmt::format(
        "  rail {}: {} ({} peers)", n, nicNames_[n], peerCount);
  }
}

void MultiRailIbgdaTransport::exchange() {
  for (auto& rail : rails_) {
    rail->exchange();
  }
}

IbgdaLocalBuffer MultiRailIbgdaTransport::registerBuffer(
    void* ptr,
    std::size_t size) {
  IbgdaLocalBuffer result;
  for (size_t i = 0; i < rails_.size(); i++) {
    auto buf = rails_[i]->registerBuffer(ptr, size);
    if (i == 0) {
      result = buf;
    }
  }
  return result;
}

void MultiRailIbgdaTransport::deregisterBuffer(void* ptr) {
  for (auto& rail : rails_) {
    rail->deregisterBuffer(ptr);
  }
}

std::vector<IbgdaRemoteBuffer> MultiRailIbgdaTransport::exchangeBuffer(
    const IbgdaLocalBuffer& localBuf) {
  auto& perRailRemote = perRailRemoteBufs_[localBuf.ptr];
  perRailRemote.resize(rails_.size());

  std::vector<IbgdaRemoteBuffer> result;
  for (size_t i = 0; i < rails_.size(); i++) {
    perRailRemote[i] = rails_[i]->exchangeBuffer(localBuf);
    if (i == 0) {
      result = perRailRemote[i];
    }
  }
  return result;
}

IbgdaRemoteBuffer MultiRailIbgdaTransport::getRemoteBufferForQp(
    void* bufferPtr,
    int qpIndex) const {
  auto [rail, localIdx] = mapQpIndex(qpIndex);
  int numQpsPerPeer = rails_[0]->totalQps() / numPeers_;
  int peerIndex = localIdx / numQpsPerPeer;
  auto it = perRailRemoteBufs_.find(bufferPtr);
  if (it == perRailRemoteBufs_.end()) {
    throw std::runtime_error("getRemoteBufferForQp: buffer not exchanged");
  }
  return it->second[rail][peerIndex];
}

P2pIbgdaTransportDevice* MultiRailIbgdaTransport::getP2pTransportDevice(
    int peerRank) const {
  // Map rank to peer index, then to rail
  int peerIndex = (peerRank < myRank_) ? peerRank : (peerRank - 1);
  int rail = railForPeer(peerIndex);
  return rails_[rail]->getP2pTransportDevice(peerRank);
}

std::pair<int, int> MultiRailIbgdaTransport::mapQpIndex(int qpIndex) const {
  // Map user-visible QP index to (railIdx, local QP index within rail).
  // User QP layout: [peer0_qp0..qpN, peer1_qp0..qpN, peer2_qp0..qpN]
  // Each peer is assigned to one rail: railForPeer(peerIndex).
  // The local QP index within the rail uses the same peer's QP range.
  int numQpsPerPeer = rails_[0]->totalQps() / numPeers_;
  int peerIndex = qpIndex / numQpsPerPeer;
  int qpWithinPeer = qpIndex % numQpsPerPeer;
  int rail = railForPeer(peerIndex);
  // Within the rail, this peer's QPs start at peerIndex * numQpsPerPeer
  int localIdx = peerIndex * numQpsPerPeer + qpWithinPeer;
  return {rail, localIdx};
}

P2pIbgdaTransportDevice* MultiRailIbgdaTransport::getP2pTransportDeviceByIndex(
    int qpIndex) const {
  auto [rail, localIdx] = mapQpIndex(qpIndex);
  return rails_[rail]->getP2pTransportDeviceByIndex(localIdx);
}

IbgdaLocalBuffer MultiRailIbgdaTransport::getLocalBufferForQp(
    void* ptr,
    int qpIndex) const {
  auto [rail, localIdx] = mapQpIndex(qpIndex);
  // registerBuffer is idempotent — on cache hit, returns the lkey and
  // ignores the size. The buffer must already be registered.
  auto& mutableRail = const_cast<MultipeerIbgdaTransport&>(*rails_[rail]);
  return mutableRail.registerBuffer(ptr, 1);
}

P2pIbgdaTransportDevice* MultiRailIbgdaTransport::getDeviceTransportPtr()
    const {
  CHECK_EQ(rails_.size(), 1)
      << "getDeviceTransportPtr() only returns rail 0's transports. "
      << "Use getP2pTransportDevice(peerRank) for multi-NIC.";
  return rails_[0]->getDeviceTransportPtr();
}

int MultiRailIbgdaTransport::numPeers() const {
  return numPeers_;
}

int MultiRailIbgdaTransport::totalQps() const {
  // User-visible QP count = numPeers * numQpsPerPeer (original config)
  // NOT sum of all rails (which double-counts peers)
  return rails_[0]->totalQps();
}

int MultiRailIbgdaTransport::myRank() const {
  return myRank_;
}
int MultiRailIbgdaTransport::nRanks() const {
  return nRanks_;
}

int MultiRailIbgdaTransport::getGidIndex() const {
  return rails_[0]->getGidIndex();
}

} // namespace comms::pipes
