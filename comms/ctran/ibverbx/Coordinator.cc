// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/Coordinator.h"

#include <folly/Singleton.h>
#include "comms/ctran/ibverbx/IbvVirtualQp.h"

namespace ibverbx {

namespace {
folly::Singleton<Coordinator> coordinatorSingleton{};
}

/*** Coordinator ***/

std::shared_ptr<Coordinator> Coordinator::getCoordinator() {
  return coordinatorSingleton.try_get();
}

// Register APIs for mapping management
void Coordinator::registerVirtualQp(
    uint32_t virtualQpNum,
    IbvVirtualQp* virtualQp) {
  virtualQpNumToVirtualQp_[virtualQpNum] = virtualQp;
}

void Coordinator::registerVirtualCq(
    uint32_t virtualCqNum,
    IbvVirtualCq* virtualCq) {
  virtualCqNumToVirtualCq_[virtualCqNum] = virtualCq;
}

void Coordinator::registerPhysicalQpToVirtualQp(
    int physicalQpNum,
    uint32_t virtualQpNum) {
  physicalQpNumToVirtualQpNum_[physicalQpNum] = virtualQpNum;
}

void Coordinator::registerVirtualQpToVirtualSendCq(
    uint32_t virtualQpNum,
    uint32_t virtualSendCqNum) {
  virtualQpNumToVirtualSendCqNum_[virtualQpNum] = virtualSendCqNum;
}

void Coordinator::registerVirtualQpToVirtualRecvCq(
    uint32_t virtualQpNum,
    uint32_t virtualRecvCqNum) {
  virtualQpNumToVirtualRecvCqNum_[virtualQpNum] = virtualRecvCqNum;
}

void Coordinator::registerVirtualQpWithVirtualCqMappings(
    IbvVirtualQp* virtualQp,
    uint32_t virtualSendCqNum,
    uint32_t virtualRecvCqNum) {
  // Extract virtual QP number from the virtual QP object
  uint32_t virtualQpNum = virtualQp->getVirtualQpNum();

  // Register the virtual QP
  registerVirtualQp(virtualQpNum, virtualQp);

  // Register all physical QP to virtual QP mappings
  for (const auto& qp : virtualQp->getQpsRef()) {
    registerPhysicalQpToVirtualQp(qp.qp()->qp_num, virtualQpNum);
  }
  // Register notify QP
  registerPhysicalQpToVirtualQp(
      virtualQp->getNotifyQpRef().qp()->qp_num, virtualQpNum);

  // Register virtual QP to virtual CQ relationships
  registerVirtualQpToVirtualSendCq(virtualQpNum, virtualSendCqNum);
  registerVirtualQpToVirtualRecvCq(virtualQpNum, virtualRecvCqNum);
}

// Access APIs for testing and internal use
const std::unordered_map<uint32_t, IbvVirtualQp*>&
Coordinator::getVirtualQpMap() const {
  return virtualQpNumToVirtualQp_;
}

const std::unordered_map<uint32_t, IbvVirtualCq*>&
Coordinator::getVirtualCqMap() const {
  return virtualCqNumToVirtualCq_;
}

const std::unordered_map<int, uint32_t>&
Coordinator::getPhysicalQpToVirtualQpMap() const {
  return physicalQpNumToVirtualQpNum_;
}

const std::unordered_map<uint32_t, uint32_t>&
Coordinator::getVirtualQpToVirtualSendCqMap() const {
  return virtualQpNumToVirtualSendCqNum_;
}

const std::unordered_map<uint32_t, uint32_t>&
Coordinator::getVirtualQpToVirtualRecvCqMap() const {
  return virtualQpNumToVirtualRecvCqNum_;
}

// Update API for move operations - only need to update pointer maps
void Coordinator::updateVirtualQpPointer(
    uint32_t virtualQpNum,
    IbvVirtualQp* newPtr) {
  virtualQpNumToVirtualQp_[virtualQpNum] = newPtr;
}

void Coordinator::updateVirtualCqPointer(
    uint32_t virtualCqNum,
    IbvVirtualCq* newPtr) {
  virtualCqNumToVirtualCq_[virtualCqNum] = newPtr;
}

void Coordinator::unregisterVirtualQp(
    uint32_t virtualQpNum,
    IbvVirtualQp* ptr) {
  // Only unregister if the pointer in the map matches the object being
  // destroyed. This handles the case where the object was moved and the map was
  // already updated with the new pointer.
  auto it = virtualQpNumToVirtualQp_.find(virtualQpNum);
  if (it == virtualQpNumToVirtualQp_.end() || it->second != ptr) {
    // Object was moved, map already updated, nothing to do
    return;
  }

  // Remove entries from all maps related to this virtual QP
  virtualQpNumToVirtualQp_.erase(virtualQpNum);
  virtualQpNumToVirtualSendCqNum_.erase(virtualQpNum);
  virtualQpNumToVirtualRecvCqNum_.erase(virtualQpNum);

  // Remove all physical QP to virtual QP mappings that point to this virtual QP
  for (auto it = physicalQpNumToVirtualQpNum_.begin();
       it != physicalQpNumToVirtualQpNum_.end();) {
    if (it->second == virtualQpNum) {
      it = physicalQpNumToVirtualQpNum_.erase(it);
    } else {
      ++it;
    }
  }
}

void Coordinator::unregisterVirtualCq(
    uint32_t virtualCqNum,
    IbvVirtualCq* ptr) {
  // Only unregister if the pointer in the map matches the object being
  // destroyed. This handles the case where the object was moved and the map was
  // already updated with the new pointer.
  auto it = virtualCqNumToVirtualCq_.find(virtualCqNum);
  if (it == virtualCqNumToVirtualCq_.end() || it->second != ptr) {
    // Object was moved, map already updated, nothing to do
    return;
  }

  // Remove the virtual CQ from the pointer map
  virtualCqNumToVirtualCq_.erase(virtualCqNum);

  // Remove all virtual QP to virtual send CQ mappings that point to this
  // virtual CQ
  for (auto it = virtualQpNumToVirtualSendCqNum_.begin();
       it != virtualQpNumToVirtualSendCqNum_.end();) {
    if (it->second == virtualCqNum) {
      it = virtualQpNumToVirtualSendCqNum_.erase(it);
    } else {
      ++it;
    }
  }

  // Remove all virtual QP to virtual recv CQ mappings that point to this
  // virtual CQ
  for (auto it = virtualQpNumToVirtualRecvCqNum_.begin();
       it != virtualQpNumToVirtualRecvCqNum_.end();) {
    if (it->second == virtualCqNum) {
      it = virtualQpNumToVirtualRecvCqNum_.erase(it);
    } else {
      ++it;
    }
  }
}

} // namespace ibverbx
