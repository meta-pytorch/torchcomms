// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvVirtualCq.h"

namespace ibverbx {

/*** IbvVirtualCq ***/

IbvVirtualCq::IbvVirtualCq(IbvCq&& physicalCq, int maxCqe)
    : physicalCq_(std::move(physicalCq)), maxCqe_(maxCqe) {
  virtualCqNum_ =
      nextVirtualCqNum_.fetch_add(1); // Assign unique virtual CQ number

  // Register the virtual CQ with Coordinator
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualCq construction!";
  coordinator->registerVirtualCq(virtualCqNum_, this);
}

IbvVirtualCq::IbvVirtualCq(IbvVirtualCq&& other) noexcept {
  physicalCq_ = std::move(other.physicalCq_);
  pendingSendVirtualWcQue_ = std::move(other.pendingSendVirtualWcQue_);
  pendingRecvVirtualWcQue_ = std::move(other.pendingRecvVirtualWcQue_);
  maxCqe_ = other.maxCqe_;
  virtualWrIdToVirtualWc_ = std::move(other.virtualWrIdToVirtualWc_);
  virtualCqNum_ = other.virtualCqNum_;

  // Update coordinator pointer mapping for this virtual CQ after move
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualCq move construction!";
  coordinator->updateVirtualCqPointer(virtualCqNum_, this);
}

IbvVirtualCq& IbvVirtualCq::operator=(IbvVirtualCq&& other) noexcept {
  if (this != &other) {
    physicalCq_ = std::move(other.physicalCq_);
    pendingSendVirtualWcQue_ = std::move(other.pendingSendVirtualWcQue_);
    pendingRecvVirtualWcQue_ = std::move(other.pendingRecvVirtualWcQue_);
    maxCqe_ = other.maxCqe_;
    virtualWrIdToVirtualWc_ = std::move(other.virtualWrIdToVirtualWc_);
    virtualCqNum_ = other.virtualCqNum_;

    // Update coordinator pointer mapping for this virtual CQ after move
    auto coordinator = Coordinator::getCoordinator();
    CHECK(coordinator)
        << "Coordinator should not be nullptr during IbvVirtualCq move construction!";
    coordinator->updateVirtualCqPointer(virtualCqNum_, this);
  }
  return *this;
}

IbvCq& IbvVirtualCq::getPhysicalCqRef() {
  return physicalCq_;
}

uint32_t IbvVirtualCq::getVirtualCqNum() const {
  return virtualCqNum_;
}

void IbvVirtualCq::enqueSendCq(VirtualWc virtualWc) {
  pendingSendVirtualWcQue_.push_back(std::move(virtualWc));
}

void IbvVirtualCq::enqueRecvCq(VirtualWc virtualWc) {
  pendingRecvVirtualWcQue_.push_back(std::move(virtualWc));
}

IbvVirtualCq::~IbvVirtualCq() {
  // Always call unregister - the coordinator will check if the pointer matches
  // and do nothing if the object was moved
  auto coordinator = Coordinator::getCoordinator();
  CHECK(coordinator)
      << "Coordinator should not be nullptr during IbvVirtualCq destruction!";
  coordinator->unregisterVirtualCq(virtualCqNum_, this);
}

} // namespace ibverbx
