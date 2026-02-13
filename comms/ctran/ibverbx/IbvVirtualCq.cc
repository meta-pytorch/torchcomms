// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvVirtualCq.h"

namespace ibverbx {

/*** IbvVirtualCq ***/

IbvVirtualCq::IbvVirtualCq(IbvCq&& physicalCq, int maxCqe) : maxCqe_(maxCqe) {
  physicalCqs_.push_back(std::move(physicalCq));
  virtualCqNum_ =
      nextVirtualCqNum_.fetch_add(1); // Assign unique virtual CQ number
}

IbvVirtualCq::IbvVirtualCq(std::vector<IbvCq>&& cqs, int maxCqe)
    : physicalCqs_(std::move(cqs)), maxCqe_(maxCqe) {
  virtualCqNum_ =
      nextVirtualCqNum_.fetch_add(1); // Assign unique virtual CQ number
}

IbvVirtualCq::IbvVirtualCq(IbvVirtualCq&& other) noexcept {
  physicalCqs_ = std::move(other.physicalCqs_);
  maxCqe_ = other.maxCqe_;
  virtualCqNum_ = other.virtualCqNum_;
  registeredQps_ = std::move(other.registeredQps_);

  // Update registered VirtualQp back-pointers to point to new location
  for (auto& [qpId, info] : registeredQps_) {
    CHECK(info.vqp != nullptr)
        << "Registered physical QP has no associated VirtualQp!";
    info.vqp->virtualCq_ = this;
  }
}

IbvVirtualCq& IbvVirtualCq::operator=(IbvVirtualCq&& other) noexcept {
  if (this != &other) {
    physicalCqs_ = std::move(other.physicalCqs_);
    maxCqe_ = other.maxCqe_;
    virtualCqNum_ = other.virtualCqNum_;
    registeredQps_ = std::move(other.registeredQps_);

    // Update registered VirtualQp back-pointers to point to new location
    for (auto& [qpId, info] : registeredQps_) {
      CHECK(info.vqp != nullptr)
          << "Registered physical QP has no associated VirtualQp!";
      info.vqp->virtualCq_ = this;
    }
  }
  return *this;
}

std::vector<IbvCq>& IbvVirtualCq::getPhysicalCqsRef() {
  return physicalCqs_;
}

uint32_t IbvVirtualCq::getVirtualCqNum() const {
  return virtualCqNum_;
}

IbvVirtualCq::~IbvVirtualCq() = default;

void IbvVirtualCq::registerPhysicalQp(
    uint32_t physicalQpNum,
    int32_t deviceId,
    IbvVirtualQp* vqp,
    bool isMultiQp,
    uint32_t virtualQpNum) {
  QpId key{.deviceId = deviceId, .qpNum = physicalQpNum};
  registeredQps_[key] = RegisteredQpInfo{
      .vqp = vqp, .isMultiQp = isMultiQp, .virtualQpNum = virtualQpNum};
}

void IbvVirtualCq::unregisterPhysicalQp(
    uint32_t physicalQpNum,
    int32_t deviceId) {
  QpId key{.deviceId = deviceId, .qpNum = physicalQpNum};
  registeredQps_.erase(key);
}

} // namespace ibverbx
