// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvQp.h"

#include <folly/json.h>
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

/*** IbvQp ***/
IbvQp::IbvQp(ibv_qp* qp) : qp_(qp) {}

IbvQp::~IbvQp() {
  if (qp_) {
    int rc = ibvSymbols.ibv_internal_destroy_qp(qp_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to destroy qp rc: {}, {}", rc, strerror(errno));
    }
  }
}

IbvQp::IbvQp(IbvQp&& other) noexcept {
  qp_ = other.qp_;
  physicalSendWrStatus_ = std::move(other.physicalSendWrStatus_);
  physicalRecvWrStatus_ = std::move(other.physicalRecvWrStatus_);
  other.qp_ = nullptr;
}

IbvQp& IbvQp::operator=(IbvQp&& other) noexcept {
  qp_ = other.qp_;
  physicalSendWrStatus_ = std::move(other.physicalSendWrStatus_);
  physicalRecvWrStatus_ = std::move(other.physicalRecvWrStatus_);
  other.qp_ = nullptr;
  return *this;
}

ibv_qp* IbvQp::qp() const {
  return qp_;
}

folly::Expected<folly::Unit, Error> IbvQp::modifyQp(
    ibv_qp_attr* attr,
    int attrMask) {
  int rc = ibvSymbols.ibv_internal_modify_qp(qp_, attr, attrMask);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

folly::Expected<std::pair<ibv_qp_attr, ibv_qp_init_attr>, Error> IbvQp::queryQp(
    int attrMask) const {
  ibv_qp_attr qpAttr{};
  ibv_qp_init_attr initAttr{};
  int rc = ibvSymbols.ibv_internal_query_qp(qp_, &qpAttr, attrMask, &initAttr);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return std::make_pair(qpAttr, initAttr);
}

void IbvQp::enquePhysicalSendWrStatus(int physicalWrId, int virtualWrId) {
  physicalSendWrStatus_.emplace_back(physicalWrId, virtualWrId);
}

void IbvQp::dequePhysicalSendWrStatus() {
  physicalSendWrStatus_.pop_front();
}

void IbvQp::dequePhysicalRecvWrStatus() {
  physicalRecvWrStatus_.pop_front();
}

void IbvQp::enquePhysicalRecvWrStatus(int physicalWrId, int virtualWrId) {
  physicalRecvWrStatus_.emplace_back(physicalWrId, virtualWrId);
}

bool IbvQp::isSendQueueAvailable(int maxMsgCntPerQp) const {
  if (maxMsgCntPerQp < 0) {
    return true;
  }
  return physicalSendWrStatus_.size() < maxMsgCntPerQp;
}

bool IbvQp::isRecvQueueAvailable(int maxMsgCntPerQp) const {
  if (maxMsgCntPerQp < 0) {
    return true;
  }
  return physicalRecvWrStatus_.size() < maxMsgCntPerQp;
}

} // namespace ibverbx
