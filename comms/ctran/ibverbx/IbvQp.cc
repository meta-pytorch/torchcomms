// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <folly/json.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/ibverbx/IbvQp.h"
#include "comms/ctran/ibverbx/Ibvcore.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

/*** IbvQp ***/
IbvQp::IbvQp(ibv_qp* qp, int32_t deviceId) : qp_(qp), deviceId_(deviceId) {}

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
  deviceId_ = other.deviceId_;
  other.qp_ = nullptr;
  other.deviceId_ = -1;
}

IbvQp& IbvQp::operator=(IbvQp&& other) noexcept {
  qp_ = other.qp_;
  physicalSendWrStatus_ = std::move(other.physicalSendWrStatus_);
  physicalRecvWrStatus_ = std::move(other.physicalRecvWrStatus_);
  deviceId_ = other.deviceId_;
  other.qp_ = nullptr;
  other.deviceId_ = -1;
  return *this;
}

ibv_qp* IbvQp::qp() const {
  return qp_;
}

int32_t IbvQp::getDeviceId() const {
  return deviceId_;
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

void IbvQp::logInfo() const {
  auto maybeQpAttrPair = queryQp(IBV_QP_STATE | IBV_QP_CAP | IBV_QP_PKEY_INDEX);
  if (maybeQpAttrPair.hasValue()) {
    auto [qpAttr, initAttr] = maybeQpAttrPair.value();
    XLOGF(
        INFO,
        "QP details: qp_num={}, qp_ptr={}, context={}, pd={}, state info: state={}, max_send_wr={}, max_recv_wr={}, max_send_sge={}, max_recv_sge={}, pkey_index={}",
        qp_->qp_num,
        fmt::ptr(qp_),
        fmt::ptr(qp_->context),
        fmt::ptr(qp_->pd),
        static_cast<int>(qpAttr.qp_state),
        qpAttr.cap.max_send_wr,
        qpAttr.cap.max_recv_wr,
        qpAttr.cap.max_send_sge,
        qpAttr.cap.max_recv_sge,
        qpAttr.pkey_index);
  } else {
    XLOGF(
        ERR,
        "QP details: qp_num={}, qp_ptr={}, context={}, pd={}, Failed to query local QP state: errno={} ({})",
        qp_->qp_num,
        fmt::ptr(qp_),
        fmt::ptr(qp_->context),
        fmt::ptr(qp_->pd),
        maybeQpAttrPair.error().errNum,
        maybeQpAttrPair.error().errStr);
  }
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
