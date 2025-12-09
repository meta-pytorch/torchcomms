// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvCq.h"
#include <folly/logging/xlog.h>
#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

/*** IbvCq ***/

IbvCq::IbvCq(ibv_cq* cq, int32_t deviceId) : cq_(cq), deviceId_(deviceId) {}

IbvCq::~IbvCq() {
  if (cq_) {
    int rc = ibvSymbols.ibv_internal_destroy_cq(cq_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to destroy cq rc: {}, {}", rc, strerror(errno));
    }
  }
}

IbvCq::IbvCq(IbvCq&& other) noexcept {
  cq_ = other.cq_;
  deviceId_ = other.deviceId_;
  other.cq_ = nullptr;
  other.deviceId_ = -1;
}

IbvCq& IbvCq::operator=(IbvCq&& other) noexcept {
  cq_ = other.cq_;
  deviceId_ = other.deviceId_;
  other.cq_ = nullptr;
  other.deviceId_ = -1;
  return *this;
}

ibv_cq* IbvCq::cq() const {
  return cq_;
}

int32_t IbvCq::getDeviceId() const {
  return deviceId_;
}

folly::Expected<folly::Unit, Error> IbvCq::reqNotifyCq(
    int solicited_only) const {
  int rc = cq_->context->ops.req_notify_cq(cq_, solicited_only);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

} // namespace ibverbx
