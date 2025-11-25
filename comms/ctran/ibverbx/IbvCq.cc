// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/ibverbx/IbvCq.h"
#include <folly/logging/xlog.h>
#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

/*** IbvCq ***/

IbvCq::IbvCq(ibv_cq* cq) : cq_(cq) {}

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
  other.cq_ = nullptr;
}

IbvCq& IbvCq::operator=(IbvCq&& other) noexcept {
  cq_ = other.cq_;
  other.cq_ = nullptr;
  return *this;
}

ibv_cq* IbvCq::cq() const {
  return cq_;
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
