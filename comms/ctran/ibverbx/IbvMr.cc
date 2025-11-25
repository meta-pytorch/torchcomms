// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "comms/ctran/ibverbx/IbvMr.h"

#include <folly/logging/xlog.h>
#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

/*** IbvMr ***/

IbvMr::IbvMr(ibv_mr* mr) : mr_(mr) {}

IbvMr::IbvMr(IbvMr&& other) noexcept {
  mr_ = other.mr_;
  other.mr_ = nullptr;
}

IbvMr& IbvMr::operator=(IbvMr&& other) noexcept {
  mr_ = other.mr_;
  other.mr_ = nullptr;
  return *this;
}

IbvMr::~IbvMr() {
  if (mr_) {
    int rc = ibvSymbols.ibv_internal_dereg_mr(mr_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to deregister mr rc: {}, {}", rc, strerror(errno));
    }
  }
}

ibv_mr* IbvMr::mr() const {
  return mr_;
}

} // namespace ibverbx
