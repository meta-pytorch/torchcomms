// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/ibverbx/IbvAh.h"

#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/utils/CtranLogger.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

/*** IbvAh ***/

IbvAh::IbvAh(ibv_ah* ah) : ah_(ah) {}

IbvAh::IbvAh(IbvAh&& other) noexcept {
  ah_ = other.ah_;
  other.ah_ = nullptr;
}

IbvAh& IbvAh::operator=(IbvAh&& other) noexcept {
  ah_ = other.ah_;
  other.ah_ = nullptr;
  return *this;
}

IbvAh::~IbvAh() {
  if (ah_) {
    int rc = ibvSymbols.ibv_internal_destroy_ah(ah_);
    if (rc != 0) {
      CTRAN_LOG(ERR, "Failed to destroy AH rc: {}, {}", rc, strerror(errno));
    }
  }
}

ibv_ah* IbvAh::ah() const {
  return ah_;
}

} // namespace ibverbx
