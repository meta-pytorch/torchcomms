// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/ibverbx/IbvSrq.h"

#include <folly/logging/xlog.h>
#include "comms/ctran/ibverbx/IbverbxSymbols.h"

namespace ibverbx {

extern IbvSymbols ibvSymbols;

/*** IbvSrq ***/

IbvSrq::IbvSrq(ibv_srq* srq) : srq_(srq) {}

IbvSrq::IbvSrq(IbvSrq&& other) noexcept {
  srq_ = other.srq_;
  other.srq_ = nullptr;
}

IbvSrq& IbvSrq::operator=(IbvSrq&& other) noexcept {
  srq_ = other.srq_;
  other.srq_ = nullptr;
  return *this;
}

IbvSrq::~IbvSrq() {
  if (srq_) {
    int rc = ibvSymbols.ibv_internal_destroy_srq(srq_);
    if (rc != 0) {
      XLOGF(ERR, "Failed to destroy SRQ rc: {}, {}", rc, strerror(errno));
    }
  }
}

ibv_srq* IbvSrq::srq() const {
  return srq_;
}

folly::Expected<folly::Unit, Error> IbvSrq::postRecv(
    ibv_recv_wr* recvWr,
    ibv_recv_wr** badRecvWr) {
  int rc = ibvSymbols.ibv_internal_post_srq_recv(srq_, recvWr, badRecvWr);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

folly::Expected<folly::Unit, Error> IbvSrq::modifySrq(
    ibv_srq_attr* srqAttr,
    int srqAttrMask) {
  int rc = ibvSymbols.ibv_internal_modify_srq(srq_, srqAttr, srqAttrMask);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return folly::unit;
}

folly::Expected<ibv_srq_attr, Error> IbvSrq::querySrq() const {
  ibv_srq_attr srqAttr{};
  int rc = ibvSymbols.ibv_internal_query_srq(srq_, &srqAttr);
  if (rc != 0) {
    return folly::makeUnexpected(Error(rc));
  }
  return srqAttr;
}

} // namespace ibverbx
