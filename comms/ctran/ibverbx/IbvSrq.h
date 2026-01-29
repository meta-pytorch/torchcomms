// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <folly/Expected.h>
#include "comms/ctran/ibverbx/IbvCommon.h"
#include "comms/ctran/ibverbx/Ibvcore.h"

namespace ibverbx {

// IbvSrq: Shared Receive Queue
// Used by DC transport for receiving messages on DCT
class IbvSrq {
 public:
  ~IbvSrq();

  // disable copy constructor
  IbvSrq(const IbvSrq&) = delete;
  IbvSrq& operator=(const IbvSrq&) = delete;

  // move constructor
  IbvSrq(IbvSrq&& other) noexcept;
  IbvSrq& operator=(IbvSrq&& other) noexcept;

  ibv_srq* srq() const;

  // Post a receive work request to the SRQ
  folly::Expected<folly::Unit, Error> postRecv(
      ibv_recv_wr* recvWr,
      ibv_recv_wr** badRecvWr);

  // Modify SRQ attributes
  folly::Expected<folly::Unit, Error> modifySrq(
      ibv_srq_attr* srqAttr,
      int srqAttrMask);

  // Query SRQ attributes
  folly::Expected<ibv_srq_attr, Error> querySrq() const;

 private:
  friend class IbvPd;

  explicit IbvSrq(ibv_srq* srq);

  ibv_srq* srq_{nullptr};
};

} // namespace ibverbx
