// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <deque>
#include <string>

#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/ibverbx/Ibverbx.h"

#include "comms/utils/commSpecs.h"

namespace ctran::ib {
class LocalVirtualConn {
 public:
  LocalVirtualConn(
      std::vector<CtranIbDevice>& devices,
      const uint64_t commHash,
      const std::string& commDesc);
  ~LocalVirtualConn() = default;

  commResult_t
  iflush(const void* dbuf, const void* ibRegElem, CtranIbRequest* req);

  commResult_t processCqe(const enum ibverbx::ibv_wc_opcode opcode);

  uint32_t qpNum(int device = 0) const;

 private:
  // CPU buffer used as src of local RDMA read
  int buf_{0};
  std::vector<ibverbx::IbvMr> ibvMrs_;
  std::vector<ibverbx::IbvQp> ibvQps_;
  std::vector<ibverbx::ibv_sge> sgs_;

  std::vector<CtranIbDevice> devices_;
  const uint64_t commHash_;
  const std::string commDesc_;

  // Track completion of outstanding flushes
  std::deque<CtranIbRequest*> outstandingReqs_;
};
} // namespace ctran::ib
