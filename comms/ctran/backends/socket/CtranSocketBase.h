// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <fmt/format.h>

#include "comms/ctran/backends/CtranCtrl.h"

class CtranSocketRequest {
 public:
  CtranSocketRequest(){};
  ~CtranSocketRequest(){};
  inline commResult_t complete() {
    state_ = COMPLETE;
    return commSuccess;
  }

  inline bool isComplete() const {
    return this->state_ == CtranSocketRequest::COMPLETE;
  }

 private:
  enum {
    INCOMPLETE,
    COMPLETE,
  } state_{INCOMPLETE};
};

struct SockPendingOp {
  enum OpType {
    UNDEFINED,
    ISEND_CTRL,
    IRECV_CTRL,
  };

 public:
  SockPendingOp(
      SockPendingOp::OpType type,
      ControlMsg& msg,
      int peerRank,
      CtranSocketRequest& req)
      : type(type), msg(msg), peerRank(peerRank), req(req) {}
  ~SockPendingOp() {}

  OpType type{UNDEFINED};

  ControlMsg& msg;
  int peerRank{-1};
  CtranSocketRequest& req;
};

template <>
struct fmt::formatter<SockPendingOp::OpType> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(SockPendingOp::OpType status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};
