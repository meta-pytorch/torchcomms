// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_BROADCAST_IMPL_H_
#define CTRAN_BROADCAST_IMPL_H_
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/utils/cvars/nccl_cvars.h"

static inline const std::string sendRecvAlgoName(
    enum NCCL_SENDRECV_ALGO algo,
    const std::vector<OpElem*>& opGroup) {
  if (algo == NCCL_SENDRECV_ALGO::ctran) {
    const auto& firstOp = opGroup.front();
    // Special case for single send/recv
    if (opGroup.size() == 1) {
      if (firstOp->type == OpElem::opType::SEND) {
        return "CtranSend";
      } else if (firstOp->type == OpElem::opType::RECV) {
        return "CtranRecv";
      }
    } else {
      // Leave it as sendrecv if more than one p2p
      return "CtranSendRecv";
    }
  }
  return "Unknown";
}

#endif
