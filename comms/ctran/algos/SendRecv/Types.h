// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h" // for CTRAN_MAX_NVL_PEERS
#include "comms/utils/commSpecs.h" // need for ncclDataType_t

namespace ctran::sendrecv {

struct SendRecvOp {
  void* buff;
  size_t nbytes;
  int peerLocalRank;
  int nGroups;
};

struct KernArgs {
  size_t numSends;
  size_t numRecvs;
  size_t numSendBlocks;
  size_t numRecvBlocks;
  SendRecvOp sends[CTRAN_MAX_NVL_PEERS];
  SendRecvOp recvs[CTRAN_MAX_NVL_PEERS];
};
} // namespace ctran::sendrecv
