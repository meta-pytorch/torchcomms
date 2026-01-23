// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h" // for CTRAN_MAX_NVL_PEERS
#include "comms/utils/commSpecs.h" // need for ncclDataType_t

// Forward declaration to avoid including P2pNvlTransportDevice.cuh in this
// header. Including that header would cause duplicate symbols when .cu files
// including this header are compiled by both device_object and
// hetero_ctran_device_lib.
namespace comms::pipes {
class P2pNvlTransportDevice;
}

namespace ctran::sendrecv {

// Max send/recv ops for P2P kernel using static array in KernArgs.
// For common batchSendRecv cases, we typically have <= 2 ops in prod.
// Exceeding this limit will fallback to list format (slower but handles
// arbitrary number of ops).
constexpr size_t kCtranMaxNvlSendRecvOps = 2;

struct SendRecvOp {
  void* buff;
  size_t nbytes;
  int nGroups;

  int peerLocalRank;
};

struct KernArgs {
  size_t numSends;
  size_t numRecvs;
  size_t numSendBlocks;
  size_t numRecvBlocks;
  // Static arrays for common cases (fast path)
  SendRecvOp sends[kCtranMaxNvlSendRecvOps];
  SendRecvOp recvs[kCtranMaxNvlSendRecvOps];
  // List format pointers for fallback when exceeding kCtranMaxNvlSendRecvOps
  SendRecvOp* sendsList;
  SendRecvOp* recvsList;
  // If true, use list format (slower path for > kCtranMaxNvlSendRecvOps ops)
  bool useList;

  // used only in SENDRECV_P2P kernel
  // If true, use block group; otherwise use warp group
  bool useBlockGroup;

  // Base pointer to pre-allocated P2pNvlTransportDevice array
  // Indexed by peerLocalRank to get the transport for each peer
  comms::pipes::P2pNvlTransportDevice* nvlTransportsBase;
};
} // namespace ctran::sendrecv
