// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"

class CtranComm;
class CtranMapperNotify;

namespace ctran::allreduce::ring {

struct HostArgs {
  int32_t rank{-1};
  int32_t leftRank{-1};
  int32_t rightRank{-1};

  size_t minShardSize{0};

  unsigned int numBlocks{0};
  unsigned int numThreads{0};

  // Enable bi-directional AllGather optimization
  bool enableBidirAg{true};

  // Forward: remote receive buffer on right
  void* rightRemBuf{nullptr};
  CtranMapperRemoteAccessKey rightRemKey;

  // Forward: receive notifications from left
  std::unique_ptr<CtranMapperNotify> leftNotify{nullptr};

  // Reverse: remote receive buffer on left (left neighbor's tmpRecvBufRev)
  void* leftRemBufRev{nullptr};
  CtranMapperRemoteAccessKey leftRemKeyRev;

  // Reverse: receive notifications from right
  std::unique_ptr<CtranMapperNotify> rightNotify{nullptr};
};

struct HostResource {
  CtranComm* comm{nullptr};

  ctran::algos::GpeKernelSync* sendCopySync{nullptr};
  ctran::algos::GpeKernelSync* recvRedCopySync{nullptr};
  ctran::algos::GpeKernelSync* partitionSync{nullptr};

  size_t chunkSize{0};
  size_t numChunks{0};
  void* tmpSendBuf{nullptr};
  void* tmpSendBufHdl{nullptr};
  void* tmpRecvBuf{nullptr};
  void* tmpRecvBufHdl{nullptr};

  // Reverse direction
  ctran::algos::GpeKernelSync* revSendCopySync{nullptr};
  ctran::algos::GpeKernelSync* revRecvCopySync{nullptr};
  void* tmpSendBufRev{nullptr};
  void* tmpSendBufRevHdl{nullptr};
  void* tmpRecvBufRev{nullptr};
  void* tmpRecvBufRevHdl{nullptr};

  // Set to true after completeHostResourceSetup() runs.
  // In CUDA graph mode, impl() is called on every replay but
  // IB resource setup must only happen once.
  bool setupComplete{false};
};

} // namespace ctran::allreduce::ring
