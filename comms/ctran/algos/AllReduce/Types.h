// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/utils/commSpecs.h"

// Forward declaration
struct KernelElem;

#define ALLREDUCE_MAX_KERNEL_ELEMS (8)

namespace ctran::allreduce {

enum class KernElemRole {
  kIntraReduceScatter,
  kInterReduceScatter,
  kIntraAllGather,
  kRemIntraReduce,
  kRemIntraBcast,
  kRemInterReduce
};

struct KernelArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  commRedOp_t redOp;
  size_t count;
  size_t nSteps;
  void* tmpbuff;
  size_t tmpbuffSize;
  // IPC imported ptr to each of the local peers' tmpRecvBuff
  void* intraNodeRemoteTmpRecvBuffs[CTRAN_MAX_NVL_PEERS];
  // IPC imported ptr to each of the local peers' RecvBuff
  void* intraNodeRemoteRecvBuffs[CTRAN_MAX_NVL_PEERS];
  KernelElem* kernelElems[ALLREDUCE_MAX_KERNEL_ELEMS];
};

} // namespace ctran::allreduce

// Ring host types are only used by CPU code (AllReduceRing.cc, CtranGpe.cc).
// Guard against nvcc host/device parses of folly in transitive includes.
// HIP sets __CUDACC__ when compiling host .cc; still need HostArgs on ROCm.
#if !defined(__CUDACC__) || \
    ((defined(__HIP__) || defined(__HIPCC__)) && !defined(__CUDA_ARCH__))

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
  ~HostResource() {
    // Release GpeKernelSyncs back to pool (sets inuse=false so pool can
    // reclaim). For graph-captured ops, this runs at graph destruction time
    // via cmdDestroy -> delete cmd -> ~OpElem -> ~HostResource, preventing
    // premature reuse by concurrent eager operations.
    if (sendCopySync) {
      sendCopySync->reset();
    }
    if (recvRedCopySync) {
      recvRedCopySync->reset();
    }
    if (partitionSync) {
      partitionSync->reset();
    }
    if (revSendCopySync) {
      revSendCopySync->reset();
    }
    if (revRecvCopySync) {
      revRecvCopySync->reset();
    }
  }

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

#endif
