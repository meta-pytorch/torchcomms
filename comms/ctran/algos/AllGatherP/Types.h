// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include <memory>
#include <vector>
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/utils/commSpecs.h"

#if !defined(__CUDACC__)
#include "comms/ctran/algos/common/BufManager.h"
#endif

using ctran::algos::GpeKernelSync;

struct CtranMapperRemoteAccessKey;
namespace ctran::allgatherp {

enum class StagingBufId {
  kSendBuf = 0,
  kRecvBuf,
  kNumBufs,
};
struct PersistArgs {
  void* recvbuff;
  void* recvHdl;
  size_t maxRecvCount;
  commDataType_t datatype;
  std::vector<void*> remoteRecvBuffs;
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys;

  // Initialization offloads the remote handle exchange to GPE thread to avoid
  // potential deadlock on mapper epoch lock, if init is called again on the
  // main thread while there is an outstanding exec. Init returns without
  // waiting for the completion of async init. Any subsequent execution call
  // should wait for its completion via the initialized_ flag, before the main
  // thread can schedule copy engine copies
  std::atomic<bool> initialized{false};
  int pinnedAlgo{0};
};

#if !defined(__CUDACC__)
struct StagingInfo {
  ctran::algos::bufmanager::RegBuf sendBuf;
  ctran::algos::bufmanager::RegBuf recvBuf;
  std::vector<ctran::algos::bufmanager::RemRegBuf> remRecvBufs;
  std::vector<int> peerRanks;
  size_t slotSize{0};
  size_t numSlots{0};
};
#endif

struct Resource {
  GpeKernelSync* pipeSync{nullptr};
  GpeKernelSync* stepDoneSync{nullptr};
#if !defined(__CUDACC__)
  std::unique_ptr<
      ctran::algos::BufManager<StagingBufId, StagingBufId::kNumBufs>>
      stagingBufMgr;
  StagingInfo stagingInfo;
#endif
};

struct PipeEndKernArgs {
  GpeKernelSync* pipeSync;
};

struct PipeSyncKernArgs {
  int stepId;
  GpeKernelSync* pipeSync;
};

struct StepDoneKernArgs {
  int stepId;
  GpeKernelSync* stepDoneSync;
};

struct PatCopyPipeEndKernArgs {
  GpeKernelSync* pipeSync;
  GpeKernelSync* stepDoneSync;
};

} // namespace ctran::allgatherp

namespace comms::pipes {
class P2pNvlTransportDevice;
}

namespace ctran::allgatherp {

struct NvlDissemKernArgs {
  void* stagingRecvBuf;
  void* recvbuff;
  int nChunks;
  size_t chunkSize;
  int nLocalRanks;
  int localRank;
  int peerNode;
  int nNodes;
  int step;
  ::comms::pipes::P2pNvlTransportDevice* nvlTransportsBase;
  uint64_t timeoutCycles;
};
} // namespace ctran::allgatherp
