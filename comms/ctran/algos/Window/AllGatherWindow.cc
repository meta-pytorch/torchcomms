// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/Window/AllGatherWindowTypes.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran;
using namespace ctran::allgatherwindow;

// Kernel declarations - must be in the same namespace as definitions in .cu
namespace ctran::allgatherwindow {
extern __global__ void ncclKernelAllGatherWindowDirect(
    int* flag,
    CtranAlgoDeviceState* devState);

extern __global__ void ncclKernelAllGatherWindowPipeStart(
    int* flag,
    CtranAlgoDeviceState* devState);

extern __global__ void ncclKernelAllGatherWindowPipeSync(
    int* flag,
    CtranAlgoDeviceState* devState,
    PipeSyncKernArgs args);

extern __global__ void ncclKernelAllGatherWindowPipeEnd(
    int* flag,
    CtranAlgoDeviceState* devState,
    PipeEndKernArgs args);

extern __global__ void ncclKernelAllGatherWindowPipe(
    int* flag,
    CtranAlgoDeviceState* devState);
} // namespace ctran::allgatherwindow

namespace {

// Get the index of the chunk in recvBuff to receive from the internode Ring
// neighbor in the rail. E.g., for nRanks = 8, nLocalRanks = 2, rank = 2, it
// would receive chunkIdx 0, 6, 4 of the recvBuff in a 3-step Ring.
inline size_t
getRecvChunkIdxInRail(int rank, int step, int nLocalRanks, int nRanks) {
  return (rank - step * nLocalRanks + nRanks) & (nRanks - 1);
}

// Helper to check if we should use NVLink for peer communication
// Returns true if peer is on the same node and NVLink is available
inline bool
useNvlinkForPeer(int peer, CtranWin* win, const ncclx::CommStateX* statex) {
  return statex->node(peer) == statex->node() && win->nvlEnabled(peer);
}

/**
 * Direct algorithm GPE function
 *
 * All-to-all PUT: each rank sends its data to all other ranks simultaneously.
 * Uses mapper->iput() for data transfer and mapper->atomicSet() for signaling.
 *
 * NVLink peers: Data already copied via nvlCeBcast in caller, synchronized by
 *               kernel barrier.
 * IB peers: PUT data via RDMA, signal completion, kernel waits for signals.
 */
commResult_t directGpeFn(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto* gpeArgs =
      reinterpret_cast<AllGatherWindowGpeArgs*>(op->allgatherWindow.args);
  CtranWin* win = gpeArgs->win;
  const size_t count = gpeArgs->count;
  const commDataType_t datatype = gpeArgs->datatype;

  CtranComm* comm = win->comm;
  const auto statex = comm->statex_.get();
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  const size_t sendSize = count * commTypeSize(datatype);

  auto mapper = comm->ctran_->mapper.get();

  CtranAlgoLogger logger("AllGatherWindowDirect", op->opCount, comm);

  // Count how many IB peers we have
  int ibPeerCount = 0;
  for (int p = 1; p < nRanks; p++) {
    const int peer = (rank + p) % nRanks;
    if (!useNvlinkForPeer(peer, win, statex)) {
      ibPeerCount++;
    }
  }

  // If no IB peers, nothing to do in GPE - all handled by NVLink + kernel
  if (ibPeerCount == 0) {
    return commSuccess;
  }

  // Get registration handle for local send buffer
  void* localMemHdl = nullptr;
  bool localReg = false;
  // Source is our slot in the recv buffer
  const void* mySlot =
      getPtr(win->winDataPtr, static_cast<size_t>(rank) * sendSize);
  FB_COMMCHECK(
      mapper->searchRegHandle(mySlot, sendSize, &localMemHdl, &localReg));
  auto guard = folly::makeGuard([localMemHdl, localReg, mapper]() {
    if (localReg) {
      FB_COMMCHECKIGNORE(mapper->deregDynamic(localMemHdl));
    }
  });

  // Vectors to track PUT requests and notify handles
  std::vector<std::unique_ptr<CtranMapperRequest>> putReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> signalReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> syncSendReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> syncRecvReqs;

  // Phase 1: Sync with IB peers to ensure they're ready
  for (int p = 1; p < nRanks; p++) {
    const int peer = (rank + p) % nRanks;
    if (!useNvlinkForPeer(peer, win, statex)) {
      CtranMapperRequest* sendReq = nullptr;
      CtranMapperRequest* recvReq = nullptr;
      FB_COMMCHECK(mapper->irecvCtrl(peer, &recvReq));
      FB_COMMCHECK(mapper->isendCtrl(peer, &sendReq));
      syncRecvReqs.push_back(std::unique_ptr<CtranMapperRequest>(recvReq));
      syncSendReqs.push_back(std::unique_ptr<CtranMapperRequest>(sendReq));
    }
  }
  for (auto& req : syncRecvReqs) {
    FB_COMMCHECK(mapper->waitRequest(req.get()));
  }
  for (auto& req : syncSendReqs) {
    FB_COMMCHECK(mapper->waitRequest(req.get()));
  }

  // Phase 2: Issue PUT operations to IB peers
  for (int p = 1; p < nRanks; p++) {
    const int peer = (rank + p) % nRanks;

    // Skip NVLink peers - already handled by nvlCeBcast in caller
    if (useNvlinkForPeer(peer, win, statex)) {
      continue;
    }

    // Calculate destination address in peer's window
    void* dstPtr = getPtr(
        win->remWinInfo[peer].dataAddr, static_cast<size_t>(rank) * sendSize);

    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->iput(
        mySlot,
        dstPtr,
        sendSize,
        peer,
        CtranMapperConfig{
            .memHdl_ = localMemHdl,
            .remoteAccessKey_ = win->remWinInfo[peer].dataRkey,
        },
        &req));
    putReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
  }

  // Phase 3: Wait for all local PUT requests to complete
  for (auto& req : putReqs) {
    FB_COMMCHECK(mapper->waitRequest(req.get()));
  }

  // Phase 4: Signal IB peers that our data has arrived
  for (int p = 1; p < nRanks; p++) {
    const int peer = (rank + p) % nRanks;

    if (useNvlinkForPeer(peer, win, statex)) {
      continue;
    }

    uint64_t signalVal = win->ctranNextSignalVal(peer);
    uint64_t* signalAddr = win->remWinInfo[peer].signalAddr + rank;

    CtranMapperRequest* req = nullptr;
    FB_COMMCHECK(mapper->atomicSet(
        signalAddr,
        signalVal,
        peer,
        CtranMapperConfig{.remoteAccessKey_ = win->remWinInfo[peer].signalRkey},
        req));
    signalReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
  }

  // Phase 5: Wait for all signal requests to complete
  for (auto& req : signalReqs) {
    FB_COMMCHECK(mapper->waitRequest(req.get()));
  }

  // Note: Waiting for signals FROM remote IB peers is NOT done here.
  // For GPU memory windows, the kernel will use cuStreamWaitValue64 or
  // spinning kernel to wait. For CPU memory windows, additional handling
  // would be needed but is not yet implemented.

  return commSuccess;
}

/**
 * Pipeline algorithm GPE function
 *
 * Ring-based pipeline: data flows through ranks in a ring pattern.
 * Each rank sends to downstream peer and receives from upstream peer.
 * Uses overlapping between inter-node PUT and intra-node NVLink broadcast.
 */
commResult_t pipelineGpeFn(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto* gpeArgs =
      reinterpret_cast<AllGatherWindowGpeArgs*>(op->allgatherWindow.args);
  CtranWin* win = gpeArgs->win;
  const size_t count = gpeArgs->count;
  const commDataType_t datatype = gpeArgs->datatype;
  Resource* resource = gpeArgs->resource;

  CtranComm* comm = win->comm;
  const auto statex = comm->statex_.get();
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  const int nLocalRanks = statex->nLocalRanks();
  const int nNodes = statex->nNodes();
  const size_t sendSize = count * commTypeSize(datatype);

  auto mapper = comm->ctran_->mapper.get();

  CtranAlgoLogger logger("AllGatherWindowPipeline", op->opCount, comm);

  // Ring topology: receive from upPeer, send to downPeer
  const int downPeer = (nRanks + rank + nLocalRanks) % nRanks;
  const int upPeer = (nRanks + rank - nLocalRanks) % nRanks;

  // Get registration handle for local data buffer (window data)
  void* localMemHdl = nullptr;
  bool localReg = false;
  FB_COMMCHECK(mapper->searchRegHandle(
      win->winDataPtr, win->dataBytes, &localMemHdl, &localReg));
  auto guard = folly::makeGuard([localMemHdl, localReg, mapper]() {
    if (localReg) {
      FB_COMMCHECKIGNORE(mapper->deregDynamic(localMemHdl));
    }
  });

  // Sync with peers before starting pipeline
  CtranMapperRequest syncSreq, syncRreq;
  FB_COMMCHECK(mapper->isendCtrl(upPeer, &syncSreq));
  FB_COMMCHECK(mapper->irecvCtrl(downPeer, &syncRreq));
  FB_COMMCHECK(mapper->waitRequest(&syncRreq));

  // Track PUT requests for each step
  std::vector<std::unique_ptr<CtranMapperRequest>> putReqs;
  std::vector<std::unique_ptr<CtranMapperRequest>> signalReqs;

  // Pipeline loop: nNodes - 1 steps for ring AllGather
  for (int step = 0; step < nNodes - 1; step++) {
    // Calculate which chunk we're sending/receiving in this step
    const size_t sendChunkIdx =
        getRecvChunkIdxInRail(rank, step, nLocalRanks, nRanks);
    const size_t offset = sendChunkIdx * sendSize;

    // First step: send our local chunk; subsequent steps: forward received
    // chunk
    const void* sendPtr = step == 0
        ? getPtr(win->winDataPtr, static_cast<size_t>(rank) * sendSize)
        : getPtr(win->winDataPtr, offset);

    // Issue PUT to downstream peer
    void* dstPtr = getPtr(win->remWinInfo[downPeer].dataAddr, offset);

    if (mapper->hasBackend(downPeer, CtranMapperBackend::IB)) {
      CtranMapperRequest* req = nullptr;
      FB_COMMCHECK(mapper->iput(
          sendPtr,
          dstPtr,
          sendSize,
          downPeer,
          CtranMapperConfig{
              .memHdl_ = localMemHdl,
              .remoteAccessKey_ = win->remWinInfo[downPeer].dataRkey,
          },
          &req));
      putReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
    }

    // Wait for data from upstream peer (previous step's PUT has arrived)
    // We wait on signal from upPeer
    const uint64_t* signalAddr = win->winSignalPtr + upPeer;
    uint64_t cmpVal = win->ctranNextWaitSignalVal(upPeer);

    // Spin-wait for signal
    const std::atomic<uint64_t>* atomicAddr =
        reinterpret_cast<const std::atomic<uint64_t>*>(signalAddr);
    while (std::atomic_load(atomicAddr) < cmpVal) {
      std::this_thread::yield();
    }

    // Signal downstream peer that we've completed this step's PUT
    uint64_t signalVal = win->ctranNextSignalVal(downPeer);
    uint64_t* downSignalAddr = win->remWinInfo[downPeer].signalAddr + rank;

    if (mapper->hasBackend(downPeer, CtranMapperBackend::IB)) {
      // Wait for PUT to complete before signaling
      if (!putReqs.empty()) {
        FB_COMMCHECK(mapper->waitRequest(putReqs.back().get()));
      }

      CtranMapperRequest* req = nullptr;
      FB_COMMCHECK(mapper->atomicSet(
          downSignalAddr,
          signalVal,
          downPeer,
          CtranMapperConfig{
              .remoteAccessKey_ = win->remWinInfo[downPeer].signalRkey},
          req));
      signalReqs.push_back(std::unique_ptr<CtranMapperRequest>(req));
    }

    // Notify kernel that this step is complete for intra-node broadcast
    if (resource->pipeSync != nullptr) {
      resource->pipeSync->post(step);
    }
  }

  // Wait for all remaining PUT and signal requests
  for (auto& req : putReqs) {
    FB_COMMCHECK(mapper->waitRequest(req.get()));
  }
  for (auto& req : signalReqs) {
    FB_COMMCHECK(mapper->waitRequest(req.get()));
  }

  // Wait for isendCtrl to complete
  FB_COMMCHECK(mapper->waitRequest(&syncSreq));

  return commSuccess;
}

/**
 * Intra-node NVLink broadcast helper
 *
 * Copies data from this rank's buffer to the same offset in all local peers'
 * buffers. Uses NVLink CE copy for efficiency.
 */
commResult_t nvlCeBcast(
    CtranComm* comm,
    CtranWin* win,
    const void* sendBuff,
    const size_t sendSize,
    const size_t recvOffset,
    cudaStream_t stream) {
  const auto statex = comm->statex_.get();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto localRank = statex->localRank();

  for (int i = 1; i < nLocalRanks; i++) {
    const int localPeer = (localRank + i) % nLocalRanks;
    const int globalPeer = statex->localRankToRank(localPeer);

    // Copy to peer's window at the specified offset
    void* dstPtr = getPtr(win->remWinInfo[globalPeer].dataAddr, recvOffset);
    FB_CUDACHECK(cudaMemcpyAsync(
        dstPtr, sendBuff, sendSize, cudaMemcpyDeviceToDevice, stream));
  }

  return commSuccess;
}

} // namespace

/**
 * Window-based AllGather using Direct algorithm
 *
 * All-to-all PUT: each rank sends its data to all other ranks simultaneously.
 * Best for small-medium message sizes and smaller rank counts.
 */
commResult_t ctranAllGatherWindowDirect(
    const void* sendbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranWin* win,
    cudaStream_t stream) {
  CtranComm* comm = win->comm;
  auto ctran = comm->ctran_.get();
  const auto statex = comm->statex_.get();
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  const size_t sendSize = sendcount * commTypeSize(datatype);
  const auto opCount = ctran->getOpCount();

  // The recv buffer is the window's data buffer
  void* recvbuff = win->winDataPtr;

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "ctranAllGatherWindowDirect: sendbuff {} recvbuff {} sendcount {} "
      "datatype {} win {} comm {} commHash {:x} [nranks={}, localRanks={}] "
      "stream={}",
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      (void*)win,
      (void*)comm,
      statex->commHash(),
      nRanks,
      statex->nLocalRanks(),
      (void*)stream);

  // Handle trivial case
  if (nRanks == 1) {
    char* mySlot = static_cast<char*>(recvbuff);
    if (sendbuff != mySlot) {
      FB_CUDACHECK(cudaMemcpyAsync(
          mySlot, sendbuff, sendSize, cudaMemcpyDefault, stream));
    }
    return commSuccess;
  }

  // Copy local data to our slot in the recv buffer
  char* mySlot = static_cast<char*>(recvbuff) + rank * sendSize;
  if (sendbuff != mySlot) {
    FB_CUDACHECK(
        cudaMemcpyAsync(mySlot, sendbuff, sendSize, cudaMemcpyDefault, stream));
  }

  // Copy to local NVLink peers
  FB_COMMCHECK(
      nvlCeBcast(comm, win, mySlot, sendSize, rank * sendSize, stream));

  // Prepare GPE args
  AllGatherWindowGpeArgs gpeArgs = {
      .win = win,
      .sendbuff = sendbuff,
      .count = sendcount,
      .datatype = datatype,
      .resource = nullptr,
  };

  // Create operation for GPE submission
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLGATHERWINDOW, stream, comm, opCount);
  op->allgatherWindow.args = &gpeArgs;

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.push_back(std::move(op));

  // Kernel config
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERWINDOW,
      stream,
      "AllGatherWindowDirect",
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = ctran->algo->getDevState();

  // Submit to GPE
  FB_COMMCHECK(ctran->gpe->submit(
      std::move(opGroup),
      directGpeFn,
      config,
      reinterpret_cast<void*>(ncclKernelAllGatherWindowDirect)));

  // After GPE completes PUT and signal, kernel waits for remote signals
  // This is done via hardware wait in the kernel

  return commSuccess;
}

/**
 * Window-based AllGather using Pipeline algorithm
 *
 * Ring-based pipeline: data flows through ranks in a ring pattern.
 * Best for large message sizes as it overlaps inter-node and intra-node.
 */
commResult_t ctranAllGatherWindowPipeline(
    const void* sendbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranWin* win,
    Resource& resource,
    cudaStream_t stream) {
  CtranComm* comm = win->comm;
  auto ctran = comm->ctran_.get();
  const auto statex = comm->statex_.get();
  const int rank = statex->rank();
  const int nRanks = statex->nRanks();
  const int nLocalRanks = statex->nLocalRanks();
  const int nNodes = statex->nNodes();
  const size_t sendSize = sendcount * commTypeSize(datatype);
  const auto opCount = ctran->getOpCount();

  // The recv buffer is the window's data buffer
  void* recvbuff = win->winDataPtr;

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "ctranAllGatherWindowPipeline: sendbuff {} recvbuff {} sendcount {} "
      "datatype {} win {} comm {} commHash {:x} [nranks={}, localRanks={}, "
      "nNodes={}] stream={}",
      sendbuff,
      recvbuff,
      sendcount,
      datatype,
      (void*)win,
      (void*)comm,
      statex->commHash(),
      nRanks,
      nLocalRanks,
      nNodes,
      (void*)stream);

  // Handle trivial case
  if (nRanks == 1) {
    char* mySlot = static_cast<char*>(recvbuff);
    if (sendbuff != mySlot) {
      FB_CUDACHECK(cudaMemcpyAsync(
          mySlot, sendbuff, sendSize, cudaMemcpyDefault, stream));
    }
    return commSuccess;
  }

  // Kernel config
  auto config = KernelConfig(
      KernelConfig::KernelType::ALLGATHERWINDOW,
      stream,
      "AllGatherWindowPipeline",
      opCount);
  config.numBlocks = 1;
  config.numThreads = 1;
  config.args.devState_d = ctran->algo->getDevState();

  // Submit inter-node ring pipeline to GPE (if multi-node)
  if (nNodes > 1) {
    AllGatherWindowGpeArgs gpeArgs = {
        .win = win,
        .sendbuff = sendbuff,
        .count = sendcount,
        .datatype = datatype,
        .resource = &resource,
    };

    auto op = std::make_unique<OpElem>(
        OpElem::opType::ALLGATHERWINDOW, stream, comm, opCount);
    op->allgatherWindow.args = &gpeArgs;

    std::vector<std::unique_ptr<struct OpElem>> opGroup;
    opGroup.push_back(std::move(op));

    if (nLocalRanks > 1) {
      // Multi-local-rank: use start kernel to kick off GPE, then overlap
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          pipelineGpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherWindowPipeStart)));
    } else {
      // Single local rank: blocking kernel until GPE completes
      FB_COMMCHECK(ctran->gpe->submit(
          std::move(opGroup),
          pipelineGpeFn,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherWindowPipe)));
    }
  }

  // Copy local data to our slot
  char* mySlot = static_cast<char*>(recvbuff) + rank * sendSize;
  if (sendbuff != mySlot) {
    FB_CUDACHECK(
        cudaMemcpyAsync(mySlot, sendbuff, sendSize, cudaMemcpyDefault, stream));
  }

  // Intra-node pipeline: broadcast local chunk, then received chunks
  if (nLocalRanks > 1) {
    const int upPeer = (nRanks + rank - nLocalRanks) & (nRanks - 1);

    // Step 0: Broadcast our local chunk to local peers
    FB_COMMCHECK(
        nvlCeBcast(comm, win, mySlot, sendSize, rank * sendSize, stream));

    // Steps 1 to nNodes-1: Broadcast received chunks from inter-node ring
    for (int step = 0; step < nNodes - 1; step++) {
      // Wait for GPE to complete this step
      PipeSyncKernArgs kernArgs = {
          .stepId = step,
          .pipeSync = resource.pipeSync,
      };
      config.algoArgs = reinterpret_cast<void*>(&kernArgs);
      FB_COMMCHECK(ctran->gpe->submit(
          {},
          nullptr,
          config,
          reinterpret_cast<void*>(ncclKernelAllGatherWindowPipeSync)));

      // Calculate offset of received chunk
      const size_t chunkIdx =
          getRecvChunkIdxInRail(upPeer, step, nLocalRanks, nRanks);
      const size_t offset = chunkIdx * sendSize;
      const void* sendPtr = getPtr(recvbuff, offset);

      // Broadcast received chunk to local peers
      FB_COMMCHECK(nvlCeBcast(comm, win, sendPtr, sendSize, offset, stream));
    }

    // Pipeline end: reset sync flags
    PipeEndKernArgs endKernArgs = {
        .pipeSync = resource.pipeSync,
    };
    config.algoArgs = reinterpret_cast<void*>(&endKernArgs);
    FB_COMMCHECK(ctran->gpe->submit(
        {},
        nullptr,
        config,
        reinterpret_cast<void*>(ncclKernelAllGatherWindowPipeEnd)));
  }

  return commSuccess;
}

/**
 * Main entry point for Window-based AllGather
 *
 * Automatically selects between Direct and Pipeline algorithms based on
 * message size and cluster topology.
 */
commResult_t ctranAllGatherWindow(
    const void* sendbuff,
    size_t sendcount,
    commDataType_t datatype,
    CtranWin* win,
    cudaStream_t stream) {
  CtranComm* comm = win->comm;
  const auto statex = comm->statex_.get();
  const int nNodes = statex->nNodes();
  const size_t sendSize = sendcount * commTypeSize(datatype);

  // Heuristic: Use Pipeline for multi-node with large messages
  // Use Direct for small messages or single-node
  const size_t pipelineThreshold = 1024 * 1024; // 1MB

  if (nNodes > 1 && sendSize >= pipelineThreshold) {
    // TODO: Initialize resource with proper GpeKernelSync allocation
    Resource resource;
    // For now, allocate pipeSync from pool (not implemented here)
    // resource.pipeSync = comm->ctran_->algo->allocGpeKernelSync();
    return ctranAllGatherWindowPipeline(
        sendbuff, sendcount, datatype, win, resource, stream);
  } else {
    return ctranAllGatherWindowDirect(
        sendbuff, sendcount, datatype, win, stream);
  }
}
