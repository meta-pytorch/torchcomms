// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>
#include <vector>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/RMA/AllGather/AllGatherWindowTypes.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran;
using namespace ctran::allgatherwindow;

namespace ctran::allgatherwindow {
extern __global__ void ncclKernelAllGatherWindowDirect(
    int* flag,
    CtranAlgoDeviceState* devState);
} // namespace ctran::allgatherwindow

namespace {

/**
 * Direct algorithm GPE function
 *
 * All-to-all PUT: each rank sends its data to all other ranks simultaneously.
 * Uses mapper->iput() for data transfer and mapper->atomicSet() for signaling.
 * Each peer is signaled immediately after its PUT completes.
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
  // Take ownership of heap-allocated args (freed when scope exits)
  std::unique_ptr<AllGatherWindowGpeArgs> gpeArgsOwner(gpeArgs);
  CtranWin* win = gpeArgs->win;
  const size_t count = gpeArgs->count;
  const commDataType_t datatype = gpeArgs->datatype;

  CtranComm* comm = win->comm;
  const auto statex = comm->statex_.get();
  const int rank = statex->rank();
  const size_t sendSize = count * commTypeSize(datatype);
  const auto& ibPeers = win->ibPeers;

  auto mapper = comm->ctran_->mapper.get();

  CtranAlgoLogger logger("AllGatherWindowDirect", op->opCount, comm);

  // If no IB peers, nothing to do in GPE - all handled by NVLink + kernel
  if (ibPeers.empty()) {
    return commSuccess;
  }

  // Use the window's pre-registered data handle (registered at window
  // construction) instead of searching per request.
  void* localMemHdl = win->dataRegHdl;

  // Source is our slot in the recv buffer
  const void* mySlot =
      getPtr(win->winDataPtr, static_cast<size_t>(rank) * sendSize);

  // Issue all PUT operations to precomputed IB peers
  std::vector<CtranMapperRequest> putReqs(ibPeers.size());
  for (size_t i = 0; i < ibPeers.size(); i++) {
    const int peer = ibPeers[i];
    void* dstPtr = getPtr(
        win->remWinInfo[peer].dataAddr, static_cast<size_t>(rank) * sendSize);

    FB_COMMCHECK(mapper->iput(
        mySlot,
        dstPtr,
        sendSize,
        peer,
        CtranMapperConfig{
            .memHdl_ = localMemHdl,
            .remoteAccessKey_ = win->remWinInfo[peer].dataRkey,
        },
        &putReqs[i]));
  }

  // Wait for all PUTs to complete
  FB_COMMCHECK(mapper->waitAllRequests(putReqs));

  // Signal all peers that data has arrived
  std::vector<CtranMapperRequest> signalReqs(ibPeers.size());
  for (size_t i = 0; i < ibPeers.size(); i++) {
    const int peer = ibPeers[i];
    uint64_t signalVal = win->ctranNextSignalVal(peer);
    uint64_t* signalAddr = win->remWinInfo[peer].signalAddr + rank;

    FB_COMMCHECK(mapper->atomicSet(
        signalAddr,
        signalVal,
        peer,
        CtranMapperConfig{.remoteAccessKey_ = win->remWinInfo[peer].signalRkey},
        &signalReqs[i]));
  }

  // Wait for all signals to complete
  FB_COMMCHECK(mapper->waitAllRequests(signalReqs));

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

    // Skip peers without NVLink (e.g., IB-only mode)
    if (!win->nvlEnabled(globalPeer)) {
      continue;
    }

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
 * IB peers: PUT via RDMA with atomic signaling.
 * NVLink peers: CE copy with kernel barrier synchronization.
 */
commResult_t ctranAllGatherWindow(
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
      "ctranAllGatherWindow: sendbuff {} recvbuff {} sendcount {} "
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

  // Prepare GPE args (heap-allocated, ownership transferred to GPE function)
  auto* gpeArgs = new AllGatherWindowGpeArgs{
      .win = win,
      .count = sendcount,
      .datatype = datatype,
  };

  // Create operation for GPE submission
  auto op = std::make_unique<OpElem>(
      OpElem::opType::ALLGATHERWINDOW, stream, comm, opCount);
  op->allgatherWindow.args = gpeArgs;

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

  return commSuccess;
}
