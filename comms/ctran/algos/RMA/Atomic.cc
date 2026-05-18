// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>

#include "Types.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran;
using ::meta::comms::colltrace::CollTraceHandleTriggerState;

extern __global__ void ncclKernelFetchAdd(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelFetchAddArgs args);

static commResult_t fetchAddImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto comm = op->comm_;
  auto win = op->fetchadd.win;
  int peerRank = op->fetchadd.peerRank;

  CtranAlgoRMALogger logger("fetchAdd", op->opCount, peerRank, win, comm);

  if (!comm->ctran_->mapper->hasBackend(peerRank, CtranMapperBackend::IB)) {
    CLOGF(ERR, "fetchAdd only supports IB backend for now");
    return commInternalError;
  }

  const size_t targetOffsetBytes = op->fetchadd.targetIndex * sizeof(uint64_t);
  void* remoteAddr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(win->remWinInfo[peerRank].dataAddr) +
      targetOffsetBytes);

  void* localMemHdl = nullptr;
  bool localReg = false;
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      op->fetchadd.resultBuf, sizeof(uint64_t), &localMemHdl, &localReg));

  CLOGF_TRACE(
      COLL,
      "fetchAddImpl: resultBuf {}, remoteAddr {} (base {} + offset {}), addVal {}",
      op->fetchadd.resultBuf,
      remoteAddr,
      win->remWinInfo[peerRank].dataAddr,
      targetOffsetBytes,
      op->fetchadd.addVal);

  auto fetchAddReq = std::make_unique<CtranMapperRequest>();
  FB_COMMCHECK(comm->ctran_->mapper->ifetchAndAdd(
      op->fetchadd.resultBuf,
      remoteAddr,
      op->fetchadd.addVal,
      peerRank,
      CtranMapperConfig{
          .memHdl_ = localMemHdl,
          .remoteAccessKey_ = win->remWinInfo[peerRank].dataRkey,
      },
      fetchAddReq.get()));

  FB_COMMCHECK(comm->ctran_->mapper->waitRequest(fetchAddReq.get()));

  if (localReg) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(localMemHdl));
  }
  return commSuccess;
}

commResult_t ctranFetchAdd(
    void* resultBuf,
    uint64_t addVal,
    size_t targetIndex,
    int peer,
    CtranWin* win,
    CtranComm* comm,
    cudaStream_t stream) {
  const auto winOpCount = win->updateOpCount(peer);
  const auto fetchAddOpCount =
      win->updateOpCount(peer, window::OpCountType::kFetchAdd);
  auto statex = comm->statex_.get();

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "CTRAN-RMA ctranFetchAdd: opCount {} winOpCount {} resultBuf {} addVal {} "
      "targetIndex {} rank {} peer {} win {} stream={}",
      fetchAddOpCount,
      winOpCount,
      resultBuf,
      addVal,
      targetIndex,
      statex->rank(),
      peer,
      (void*)win,
      (void*)stream);

  if (!win->isAtomicCapable()) {
    CLOGF(
        ERR,
        "ctranFetchAdd requires an atomic_capable window (data buffer must be 8-byte aligned and size must be a multiple of 8 bytes)");
    return commInvalidArgument;
  }

  auto remoteDataAddr =
      reinterpret_cast<uintptr_t>(win->remWinInfo[peer].dataAddr);
  if (remoteDataAddr % sizeof(uint64_t) != 0) {
    CLOGF(
        ERR,
        "Remote window data buffer {} is not 8-byte aligned for peer {}",
        remoteDataAddr,
        peer);
    return commInvalidArgument;
  }

  size_t peerWinSize = win->getDataSize(peer);
  if (peerWinSize % sizeof(uint64_t) != 0) {
    CLOGF(
        ERR,
        "Peer {}'s window size {} is not a multiple of 8 bytes, not suitable for atomic operations",
        peer,
        peerWinSize);
    return commInvalidArgument;
  }

  size_t targetOffsetBytes = targetIndex * sizeof(uint64_t);
  if ((targetOffsetBytes + sizeof(uint64_t)) > peerWinSize) {
    CLOGF(
        ERR,
        "Invalid targetIndex {}: offset {} + 8 bytes exceeds peer {}'s window size {}",
        targetIndex,
        targetOffsetBytes,
        peer,
        peerWinSize);
    return commInvalidArgument;
  }

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::FETCHADD, stream, "FetchAdd", fetchAddOpCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;

  if (statex->node(peer) == statex->node() && win->nvlEnabled(peer)) {
    uint64_t* remoteAddr = reinterpret_cast<uint64_t*>(
        reinterpret_cast<uintptr_t>(win->remWinInfo[peer].dataAddr) +
        targetOffsetBytes);

    auto colltraceHandle = meta::comms::colltrace::getCollTraceHandleRMA(
        comm, opGroup, config, true);
    colltraceHandle->trigger(CollTraceHandleTriggerState::BeforeEnqueueKernel);

    CtranKernelFetchAddArgs kernArgs{
        .remoteAddr = remoteAddr,
        .addVal = addVal,
        .resultAddr = reinterpret_cast<uint64_t*>(resultBuf),
    };
    config.algoArgs = reinterpret_cast<void*>(&kernArgs);

    FB_COMMCHECK(comm->ctran_->gpe->submit(
        std::move(opGroup),
        fetchAddImpl,
        config,
        reinterpret_cast<void*>(ncclKernelFetchAdd)));

    colltraceHandle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    return commSuccess;
  }

  // IB path: GPE callback does the work; kernel only coordinates with GPE.
  // Set remoteAddr to nullptr so the kernel skips the NVL atomic.
  CtranKernelFetchAddArgs kernArgs{};
  config.algoArgs = reinterpret_cast<void*>(&kernArgs);

  struct OpElem* op =
      new struct OpElem(OpElem::opType::FETCHADD, comm, fetchAddOpCount);
  op->fetchadd.resultBuf = resultBuf;
  op->fetchadd.targetIndex = targetIndex;
  op->fetchadd.addVal = addVal;
  op->fetchadd.peerRank = peer;
  op->fetchadd.win = win;
  opGroup.push_back(std::unique_ptr<struct OpElem>(op));

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      fetchAddImpl,
      config,
      reinterpret_cast<void*>(ncclKernelFetchAdd)));
  return commSuccess;
}
