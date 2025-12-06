// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>

#include "Types.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/tracing/CollTraceWrapper.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran;
using meta::comms::colltrace::CollTraceHandleTriggerState;

extern __global__ void ncclKernelPutNotify(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelPutNotifyArgs args);

extern __global__ void ncclKernelWaitNotify(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelWaitNotifyArgs args);

extern __global__ void ncclKernelPutSignal(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelPutSignalArgs args);

extern __global__ void ncclKernelPut(int* flag, CtranAlgoDeviceState* devState);

extern __global__ void ncclKernelWaitSignal(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelWaitSignalArgs args);

extern __global__ void ncclKernelSignal(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelSignalArgs args);

// Helper function to check if displacement exceeds window bounds
inline static commResult_t checkDisplacementBounds(
    size_t disp,
    size_t elemSize,
    size_t elemCount,
    size_t winSize) {
  size_t dispNbytes = disp * elemSize;
  size_t totalBytes = dispNbytes + (elemSize * elemCount);

  if (totalBytes > winSize) {
    CLOGF(
        ERR,
        "Invalid displacement from {} bytes to {} bytes exceeding the window size {}",
        dispNbytes,
        totalBytes,
        winSize);
    return commInvalidArgument;
  }
  return commSuccess;
}

inline static commResult_t checkSignalDisplacement(
    size_t disp,
    size_t signalSize) {
  if (disp > signalSize) {
    CLOGF(
        ERR,
        "Invalid displacement {} exceeding the signal buffer size {}",
        disp,
        signalSize);
    return commInvalidArgument;
  }
  return commSuccess;
}

static commResult_t putNotifyImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto comm = op->comm_;
  auto win = op->putnotify.win;
  int peerRank = op->putnotify.peerRank;

  CtranAlgoRMALogger logger(
      op->putnotify.notify ? "ctranPutSignal" : "ctranPut",
      op->opCount,
      peerRank,
      win,
      comm);

  // If NVL is not enabled, the IB backend must be available
  if (!win->nvlEnabled(peerRank) &&
      !comm->ctran_->mapper->hasBackend(peerRank, CtranMapperBackend::IB)) {
    CLOGF(ERR, "Put notify doesn't have IB backend");
    return commInternalError;
  }

  size_t putSize = op->putnotify.count * commTypeSize(op->putnotify.datatype);
  size_t targetDispNbytes =
      op->putnotify.targetDisp * commTypeSize(op->putnotify.datatype);
  void* dstPtr = reinterpret_cast<void*>(
      reinterpret_cast<size_t>(win->remWinInfo[peerRank].dataAddr) +
      targetDispNbytes);

  // Get registration handle for local send buffer
  void* localMemHdl = nullptr;
  bool localReg = false;
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      op->putnotify.sendbuff, putSize, &localMemHdl, &localReg));

  CLOGF_TRACE(
      COLL,
      "putNotifyImpl: sbuf {}, rbuf {} (base {} + offset {}), size {}",
      op->putnotify.sendbuff,
      dstPtr,
      win->remWinInfo[peerRank].dataAddr,
      targetDispNbytes,
      putSize);

  CtranMapperRequest* req = nullptr;
  FB_COMMCHECK(comm->ctran_->mapper->iput(
      op->putnotify.sendbuff,
      dstPtr,
      putSize,
      peerRank,
      CtranMapperConfig{
          .memHdl_ = localMemHdl,
          .remoteAccessKey_ = win->remWinInfo[peerRank].dataRkey,
          .notify_ = op->putnotify.notify,
      },
      &req));

  auto putReq = std::unique_ptr<CtranMapperRequest>(req);
  FB_COMMCHECK(comm->ctran_->mapper->waitRequest(putReq.get()));

  // Deregister the sendbuffer if it is automatically registered by the
  // mapper->searchRegHandle()
  if (localReg) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(localMemHdl));
  }
  return commSuccess;
}

static commResult_t waitNotifyImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  int peerRank = op->waitnotify.peerRank;
  auto comm = op->comm_;
  auto win = op->waitnotify.win;
  CtranAlgoRMALogger logger(
      "ctranWaitNotify", op->opCount, peerRank, win, comm);

  if (!win->nvlEnabled(peerRank) &&
      !comm->ctran_->mapper->hasBackend(peerRank, CtranMapperBackend::IB)) {
    CLOGF(ERR, "Wait notify doesn't have IB backend");
    return commInternalError;
  }

  CLOGF_TRACE(COLL, "waitNotifyImpl: peerRank={}", op->waitnotify.peerRank);

  std::unique_ptr<CtranMapperNotify> notify =
      std::make_unique<CtranMapperNotify>();
  // we don't have the local RegElem so we cannot call mapper's initNotify,
  // directly update the notify object instead
  notify->update(peerRank, nullptr, CtranMapperBackend::IB);
  FB_COMMCHECK(comm->ctran_->mapper->waitNotify(notify.get()));
  return commSuccess;
}

static commResult_t putSignalImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto comm = op->comm_;
  auto win = op->putsignal.win;
  int peerRank = op->putsignal.peerRank;

  CtranAlgoRMALogger logger(
      op->putsignal.signalAddr != nullptr ? "ctranPutSignal" : "ctranPut",
      op->opCount,
      peerRank,
      win,
      comm);

  // The IB backend must be available
  if (!comm->ctran_->mapper->hasBackend(peerRank, CtranMapperBackend::IB)) {
    CLOGF(ERR, "Put signal doesn't have IB backend");
    return commInternalError;
  }

  size_t putSize = op->putsignal.count * commTypeSize(op->putsignal.datatype);
  size_t targetDispNbytes =
      op->putsignal.targetDisp * commTypeSize(op->putsignal.datatype);
  void* dstPtr = reinterpret_cast<void*>(
      reinterpret_cast<size_t>(win->remWinInfo[peerRank].dataAddr) +
      targetDispNbytes);

  // Get registration handle for local send buffer
  void* localMemHdl = nullptr;
  bool localReg = false;
  FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
      op->putsignal.sendbuff, putSize, &localMemHdl, &localReg));

  CLOGF_TRACE(
      COLL,
      "putSignalImpl: sbuf {}, rbuf {} (base {} + offset {}), size {}, signalAddr {} signalVal {}",
      op->putsignal.sendbuff,
      dstPtr,
      win->remWinInfo[peerRank].dataAddr,
      targetDispNbytes,
      putSize,
      (void*)op->putsignal.signalAddr,
      op->putsignal.signalVal);

  CtranMapperRequest* req = nullptr;

  FB_COMMCHECK(comm->ctran_->mapper->iput(
      op->putsignal.sendbuff,
      dstPtr,
      putSize,
      peerRank,
      CtranMapperConfig{
          .memHdl_ = localMemHdl,
          .remoteAccessKey_ = win->remWinInfo[peerRank].dataRkey,
      },
      &req));

  auto putReq = std::unique_ptr<CtranMapperRequest>(req);
  FB_COMMCHECK(comm->ctran_->mapper->waitRequest(putReq.get()));

  CtranMapperRequest signalReq = CtranMapperRequest();
  if (op->putsignal.signalAddr != nullptr) {
    // flush the iput to make sure the signal is sent after the data
    FB_COMMCHECK(comm->ctran_->mapper->atomicSet(
        op->putsignal.signalAddr,
        op->putsignal.signalVal,
        peerRank,
        CtranMapperConfig{
            .remoteAccessKey_ = win->remWinInfo[peerRank].signalRkey},
        &signalReq));
    FB_COMMCHECK(comm->ctran_->mapper->waitRequest(&signalReq));
  }

  // Deregister the sendbuffer if it is automatically registered by the
  // mapper->searchRegHandle()
  if (localReg) {
    FB_COMMCHECK(comm->ctran_->mapper->deregDynamic(localMemHdl));
  }
  return commSuccess;
}
static commResult_t signalImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto comm = op->comm_;
  auto win = op->signal.win;
  int peerRank = op->signal.peerRank;

  CtranAlgoRMALogger logger("ctranSignal", op->opCount, peerRank, win, comm);

  // The IB backend must be available
  if (!comm->ctran_->mapper->hasBackend(peerRank, CtranMapperBackend::IB)) {
    CLOGF(ERR, "Signal doesn't have IB backend");
    return commInternalError;
  }

  CLOGF_TRACE(
      COLL,
      "signalImpl: peer {} signalAddr {} signalVal {}",
      op->signal.peerRank,
      (void*)op->signal.signalAddr,
      op->signal.signalVal);

  CtranMapperRequest signalReq = CtranMapperRequest();
  FB_COMMCHECK(comm->ctran_->mapper->atomicSet(
      op->signal.signalAddr,
      op->signal.signalVal,
      peerRank,
      CtranMapperConfig{
          .remoteAccessKey_ = win->remWinInfo[peerRank].signalRkey},
      &signalReq));
  FB_COMMCHECK(comm->ctran_->mapper->waitRequest(&signalReq));

  return commSuccess;
}

static commResult_t waitSignalImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto comm = op->comm_;
  auto win = op->waitsignal.win;
  const std::atomic<uint64_t>* addr =
      reinterpret_cast<const std::atomic<uint64_t>*>(op->waitsignal.signalAddr);
  CtranAlgoRMALogger logger("ctranWaitSignal", op->opCount, -1, win, comm);
  CLOGF_TRACE(
      COLL,
      "waitSignalImpl: signalAddr {}, cmpVal={}",
      (void*)const_cast<uint64_t*>(op->waitsignal.signalAddr),
      op->waitsignal.cmpVal);
  if (op->waitsignal.cmpOp == commCmpEQ) {
    while (std::atomic_load(addr) != op->waitsignal.cmpVal) {
      std::this_thread::yield();
    };
  } else if (op->waitsignal.cmpOp == commCmpGE) {
    while (std::atomic_load(addr) < op->waitsignal.cmpVal) {
      std::this_thread::yield();
    };
  } else if (op->waitsignal.cmpOp == commCmpLE) {
    while (std::atomic_load(addr) > op->waitsignal.cmpVal) {
      std::this_thread::yield();
    };
  } else {
    CLOGF(ERR, "waitSignalImpl: invalid cmpOp {}", op->waitsignal.cmpOp);
    return commInvalidArgument;
  }

  return commSuccess;
}

commResult_t ctranPutSignal(
    const void* originBuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    size_t targetDisp,
    CtranWin* win,
    cudaStream_t stream,
    bool signal) {
  CtranComm* comm = win->comm;
  const auto winOpCount = win->updateOpCount(peer);
  const auto putOpCount = win->updateOpCount(peer, window::OpCountType::kPut);
  auto statex = comm->statex_.get();
  CTRAN_RMA_INFO(
      signal ? "ctranPutSignal" : "ctranPut",
      putOpCount,
      winOpCount,
      originBuff,
      targetDisp,
      count,
      datatype,
      statex->rank(),
      peer,
      win,
      comm,
      stream);

  // Check if the target displacement exceeds the window size
  FB_COMMCHECK(checkDisplacementBounds(
      targetDisp, commTypeSize(datatype), count, win->dataBytes));
  size_t targetDispNbytes = targetDisp * commTypeSize(datatype);
  size_t countNbytes = count * commTypeSize(datatype);
  uint64_t* signalAddr = nullptr;
  uint64_t signalVal = 0;
  if (signal) {
    signalVal = win->ctranNextSignalVal(peer);
    signalAddr = win->remWinInfo[peer].signalAddr + statex->rank();
  }

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::PUTSIGNAL, stream, "PutSignal", putOpCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.clear();

  // Use direct copy if peer is on the same host and has NVL enabled.
  // Otherwise, do put & signal via network
  CtranKernelPutSignalArgs kernArgs = {.signalAddr = nullptr, .signalVal = 0};
  if (statex->node(peer) == statex->node() && win->nvlEnabled(peer)) {
    // Single-node direct cudaMemcpy
    if (count > 0) {
      void* dstPtr = reinterpret_cast<void*>(
          reinterpret_cast<size_t>(win->remWinInfo[peer].dataAddr) +
          targetDispNbytes);
      // CollTrace tracing logic for local + no signal case. In this case the
      // put will not trigger gpe->submit, so we need to record manually in the
      // algo. In other cases, this handle would be a no-op and the tracing
      // will take place in the gpe function.
      auto colltraceHandle = meta::comms::colltrace::getCollTraceHandleRMA(
          comm, opGroup, config, !signal);
      colltraceHandle->trigger(
          CollTraceHandleTriggerState::BeforeEnqueueKernel);

      FB_CUDACHECK(cudaMemcpyAsync(
          dstPtr, originBuff, countNbytes, cudaMemcpyDefault, stream));

      colltraceHandle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    }
    if (signal) {
      // Use atomic_store to signal the remote peer
      kernArgs.signalAddr = signalAddr;
      kernArgs.signalVal = signalVal;
    } else {
      // No signal, just return
      return commSuccess;
    }
  } else {
    // X-node put & signal via network
    // Create an op for GPE thread to complete put and signal
    struct OpElem* op =
        new struct OpElem(OpElem::opType::PUTSIGNAL, comm, putOpCount);
    op->putsignal.sendbuff = originBuff;
    op->putsignal.targetDisp = targetDisp;
    op->putsignal.count = count;
    op->putsignal.datatype = datatype;
    op->putsignal.signalAddr = signalAddr;
    op->putsignal.signalVal = signalVal;
    op->putsignal.peerRank = peer;
    op->putsignal.win = win;
    opGroup.push_back(std::unique_ptr<struct OpElem>(op));
  }

  void* func = nullptr;
  if (signal) {
    func = reinterpret_cast<void*>(ncclKernelPutSignal);
    config.algoArgs = reinterpret_cast<void*>(&kernArgs);
  } else {
    func = reinterpret_cast<void*>(ncclKernelPut);
    config.algoArgs = nullptr;
  }
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup), putSignalImpl, config, func));

  return commSuccess;
}

commResult_t ctranWaitSignal(int peer, CtranWin* win, cudaStream_t stream) {
  CtranComm* comm = win->comm;
  auto statex = comm->statex_.get();
  auto waitSignalVal = win->ctranNextWaitSignalVal(peer);
  const auto winOpCount = win->updateOpCount(peer);
  const auto waitOpCount =
      win->updateOpCount(peer, window::OpCountType::kWaitSignal);
  CTRAN_RMA_INFO(
      "ctranWaitSignal",
      waitOpCount,
      winOpCount,
      nullptr,
      peer, // signal disp
      size_t(1), // signal count
      commDataType_t::commUint64, // signal datatype
      statex->rank(),
      peer,
      win,
      comm,
      stream);

  const uint64_t* signalAddr = win->winSignalPtr + peer;

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::WAITSIGNAL, stream, "WaitSignal", waitOpCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.clear();

  CtranKernelWaitSignalArgs kernArgs = {
      .signalAddr = nullptr, .cmpVal = 0, .cmpOp = commCmpOp_t::commCmpGE};
  config.algoArgs = reinterpret_cast<void*>(&kernArgs);
  if (win->isGpuMem()) {
    // if GPU memory, use kernel to atomic wait on the local signal value update
    // from remote rank, either from NVL or IB
    kernArgs.signalAddr = const_cast<uint64_t*>(signalAddr);
    kernArgs.cmpVal = waitSignalVal;
  } else {
    // if CPU memory, GPE thread to atomic wait for update from remote rank via
    // IB.
    struct OpElem* op =
        new struct OpElem(OpElem::opType::WAITSIGNAL, comm, waitOpCount);
    op->waitsignal.win = win;
    op->waitsignal.signalAddr = signalAddr;
    op->waitsignal.cmpVal = waitSignalVal;
    op->waitsignal.cmpOp = commCmpOp_t::commCmpGE;
    opGroup.push_back(std::unique_ptr<struct OpElem>(op));
  }

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      waitSignalImpl,
      config,
      reinterpret_cast<void*>(ncclKernelWaitSignal)));
  return commSuccess;
}

commResult_t ctranPutSignal_v2(
    const void* originBuff,
    size_t targetDisp,
    size_t count,
    commDataType_t datatype,
    size_t signalDisp,
    uint64_t signalVal,
    int peer,
    CtranWin* win,
    cudaStream_t stream,
    bool signal) {
  CtranComm* comm = win->comm;
  const auto winOpCount = win->updateOpCount(peer);
  const auto putOpCount = win->updateOpCount(peer, window::OpCountType::kPut);
  auto statex = comm->statex_.get();
  CTRAN_RMA_INFO(
      signal ? "ctranPutSignal_v2" : "ctranPut_v2",
      putOpCount,
      winOpCount,
      originBuff,
      targetDisp,
      count,
      datatype,
      statex->rank(),
      peer,
      win,
      comm,
      stream);

  // Check if the target displacement exceeds the window size
  FB_COMMCHECK(checkDisplacementBounds(
      targetDisp, commTypeSize(datatype), count, win->dataBytes));
  size_t targetDispNbytes = targetDisp * commTypeSize(datatype);
  size_t countNbytes = count * commTypeSize(datatype);
  uint64_t* signalAddr = nullptr;
  if (signal) {
    FB_COMMCHECK(checkSignalDisplacement(signalDisp, win->signalSize));
    signalAddr = win->remWinInfo[peer].signalAddr + signalDisp;
  }

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::PUTSIGNAL, stream, "PutSignal", putOpCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.clear();

  // Use direct copy if peer is on the same host and has NVL enabled.
  // Otherwise, do put & signal via IB
  CtranKernelPutSignalArgs kernArgs = {.signalAddr = nullptr, .signalVal = 0};
  if (statex->node(peer) == statex->node() && win->nvlEnabled(peer)) {
    // Single-node direct cudaMemcpy
    if (count > 0) {
      void* dstPtr = reinterpret_cast<void*>(
          reinterpret_cast<size_t>(win->remWinInfo[peer].dataAddr) +
          targetDispNbytes);
      // CollTrace tracing logic for local + no signal case. In this case the
      // put will not trigger gpe->submit, so we need to record manually in the
      // algo. In other cases, this handle would be a no-op and the tracing
      // will take place in the gpe function.
      auto colltraceHandle = meta::comms::colltrace::getCollTraceHandleRMA(
          comm, opGroup, config, !signal);
      colltraceHandle->trigger(
          CollTraceHandleTriggerState::BeforeEnqueueKernel);

      FB_CUDACHECK(cudaMemcpyAsync(
          dstPtr, originBuff, countNbytes, cudaMemcpyDefault, stream));

      colltraceHandle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    }
    if (signal) {
      // Use atomic_store to signal the remote peer
      kernArgs.signalAddr = signalAddr;
      kernArgs.signalVal = signalVal;
    } else {
      // No signal, just return
      return commSuccess;
    }
  } else {
    // X-node put & signal via network
    // Create an op for GPE thread to complete put and signal
    struct OpElem* op =
        new struct OpElem(OpElem::opType::PUTSIGNAL, comm, putOpCount);
    op->putsignal.sendbuff = originBuff;
    op->putsignal.targetDisp = targetDisp;
    op->putsignal.count = count;
    op->putsignal.datatype = datatype;
    op->putsignal.signalAddr = signalAddr;
    op->putsignal.signalVal = signalVal;
    op->putsignal.peerRank = peer;
    op->putsignal.win = win;
    opGroup.push_back(std::unique_ptr<struct OpElem>(op));
  }

  void* func = nullptr;
  if (signal) {
    func = reinterpret_cast<void*>(ncclKernelPutSignal);
    config.algoArgs = reinterpret_cast<void*>(&kernArgs);
  } else {
    func = reinterpret_cast<void*>(ncclKernelPut);
    config.algoArgs = nullptr;
  }
  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup), putSignalImpl, config, func));

  return commSuccess;
}

commResult_t ctranWaitSignal_v2(
    size_t signalDisp,
    uint64_t cmpVal,
    commCmpOp_t cmpOp,
    CtranWin* win,
    cudaStream_t stream) {
  CtranComm* comm = win->comm;
  auto statex = comm->statex_.get();
  // FIXME: log ctranWaitSignal with CTRAN_RMA_INFO(add signalDisp and cmpVal)
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "CTRAN-RMA WaitSignal_v2: cmpValue {} signalDisp {} rank {} "
      "win {} winSigBase {} comm {} commHash {:x} [nranks={} localRanks={}] stream={}",
      cmpVal,
      signalDisp,
      statex->rank(),
      (void*)win,
      (void*)win->winSignalPtr,
      (void*)win->comm,
      statex->commHash(),
      statex->nRanks(),
      statex->nLocalRanks(),
      (void*)stream);

  // Check if the signal displacement exceeds the window size
  FB_COMMCHECK(checkSignalDisplacement(signalDisp, win->signalSize));
  const uint64_t* signalAddr = win->winSignalPtr + signalDisp;

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::WAITSIGNAL, stream, "WaitSignal", 0);
  config.args.devState_d = comm->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.clear();

  CtranKernelWaitSignalArgs kernArgs = {
      .signalAddr = nullptr, .cmpVal = 0, .cmpOp = cmpOp};
  config.algoArgs = reinterpret_cast<void*>(&kernArgs);
  if (win->isGpuMem()) {
    // if GPU memory, use kernel to atomic wait on the local signal value update
    // from remote rank, either from NVL or IB
    kernArgs.signalAddr = const_cast<uint64_t*>(signalAddr);
    kernArgs.cmpVal = cmpVal;
  } else {
    // if CPU memory, GPE thread to atomic wait for update from remote rank via
    // IB.
    struct OpElem* op = new struct OpElem(OpElem::opType::WAITSIGNAL, comm, 0);
    op->waitsignal.win = win;
    op->waitsignal.signalAddr = signalAddr;
    op->waitsignal.cmpVal = cmpVal;
    op->waitsignal.cmpOp = cmpOp;
    opGroup.push_back(std::unique_ptr<struct OpElem>(op));
  }

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      waitSignalImpl,
      config,
      reinterpret_cast<void*>(ncclKernelWaitSignal)));
  return commSuccess;
}

commResult_t ctranSignal(int peer, CtranWin* win, cudaStream_t stream) {
  CtranComm* comm = win->comm;
  const auto winOpCount = win->updateOpCount(peer);
  const auto sigOpCount =
      win->updateOpCount(peer, window::OpCountType::kSignal);
  auto statex = comm->statex_.get();

  auto signalVal = win->ctranNextSignalVal(peer);
  CTRAN_RMA_INFO(
      "ctranSignal",
      sigOpCount,
      winOpCount,
      nullptr,
      statex->rank(),
      1,
      commUint64,
      statex->rank(),
      peer,
      win,
      comm,
      stream);

  uint64_t* signalAddr = win->remWinInfo[peer].signalAddr + statex->rank();

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::SIGNAL, stream, "Signal", sigOpCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.clear();

  // Use cuda atomic store if peer is on the same host and has NVL enabled.
  // Otherwise, do signal via IB in GPE thread
  CtranKernelSignalArgs kernArgs = {.signalAddr = nullptr, .signalVal = 0};
  config.algoArgs = reinterpret_cast<void*>(&kernArgs);
  if (statex->node(peer) == statex->node() && win->nvlEnabled(peer)) {
    kernArgs.signalAddr = signalAddr;
    kernArgs.signalVal = signalVal;
  } else {
    struct OpElem* op =
        new struct OpElem(OpElem::opType::SIGNAL, comm, sigOpCount);
    op->signal.signalAddr = signalAddr;
    op->signal.signalVal = signalVal;
    op->signal.peerRank = peer;
    op->signal.win = win;
    opGroup.push_back(std::unique_ptr<struct OpElem>(op));
  }

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      signalImpl,
      config,
      reinterpret_cast<void*>(ncclKernelSignal)));
  return commSuccess;
}
