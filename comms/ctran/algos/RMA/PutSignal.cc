// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <memory>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/RMA/Types.h"
#include "comms/ctran/algos/RMA/WaitSignalImpl.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran;
using meta::comms::colltrace::CollTraceHandleTriggerState;

extern __global__ void ncclKernelPutNotify(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::rma::KernelPutNotifyArgs args);

extern __global__ void ncclKernelWaitNotify(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::rma::KernelWaitNotifyArgs args);

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

  // Skip data transfer if count is 0 (signal-only, e.g. ready barrier)
  void* localMemHdl = nullptr;
  bool localReg = false;
  if (putSize > 0) {
    // Get registration handle for local send buffer
    FB_COMMCHECK(comm->ctran_->mapper->searchRegHandle(
        op->putsignal.sendbuff, putSize, &localMemHdl, &localReg));

    CLOGF_TRACE(
        COLL,
        "putSignalImpl: sbuf {}, rbuf {} (base {} + offset {}), size {}, signalAddr {}",
        op->putsignal.sendbuff,
        dstPtr,
        win->remWinInfo[peerRank].dataAddr,
        targetDispNbytes,
        putSize,
        (void*)op->putsignal.signalAddr);

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
  }

  CtranMapperRequest signalReq = CtranMapperRequest();
  if (op->putsignal.signalAddr != nullptr) {
    // The kernel has already incremented the per-peer counter
    // (KernelWaitGpeTerminate serializes kernel and GPE). Acquire pairs
    // with the kernel's system-scope release.
    uint64_t signalVal =
        __atomic_load_n(op->putsignal.signalCounter, __ATOMIC_ACQUIRE);
    // flush the iput to make sure the signal is sent after the data
    FB_COMMCHECK(comm->ctran_->mapper->atomicSet(
        op->putsignal.signalAddr,
        signalVal,
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
      "signalImpl: peer {} signalAddr {}",
      op->signal.peerRank,
      (void*)op->signal.signalAddr);

  uint64_t signalVal =
      __atomic_load_n(op->signal.signalCounter, __ATOMIC_ACQUIRE);

  CtranMapperRequest signalReq = CtranMapperRequest();
  FB_COMMCHECK(comm->ctran_->mapper->atomicSet(
      op->signal.signalAddr,
      signalVal,
      peerRank,
      CtranMapperConfig{
          .remoteAccessKey_ = win->remWinInfo[peerRank].signalRkey},
      &signalReq));
  FB_COMMCHECK(comm->ctran_->mapper->waitRequest(&signalReq));

  return commSuccess;
}

static commResult_t waitSignalSpinningKernelImpl(
    const std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  struct OpElem* op = opGroup.front().get();
  auto comm = op->comm_;
  auto win = op->waitsignal.win;
  const std::atomic<uint64_t>* addr =
      reinterpret_cast<const std::atomic<uint64_t>*>(op->waitsignal.signalAddr);
  CtranAlgoRMALogger logger("ctranWaitSignal", op->opCount, -1, win, comm);
  // The kernel increments the wait counter before KernelStartGpe,
  // so this load always observes the post-increment value.
  uint64_t cmpVal =
      __atomic_load_n(op->waitsignal.signalCounter, __ATOMIC_ACQUIRE);
  CLOGF_TRACE(
      COLL,
      "waitSignalSpinningKernelImpl: signalAddr {}, cmpVal={}",
      (void*)const_cast<uint64_t*>(op->waitsignal.signalAddr),
      cmpVal);
  while (std::atomic_load(addr) < cmpVal) {
    std::this_thread::yield();
  };

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

  // Check if the target displacement exceeds the remote peer's window size
  FB_COMMCHECK(checkDisplacementBounds(
      targetDisp, commTypeSize(datatype), count, win->getDataSize(peer)));
  size_t targetDispNbytes = targetDisp * commTypeSize(datatype);
  size_t countNbytes = count * commTypeSize(datatype);
  uint64_t* signalAddr = nullptr;
  if (signal) {
    signalAddr = win->remWinInfo[peer].signalAddr + statex->rank();
  }

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::PUTSIGNAL, stream, "PutSignal", putOpCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.clear();

  // Use direct copy if peer is on the same host and has NVL enabled.
  // Otherwise, do put & signal via network
  bool isNvl = statex->node(peer) == statex->node() && win->nvlEnabled(peer);
  // The kernel always increments the per-peer signal counter to get a
  // monotonically increasing value. KernelWaitGpeTerminate serializes
  // kernel execution with GPE processing, so the GPE thread always sees
  // the correct counter value for the current op.
  CtranKernelPutSignalArgs kernArgs = {
      .signalAddr = nullptr,
      .signalCounter = signal ? &win->signalCounters[peer] : nullptr,
      .signalCounterSystemScope = !isNvl};
  if (isNvl) {
    // Single-node direct cudaMemcpy
    if (count > 0) {
      void* dstPtr = reinterpret_cast<void*>(
          reinterpret_cast<size_t>(win->remWinInfo[peer].dataAddr) +
          targetDispNbytes);
      auto colltraceHandle = meta::comms::colltrace::getCollTraceHandleRMA(
          comm, opGroup, config, !signal);
      colltraceHandle->trigger(
          CollTraceHandleTriggerState::BeforeEnqueueKernel);

      FB_CUDACHECK(cudaMemcpyAsync(
          dstPtr, originBuff, countNbytes, cudaMemcpyDefault, stream));

      colltraceHandle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
    }
    if (signal) {
      kernArgs.signalAddr = signalAddr;
    } else {
      return commSuccess;
    }
  } else {
    // X-node put & signal via network
    struct OpElem* op =
        new struct OpElem(OpElem::opType::PUTSIGNAL, comm, putOpCount);
    op->putsignal.sendbuff = originBuff;
    op->putsignal.targetDisp = targetDisp;
    op->putsignal.count = count;
    op->putsignal.datatype = datatype;
    op->putsignal.signalAddr = signalAddr;
    op->putsignal.signalCounter = &win->signalCounters[peer];
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

// Hardware-accelerated wait. Increments the wait counter internally so
// the counter only advances when we know the HW path will be attempted.
// Returns commInvalidUsage if HW wait is unavailable (caller can fall
// back to spinning kernel without double-increment risk).
static commResult_t
waitSignalDriverApi(int peer, CtranWin* win, cudaStream_t stream) {
#if CUDART_VERSION >= 11070
  if (!ctran::utils::canUse64BitStreamMemOps()) {
    return commInvalidUsage;
  }

  const uint64_t* signalAddr = win->winSignalPtr + peer;
  // Peek at the next expected value without incrementing. Only commit the
  // counter advance after confirming the HW wait was enqueued successfully,
  // so the spinning-kernel fallback won't double-increment.
  uint64_t cmpVal =
      __atomic_load_n(&win->waitCounters[peer], __ATOMIC_RELAXED) + 1;

  CUresult result = FB_CUPFN(cuStreamWaitValue64)(
      (CUstream)stream,
      (CUdeviceptr)signalAddr,
      cmpVal,
      CU_STREAM_WAIT_VALUE_GEQ);

  if (result != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    FB_CUPFN(cuGetErrorString)(result, &errStr);

    if (result == CUDA_ERROR_NOT_SUPPORTED) {
      CLOGF(
          WARN,
          "CTRAN RMA: Hardware wait not supported ({}), falling back to spinning kernel",
          errStr ? errStr : "unknown error");
      return commInvalidUsage;
    }

    CLOGF(
        ERR,
        "CTRAN RMA: Hardware wait failed with error ({}), not falling back",
        errStr ? errStr : "unknown error");
    return commInternalError;
  }

  // HW wait enqueued — commit the counter increment
  win->nextWaitCounter(peer);
  return commSuccess;
#else
  return commInvalidUsage;
#endif
}

// Spinning kernel wait. The kernel always increments the per-peer wait
// counter to derive the compare value — no capture-mode branching needed.
commResult_t waitSignalSpinningKernel(
    int peer,
    CtranWin* win,
    cudaStream_t stream,
    uint64_t waitOpCount) {
  CtranComm* comm = win->comm;

  const uint64_t* signalAddr = win->winSignalPtr + peer;

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::WAITSIGNAL, stream, "WaitSignal", waitOpCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();
  config.canConcurrent = true;

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.clear();

  CtranKernelWaitSignalArgs kernArgs = {
      .signalAddr = nullptr,
      .signalCounter = &win->waitCounters[peer],
  };
  config.algoArgs = reinterpret_cast<void*>(&kernArgs);
  if (win->isGpuMem()) {
    kernArgs.signalAddr = const_cast<uint64_t*>(signalAddr);
  } else {
    struct OpElem* op =
        new struct OpElem(OpElem::opType::WAITSIGNAL, comm, waitOpCount);
    op->waitsignal.win = win;
    op->waitsignal.signalAddr = signalAddr;
    op->waitsignal.signalCounter = &win->waitCounters[peer];
    opGroup.push_back(std::unique_ptr<struct OpElem>(op));
  }

  FB_COMMCHECK(comm->ctran_->gpe->submit(
      std::move(opGroup),
      waitSignalSpinningKernelImpl,
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

  uint64_t* signalAddr = win->remWinInfo[peer].signalAddr + statex->rank();

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

  KernelConfig config = KernelConfig(
      KernelConfig::KernelType::SIGNAL, stream, "Signal", sigOpCount);
  config.args.devState_d = comm->ctran_->algo->getDevState();

  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  opGroup.clear();

  bool isSigNvl = statex->node(peer) == statex->node() && win->nvlEnabled(peer);
  CtranKernelSignalArgs kernArgs = {
      .signalAddr = nullptr,
      .signalCounter = &win->signalCounters[peer],
      .signalCounterSystemScope = !isSigNvl};
  config.algoArgs = reinterpret_cast<void*>(&kernArgs);
  if (isSigNvl) {
    kernArgs.signalAddr = signalAddr;
  } else {
    struct OpElem* op =
        new struct OpElem(OpElem::opType::SIGNAL, comm, sigOpCount);
    op->signal.signalAddr = signalAddr;
    op->signal.signalCounter = &win->signalCounters[peer];
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

// Hardware-accelerated wait with automatic fallback to spinning kernel
// This is the recommended API for best performance
// Tries hardware wait first (CUDA 11.7+), falls back to spinning kernel
commResult_t ctranWaitSignal(int peer, CtranWin* win, cudaStream_t stream) {
  CtranComm* comm = win->comm;
  auto statex = comm->statex_.get();

  // Track op counts for BOTH implementations
  const auto winOpCount = win->updateOpCount(peer);
  const auto waitOpCount =
      win->updateOpCount(peer, window::OpCountType::kWaitSignal);

  // Log this operation for BOTH implementations
  CTRAN_RMA_INFO(
      "ctranWaitSignal",
      waitOpCount,
      winOpCount,
      nullptr,
      peer,
      size_t(1),
      commDataType_t::commUint64,
      statex->rank(),
      peer,
      win,
      comm,
      stream);

  // For eager GPU-memory, try hardware-accelerated wait first.
  // waitSignalDriverApi increments the counter internally, only after
  // confirming HW support — so commInvalidUsage means the counter
  // was NOT incremented and we can safely fall back.
  if (win->isGpuMem()) {
    cudaStreamCaptureStatus captureStatus{};
    cudaStreamGetCaptureInfo(stream, &captureStatus, nullptr);
    if (captureStatus != cudaStreamCaptureStatusActive) {
      auto colltraceHandle = meta::comms::colltrace::getCollTraceHandleRMA(
          comm,
          {},
          KernelConfig{
              KernelConfig::KernelType::WAITSIGNAL,
              stream,
              "WaitSignal",
              waitOpCount},
          true);
      colltraceHandle->trigger(
          CollTraceHandleTriggerState::BeforeEnqueueKernel);

      commResult_t hwResult = waitSignalDriverApi(peer, win, stream);

      colltraceHandle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);

      if (hwResult == commSuccess) {
        CLOGF_TRACE(
            COLL,
            "CTRAN RMA: WaitSignal successful using hardware acceleration");
        return commSuccess;
      }

      if (hwResult != commInvalidUsage) {
        return hwResult;
      }
    }
  }
  CLOGF(
      INFO,
      "CTRAN RMA: WaitSignal falling back to spinning kernel (peer={})",
      peer);
  return waitSignalSpinningKernel(peer, win, stream, waitOpCount);
}
