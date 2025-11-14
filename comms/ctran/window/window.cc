// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevMemType.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/ctran/window/Types.h"
#include "comms/utils/logger/LogUtils.h"

using ctran::window::RemWinInfo;

namespace ctran {
CtranWin::CtranWin(
    CtranComm* comm,
    size_t size,
    size_t signalSize,
    DevMemType bufType)
    : comm(comm), dataSize(size), signalSize(signalSize), bufType_(bufType) {};

commResult_t CtranWin::exchange() {
  auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();
  const auto myRank = statex->rank();

  remWinInfo.resize(nRanks);

  auto mapper = comm->ctran_->mapper.get();
  CtranMapperEpochRAII epochRAII(mapper);

  // Registration via ctran mapper.
  FB_COMMCHECK(comm->ctran_->mapper->regMem(
      winBasePtr,
      range,
      &(segHdl),
      true,
      true, /* NCCL managed buffer */
      &regHdl));

  // Handshake with other peers for registration exchange and network
  // connection setup
  std::vector<void*> remoteBufs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteAccessKeys(nRanks);
  FB_COMMCHECK(
      mapper->allGatherCtrl(winBasePtr, regHdl, remoteBufs, remoteAccessKeys));
  for (auto r = 0; r < nRanks; r++) {
    remWinInfo[r].addr = remoteBufs[r];
    remWinInfo[r].rkey = remoteAccessKeys[r];
    remWinInfo[r].signalAddr = reinterpret_cast<uint64_t*>(
        reinterpret_cast<size_t>(remoteBufs[r]) + dataSize);
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-WINDOW: Rank {} exchanged remote windowInfo in win {} comm {} commHash {:x}:",
      myRank,
      (void*)this,
      (void*)comm,
      statex->commHash());

  for (int i = 0; i < nRanks; ++i) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-WINDOW     Peer {}: addr {} rkey {}",
        i,
        (void*)remWinInfo[i].addr,
        myRank == i ? "(local)" : remWinInfo[i].rkey.toString());
  }

  // A barrier among ranks after importing handles to prevent accessing window
  // memory space while other ranks are still importing.
  FB_COMMCHECK(mapper->barrier());
  return commSuccess;
}

commResult_t CtranWin::allocate() {
  auto statex = comm->statex_.get();
  const auto myRank = statex->rank();

  if (winBasePtr != nullptr) {
    FB_ERRORRETURN(commInternalError, "CtranWin already allocated.");
  }

  void* addr = nullptr;
  CUmemGenericAllocationHandle allocHandle;

  size_t allocSize = dataSize + signalSize * sizeof(uint64_t);
  if (bufType_ == DevMemType::kHostPinned) {
    FB_CUDACHECK(cudaMallocHost(&addr, allocSize));
    range = allocSize;
  } else {
    FB_COMMCHECK(
        utils::commCuMemAlloc(
            &addr,
            &allocHandle,
            utils::getCuMemAllocHandleType(),
            allocSize,
            &comm->logMetaData_,
            "allocate"));

    // query the actually allocated range of the memory
    CUdeviceptr pbase = 0;
    FB_CUCHECK(cuMemGetAddressRange(&pbase, &range, (CUdeviceptr)addr));
  }

  winBasePtr = addr;
  winBaseSignalPtr =
      reinterpret_cast<uint64_t*>(reinterpret_cast<size_t>(addr) + dataSize);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-WINDOW: Rank {} allocated local winBase {} winSigBase {} "
      "dataSize {} signalSize {} win {} comm {} commHash {:x} [nnodes={} nranks={} localRanks={}]",
      myRank,
      winBasePtr,
      (void*)winBaseSignalPtr,
      dataSize,
      signalSize,
      (void*)this,
      (void*)comm,
      statex->commHash(),
      statex->nNodes(),
      statex->nRanks(),
      statex->nLocalRanks());
  return commSuccess;
}

commResult_t CtranWin::free() {
  auto statex = comm->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(commInternalError, "Empty communicator statex.");
  }
  CtranMapperEpochRAII epochRAII(comm->ctran_->mapper.get());

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-WINDOW: Rank {} free win {} comm {} commHash {:x}",
      statex->rank(),
      (void*)this,
      (void*)comm,
      statex->commHash());

  // A barrier among ranks before freeing window to prevent peer ranks accessing
  // the window after it is freed.
  // NOTE: the window object is not aware of CUDA streams, users need to
  // ensure the host process waits for CUDA streams where put/wait operations
  // are launched.
  FB_COMMCHECK(comm->ctran_->mapper->barrier());

  if (segHdl != nullptr) {
    FB_COMMCHECK(
        comm->ctran_->mapper->deregMem(segHdl, true /* skipRemRelease */));
  }

  // Deregister remote memory for ctran mapper
  auto nRanks = statex->nRanks();
  for (auto i = 0; i < nRanks; ++i) {
    if (i != statex->rank()) {
      FB_COMMCHECK(comm->ctran_->mapper->deregRemReg(&remWinInfo[i].rkey));
    }
  }
  // Release local memory
  if (bufType_ == DevMemType::kHostPinned) {
    FB_CUDACHECK(cudaFreeHost(winBasePtr));
  } else {
    FB_COMMCHECK(utils::commCuMemFree(remWinInfo[statex->rank()].addr));
  }

  return commSuccess;
}

bool CtranWin::nvlEnabled(int rank) const {
  return bufType_ != DevMemType::kHostPinned &&
      comm->ctran_->mapper->hasBackend(rank, CtranMapperBackend::NVL);
}

commResult_t ctranWinAllocate(
    size_t size,
    CtranComm* comm,
    void** baseptr,
    CtranWin** win,
    const meta::comms::Hints& hints) {
  if (size < CTRAN_MIN_REGISTRATION_SIZE) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "ctranWinAllocate size {} is smaller than {}, resize to CTRAN_MIN_REGISTRATION_SIZE",
        size,
        CTRAN_MIN_REGISTRATION_SIZE);
    size = CTRAN_MIN_REGISTRATION_SIZE;
  }
  // Round up size to be divisible by 8 (size % 8 == 0)
  size = (size + 7) & ~7;
  std::string locationRes;
  std::string sigBufSize;

  hints.get("window_buffer_location", locationRes);
  hints.get("window_signal_bufsize", sigBufSize);

  CtranWin* newWin = new CtranWin(
      comm,
      size,
      sigBufSize.empty() ? NCCL_CTRAN_WIN_SIGNAL_SIZE : std::stoi(sigBufSize),
      locationRes == "cpu" ? DevMemType::kHostPinned : DevMemType::kCumem);
  FB_COMMCHECK(newWin->allocate());
  FB_COMMCHECK(newWin->exchange());
  if (baseptr) {
    *baseptr = newWin->winBasePtr;
  }
  *win = newWin;
  return commSuccess;
}

commResult_t ctranWinSharedQuery(int rank, CtranWin* win, void** addr) {
  CtranComm* comm = win->comm;
  if (rank == comm->statex_->rank() || win->nvlEnabled(rank)) {
    *addr = win->remWinInfo[rank].addr;
  } else {
    // If the remote rank is not supported by NVL path (either on a different
    // node, or it is CPU memory), return nullptr so that user should not
    // directly access the memory.
    *addr = nullptr;
  }
  return commSuccess;
}

commResult_t ctranWinFree(CtranWin* win) {
  FB_COMMCHECK(win->free());
  CtranWin* win_ = static_cast<CtranWin*>(win);
  delete win_;
  return commSuccess;
}

} // namespace ctran
