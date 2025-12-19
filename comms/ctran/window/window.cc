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
CtranWin::CtranWin(CtranComm* comm, size_t size, DevMemType bufType)
    : comm(comm), dataBytes(size), bufType_(bufType) {
  if (comm == nullptr) {
    FB_CHECKABORT(
        commInternalError, "CtranWin: comm is nullptr when creating window.");
  }
  signalSize = comm->statex_.get()->nRanks();
  signalVal.resize(signalSize);
  waitSignalVal.resize(signalSize);
  for (auto& val : signalVal)
    val.store(1);
  for (auto& val : waitSignalVal)
    val.store(1);
}

commResult_t CtranWin::exchange() {
  auto statex = comm->statex_.get();
  const auto nRanks = statex->nRanks();
  const auto myRank = statex->rank();

  remWinInfo.resize(nRanks);

  auto mapper = comm->ctran_->mapper.get();
  CtranMapperEpochRAII epochRAII(mapper);
  // Registration via ctran mapper.
  FB_COMMCHECK(mapper->regMem(
      winBasePtr,
      range_,
      &(baseSegHdl),
      true,
      true, /* NCCL managed buffer */
      &baseRegHdl));

  if (allocDataBuf_) {
    dataSegHdl = baseSegHdl;
    dataRegHdl = baseRegHdl;
  } else {
    // if data buffer is provided by user, we need to register it
    FB_COMMCHECK(mapper->regMem(
        winDataPtr,
        dataBytes,
        &(dataSegHdl),
        true,
        true, /* NCCL managed buffer */
        &dataRegHdl));
  }

  // Handshake with other peers for registration exchange and network
  // connection setup
  std::vector<void*> remoteBaseBufs(nRanks);
  std::vector<void*> remoteUserBufs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteBaseBufAccessKeys(
      nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteUserBufAccessKeys(
      nRanks);

  FB_COMMCHECK(mapper->allGatherCtrl(
      winBasePtr, baseRegHdl, remoteBaseBufs, remoteBaseBufAccessKeys));

  if (!allocDataBuf_) {
    // if data buffer is provided by user, extra round of handler exchange is
    // needed
    FB_COMMCHECK(mapper->allGatherCtrl(
        winDataPtr, dataRegHdl, remoteUserBufs, remoteUserBufAccessKeys));
  }

  for (auto r = 0; r < nRanks; r++) {
    if (allocDataBuf_) {
      remWinInfo[r].dataAddr = remoteBaseBufs[r];
      remWinInfo[r].dataRkey = remoteBaseBufAccessKeys[r];
      remWinInfo[r].signalAddr = reinterpret_cast<uint64_t*>(
          reinterpret_cast<size_t>(remoteBaseBufs[r]) + dataBytes);
      remWinInfo[r].signalRkey = remoteBaseBufAccessKeys[r];
    } else {
      remWinInfo[r].dataAddr = remoteUserBufs[r];
      remWinInfo[r].dataRkey = remoteUserBufAccessKeys[r];
      remWinInfo[r].signalAddr = reinterpret_cast<uint64_t*>(remoteBaseBufs[r]);
      remWinInfo[r].signalRkey = remoteBaseBufAccessKeys[r];
    }
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
        (void*)remWinInfo[i].dataAddr,
        myRank == i ? "(local)" : remWinInfo[i].dataRkey.toString());
  }

  // A barrier among ranks after importing handles to prevent accessing window
  // memory space while other ranks are still importing.
  FB_COMMCHECK(mapper->barrier());
  return commSuccess;
}

commResult_t CtranWin::allocate(void* userBufPtr) {
  auto statex = comm->statex_.get();
  const auto myRank = statex->rank();

  if (winBasePtr != nullptr) {
    FB_ERRORRETURN(commInternalError, "CtranWin already allocated.");
  }

  // If no buffer is provided by the user, the Window object is responsible for
  // allocating a new buffer internally.
  allocDataBuf_ = userBufPtr == nullptr ? true : false;

  void* addr = nullptr;
  CUmemGenericAllocationHandle allocHandle;
  auto signalBytes = signalSize * sizeof(uint64_t);
  size_t allocSize = allocDataBuf_ ? dataBytes + signalBytes : signalBytes;
  if (isGpuMem()) {
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
    FB_CUCHECK(cuMemGetAddressRange(&pbase, &range_, (CUdeviceptr)addr));
  } else {
    FB_CUDACHECK(cudaMallocHost(&addr, allocSize));
    range_ = allocSize;
  }

  winBasePtr = addr;

  if (allocDataBuf_) {
    winDataPtr = addr;
    winSignalPtr =
        reinterpret_cast<uint64_t*>(reinterpret_cast<size_t>(addr) + dataBytes);
  } else {
    winDataPtr = userBufPtr;
    winSignalPtr = reinterpret_cast<uint64_t*>(reinterpret_cast<size_t>(addr));
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-WINDOW: Rank {} allocated local window data buffer base {} signal buffer base {} "
      "dataBytes {} signalSize {} win {} comm {} commHash {:x} [nnodes={} nranks={} localRanks={}]",
      myRank,
      winDataPtr,
      (void*)winSignalPtr,
      dataBytes,
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
  auto mapper = comm->ctran_->mapper.get();
  CtranMapperEpochRAII epochRAII(mapper);

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
  FB_COMMCHECK(mapper->barrier());

  auto nRanks = statex->nRanks();

  // utils funcs to deregister memory
  auto deregMemIfNotNull = [&](void* segHdl) {
    if (segHdl != nullptr) {
      FB_COMMCHECK(mapper->deregMem(segHdl, true /* skipRemRelease */));
    }
    return commSuccess;
  };

  // utils func to free memory
  auto freeMem = [&](void* addr) {
    if (isGpuMem()) {
      FB_COMMCHECK(utils::commCuMemFree(addr));
    } else {
      FB_CUDACHECK(cudaFreeHost(addr));
    }
    return commSuccess;
  };

  // deregistr buffer
  deregMemIfNotNull(baseSegHdl);
  // deregister remote buf
  for (auto i = 0; i < nRanks; ++i) {
    if (i != statex->rank()) {
      // the signal buffer is always allocated by window internally, so we only
      // need to dereg using the signalRkey
      FB_COMMCHECK(mapper->deregRemReg(&remWinInfo[i].signalRkey));
    }
  }

  // if data buffer is provided by user, we need to dereg the data buffer
  if (!allocDataBuf_) {
    deregMemIfNotNull(dataSegHdl);
    for (auto i = 0; i < nRanks; ++i) {
      if (i != statex->rank()) {
        FB_COMMCHECK(mapper->deregRemReg(&remWinInfo[i].dataRkey));
      }
    }
  }
  freeMem(winBasePtr);

  return commSuccess;
}

bool CtranWin::nvlEnabled(int rank) const {
  return isGpuMem() &&
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
  // Round up data buffer size to be divisible by 8.
  // This ensures that when data and signal buffers (n * uint64_t) are allocated
  // contiguously, the signal buffer remains 8-byte aligned for atomic
  // operations (assuming the base allocation is 8-byte aligned).
  size = (size + 7) & ~7;
  std::string locationRes;
  std::string sigBufSize;

  hints.get("window_buffer_location", locationRes);

  CtranWin* newWin = new CtranWin(
      comm,
      size,
      locationRes == "cpu" ? DevMemType::kHostPinned : DevMemType::kCumem);
  FB_COMMCHECK(newWin->allocate(nullptr));
  FB_COMMCHECK(newWin->exchange());
  if (baseptr) {
    *baseptr = newWin->winDataPtr;
  }
  *win = newWin;
  return commSuccess;
}

commResult_t checkUserBufType(const DevMemType bufType) {
  if (bufType == DevMemType::kCumem || bufType == DevMemType::kHostPinned ||
      bufType == DevMemType::kHostUnregistered ||
      bufType == DevMemType::kCudaMalloc) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-WINDOW: Buffer Type {} is provided by user while registering window",
        devMemTypeStr(bufType));
    return commSuccess;
  }
  CLOGF(
      ERR,
      "CTRAN-WINDOW: Unsupported buffer type {} provided when registering window. Supported buffer types are kCumem, kHostPinned, kHostUnregistered, kCudaMalloc",
      devMemTypeStr(bufType));

  return commInvalidUsage;
}

commResult_t ctranWinRegister(
    const void* databuf,
    size_t size,
    CtranComm* comm,
    CtranWin** win,
    const meta::comms::Hints& hints) {
  if (databuf == nullptr) {
    FB_ERRORRETURN(
        commInternalError,
        "CtranWin: Valid data buffer must be provided while the ctranWinRegister is used.");
  }

  DevMemType userBufType =
      DevMemType::kCumem; // will be overwritten by getDevMemType
  FB_COMMCHECK(getDevMemType(databuf, comm->statex_->cudaDev(), userBufType));
  FB_COMMCHECK(checkUserBufType(userBufType));

  CtranWin* newWin = new struct CtranWin(
      comm,
      // byte size of user provided data buffer
      size,
      // if user buffer is on host CPU, allocate kHostPinned buffer for
      // signal otherwise is on GPU device, allocate kCumem buffer for signal
      userBufType);

  FB_COMMCHECK(newWin->allocate((void*)databuf));

  FB_COMMCHECK(newWin->exchange()); // register and exchange both signal
                                    // & data buffer
  *win = newWin;
  return commSuccess;
}

commResult_t ctranWinSharedQuery(int rank, CtranWin* win, void** addr) {
  CtranComm* comm = win->comm;

  // Validate rank is within valid bounds
  if (rank < 0 || rank >= comm->statex_->nRanks()) {
    CLOGF(
        ERR,
        "CTRAN-WINDOW: Invalid rank {} for sharedQuery (valid range: [0, {}))",
        rank,
        comm->statex_->nRanks());
    *addr = nullptr;
    return commInvalidArgument;
  }

  if (rank == comm->statex_->rank() || win->nvlEnabled(rank)) {
    *addr = win->remWinInfo[rank].dataAddr;
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
