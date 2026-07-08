// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <memory>
#include <numeric>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/DevMemType.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/ctran/window/Types.h"
#if defined(ENABLE_PRIMS)
#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/prims/window/DeviceWindow.cuh"
#include "comms/prims/window/HostWindow.h"
#endif
#include "comms/utils/logger/LogUtils.h"

using ctran::window::RemWinInfo;

namespace ctran {
namespace {

commResult_t getWindowMapper(CtranComm* comm, CtranMapper** mapperOut) {
  if (comm == nullptr || comm->statex_ == nullptr || mapperOut == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "CTRAN-WINDOW: window communicator/statex or mapper output is null");
  }

  auto* resourceComm = comm->resourceComm();
  if (!ctranInitialized(resourceComm) ||
      resourceComm->ctran_->mapper == nullptr) {
    FB_ERRORRETURN(
        commInternalError,
        "CTRAN-WINDOW: window resource communicator has no initialized mapper");
  }

  *mapperOut = resourceComm->ctran_->mapper.get();
  return commSuccess;
}

commResult_t getWindowResourceRanks(
    CtranComm* comm,
    std::vector<int>* ranksOut) {
  if (comm == nullptr || comm->statex_ == nullptr || ranksOut == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "CTRAN-WINDOW: window communicator/statex or rank output is null");
  }

  const int nRanks = comm->statex_->nRanks();
  ranksOut->resize(nRanks);
  if (!comm->isSplitShare()) {
    std::iota(ranksOut->begin(), ranksOut->end(), 0);
    return commSuccess;
  }

  const auto& resourceRanks = comm->parentRanks();
  if (resourceRanks.size() != static_cast<size_t>(nRanks)) {
    FB_ERRORRETURN(
        commInternalError,
        "CTRAN-WINDOW: split-share window rank map size {} does not match nRanks {}",
        resourceRanks.size(),
        nRanks);
  }

  *ranksOut = resourceRanks;
  return commSuccess;
}

commResult_t getWindowResourceRank(CtranComm* comm, int rank, int* rankOut) {
  if (comm == nullptr || comm->statex_ == nullptr || rankOut == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "CTRAN-WINDOW: window communicator/statex or rank output is null");
  }
  if (rank < 0 || rank >= comm->statex_->nRanks()) {
    FB_ERRORRETURN(
        commInvalidArgument, "CTRAN-WINDOW: rank {} is out of range", rank);
  }
  if (!comm->isSplitShare()) {
    *rankOut = rank;
    return commSuccess;
  }

  const auto& resourceRanks = comm->parentRanks();
  if (resourceRanks.size() != static_cast<size_t>(comm->statex_->nRanks())) {
    FB_ERRORRETURN(
        commInternalError,
        "CTRAN-WINDOW: split-share window rank map size {} does not match nRanks {}",
        resourceRanks.size(),
        comm->statex_->nRanks());
  }
  *rankOut = resourceRanks[rank];
  return commSuccess;
}

commResult_t windowBarrier(CtranComm* comm, CtranMapper* mapper) {
  if (comm == nullptr || comm->statex_ == nullptr || mapper == nullptr) {
    FB_ERRORRETURN(
        commInvalidArgument,
        "CTRAN-WINDOW: cannot run window barrier without comm/statex/mapper");
  }
  if (!comm->isSplitShare()) {
    return mapper->barrier();
  }

  std::vector<int> ranks;
  FB_COMMCHECK(getWindowResourceRanks(comm, &ranks));
  if (ranks.size() <= 1) {
    return commSuccess;
  }

  const int myResourceRank = comm->resourceComm()->statex_->rank();
  bool foundSelf = false;
  for (const int rank : ranks) {
    if (rank == myResourceRank) {
      foundSelf = true;
      break;
    }
  }
  if (!foundSelf) {
    FB_ERRORRETURN(
        commInternalError,
        "CTRAN-WINDOW: split-share window ranks do not contain resource rank {}",
        myResourceRank);
  }

  std::vector<CtranMapperRequest> reqs((ranks.size() - 1) * 2);
  int reqIdx = 0;
  for (const int peerRank : ranks) {
    if (peerRank == myResourceRank) {
      continue;
    }
    FB_COMMCHECK(mapper->irecvCtrl(peerRank, &reqs[reqIdx++]));
    FB_COMMCHECK(mapper->isendCtrl(peerRank, &reqs[reqIdx++]));
  }
  for (auto& req : reqs) {
    FB_COMMCHECK(mapper->waitRequest(&req));
  }
  return commSuccess;
}

} // namespace

// Defined here (not in header) so that unique_ptr<HostWindow> destructor
// sees the complete HostWindow type.
CtranWin::~CtranWin() = default;

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

  CtranMapper* mapper = nullptr;
  FB_COMMCHECK(getWindowMapper(comm, &mapper));
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
    // User-provided buffer: use globalRegisterWithPtr to cache and register
    // multi-segment buffers (e.g., expandable segments) that mapper->regMem
    // cannot handle (it asserts single-segment). forceReg=true ensures
    // registration happens immediately; ncclManaged=true enables NVL IPC
    // handle creation for cudaMalloc buffers. searchRegHandle then finds the
    // RegElem via fast path for use in allGatherCtrl handle exchange.
    FB_COMMCHECK(
        ctran::globalRegisterWithPtr(
            winDataPtr,
            dataBytes,
            true /* forceReg */,
            true /* ncclManaged */));
    bool dynamicRegist = false;
    FB_COMMCHECK(mapper->searchRegHandle(
        winDataPtr, dataBytes, &dataRegHdl, &dynamicRegist));
    // globalRegisterWithPtr with forceReg=true guarantees the RegElem is
    // already created, so searchRegHandle should find it via fast path
    // (not dynamic registration).
    FB_CHECKABORT(
        !dynamicRegist,
        "Unexpected dynamic registration for window data buffer {} len {}",
        winDataPtr,
        dataBytes);
  }

  // Exchange each rank's data buffer size via bootstrap allGather
  // This populates remWinInfo[r].dataBytes for all ranks
  std::vector<size_t> allRankSizes(nRanks);
  allRankSizes[myRank] = dataBytes;
  auto resFuture = comm->bootstrap_->allGather(
      allRankSizes.data(), sizeof(size_t), myRank, nRanks);
  FB_COMMCHECK(static_cast<commResult_t>(std::move(resFuture).get()));

  // Handshake with other peers for registration exchange and network
  // connection setup
  std::vector<void*> remoteBaseBufs(nRanks);
  std::vector<void*> remoteUserBufs(nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteBaseBufAccessKeys(
      nRanks);
  std::vector<struct CtranMapperRemoteAccessKey> remoteUserBufAccessKeys(
      nRanks);
  std::vector<int> exchangeRanks;
  FB_COMMCHECK(getWindowResourceRanks(comm, &exchangeRanks));
  if (comm->isSplitShare()) {
    const auto resourceNRanks = comm->resourceComm()->statex_->nRanks();
    remoteBaseBufs.resize(resourceNRanks);
    remoteUserBufs.resize(resourceNRanks);
    remoteBaseBufAccessKeys.resize(resourceNRanks);
    remoteUserBufAccessKeys.resize(resourceNRanks);
  }

  FB_COMMCHECK(mapper->allGatherCtrl(
      winBasePtr,
      baseRegHdl,
      exchangeRanks,
      remoteBaseBufs,
      remoteBaseBufAccessKeys));

  if (!allocDataBuf_) {
    // if data buffer is provided by user, extra round of handler exchange is
    // needed
    FB_COMMCHECK(mapper->allGatherCtrl(
        winDataPtr,
        dataRegHdl,
        exchangeRanks,
        remoteUserBufs,
        remoteUserBufAccessKeys));
  }

  for (auto r = 0; r < nRanks; r++) {
    const int exchangeRank = exchangeRanks[r];
    remWinInfo[r].dataBytes = allRankSizes[r];
    if (allocDataBuf_) {
      remWinInfo[r].dataAddr = remoteBaseBufs[exchangeRank];
      remWinInfo[r].dataRkey = remoteBaseBufAccessKeys[exchangeRank];
      remWinInfo[r].signalAddr = reinterpret_cast<uint64_t*>(
          reinterpret_cast<size_t>(remoteBaseBufs[exchangeRank]) +
          allRankSizes[r]);
      remWinInfo[r].signalRkey = remoteBaseBufAccessKeys[exchangeRank];
    } else {
      remWinInfo[r].dataAddr = remoteUserBufs[exchangeRank];
      remWinInfo[r].dataRkey = remoteUserBufAccessKeys[exchangeRank];
      remWinInfo[r].signalAddr =
          reinterpret_cast<uint64_t*>(remoteBaseBufs[exchangeRank]);
      remWinInfo[r].signalRkey = remoteBaseBufAccessKeys[exchangeRank];
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
        "CTRAN-WINDOW     Peer {}: addr {} size {} rkey {}",
        i,
        (void*)remWinInfo[i].dataAddr,
        allRankSizes[i],
        myRank == i ? "(local)" : remWinInfo[i].dataRkey.toString());
  }

  // A barrier among ranks after importing handles to prevent accessing window
  // memory space while other ranks are still importing.
  FB_COMMCHECK(windowBarrier(comm, mapper));
  return commSuccess;
}

bool CtranWin::allGatherPSupported(CtranComm* comm) {
  if (comm == nullptr || comm->isSplitShare() || !ctranInitialized(comm)) {
    return false;
  }
  auto statex = comm->statex_.get();
  auto mapper = comm->ctran_->mapper.get();
  const auto myRank = statex->rank();
  for (int rank = 0; rank < statex->nRanks(); rank++) {
    if (rank != myRank &&
        mapper->getBackend(rank) == CtranMapperBackend::UNSET) {
      return false;
    }
  }
  return true;
}

bool CtranWin::allToAllWinSupported(CtranComm* comm) {
  return allGatherPSupported(comm);
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
      "CTRAN-WINDOW: Rank {} window buffer is {} window data buffer base {} signal buffer base {} "
      "dataBytes {} signalSize {} win {} comm {} commHash {:x} [nnodes={} nranks={} localRanks={}]",
      myRank,
      allocDataBuf_ ? "Allocated" : "User Provided",
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

commResult_t CtranWin::free(bool skipBarrier) {
  auto statex = comm->statex_.get();
  if (statex == nullptr) {
    FB_ERRORRETURN(commInternalError, "Empty communicator statex.");
  }
  CtranMapper* mapper = nullptr;
  FB_COMMCHECK(getWindowMapper(comm, &mapper));
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
  // the window after it is freed. Skipped when called from deferred cleanup at
  // comm destruction (all communication is already finalized).
  // NOTE: the window object is not aware of CUDA streams, users need to
  // ensure the host process waits for CUDA streams where put/wait operations
  // are launched.
  if (!skipBarrier) {
    FB_COMMCHECK(windowBarrier(comm, mapper));
  }

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
      FB_COMMCHECK(mapper->deregRemReg(&remWinInfo[i].signalRkey));
    }
  }

  // User-provided data buffer: deregister locally without remote IPC
  // notifications. Remove from export cache first (so mapper destructor
  // won't access the freed RegElem), then free segments locally, then
  // release locally-imported remote handles.
  if (!allocDataBuf_) {
    mapper->removeFromExportCache(dataRegHdl);
    FB_COMMCHECK(
        ctran::globalDeregisterWithPtr(
            winDataPtr, dataBytes, true /* skipRemRelease */));
    for (auto i = 0; i < nRanks; ++i) {
      if (i != statex->rank()) {
        FB_COMMCHECK(mapper->deregRemReg(&remWinInfo[i].dataRkey));
      }
    }
  }

#if defined(ENABLE_PRIMS)
  // HostWindow handles cleanup via RAII
  hostWindow_.reset();
#endif

  freeMem(winBasePtr);

  return commSuccess;
}

bool CtranWin::nvlEnabled(int rank) const {
  CtranMapper* mapper = nullptr;
  int resourceRank = -1;
  if (getWindowMapper(comm, &mapper) != commSuccess ||
      getWindowResourceRank(comm, rank, &resourceRank) != commSuccess) {
    return false;
  }
  return isGpuMem() &&
      mapper->hasBackend(resourceRank, CtranMapperBackend::NVL);
}

#if defined(ENABLE_PRIMS)
commResult_t CtranWin::getDeviceWin(
    comms::prims::DeviceWindow* devWin,
    const comms::prims::WindowConfig& config) {
  auto* transport = comm->multiPeerTransport_.get();
  if (!transport) {
    FB_ERRORRETURN(
        commInternalError, "getDeviceWin: multiPeerTransport is null.");
  }

  if (!hostWindow_) {
    const auto myRank = transport->my_rank();

    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-WINDOW: Rank {} creating HostWindow with signalCount={} "
        "counterCount={} barrierCount={} dataPtr={} dataBytes={}",
        myRank,
        config.peerSignalCount,
        config.peerCounterCount,
        config.barrierCount,
        winDataPtr,
        dataBytes);

    hostWindow_ = std::make_unique<comms::prims::HostWindow>(
        *transport, config, winDataPtr, dataBytes);

    hostWindow_->exchange();

    CLOGF_SUBSYS(
        INFO, INIT, "CTRAN-WINDOW: Rank {} device window built", myRank);
  }

  new (devWin) comms::prims::DeviceWindow(hostWindow_->getDeviceWindow());
  return commSuccess;
}
#endif // ENABLE_PRIMS

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
  auto winGuard = folly::makeGuard([&newWin]() {
    // On any early error return, release partial resources (SW + backend
    // dereg, no barrier) and delete the window so it does not leak.
    (void)newWin->free(/*skipBarrier=*/true);
    delete newWin;
  });
  newWin->setAtomicCapable(true);
  FB_COMMCHECK(newWin->allocate(nullptr));
  FB_COMMCHECK(newWin->exchange());
  if (baseptr) {
    *baseptr = newWin->winDataPtr;
  }
  *win = newWin;
  winGuard.dismiss();
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
  auto winGuard = folly::makeGuard([&newWin]() {
    // On any early error return, release partial resources (SW + backend
    // dereg, no barrier) and delete the window so it does not leak.
    (void)newWin->free(/*skipBarrier=*/true);
    delete newWin;
  });
  newWin->setAtomicCapable(
      reinterpret_cast<uintptr_t>(databuf) % sizeof(uint64_t) == 0 &&
      size % sizeof(uint64_t) == 0);

  FB_COMMCHECK(newWin->allocate((void*)databuf));

  FB_COMMCHECK(newWin->exchange()); // register and exchange both signal
                                    // & data buffer
  *win = newWin;
  winGuard.dismiss();
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
