// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/ib/CtranIb.h"

#include <unistd.h>

#include <fmt/core.h>
#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "comms/ctran/backends/ib/CtranIbBase.h"
#include "comms/ctran/backends/ib/CtranIbLocalVc.h"
#include "comms/ctran/backends/ib/CtranIbVc.h"
#include "comms/ctran/backends/ib/ibutils.h"
#include "comms/ctran/bootstrap/Socket.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/utils/ArgCheck.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/ctran/utils/SkipDestroyUtil.h"
#include "comms/utils/MemUtils.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"
#include "folly/synchronization/CallOnce.h"

using namespace ctran::ib;
using namespace ctran::ibvwrap;

namespace {
#define CTRAN_IB_ANY_PORT -1
const std::string kCtranIbLogEventName{"CtranIb-QpExchange"};

const uint64_t kBootstrapMagic = 0xfaceb00cdeadbeef;
}; // namespace

thread_local std::unordered_map<void*, std::atomic_bool> epochLockedFlags;

commResult_t checkEpochLock(CtranIb* ctranIb) {
  if (NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK &&
      NCCL_CTRAN_IB_EPOCH_LOCK_ENABLE && !epochLockedFlags[ctranIb].load()) {
    CLOGF(
        ERR,
        "NCCL_CTRAN_IB_EPOCH_LOCK_ENABLE is set to true, but the critical section is called "
        "without acquiring the epoch lock on the calling thread. It is likely a NCCL bug.");
    return commInternalError;
  }
  return commSuccess;
}

CtranIbSingleton& CtranIbSingleton::getInstance(void) {
  static CtranIbSingleton s;
  return s;
}

CtranIbSingleton::CtranIbSingleton(void) {
  auto ibvInitResult = ibverbx::ibvInit();
  FOLLY_EXPECTED_CHECKTHROW(ibvInitResult);
  auto maybeDeviceList = ibverbx::IbvDevice::ibvGetDeviceList(
      NCCL_IB_HCA, NCCL_IB_HCA_PREFIX, CTRAN_IB_ANY_PORT);
  FOLLY_EXPECTED_CHECKTHROW(maybeDeviceList);
  ibvDevices = std::move(*maybeDeviceList);

  if (ibvDevices.size() < NCCL_CTRAN_IB_DEVICES_PER_RANK) {
    CLOGF(
        WARN,
        "CTRAN-IB : active device found {} is less than requested device count {}",
        ibvDevices.size(),
        NCCL_CTRAN_IB_DEVICES_PER_RANK);
    throw std::bad_alloc();
  }

  for (auto i = 0; i < this->ibvDevices.size(); i++) {
    auto maybePd = this->ibvDevices[i].allocPd();
    FOLLY_EXPECTED_CHECKTHROW(maybePd);
    this->ibvPds.wlock()->push_back(std::move(*maybePd));
  }

  if (NCCL_IB_ASYNC_EVENT_LOOP == NCCL_IB_ASYNC_EVENT_LOOP::ctran) {
    int cudaDev;
    FB_CUDACHECKIGNORE(cudaGetDevice(&cudaDev));
    this->ibAsyncEventThread_ = std::thread{ibAsyncEventHandler, cudaDev};
  }

  this->devBytes_.resize(this->ibvDevices.size());
  for (auto& p : this->devBytes_) {
    p = std::make_unique<std::atomic<size_t>>(0);
  }

  return;
}

commResult_t CtranIbSingleton::destroy() {
  // No-op if already destroyed and skipped
  if (destroySkipped_) {
    return commSuccess;
  }

  // Ask eventHandler thread to stop; since the thread merely handle local
  // operation, it should return immediately and not hang. Thus, stop no matter
  // the commAbort scope.
  this->stopIbAsyncEventHandler();

  // Report traffic if enabled
  if (NCCL_SLOW_RANK_ENABLE) {
    for (int i = 0; i < devBytes_.size(); i++) {
      CLOGF_SUBSYS(
          INFO,
          INIT,
          "CTRAN-IB: [traffic profiling] cudaDev {} total traffic: {} bytes",
          i,
          devBytes_[i]->load());
    }
  }

  // Skip remaining resource leak check and cleanup if commAbort is
  // intentionally skipped. This avoids annoying WARN logs.
  if (ctran::utils::getSkipDestroyCtran()) {
    destroySkipped_ = true;
    return commSuccess;
  }

  // Below reports any resource leak and cleanup network resource.
  // NOTE: If any outstanding comm or IB registration exists, it is known that
  // the resource cleanup calls can fail.
  this->comms_.withRLock([&](auto& comms) {
    if (comms.size()) {
      for (auto& it : comms) {
        CLOGF(
            WARN,
            "CTRAN-IB: communicator {} are still alive when CtranIbSingleton is destroyed.",
            (void*)it);
      }
    }
  });

  size_t activeRegCount = getActiveRegCount();
  if (activeRegCount > 0) {
    CLOGF(
        WARN,
        "CTRAN-IB: {} active buffer registrations when CtranIbSingleton is destroyed.",
        activeRegCount);
  }

  return commSuccess;
}

CtranIbSingleton::~CtranIbSingleton() {
  FB_COMMCHECKIGNORE(destroy());
  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

bool CtranIbSingleton::getDevToDmaBufSupport(int cudaDev) {
  std::lock_guard<std::mutex> guard(this->dmaBufSupportMutex_);
  auto result = devToDmaBufSupport.emplace(cudaDev, false);
  if (result.second) {
    // Successful insertion means the device is not in the map. Let us
    // initialize it explicitly.
    result.first->second =
        (ctran::utils::commCudaLibraryInit() == commSuccess) &&
        (ctran::utils::dmaBufDriverSupport(cudaDev) == commSuccess);
  }
  return result.first->second;
}

size_t CtranIbSingleton::getDeviceTrafficSnapshot(const int cudaDev) {
  return devBytes_[cudaDev]->load();
}

/* static */
IVerbsWrapper* CtranIbSingleton::getVerbsPtr() {
  CtranIbSingleton& s = CtranIbSingleton::getInstance();
  return s.verbsUtils->verbsPtr.get();
}

void CtranIbSingleton::recordDeviceTraffic(
    ibverbx::ibv_context* ctx,
    const int cudaDev,
    size_t nbytes) {
  *this->devBytes_[cudaDev] += nbytes;
}

void CtranIbSingleton::commRef(CtranComm* comm) {
  auto comms = this->comms_.wlock();
  comms->insert(comm);
}

void CtranIbSingleton::commDeref(CtranComm* comm) {
  auto comms = this->comms_.wlock();
  comms->erase(comm);
}

size_t CtranIbSingleton::getActiveRegCount() {
  return activeRegCount_.load();
}

void CtranIbSingleton::incActiveRegCount() {
  activeRegCount_++;
}

void CtranIbSingleton::decActiveRegCount() {
  activeRegCount_--;
}

// startIbAsyncEventHandler provided for testing
void CtranIbSingleton::startIbAsyncEventHandler(const int cudaDev) {
  if (!this->ibAsyncEventThread_.joinable()) {
    this->stopIbAsyncEventHandlerFlag = false;
    this->ibAsyncEventThread_ = std::thread{ibAsyncEventHandler, cudaDev};
  }
}

void CtranIbSingleton::stopIbAsyncEventHandler() {
  if (!this->ibAsyncEventThread_.joinable()) {
    return;
  }

  // ask stopIbAsyncEventHandler() to stop.
  this->stopIbAsyncEventHandlerFlag = true;
  this->ibAsyncEventThread_.join();
}

/* static */
bool CtranIbSingleton::getStopIbAsyncEventHandlerFlag() {
  CtranIbSingleton& s = CtranIbSingleton::getInstance();
  return s.stopIbAsyncEventHandlerFlag.load();
}

/*
 * IB async event thread - single thread per process
 * If, in the future, we have more than one port,
 * this thread can be modified to poll all ports.
 */
void CtranIbSingleton::ibAsyncEventHandler(const int cudaDev) {
  FB_CUDACHECKIGNORE(cudaSetDevice(cudaDev));
  commNamedThreadStart("CTranIbAsyncEventHandler");

  CtranIbSingleton& s = CtranIbSingleton::getInstance();
  auto verbsPtr = CtranIbSingleton::getVerbsPtr();
  const auto devName = s.ibvDevices[cudaDev].device()->name;
  const auto port = s.ibvDevices[cudaDev].port();

  auto ib = std::make_unique<IbUtils>();
  std::string errorMessage = "initialized";

  while (!s.stopIbAsyncEventHandlerFlag && !ib->linkDownTimeout()) {
    auto ibvContext = s.ibvDevices[cudaDev].context();
    // this call will block until there is an async event or error
    if (commSuccess != ib->pollForAsyncEvent(ibvContext, verbsPtr)) {
      return;
    }

    ibverbx::ibv_async_event event;
    // if there is no async event to read, ibv_get_async_event will return -1
    // this would be a code bug if it happens
    if (commSuccess != verbsPtr->ibv_get_async_event(ibvContext, &event)) {
      errorMessage = "ibv_get_async_event call failed";
      break;
    }

    if (commSuccess != verbsPtr->ibv_ack_async_event(&event)) {
      errorMessage = "ibv_ack_async_event call failed";
      break;
    }

    if (commSuccess !=
        ib->triageIbAsyncEvents(event.event_type, devName, port)) {
      break;
    }
  }
  CLOGF_SUBSYS(INFO, INIT, "CTRAN-IB: Exiting ibAsyncEventHandler");
}

CtranIb::CtranIb(
    CtranComm* comm,
    CtranCtrlManager* ctrlMgr,
    std::optional<bool> enableLocalFlush)
    : comm(comm) {
  // enableLocalFlush: whether to support local flush. If true, CtranIb
  // will enable resource required for local flush.
  bool enableLocalFlush_ = false;
  if (enableLocalFlush.has_value()) {
    // Honor user specified value
    enableLocalFlush_ = enableLocalFlush.value();
  } else {
#if defined(USE_ROCM)
    // AMD GPUs always require local flush
    // https://ontrack.amd.com/browse/FBA-633
    enableLocalFlush_ = true;
#else
    // Turn on flush for NVidia GPUs older than H100
    enableLocalFlush_ = comm->statex_->cudaArch() < 900;
#endif
  }
  init(
      comm,
      comm->statex_->rank(),
      comm->statex_->cudaDev(),
      comm->statex_->commHash(),
      comm->statex_->commDesc(),
      ctrlMgr,
      enableLocalFlush_);
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: Initialized {} from comm {}",
      (void*)this,
      (void*)comm);
}

CtranIb::CtranIb(
    int rank,
    int cudaDev,
    uint64_t commHash,
    const std::string& commDesc,
    CtranCtrlManager* ctrlMgr,
    // FIXME: Unlike IB with comm, we require user to always specify
    // enableLocalFlush because we don't have access to statex->cudaArch() for
    // default config detection
    bool enableLocalFlush,
    const BootstrapMode bootstrapMode,
    std::optional<const SocketServerAddr*> qpServerAddr,
    std::shared_ptr<Abort> abortCtrl) {
  init(
      nullptr,
      rank,
      cudaDev,
      commHash,
      commDesc,
      ctrlMgr,
      enableLocalFlush,
      bootstrapMode,
      qpServerAddr,
      abortCtrl);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: Initialized {} from rank {} cudaDev {} commHash {:x} commDesc {}",
      (void*)this,
      rank,
      cudaDev,
      commHash,
      commDesc);
}

void CtranIb::init(
    CtranComm* comm,
    int rank,
    int cudaDev,
    uint64_t commHash,
    const std::string& commDesc,
    CtranCtrlManager* ctrlMgr,
    bool enableLocalFlush,
    const BootstrapMode bootstrapMode,
    std::optional<const SocketServerAddr*> qpServerAddr,
    std::shared_ptr<Abort> abortCtrl) {
  bool foundPort = false;
  this->comm = comm;
  this->abortCtrl_ = comm ? comm->getAbort() : abortCtrl;
  this->rank = rank;
  this->cudaDev = cudaDev;
  this->commHash = commHash;
  this->commDesc = commDesc;
  this->ctrlMgr = ctrlMgr;
  this->ncclLogData = CommLogData{
      .commId = comm ? comm->logMetaData_.commId : 0,
      .commHash = commHash,
      .commDesc = commDesc,
      .rank = rank,
      .nRanks = comm ? comm->statex_->nRanks() : 1};
  this->enableLocalFlush = enableLocalFlush;
  this->bootstrapMode = bootstrapMode;
  this->devices.resize(NCCL_CTRAN_IB_DEVICES_PER_RANK);
  this->cqs.reserve(NCCL_CTRAN_IB_DEVICES_PER_RANK);
  FB_COMMCHECKTHROW(this->setPgToTrafficClassMap());

  CtranIbSingleton& s = CtranIbSingleton::getInstance();

  const bool dmaBufSupported = s.getDevToDmaBufSupport(cudaDev);

  const bool commAbortEnabled = comm && comm->abortEnabled();
  if (commAbortEnabled && abortCtrl->Enabled()) {
    throw ::ctran::utils::Exception(
        "CtranIB::init called on two different enabled Abort controls",
        commInternalError,
        this->rank,
        this->commHash,
        this->commDesc);
  }

  if (NCCL_CTRAN_IB_DEVICES_PER_RANK > CTRAN_MAX_IB_DEVICES_PER_RANK) {
    std::string msg = "NCCL_CTRAN_IB_DEVICES_PER_RANK (" +
        std::to_string(NCCL_CTRAN_IB_DEVICES_PER_RANK) +
        ") exceeds CTRAN_MAX_IB_DEVICES_PER_RANK (" +
        std::to_string(CTRAN_MAX_IB_DEVICES_PER_RANK) + ")";
    CLOGF(ERR, "CTRAN-IB: {}", msg);
    throw std::runtime_error(msg.c_str());
  }

  // assume NCCL_CTRAN_IB_DEVICES_PER_RANK contexts per cuda device
  if (cudaDev * NCCL_CTRAN_IB_DEVICES_PER_RANK * NCCL_CTRAN_IB_DEVICE_STRIDE >=
      s.ibvDevices.size()) {
    std::string msg = "cudaDev (" + std::to_string(cudaDev) +
        ") * NCCL_CTRAN_IB_DEVICES_PER_RANK * NCCL_CTRAN_IB_DEVICE_STRIDE exceeds the number of contexts (" +
        std::to_string(s.ibvDevices.size()) + ")";
    CLOGF(ERR, "CTRAN-IB: {}", msg);
    throw std::runtime_error(msg.c_str());
  }

  for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; ++device) {
    int singletonDevIdx =
        cudaDev * NCCL_CTRAN_IB_DEVICES_PER_RANK * NCCL_CTRAN_IB_DEVICE_STRIDE +
        device;
    devices[device].ibvDevice = &s.ibvDevices[singletonDevIdx];
    devices[device].ibvPd = &s.ibvPds.rlock()->at(singletonDevIdx);

    ibverbx::ibv_device_attr devAttr;
    auto maybeDeviceAttr = devices[device].ibvDevice->queryDevice();
    FOLLY_EXPECTED_CHECKTHROW(maybeDeviceAttr);
    devAttr = std::move(*maybeDeviceAttr);

    // Found available port for the given device
    for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
      ibverbx::ibv_port_attr portAttr;
      auto maybePortAttr = s.ibvDevices[singletonDevIdx].queryPort(port);
      if (maybePortAttr.hasError()) {
        CLOGF(
            WARN,
            "CTRAN-IB : Unable to query port {} on device {}",
            port,
            s.ibvDevices[singletonDevIdx].device()->name);
        continue;
      }
      portAttr = std::move(*maybePortAttr);
      if (portAttr.state != ibverbx::IBV_PORT_ACTIVE) {
        continue;
      }
      if (portAttr.link_layer != ibverbx::IBV_LINK_LAYER_INFINIBAND &&
          portAttr.link_layer != ibverbx::IBV_LINK_LAYER_ETHERNET) {
        continue;
      }
      if (s.ibvDevices[singletonDevIdx].port() == CTRAN_IB_ANY_PORT ||
          port == s.ibvDevices[singletonDevIdx].port()) {
        devices[device].port = port;
        devices[device].devName = s.ibvDevices[singletonDevIdx].device()->name;
        foundPort = true;
        break;
      }
    }

    if (foundPort) {
      CLOGF_SUBSYS(
          INFO,
          INIT,
          "CTRAN-IB: using device {}, port {} commHash {:x} commDesc {}",
          s.ibvDevices[singletonDevIdx].device()->name,
          devices[device].port,
          commHash,
          commDesc);
    } else {
      CLOGF(
          WARN,
          "CTRAN-IB : No active port found on device {}. Disable IB backend.",
          s.ibvDevices[singletonDevIdx].device()->name);
      throw std::runtime_error(
          std::string("CTRAN-IB : No active port found on device ") +
          s.ibvDevices[singletonDevIdx].device()->name);
    }

    /* The max CQEs would not be enough for us in the worst case, where
     * we have a lot of VCs, and there is a lot of posted messages on
     * each of the VCs.  Static partitioning would reduce the number of
     * CQEs available to each VC in the common case.  Instead, we are
     * making an assumption here that the progress thread will pull out
     * completion entries fast enough that we will never overflow the
     * CQ. */

#if defined(USE_ROCM)
    // We found that IB initialization would fail with Thor2 NIC
    // if maxCqe is not smaller than devAttr.max_cqe.
    maxCqe = devAttr.max_cqe - 1;
#else
    maxCqe = devAttr.max_cqe;
#endif

    // Skip lock for cq and localVc in constructor since no other thread can
    // access it yet.
    auto maybeCq =
        devices[device].ibvDevice->createCq(maxCqe, nullptr, nullptr, 0);
    FOLLY_EXPECTED_CHECKTHROW(maybeCq);
    cqs.emplace_back(std::move(*maybeCq));
    devices[device].ibvCq = &cqs[device];
    // FIXME: use initRemoteTransStates() to create cq
  }

  if (enableLocalFlush) {
    localVc = std::make_unique<LocalVirtualConn>(devices, commHash, commDesc);
  }

  // Record reference to CtranIbSingleton
  if (comm) {
    s.commRef(comm);
    // TODO: we would need track non-comm transport usage and its leak as well.
    // It is because, e.g., if FTAR doesn't release transport at failure,
    // ibv_destroy_pd will fail.
  }

  // Reset flags for invalid use case check in epochLock() and epochUnlock()
  epochLockedFlags[this].store(false);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: dmabuf support for device {} is {}",
      cudaDev,
      dmaBufSupported);

  // assign the lock-free vcStateMaps ptr for fast path access when applicable
  {
    auto locked = vcStateMaps.wlock();
    vcStateMapsPtr = &(*locked);
  }

  if (comm) {
    // initialize connection map before preConnect() is called by any algorithm
    connectedPeerMap.resize(comm->statex_->nRanks(), false);
  }

  // Optionally start internal bootstrap service.
  // If kExternal, external callsite would explicitly manage it and thus we skip
  // the internal bootstrap.
  if (this->bootstrapMode == BootstrapMode::kExternal) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-IB: skip internal bootstrap for IB backend {} commHash {:x} commDesc {}",
        (void*)this,
        commHash,
        commDesc);
  } else {
    bootstrapStart(qpServerAddr);
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: created IB backend {} for commHash {:x} commDesc {}",
      (void*)this,
      commHash,
      commDesc);
}

void CtranIb::bootstrapStart(
    std::optional<const SocketServerAddr*> qpServerAddr) {
  // Setup the listen socket
  std::string* ifnamePtr = nullptr;
  folly::SocketAddress addrSockAddr;
  if (this->bootstrapMode == BootstrapMode::kDefaultServer) {
    ifnamePtr = &NCCL_SOCKET_IFNAME;
    // Use default NCCL socket ifname
    auto maybeAddr = ctran::bootstrap::getInterfaceAddress(
        NCCL_SOCKET_IFNAME, NCCL_SOCKET_IPADDR_PREFIX);
    if (maybeAddr.hasError()) {
      CLOGF(WARN, "CTRAN-IB: No socket interfaces found");
      throw std::runtime_error("CTRAN-IB : No socket interfaces found");
    }

    addrSockAddr = folly::SocketAddress(maybeAddr.value(), 0 /* port */);
  } else {
    FB_CHECKABORT(
        qpServerAddr.has_value(),
        "CTRAN-IB: Expect bootstrap with specified server address, but is not provided. It indicates a COMM bug");

    auto qpServerAddrPtr = qpServerAddr.value();
    // use provided addr(i.e. ip, port, host) to initialize ctranIB
    addrSockAddr = toSocketAddress(*qpServerAddrPtr);
    ifnamePtr = const_cast<std::string*>(&qpServerAddrPtr->ifName);
  }

  FB_SYSCHECKTHROW(this->listenSocket.bindAndListen(addrSockAddr, *ifnamePtr));
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: Rank {} created listen socket with {} listenAddr {} ifname {}",
      rank,
      bootstrapMode == BootstrapMode::kSpecifiedServer ? "specified"
                                                       : "self-finding",
      this->listenSocket.getListenAddress()->describe().c_str(),
      *ifnamePtr);

  // Exchange listen sock address among all ranks
  if (comm) {
    allListenSocketAddrs.resize(comm->statex_->nRanks());
    auto maybeListenAddr = this->listenSocket.getListenAddress();
    if (maybeListenAddr.hasError()) {
      FB_SYSCHECKTHROW(maybeListenAddr.error());
    }
    maybeListenAddr->getAddress(&allListenSocketAddrs[rank]);

    auto resFuture = comm->bootstrap_->allGather(
        allListenSocketAddrs.data(),
        sizeof(allListenSocketAddrs.at(0)),
        comm->statex_->rank(),
        comm->statex_->nRanks());
    FB_COMMCHECKTHROW(static_cast<commResult_t>(std::move(resFuture).get()));
  }

  this->listenThread = std::thread{bootstrapAccept, this};
}

void CtranIb::regCtrlCb(std::unique_ptr<CtranCtrlManager>& ctrlMgr) {
  // no control message callback to register; no-op.
}

std::string CtranIb::getIbDevName(int device) const {
  return std::string(this->devices.at(device).devName);
}

int CtranIb::getIbDevPort(int device) const {
  return this->devices.at(device).port;
}

CtranIb::~CtranIb(void) {
  CtranIbSingleton& s = CtranIbSingleton::getInstance();

  if (bootstrapMode != BootstrapMode::kExternal) {
    listenSocket.shutdown();
    listenThread.join();
  }

  FB_COMMCHECKIGNORE(releaseRemoteTransStates(true /* fromDestructor */));

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: destroyed IB backend {} for commHash {:x} commDesc {}",
      (void*)this,
      commHash,
      commDesc);

  // this comm is being destroyed, thus dereference from CtranIbSingleton
  if (this->comm) {
    s.commDeref(this->comm);
  }

  // Do not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

// Checks for invalid use cases of epochLock() and epochUnlock():
// 1. Always enabled check at lock/unlock time via thread local
// epochLockedFlags:
// - A thread locks twice
// - A thread unlocks without outstanding lock
// 2. Optionally enabled check at every critical path entry (search
// checkEpochLock()):
// - A thread invokes a critical path call without an outstanding lock
commResult_t CtranIb::epochLock() {
  if (!NCCL_CTRAN_IB_EPOCH_LOCK_ENABLE) {
    return commSuccess;
  }

  if (epochLockedFlags[this].load()) {
    CLOGF(
        ERR,
        "epochLock() already called on the same thread without epochUnlock(). It is likely a COMM bug.");
    return commInternalError;
  }

  epochMutex.lock();
  epochLockedFlags[this].store(true);

  return commSuccess;
}

commResult_t CtranIb::epochTryLock() {
  if (!NCCL_CTRAN_IB_EPOCH_LOCK_ENABLE) {
    return commSuccess;
  }

  if (epochLockedFlags[this].load()) {
    CLOGF(
        ERR,
        "epochLock() already called on the same thread without epochUnlock(). It is likely a COMM bug.");
    return commInternalError;
  }

  if (epochMutex.try_lock()) {
    epochLockedFlags[this].store(true);
    return commSuccess;
  } else {
    return commInProgress;
  }
}

commResult_t CtranIb::epochUnlock() {
  if (!NCCL_CTRAN_IB_EPOCH_LOCK_ENABLE) {
    return commSuccess;
  }

  if (!epochLockedFlags[this].load()) {
    CLOGF(
        ERR,
        "epochUnlock() is called without an outstanding epochLock() on the thread. It is likely a COMM bug.");
    return commInternalError;
  }

  epochLockedFlags[this].store(false);
  epochMutex.unlock();
  return commSuccess;
}

commResult_t CtranIb::regMem(
    const void* buf,
    const size_t len,
    const int cudaDev,
    void** ibRegElem) {
  commResult_t res = commSuccess;
  if (len < CTRAN_MIN_REGISTRATION_SIZE && NCCL_CTRAN_REGISTRATION_SIZE_CHECK) {
    CLOGF(
        ERR,
        "CTRAN-IB: cannot register buffer, size ({}) < {}",
        len,
        CTRAN_MIN_REGISTRATION_SIZE);
    return commSystemError;
  }

  CtranIbSingleton& s = CtranIbSingleton::getInstance();
  const auto dmaBufSupport = s.getDevToDmaBufSupport(cudaDev);

  bool useDmaBuf = dmaBufSupport && NCCL_CTRAN_IB_DMABUF_ENABLE;
  int dmaBufFd = -1;
  if (useDmaBuf) {
    // Check if the buffer is backed by cumem if dmaBuf is enabled;
    // As of now, it would return a valid fd if the buffer is backed by cumem
    // and is aligned with 4KB; otherwise -1 is returned. In such a case, we
    // would fall back to ibv_reg_mr.
    dmaBufFd = ctran::utils::getCuMemDmaBufFd(buf, len);
    useDmaBuf = (dmaBufFd != -1);
  }

  CLOGF_TRACE(
      ALLOC,
      "CTRAN-IB: regMem buf={}, len={}, useDmaBuf={}, dmaBufSupport={}",
      buf,
      len,
      useDmaBuf,
      dmaBufSupport);

  auto mrs = new std::vector<ibverbx::IbvMr>();
  mrs->reserve(NCCL_CTRAN_IB_DEVICES_PER_RANK);
  ibverbx::ibv_access_flags access = static_cast<ibverbx::ibv_access_flags>(
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_REMOTE_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_READ | ibverbx::IBV_ACCESS_REMOTE_ATOMIC);

  for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
    const auto pdIdx =
        cudaDev * NCCL_CTRAN_IB_DEVICES_PER_RANK * NCCL_CTRAN_IB_DEVICE_STRIDE +
        device;
    if (useDmaBuf) {
      auto maybeDmabufMr = s.ibvPds.rlock()->at(pdIdx).regDmabufMr(
          0, len, reinterpret_cast<uint64_t>(buf), dmaBufFd, access);
      FOLLY_EXPECTED_CHECKGOTO(maybeDmabufMr, fail);
      mrs->emplace_back(std::move(*maybeDmabufMr));
    } else {
      // fall back to ibv_reg_mr
      if (comms::utils::cumem::isBackedByMultipleCuMemAllocations(
              buf, cudaDev, len)) {
        CLOGF(
            ERR,
            "CTRAN-IB: Memory region (buf {}, len {}) spans multiple cuMem allocations, ibv_reg_mr may fail!",
            buf,
            len);
        return commInvalidUsage;
      }
      auto maybeMr = s.ibvPds.rlock()->at(pdIdx).regMr((void*)buf, len, access);
      FOLLY_EXPECTED_CHECKGOTO(maybeMr, fail);
      mrs->emplace_back(std::move(*maybeMr));
    }
  }
  if (dmaBufFd != -1) {
    (void)close(dmaBufFd);
  }
  *ibRegElem = reinterpret_cast<void*>(mrs);
  s.incActiveRegCount();

  return res;

fail:
  CLOGF(
      ERR,
      "CTRAN-IB: buffer registration failed: buf = {}, len = {}, dmaBufFd = {}, dmaBufSupport = {}, NCCL_CTRAN_IB_DMABUF_ENABLE = {}, useDmaBuf = {}",
      buf,
      len,
      dmaBufFd,
      dmaBufSupport,
      NCCL_CTRAN_IB_DMABUF_ENABLE,
      useDmaBuf);
  return commSystemError;
}

commResult_t CtranIb::deregMem(void* ibRegElem) {
  auto mrs = reinterpret_cast<std::vector<ibverbx::IbvMr>*>(ibRegElem);

  CtranIbSingleton& s = CtranIbSingleton::getInstance();
  s.decActiveRegCount();

  delete mrs;

  return commSuccess;
}

commResult_t CtranIb::iflush(
    const void* dbuf,
    const void* localRegHdl,
    CtranIbRequest* req) {
  FB_COMMCHECK(checkEpochLock(this));

  if (enableLocalFlush) {
    CTRAN_IB_PER_OBJ_LOCK_GUARD(localVcMutex, {
      auto& vc = localVc;
      return vc->iflush(dbuf, localRegHdl, req);
    });
  } else {
    req->complete();
    return commSuccess;
  }
}

commResult_t CtranIb::getVcConfig(int peerRank, CtranIbVcConfig_t& config) {
  std::shared_ptr<CtranIbVirtualConn> vc = getVcImpl(peerRank);
  FB_COMMCHECK(checkValidVc(vc, peerRank));

  config = {
      vc->getQpScalingTh(),
      vc->getMaxNumQp(),
      vc->getVcMode(),
      vc->getMaxQpMsgs()};
  return commSuccess;
}

commResult_t CtranIb::releaseRemoteTransStates(bool fromDestructor) {
  // Ensure no other thread is accessing the CtranIb object while releasing
  // resource. Skip epochLock if called from destructor
  CtranIbEpochRAII epochRAII(fromDestructor ? nullptr : this);

  { // Explicitly release all virtual connections before destroying cq.
    auto lockedVcStateMaps = vcStateMaps.wlock();
    lockedVcStateMaps->rankToVcMap.clear();
    lockedVcStateMaps->qpToVcMap.clear();
    vcStateMapsPtr = nullptr;
  }

  // Epoch lock only ensures no external access to CtranIb while releasing
  // resources; We still need per-object lock here to ensure the internal
  // listenThread doesn't read garbage data
  // Release local VC since we need destroy the shared cq
  {
    std::unique_lock<std::mutex> lock(localVcMutex);
    localVc.reset();
  }

  // Clear pending ops - The cache is managed by ctranIB rather than ibverbs,
  // so it can be sent via a new established qp/cq if we don't clear it when
  // reset CtranIB.
  {
    std::unique_lock<std::mutex> lock(pendingOpsMutex);
    rankToPendingOpsMap.clear();
  }

  this->connectedPeerMap.clear();
  return commSuccess;
}

// Reset CtranIb backend qps and cq state
commResult_t CtranIb::initRemoteTransStates(void) {
  // Ensure no other thread is accessing the CtranIb object while releasing
  // resource
  CtranIbEpochRAII epochRAII(this);

  // Epoch lock only ensures no external access to CtranIb while releasing
  // resources; We still need per-object lock here to ensure the internal
  // listenThread doesn't read garbage data

  this->cqs.reserve(NCCL_CTRAN_IB_DEVICES_PER_RANK);

  // create a new cq
  {
    std::unique_lock<std::mutex> lock(cqMutex);
    for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
      auto createCqResult =
          devices[device].ibvDevice->createCq(maxCqe, nullptr, nullptr, 0);
      FOLLY_EXPECTED_CHECK(createCqResult);
      cqs.emplace_back(std::move(*createCqResult));
      devices[device].ibvCq = &this->cqs[device];
      FB_COMMCHECK(
          ctran::PtrCheck(
              devices[device].ibvCq, "initRemoteTransStates", "cq"));
    }
  }

  // create local VC
  {
    std::unique_lock<std::mutex> lock(localVcMutex);
    localVc = std::make_unique<LocalVirtualConn>(devices, commHash, commDesc);
  }

  cqMutex.unlock();
  localVcMutex.unlock();

  return commSuccess;
}

commResult_t CtranIb::preConnect(const std::unordered_set<int>& peerRanks) {
  std::shared_ptr<CtranIbVirtualConn> vc = nullptr;
  bool newConnection = false;
  // if map is empty, it means we don't know comm size and peerAddr
  // FIXME: revisit preConnect for CtranEx if needed
  if (connectedPeerMap.empty()) {
    return commSuccess;
  }
  // Actively connect to peers with larger rank number, if not already connected
  for (int peerRank : peerRanks) {
    FB_COMMCHECK(checkValidPeer(peerRank));
    if (rank < peerRank && getVcImpl(peerRank) == nullptr) {
      FB_COMMCHECK(bootstrapConnect(peerRank));
    }
  }
  // check if all requested peers are connected
  for (int peerRank : peerRanks) {
    if (!connectedPeerMap.at(peerRank)) {
      while (getVcImpl(peerRank) == nullptr) {
        // wait listening thread to establish connections with rest of peers
        // with smaller rank number
      }
      connectedPeerMap.at(peerRank) = true;
      newConnection = true;
    }
  }

  if (newConnection) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-IB: Rank-{} Pre-connected peers: [{}]",
        rank,
        unorderedSetToStr(peerRanks));
  }

  return commSuccess;
}

commResult_t CtranIb::connectVc(
    ctran::bootstrap::Socket& sock,
    const bool isServer,
    const int peerRank) {
  // Create a new VC for the peer
  FB_COMMCHECK(this->checkValidPeer(peerRank));
  std::shared_ptr<CtranIbVirtualConn> vc = createVc(peerRank);

  std::string localBusCard, remoteBusCard;
  {
    // No need to lock since VC is not yet exposed to local rank. Lock to
    // simply follow VC thread-safety semantics.
    const std::lock_guard<std::mutex> lock(vc->mutex);

    /* exchange business cards */
    std::size_t size = vc->getBusCardSize();
    localBusCard.resize(size);
    remoteBusCard.resize(size);
    FB_COMMCHECK(vc->getLocalBusCard(localBusCard.data()));
    if (isServer) {
      FB_SYSCHECKRETURN(sock.recv(remoteBusCard.data(), size), commRemoteError);
      FB_SYSCHECKRETURN(sock.send(localBusCard.data(), size), commRemoteError);
    } else {
      FB_SYSCHECKRETURN(sock.send(localBusCard.data(), size), commRemoteError);
      FB_SYSCHECKRETURN(sock.recv(remoteBusCard.data(), size), commRemoteError);
    }
  }

  connectVcDirect(remoteBusCard, peerRank);

  // Ack that the connection is fully established.
  // Ensure remote rank don't use the VC before local setupVc and
  // vcStateMaps update.
  int ack{0};
  if (isServer) {
    FB_SYSCHECKRETURN(sock.send(&ack, sizeof(int)), commRemoteError);
    FB_SYSCHECKRETURN(sock.recv(&ack, sizeof(int)), commRemoteError);
  } else {
    FB_SYSCHECKRETURN(sock.recv(&ack, sizeof(int)), commRemoteError);
    FB_SYSCHECKRETURN(sock.send(&ack, sizeof(int)), commRemoteError);
  }
  return commSuccess;
}

std::string CtranIb::getLocalVcIdentifier(const int peerRank) {
  std::string localBusCard;
  auto vc = createVc(peerRank);
  {
    const std::lock_guard<std::mutex> lock(vc->mutex);
    localBusCard.resize(vc->getBusCardSize());
    FB_COMMCHECKTHROW(vc->getLocalBusCard(localBusCard.data()));
  }
  return localBusCard;
}

commResult_t CtranIb::connectVcDirect(
    const std::string& remoteVcIdentifier,
    const int peerRank) {
  // Connect the VC
  auto vc = createVc(peerRank);
  {
    const std::lock_guard<std::mutex> lock(vc->mutex);
    FB_COMMCHECKTHROW(vc->setupVc((void*)remoteVcIdentifier.data()));
  }

  uint32_t controlQp = vc->getControlQpNum();
  uint32_t notifyQp = vc->getNotifyQpNum();
  uint32_t atomicQp = vc->getAtomicQpNum();
  std::vector<uint32_t> dataQps = vc->getDataQpNums();

  // Till now VC is not yet exposed to local rank. Local rank can use the VC
  // once updated the vcStateMaps.
  FB_COMMCHECKTHROW(updateVcState(vc, peerRank));

  // Remove from pendingVcs_ since the VC is now established
  pendingVcs_.erase(peerRank);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: Established connection: commHash {:x}, commDesc {}, "
      "vc {}, rank {}, peer {}, control qpn {}, notify qpn {}, atomic qpn {}, data qpns {}",
      commHash,
      commDesc,
      (void*)vc.get(),
      rank,
      peerRank,
      controlQp,
      notifyQp,
      atomicQp,
      vecToStr(dataQps));

  return commSuccess;
}

void CtranIb::bootstrapAccept(CtranIb* ib) {
  // Set cudaDev for logging
  FB_CUDACHECKTHROW(cudaSetDevice(ib->cudaDev));
  commNamedThreadStart(
      "CTranIbListen", ib->rank, ib->commHash, ib->commDesc.c_str(), __func__);
  while (1) {
    // Accept a connection from a peer. Socket will automatically closed when
    // it'll go out of scope (part of its destructor)
    auto maybeSocket = ib->listenSocket.accept();
    if (maybeSocket.hasError()) {
      if (ib->listenSocket.hasShutDown()) {
        break; // listen socket is closed
      }
      FB_SYSCHECKTHROW(maybeSocket.error());
    }

    auto& socket = maybeSocket.value();

    uint64_t magic{0};
    FB_SYSCHECKTHROW(socket.recv(&magic, sizeof(uint64_t)));
    if (magic != kBootstrapMagic) {
      CLOGF(
          WARN,
          "CTRAN-IB: Invalid magic - received {:x} but expected {:x} for commHash {:x} commDesc {}. "
          "Likely unexpected connection attempt. Ignoring. Local Addr: {},  Peer Addr: {}",
          magic,
          kBootstrapMagic,
          ib->commHash,
          ib->commDesc,
          socket.getLocalAddress().describe(),
          socket.getPeerAddress().describe());
      continue;
    }

    int peerRank;
    FB_SYSCHECKTHROW(socket.recv(&peerRank, sizeof(int)));
    FB_COMMCHECKTHROW(ib->connectVc(socket, true, peerRank));
  }

  CLOGF(
      INFO,
      "CTRAN-IB: Listen thread terminating, rank {} commHash {:x} commDesc {}",
      ib->rank,
      ib->commHash,
      ib->commDesc);
  return;
}

commResult_t CtranIb::bootstrapConnect(
    int peerRank,
    std::optional<const SocketServerAddr*> peerAddr) {
  folly::SocketAddress peerSockAddr;
  const std::string* clientIfName;
  // When peer server address is passed, connect to it directly.
  // Otherwise, use the pre-exchanged listen socket address which requires an
  // associated communicator.
  if (peerAddr.has_value()) {
    auto peerAddrPtr = peerAddr.value();
    peerSockAddr = toSocketAddress(*peerAddrPtr);
    // always use the same ifname as remote server
    clientIfName = &peerAddrPtr->ifName;
  } else {
    FB_CHECKABORT(
        allListenSocketAddrs.size() > 0,
        "Peer address is not specified, but pre-exchanged listen sockets is empty. It indicates a COMM internal bug.");
    peerSockAddr = toSocketAddress(allListenSocketAddrs[peerRank]);
    clientIfName = &NCCL_CLIENT_SOCKET_IFNAME;
  }

  NcclScubaEvent scubaEvent(kCtranIbLogEventName, &ncclLogData);
  scubaEvent.startAndRecord();
  SCOPE_EXIT {
    scubaEvent.stopAndRecord();
  };

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: Establishing connection: commHash {:x}, commDesc {}, rank {}, peer {}, peer listenAddr {} clientIfName {}",
      commHash,
      commDesc,
      rank,
      peerRank,
      peerSockAddr.describe(),
      *clientIfName);

  // Send SETUP command to remote listenThread
  ctran::bootstrap::Socket sock;
  FB_SYSCHECKRETURN(
      sock.connect(
          peerSockAddr,
          *clientIfName,
          std::chrono::milliseconds(NCCL_SOCKET_RETRY_SLEEP_MSEC),
          NCCL_SOCKET_RETRY_CNT),
      commRemoteError);
  FB_SYSCHECKRETURN(
      sock.send(&kBootstrapMagic, sizeof(uint64_t)), commRemoteError);
  FB_SYSCHECKRETURN(sock.send(&rank, sizeof(int)), commRemoteError);
  return connectVc(sock, false, peerRank);
}

const char* CtranIb::ibv_wc_status_str(enum ibverbx::ibv_wc_status status) {
  switch (status) {
    case ibverbx::IBV_WC_SUCCESS:
      return "success";
    case ibverbx::IBV_WC_LOC_LEN_ERR:
      return "local length error";
    case ibverbx::IBV_WC_LOC_QP_OP_ERR:
      return "local QP operation error";
    case ibverbx::IBV_WC_LOC_EEC_OP_ERR:
      return "local EE context operation error";
    case ibverbx::IBV_WC_LOC_PROT_ERR:
      return "local protection error";
    case ibverbx::IBV_WC_WR_FLUSH_ERR:
      return "Work Request Flushed Error";
    case ibverbx::IBV_WC_MW_BIND_ERR:
      return "memory management operation error";
    case ibverbx::IBV_WC_BAD_RESP_ERR:
      return "bad response error";
    case ibverbx::IBV_WC_LOC_ACCESS_ERR:
      return "local access error";
    case ibverbx::IBV_WC_REM_INV_REQ_ERR:
      return "remote invalid request error";
    case ibverbx::IBV_WC_REM_ACCESS_ERR:
      return "remote access error";
    case ibverbx::IBV_WC_REM_OP_ERR:
      return "remote operation error";
    case ibverbx::IBV_WC_RETRY_EXC_ERR:
      return "transport retry counter exceeded";
    case ibverbx::IBV_WC_RNR_RETRY_EXC_ERR:
      return "RNR retry counter exceeded";
    case ibverbx::IBV_WC_LOC_RDD_VIOL_ERR:
      return "local RDD violation error";
    case ibverbx::IBV_WC_REM_INV_RD_REQ_ERR:
      return "remote invalid RD request";
    case ibverbx::IBV_WC_REM_ABORT_ERR:
      return "aborted error";
    case ibverbx::IBV_WC_INV_EECN_ERR:
      return "invalid EE context number";
    case ibverbx::IBV_WC_INV_EEC_STATE_ERR:
      return "invalid EE context state";
    case ibverbx::IBV_WC_FATAL_ERR:
      return "fatal error";
    case ibverbx::IBV_WC_RESP_TIMEOUT_ERR:
      return "response timeout error";
    case ibverbx::IBV_WC_GENERAL_ERR:
      return "general error";
    default:
      return "unrecognized error";
  }
}

commResult_t CtranIb::updateVcState(
    std::shared_ptr<CtranIbVirtualConn> vc,
    int peerRank) {
  auto locked = vcStateMaps.wlock();
  if (locked->rankToVcMap.find(peerRank) != locked->rankToVcMap.end()) {
    CLOGF(
        ERR,
        "CTRAN-IB: VirtualConnection (VC) already exists for peerRank {} in pimpl {} commHash {:x}, commDesc {}. It likely indicates a COMM bug.",
        peerRank,
        (void*)this,
        commHash,
        commDesc);
    return commInternalError;
  }

  locked->rankToVcMap[peerRank] = vc;
  QpUniqueId controlQpId = std::make_pair(vc->getControlQpNum(), 0);
  if (checkAndInsertQpToVcMap(locked->qpToVcMap, controlQpId, vc) !=
      commSuccess) {
    return commInternalError;
  }
  QpUniqueId notifyQpId = std::make_pair(vc->getNotifyQpNum(), 0);
  if (checkAndInsertQpToVcMap(locked->qpToVcMap, notifyQpId, vc) !=
      commSuccess) {
    return commInternalError;
  }
  QpUniqueId atomicQpId = std::make_pair(vc->getAtomicQpNum(), 0);
  if (checkAndInsertQpToVcMap(locked->qpToVcMap, atomicQpId, vc) !=
      commSuccess) {
    return commInternalError;
  }

  std::vector<uint32_t> dataQps = vc->getDataQpNums();
  for (int qpIdx = 0; qpIdx < dataQps.size(); qpIdx++) {
    int device = vc->getIbDevFromQpIdx(qpIdx);
    QpUniqueId qpId = std::make_pair(dataQps[qpIdx], device);
    if (checkAndInsertQpToVcMap(locked->qpToVcMap, qpId, vc) != commSuccess) {
      return commInternalError;
    }
  }

  return commSuccess;
}

std::shared_ptr<CtranIbVirtualConn> CtranIb::createVc(int peerRank) {
  std::lock_guard<std::mutex> lock(cqMutex);
  auto it = pendingVcs_.emplace(peerRank, nullptr);
  if (!it.second) {
    return it.first->second;
  }
  // Create a new VC and assign
  it.first->second = std::make_shared<CtranIbVirtualConn>(
      devices, peerRank, comm, ctrlMgr, getPgToTrafficClassValue(), cudaDev);
  return it.first->second;
}

commResult_t CtranIb::setPgToTrafficClassMap() {
  for (const auto& pgTrafficClassPairStr : NCCL_CTRAN_IB_PG_TRAFFIC_CLASS) {
    std::vector<std::string> pgTrafficClassPair;
    folly::split(":", pgTrafficClassPairStr, pgTrafficClassPair);
    if (pgTrafficClassPair.size() != 2) {
      CLOGF(
          ERR,
          "CTRAN-IB: Invalid PG->Traffic class pair {} in pimpl {} commHash {:x}, commDesc {}.",
          pgTrafficClassPairStr,
          (void*)this,
          commHash,
          commDesc);
      return commInternalError;
    }
    std::string tcStr = pgTrafficClassPair[1];
    auto tc = folly::tryTo<uint32_t>(tcStr);
    if (!tc.hasValue()) {
      CLOGF(
          ERR,
          "CTRAN-IB: Invalid Traffic Class value provided {} in pimpl {} commHash {:x}, commDesc {}.",
          tcStr,
          (void*)this,
          commHash,
          commDesc);
      return commInternalError;
    }
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-IB: commHash {:x}, commDesc {} override traffic class to {}",
        commHash,
        commDesc,
        tc.value());
    pgToTrafficClassMap_[pgTrafficClassPair[0]] = tc.value();
  }
  return commSuccess;
}

uint32_t CtranIb::getPgToTrafficClassValue() const {
  std::vector<std::string> pgCommDescPair;
  folly::split(":", commDesc, pgCommDescPair);
  auto it = pgToTrafficClassMap_.find(pgCommDescPair[0]);
  if (it != pgToTrafficClassMap_.end()) {
    return it->second;
  }
  return NCCL_IB_TC;
}

static folly::once_flag enabledFlag;
static bool enabled;

bool CtranIb::isEnabled() {
  folly::call_once(enabledFlag, [&]() {
    auto it = std::find(
        NCCL_CTRAN_BACKENDS.begin(),
        NCCL_CTRAN_BACKENDS.end(),
        NCCL_CTRAN_BACKENDS::ib);
    enabled = it != NCCL_CTRAN_BACKENDS.end();
  });
  return enabled;
}
