// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/BootstrapExternal.h"
#include "comms/ctran/backends/ib/BootstrapInternal.h"
#include "comms/ctran/backends/ib/CtranIb.h"

#include <fmt/core.h>
#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
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
constexpr int kH100CudaArch = 900;
constexpr int kGb300CudaArch = 1030;
}; // namespace

thread_local std::unordered_map<void*, std::atomic_bool> epochLockedFlags;

bool CtranIb::shouldEnableLocalFlushByDefault(int cudaArch) {
#if defined(USE_ROCM)
  // AMD GPUs always require local flush.
  // https://ontrack.amd.com/browse/FBA-633
  return true;
#else
  // Turn on flush by default for NVidia GPUs older than H100 and for GB300.
  // Multi-NIC topologies with cross-NIC DMA ordering hazards must opt in via
  // NCCL_CTRAN_NET_FORCE_FLUSH=1.
  // TODO: Replace the GB300 CUDA-arch proxy with a topology query similar to
  // baseline ncclTopoNeedFlush(); CUDA arch is not precise topology detection.
  return cudaArch < kH100CudaArch || cudaArch == kGb300CudaArch ||
      NCCL_CTRAN_NET_FORCE_FLUSH;
#endif
}

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

// folly::Singleton instance for CtranIbSingleton
static folly::Singleton<CtranIbSingleton> ctranIbSingleton;

std::shared_ptr<CtranIbSingleton> CtranIbSingleton::getInstance() {
  return ctranIbSingleton.try_get();
}

CtranIbSingleton::CtranIbSingleton() {
  auto ibvInitResult = ibverbx::ibvInit();
  try {
    FOLLY_EXPECTED_CHECKTHROW_EX_NOCOMM(ibvInitResult);
  } catch (const ctran::utils::Exception& e) {
    CLOGF(
        ERR,
        "CTRAN-IB: Failed to initialize ibverbs (libibverbs.so): {}. "
        "Set NCCL_CTRAN_BACKENDS=nvl,socket to use alternative backends.",
        e.what());
    throw;
  }
  auto maybeDeviceList = ibverbx::IbvDevice::ibvGetDeviceList(
      NCCL_IB_HCA, NCCL_IB_HCA_PREFIX, CTRAN_IB_ANY_PORT, NCCL_IB_DATA_DIRECT);
  try {
    FOLLY_EXPECTED_CHECKTHROW_EX_NOCOMM(maybeDeviceList);
  } catch (const ctran::utils::Exception& e) {
    CLOGF(
        ERR,
        "CTRAN-IB: Failed to get InfiniBand device list: {}. "
        "Set NCCL_CTRAN_BACKENDS=nvl,socket to use alternative backends.",
        e.what());
    throw;
  }
  ibvDevices = std::move(*maybeDeviceList);

  if (ibvDevices.size() < NCCL_CTRAN_IB_DEVICES_PER_RANK) {
    std::string msg = fmt::format(
        "CTRAN-IB: Found {} InfiniBand device(s) but {} required "
        "(NCCL_CTRAN_IB_DEVICES_PER_RANK). "
        "Set NCCL_CTRAN_BACKENDS=nvl,socket to use alternative backends.",
        ibvDevices.size(),
        NCCL_CTRAN_IB_DEVICES_PER_RANK);
    CLOGF(ERR, msg);
    throw ctran::utils::Exception(msg, commSystemError);
  }

  for (auto i = 0; i < this->ibvDevices.size(); i++) {
    auto maybePd = this->ibvDevices[i].allocPd();
    FOLLY_EXPECTED_CHECKTHROW_EX_NOCOMM(maybePd);
    this->ibvPds_.push_back(std::move(*maybePd));
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
const ibverbx::IbvPd& CtranIbSingleton::getIbvPd(size_t idx) const {
  return ibvPds_.at(idx);
}

IVerbsWrapper* CtranIbSingleton::getVerbsPtr() {
  auto s = CtranIbSingleton::getInstance();
  CHECK_VALID_IB_SINGLETON(s);
  return s->verbsUtils->verbsPtr.get();
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
  auto s = CtranIbSingleton::getInstance();
  CHECK_VALID_IB_SINGLETON(s);
  return s->stopIbAsyncEventHandlerFlag.load();
}

/*
 * IB async event thread - single thread per process
 * If, in the future, we have more than one port,
 * this thread can be modified to poll all ports.
 */
void CtranIbSingleton::ibAsyncEventHandler(const int cudaDev) {
  FB_CUDACHECKIGNORE(cudaSetDevice(cudaDev));
  commNamedThreadStart("CTranIbAsyncEventHandler");

  auto s = CtranIbSingleton::getInstance();
  CHECK_VALID_IB_SINGLETON(s);
  auto verbsPtr = CtranIbSingleton::getVerbsPtr();
  const auto devName = s->ibvDevices[cudaDev].device()->name;
  const auto port = s->ibvDevices[cudaDev].port();

  auto ib = std::make_unique<IbUtils>();
  std::string errorMessage = "initialized";

  while (!s->stopIbAsyncEventHandlerFlag && !ib->linkDownTimeout()) {
    auto ibvContext = s->ibvDevices[cudaDev].context();
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
    std::optional<bool> enableLocalFlush,
    std::shared_ptr<ctran::bootstrap::ISocketFactory> socketFactory,
    std::optional<int> maxNumCqe)
    : comm(comm) {
  init(
      comm,
      comm->statex_->rank(),
      comm->statex_->cudaDev(),
      comm->statex_->commHash(),
      comm->statex_->commDesc(),
      // Honor a caller-specified value; otherwise use the platform default.
      enableLocalFlush.has_value()
          ? *enableLocalFlush
          : shouldEnableLocalFlushByDefault(comm->statex_->cudaArch()),
      BootstrapMode::kDefaultServer,
      std::nullopt,
      ::ctran::utils::createAbort(/*enabled=*/false),
      socketFactory,
      maxNumCqe);
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: Initialized {} from comm {} enableLocalFlush {}",
      (void*)this,
      (void*)comm,
      this->enableLocalFlush_);
}

CtranIb::CtranIb(
    int rank,
    int cudaDev,
    uint64_t commHash,
    const std::string& commDesc,
    // FIXME: Unlike IB with comm, we require user to always specify
    // enableLocalFlush because we don't have access to statex->cudaArch() for
    // default config detection
    bool enableLocalFlush,
    const BootstrapMode bootstrapMode,
    std::optional<const SocketServerAddr*> qpServerAddr,
    std::shared_ptr<Abort> abortCtrl,
    std::shared_ptr<ctran::bootstrap::ISocketFactory> socketFactory,
    std::optional<int> maxNumCqe) {
  init(
      nullptr,
      rank,
      cudaDev,
      commHash,
      commDesc,
      enableLocalFlush,
      bootstrapMode,
      qpServerAddr,
      abortCtrl,
      socketFactory,
      maxNumCqe);

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: Initialized {} from rank {} cudaDev {} commHash {:x} commDesc {} enableLocalFlush {}",
      (void*)this,
      rank,
      cudaDev,
      commHash,
      commDesc,
      this->enableLocalFlush_);
}

void CtranIb::init(
    CtranComm* comm,
    int rank,
    int cudaDev,
    uint64_t commHash,
    const std::string& commDesc,
    bool enableLocalFlush,
    const BootstrapMode bootstrapMode,
    std::optional<const SocketServerAddr*> qpServerAddr,
    std::shared_ptr<Abort> abortCtrl,
    std::shared_ptr<ctran::bootstrap::ISocketFactory> socketFactory,
    std::optional<int> maxNumCqe) {
  bool foundPort = false;
  this->comm = comm;
  this->rank = rank;
  this->abortCtrl_ = comm ? comm->getAbort() : abortCtrl;
  this->cudaDev = cudaDev;
  this->commHash = commHash;
  this->commDesc = commDesc;
  this->ncclLogData = CommLogData{
      .commId = comm ? comm->logMetaData_.commId : 0,
      .commHash = commHash,
      .commDesc = commDesc,
      .rank = rank,
      .nRanks = comm ? comm->statex_->nRanks() : 1};
  this->enableLocalFlush_ = enableLocalFlush;
  this->bootstrapMode = bootstrapMode;
  this->devices.resize(NCCL_CTRAN_IB_DEVICES_PER_RANK);
  this->cqs.reserve(NCCL_CTRAN_IB_DEVICES_PER_RANK);
  FB_COMMCHECKTHROW_EX(this->setPgToTrafficClassMap(), this->ncclLogData);

  auto s = CtranIbSingleton::getInstance();
  CHECK_VALID_IB_SINGLETON(s);

  const bool dmaBufSupported = s->getDevToDmaBufSupport(cudaDev);

  if (socketFactory != nullptr) {
    socketFactory_ = socketFactory; // ISocketFactory explicitly specified.
  } else if (abortCtrl_->Enabled()) {
    socketFactory_ = // Abort is enabled, so we want AbortableSockets.
        std::make_shared<ctran::bootstrap::AbortableSocketFactory>();
  } else { // Fall back to old, non-abortable sockets.
    socketFactory_ = std::make_shared<ctran::bootstrap::SocketFactory>();
  }

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
    throw ::ctran::utils::Exception(
        msg.c_str(),
        commInvalidArgument,
        this->rank,
        this->commHash,
        this->commDesc);
  }

  // assume NCCL_CTRAN_IB_DEVICES_PER_RANK contexts per cuda device
  if (cudaDev * NCCL_CTRAN_IB_DEVICES_PER_RANK * NCCL_CTRAN_IB_DEVICE_STRIDE >=
      s->ibvDevices.size()) {
    std::string msg = "cudaDev (" + std::to_string(cudaDev) +
        ") * NCCL_CTRAN_IB_DEVICES_PER_RANK * NCCL_CTRAN_IB_DEVICE_STRIDE exceeds the number of contexts (" +
        std::to_string(s->ibvDevices.size()) + ")";
    CLOGF(ERR, "CTRAN-IB: {}", msg);
    throw ::ctran::utils::Exception(
        msg.c_str(),
        commSystemError,
        this->rank,
        this->commHash,
        this->commDesc);
  }

  for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; ++device) {
    int singletonDevIdx =
        cudaDev * NCCL_CTRAN_IB_DEVICES_PER_RANK * NCCL_CTRAN_IB_DEVICE_STRIDE +
        device;
    devices[device].ibvDevice = &s->ibvDevices[singletonDevIdx];
    devices[device].ibvPd = &s->getIbvPd(singletonDevIdx);

    ibverbx::ibv_device_attr devAttr;
    auto maybeDeviceAttr = devices[device].ibvDevice->queryDevice();
    FOLLY_EXPECTED_CHECKTHROW_EX(maybeDeviceAttr, ncclLogData);
    devAttr = std::move(*maybeDeviceAttr);

    // Found available port for the given device
    for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
      ibverbx::ibv_port_attr portAttr;
      auto maybePortAttr = s->ibvDevices[singletonDevIdx].queryPort(port);
      if (maybePortAttr.hasError()) {
        CLOGF(
            WARN,
            "CTRAN-IB : Unable to query port {} on device {}",
            port,
            s->ibvDevices[singletonDevIdx].device()->name);
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
      if (s->ibvDevices[singletonDevIdx].port() == CTRAN_IB_ANY_PORT ||
          port == s->ibvDevices[singletonDevIdx].port()) {
        devices[device].port = port;
        devices[device].devName = s->ibvDevices[singletonDevIdx].device()->name;
        foundPort = true;
        break;
      }
    }

    if (foundPort) {
      CLOGF_SUBSYS(
          INFO,
          INIT,
          "CTRAN-IB: using device {}, port {} commHash {:x} commDesc {}",
          s->ibvDevices[singletonDevIdx].device()->name,
          devices[device].port,
          commHash,
          commDesc);
    } else {
      CLOGF(
          WARN,
          "CTRAN-IB : No active port found on device {}. Disable IB backend.",
          s->ibvDevices[singletonDevIdx].device()->name);
      throw ::ctran::utils::Exception(
          std::string("CTRAN-IB : No active port found on device ") +
              s->ibvDevices[singletonDevIdx].device()->name,
          commSystemError,
          this->rank,
          this->commHash,
          this->commDesc);
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

    // Cap CQ size to avoid excessive memory usage.
    // Per-transport maxNumCqe takes precedence over env
    // NCCL_CTRAN_IB_MAX_NUM_CQE. Default -1 (or nullopt) to use the
    // device-reported maximum.
    const int cqeCap = maxNumCqe.value_or(NCCL_CTRAN_IB_MAX_NUM_CQE);
    if (cqeCap > 0 && maxCqe > cqeCap) {
      CLOGF(
          INFO,
          "CTRAN-IB: Capping CQ size from {} to {} ({}) to reduce memory overhead",
          maxCqe,
          cqeCap,
          maxNumCqe.has_value() ? "per-transport"
                                : "NCCL_CTRAN_IB_MAX_NUM_CQE");
      maxCqe = cqeCap;
    } else {
      CLOGF(INFO, "CTRAN-IB: CQ size is {}", maxCqe);
    }

    // Skip lock for cq and localVc in constructor since no other thread can
    // access it yet.
    auto maybeCq =
        devices[device].ibvDevice->createCq(maxCqe, nullptr, nullptr, 0);
    FOLLY_EXPECTED_CHECKTHROW_EX(maybeCq, ncclLogData);
    cqs.emplace_back(std::move(*maybeCq));
    devices[device].ibvCq = &cqs[device];
    // FIXME: use initRemoteTransStates() to create cq
  }

  if (enableLocalFlush) {
    localVc = std::make_unique<LocalVirtualConn>(devices, ncclLogData);
  }

  // Record reference to CtranIbSingleton
  if (comm) {
    s->commRef(comm);
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

  // Initialize the per-peer VC state (rank/qp maps, connectedPeerMap, and
  // lock-free fast-path pointer). Must run before any incoming connection is
  // accepted (i.e., before bootstrap_->start() spawns the listen thread).
  vcState_.init(this, this->ncclLogData, comm ? comm->statex_->nRanks() : 0);

  // Derive per-peer-VC sizing from cvars via the unified VcLayout utility.
  // Legacy single-VC (NCCL_CTRAN_IB_NUM_VCS_PER_RANK <= 0) is normalized to
  // 1, so the layout (and per-VC active-NIC vector) is well-defined in
  // every configuration.
  int maxVcsPerPeer = NCCL_CTRAN_IB_NUM_VCS_PER_RANK;
  if (maxVcsPerPeer <= 0) {
    maxVcsPerPeer = 1;
  }
  if (NCCL_CTRAN_IB_MAX_QPS < maxVcsPerPeer) {
    std::string msg = fmt::format(
        "CTRAN-IB: invalid VC sizing: NCCL_CTRAN_IB_MAX_QPS={} must be "
        ">= NCCL_CTRAN_IB_NUM_VCS_PER_RANK={} so that each per-peer VC "
        "gets at least one data QP.",
        NCCL_CTRAN_IB_MAX_QPS,
        maxVcsPerPeer);
    CLOGF(ERR, msg);
    throw ctran::utils::Exception(msg, commInvalidArgument);
  }
  vcLayout_ =
      ctran::ib::VcLayout(NCCL_CTRAN_IB_DEVICES_PER_RANK, maxVcsPerPeer);
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: VC layout: {} (numNics={}, NCCL_CTRAN_IB_MAX_QPS={}). "
      "Per-VC data-QP count is determined per connection class inside CtranIbVirtualConn.",
      vcLayout_.describe(),
      NCCL_CTRAN_IB_DEVICES_PER_RANK,
      NCCL_CTRAN_IB_MAX_QPS);

  // Optionally start internal bootstrap service.
  // If kExternal, external callsite would explicitly manage it and thus we skip
  // the internal bootstrap.
  if (this->bootstrapMode == BootstrapMode::kExternal) {
    externalBootstrap_ = std::make_unique<ctran::ib::BootstrapExternal>(
        vcState_,
        devices,
        comm,
        cudaDev,
        commHash,
        commDesc,
        ncclLogData,
        getPgToTrafficClassValue());
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "CTRAN-IB: skip internal bootstrap for IB backend {} commHash {:x} commDesc {}",
        (void*)this,
        commHash,
        commDesc);
  } else {
    bootstrap_ = std::make_unique<Bootstrap>(
        vcState_,
        vcLayout_,
        socketFactory_,
        abortCtrl_,
        ncclLogData,
        comm,
        devices,
        getPgToTrafficClassValue(),
        cudaDev,
        rank,
        commHash,
        commDesc);
    bootstrap_->start(qpServerAddr);
  }

  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-IB: created IB backend {} for commHash {:x} commDesc {}",
      (void*)this,
      commHash,
      commDesc);
}

std::string CtranIb::getIbDevName(int device) const {
  return std::string(this->devices.at(device).devName);
}

int CtranIb::getIbDevPort(int device) const {
  return this->devices.at(device).port;
}

CtranIb::~CtranIb(void) {
  auto s = CtranIbSingleton::getInstance();
  CHECK_VALID_IB_SINGLETON(s);

  if (bootstrap_) {
    bootstrap_->shutdown();
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
    s->commDeref(this->comm);
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

  auto s = CtranIbSingleton::getInstance();
  CHECK_VALID_IB_SINGLETON(s);
  const auto dmaBufSupport = s->getDevToDmaBufSupport(cudaDev);

  bool useDmaBuf = dmaBufSupport && NCCL_CTRAN_IB_DMABUF_ENABLE;

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
    const auto& pd = s->getIbvPd(pdIdx);
    int dmaBufFd = useDmaBuf
        ? ctran::utils::getCuMemDmaBufFd(buf, len, pd.useDataDirect())
        : -1;
    auto makeErrorInfo = [&]() {
      return fmt::format(
          "CTRAN-IB: buffer registration failed: cudaDev={}, nicIdx={}, "
          "nicName={}, pdIdx={}, buf={}, len={}, dmaBufSupport={}, "
          "useDataDirect={}. If dmaBufSupport=true but registration fails "
          "with error 524 (ENOMEDIUM), the GPU and NIC are not PCIe-close - "
          "check NCCL_IB_HCA order for GPU-to-NIC affinity",
          cudaDev,
          device,
          pd.getDeviceName(),
          pdIdx,
          buf,
          len,
          dmaBufSupport,
          pd.useDataDirect());
    };
    if (useDmaBuf && dmaBufFd != -1) {
      auto maybeDmabufMr = pd.regDmabufMr(
          0, len, reinterpret_cast<uint64_t>(buf), dmaBufFd, access);
      FOLLY_EXPECTED_CHECKGOTO(maybeDmabufMr, fail, makeErrorInfo());
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
      auto maybeMr = pd.regMr((void*)buf, len, access);
      FOLLY_EXPECTED_CHECKGOTO(maybeMr, fail, makeErrorInfo());
      mrs->emplace_back(std::move(*maybeMr));
    }
    if (dmaBufFd != -1) {
      (void)close(dmaBufFd);
    }
  }

  *ibRegElem = reinterpret_cast<void*>(mrs);
  s->incActiveRegCount();

  return res;

fail:
  return commSystemError;
}

commResult_t CtranIb::deregMem(void* ibRegElem) {
  auto mrs = reinterpret_cast<std::vector<ibverbx::IbvMr>*>(ibRegElem);

  auto s = CtranIbSingleton::getInstance();
  CHECK_VALID_IB_SINGLETON(s);
  s->decActiveRegCount();

  delete mrs;

  return commSuccess;
}

commResult_t CtranIb::iflush(
    const void* dbuf,
    const void* localRegHdl,
    CtranIbRequest* req) {
  FB_COMMCHECK(checkEpochLock(this));

  if (enableLocalFlush_) {
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
  std::shared_ptr<CtranIbVirtualConn> vc = vcState_.getVc(peerRank);
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

  // Explicitly release all virtual connections (and clear connectedPeerMap)
  // before destroying cq.
  vcState_.releaseAll();

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

    // Clear old CQs before creating new ones (fixes accumulation on
    // reconfigure).
    for (int device = 0; device < NCCL_CTRAN_IB_DEVICES_PER_RANK; device++) {
      devices[device].ibvCq = nullptr;
    }
    cqs.clear(); // IbvCq destructors call ibv_destroy_cq

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
    localVc = std::make_unique<LocalVirtualConn>(devices, ncclLogData);
  }

  return commSuccess;
}

commResult_t CtranIb::preConnect(const std::unordered_set<int>& peerRanks) {
  bool newConnection = false;
  // if map is empty, it means we don't know comm size and peerAddr
  if (!vcState_.hasConnectedPeerMap()) {
    return commSuccess;
  }
  // Actively connect to peers with larger rank number, if not already connected
  for (int peerRank : peerRanks) {
    FB_COMMCHECK(checkValidPeer(peerRank));
    if (rank < peerRank && vcState_.getVc(peerRank) == nullptr) {
      FB_COMMCHECK(bootstrap_->connect(peerRank, std::nullopt));
    }
  }
  // check if all requested peers are connected
  for (int peerRank : peerRanks) {
    if (!vcState_.isPeerConnected(peerRank)) {
      while (vcState_.getVc(peerRank) == nullptr) {
        // wait listening thread to establish connections with rest of peers
        // with smaller rank number
      }
      vcState_.markPeerConnected(peerRank);
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

folly::Expected<folly::SocketAddress, int>
CtranIb::getListenSocketListenAddr() {
  return bootstrap_->getListenAddress();
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

// ============================================================
// Per-peer VC API implementation
// ============================================================

const std::vector<std::shared_ptr<CtranIbVirtualConn>>& CtranIb::connectVcs(
    int peerRank,
    std::optional<const SocketServerAddr*> peerServerAddr) {
  // Fast path: per-peer VCs already published. References to values in
  // std::unordered_map are stable across insertions, so the returned
  // reference remains valid until releaseAll().
  if (const auto* vcs = vcState_.tryGetVcs(peerRank)) {
    return *vcs;
  }

  if (rank < peerRank) {
    // Smaller rank initiates the rendezvous. connectVcs(peer, ...) must be
    // called from a single thread per peer; concurrent per-peer callers
    // are not supported.
    FB_CHECKABORT(
        bootstrap_ != nullptr,
        "CTRAN-IB: connectVcs() called on externally-bootstrapped instance");
    // bootstrap_->connect internally exchanges vcLayout_.maxVcsPerPeer
    // VCs and publishes them into vcState_ (legacy single-VC callers
    // read .front()).
    const auto err = bootstrap_->connect(peerRank, peerServerAddr);
    FB_CHECKABORT(
        err == commSuccess,
        "CTRAN-IB: bootstrap_->connect failed for peerRank " +
            std::to_string(peerRank));
  }

  // Smaller-rank gets and returns vc created by connect. Larger-rank side spins
  // until the listen thread publishes (mirrors preConnect()). The abort check
  // lets teardown / explicit abort unblock us so we don't loop forever after
  // releaseAll().
  while (true) {
    if (const auto* vcs = vcState_.tryGetVcs(peerRank)) {
      return *vcs;
    }
    if (abortCtrl_ && abortCtrl_->Test()) {
      static const std::vector<std::shared_ptr<CtranIbVirtualConn>> empty;
      return empty;
    }
    std::this_thread::yield();
  }
}
