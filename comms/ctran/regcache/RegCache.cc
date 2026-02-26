// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/regcache/RegCache.h"
#include <folly/Singleton.h>
#include <folly/system/ThreadName.h>

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#ifdef CTRAN_DISABLE_TCPDM
#include "comms/ctran/backends/mock/CtranTcpDmMock.h"
#else
#include "comms/ctran/backends/tcpdevmem/CtranTcpDm.h"
#endif
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/alloc.h"

static folly::Singleton<ctran::RegCache> regCacheSingleton;
std::shared_ptr<ctran::RegCache> ctran::RegCache::getInstance() {
  return regCacheSingleton.try_get();
}

ctran::RegCache::RegCache(void) {
  init();
}

ctran::RegCache::~RegCache(void) {
  // Define separate destroy() function to allow test to call it explicitly
  FB_COMMCHECKIGNORE(destroy());
}

static const std::unordered_map<ctran::regcache::EventType, std::string>
    RegCacheEventNameMap = {
        {ctran::regcache::kCacheSegEvent, "CACHE"},
        {ctran::regcache::kRegMemEvent, "REG"},
        {ctran::regcache::kDeregMemEvent, "DEREG"},
        {ctran::regcache::kDynamicRegMemEvent, "DYNAMIC_REG"},
        {ctran::regcache::kAsyncRegMemEvent, "ASYNC_REG"},
};

// Currently cached and registered buffers.
// The number may change overtime if any buffer is deregistered.
static const std::vector<ctran::regcache::EventType> currentEvents = {
    ctran::regcache::EventType::kCacheSegEvent,
    ctran::regcache::EventType::kRegMemEvent,
};

static const std::vector<ctran::regcache::EventType> latencyEvents = {
    ctran::regcache::EventType::kRegMemEvent,
    ctran::regcache::EventType::kDeregMemEvent,
};

static const std::vector<ctran::regcache::EventType> totalEvents = {
    ctran::regcache::EventType::kCacheSegEvent,
    ctran::regcache::EventType::kRegMemEvent,
    ctran::regcache::EventType::kDeregMemEvent,
    ctran::regcache::EventType::kDynamicRegMemEvent,
    ctran::regcache::EventType::kAsyncRegMemEvent,
};

ctran::regcache::Profiler::Profiler() {
  reset();
}

void ctran::regcache::Profiler::reset(void) {
  for (const auto type : latencyEvents) {
    latencyMap[type] = 0;
  }
  for (const auto type : totalEvents) {
    totalCountMap[type] = 0;
  }
  for (const auto type : currentEvents) {
    currentCountMap[type] = 0;
  }
}

void ctran::regcache::Profiler::record(
    ctran::regcache::EventType type,
    CtranMapperTimer& dur) {
  if (NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT < 0) {
    return;
  }

  if (std::find(latencyEvents.begin(), latencyEvents.end(), type) !=
      latencyEvents.end()) {
    latencyMap.at(type) += dur.durationUs();
  }

  record(type);
}

void ctran::regcache::Profiler::record(ctran::regcache::EventType type) {
  if (NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT < 0) {
    return;
  }

  if (std::find(totalEvents.begin(), totalEvents.end(), type) !=
      totalEvents.end()) {
    totalCountMap.at(type)++;
  }

  if (type == ctran::regcache::EventType::kFreeSegEvent) {
    currentCountMap.at(ctran::regcache::EventType::kCacheSegEvent)--;
  } else if (type == ctran::regcache::EventType::kDeregMemEvent) {
    currentCountMap.at(ctran::regcache::EventType::kRegMemEvent)--;
  } else if (
      std::find(currentEvents.begin(), currentEvents.end(), type) !=
      currentEvents.end()) {
    currentCountMap.at(type)++;
  }

  // Allow periodical snapshot report during long job running
  if (type == ctran::regcache::EventType::kRegMemEvent &&
      NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT > 0 &&
      (totalCountMap.at(type) % NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT ==
       0)) {
    reportSnapshot();
  }
}

void ctran::regcache::Profiler::reportSnapshot(void) const {
  const std::string prefix = "CTRAN-REGCACHE RegCache Snapshot";
  for (const auto type : totalEvents) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "[{}] Total count of {}: {}",
        prefix.c_str(),
        RegCacheEventNameMap.at(type).c_str(),
        totalCountMap.at(type));
  }
  for (const auto type : latencyEvents) {
    auto count = totalCountMap.at(type);
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "[{}] Average latency (us) of {}: {:.2f}",
        prefix.c_str(),
        RegCacheEventNameMap.at(type).c_str(),
        latencyMap.at(type) / count);
  }

  for (const auto type : currentEvents) {
    CLOGF_SUBSYS(
        INFO,
        INIT,
        "[{}] Current count of {}: {}",
        prefix.c_str(),
        RegCacheEventNameMap.at(type).c_str(),
        currentCountMap.at(type));
  }
}

ctran::regcache::Snapshot ctran::regcache::Profiler::getSnapshot() const {
  ctran::regcache::Snapshot snapshot;
  snapshot.currentNumCache =
      currentCountMap.at(ctran::regcache::EventType::kCacheSegEvent);
  snapshot.currentNumReg =
      currentCountMap.at(ctran::regcache::EventType::kRegMemEvent);
  snapshot.totalNumCache =
      totalCountMap.at(ctran::regcache::EventType::kCacheSegEvent);
  snapshot.totalNumReg =
      totalCountMap.at(ctran::regcache::EventType::kRegMemEvent);
  snapshot.totalNumDereg =
      totalCountMap.at(ctran::regcache::EventType::kDeregMemEvent);
  snapshot.totalNumDynamicReg =
      totalCountMap.at(ctran::regcache::EventType::kDynamicRegMemEvent);
  snapshot.totalNumAsyncReg =
      totalCountMap.at(ctran::regcache::EventType::kAsyncRegMemEvent);
  snapshot.regMemLatency =
      latencyMap.at(ctran::regcache::EventType::kRegMemEvent) /
      snapshot.totalNumReg;
  snapshot.deregMemLatency =
      latencyMap.at(ctran::regcache::EventType::kDeregMemEvent) /
      snapshot.totalNumDereg;
  return snapshot;
}

void ctran::RegCache::init() {
  // Acquire a reference to CtranIbSingleton to establish dependency ordering.
  // By holding this shared_ptr, we ensure that CtranIbSingleton is destroyed
  // AFTER RegCache during program shutdown, preventing use-after-free when
  // RegCache::destroy() calls CtranIb::deregMem().
  ibSingleton_ = CtranIbSingleton::getInstance();
  CHECK_VALID_IB_SINGLETON(ibSingleton_);

  // Initialize global backends from environment variable.
  // The NCCL_CTRAN_BACKENDS cvar is already parsed at cvar initialization time.
  // This allows registration to work without requiring a communicator.
  globalBackends_.resize(CommBackend::NUM_BACKENDS, false);
  for (const auto& backend : NCCL_CTRAN_BACKENDS) {
    switch (backend) {
      case NCCL_CTRAN_BACKENDS::ib:
        globalBackends_[CommBackend::IB] = true;
        break;
      case NCCL_CTRAN_BACKENDS::nvl:
        globalBackends_[CommBackend::NVL] = true;
        break;
      case NCCL_CTRAN_BACKENDS::socket:
        globalBackends_[CommBackend::SOCKET] = true;
        break;
      case NCCL_CTRAN_BACKENDS::tcpdm:
        globalBackends_[CommBackend::TCPDM] = true;
        break;
    }
  }
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTRAN-REGCACHE: Global backends initialized from NCCL_CTRAN_BACKENDS: "
      "IB={} NVL={} SOCKET={} TCPDM={}",
      static_cast<bool>(globalBackends_[CommBackend::IB]),
      static_cast<bool>(globalBackends_[CommBackend::NVL]),
      static_cast<bool>(globalBackends_[CommBackend::SOCKET]),
      static_cast<bool>(globalBackends_[CommBackend::TCPDM]));

  if (NCCL_CTRAN_REGISTER == NCCL_CTRAN_REGISTER::async &&
      !asyncRegThread_.joinable()) {
    int cudaDev;
    FB_CUDACHECKTHROW_EX_NOCOMM(cudaGetDevice(&cudaDev));
    asyncRegThread_ = std::thread{&RegCache::asyncRegThreadFn, this, cudaDev};
  }
}

commResult_t ctran::RegCache::destroy() {
  {
    // Warn if user missed any buffer registration.
    // RegCache holds a shared_ptr to CtranIbSingleton which guarantees
    // IB singleton stays alive during deregistration.
    auto [segmentsAvl, regElemsMaps] =
        folly::acquireLocked(segmentsAvl_, regElemsMaps_);
    auto& regHdlToElemMap = regElemsMaps->regHdlToElemMap;
    if (segmentsAvl->size() > 0 || regHdlToElemMap.size() > 0) {
      CLOGF(
          WARN,
          "Total {}/{} remaining segments are still in RegCache at destroy time. ",
          segmentsAvl->size(),
          regHdlToElemMap.size());
    }

    auto it = regHdlToElemMap.begin();
    while (!regHdlToElemMap.empty()) {
      auto& regElem = it->second;
      CLOGF_TRACE(
          ALLOC,
          "Remaining regElem {} buf {} len {} isDynamic {} in regHdlToElemMap",
          (void*)regElem.get(),
          regElem->buf,
          regElem->len,
          regElem->isDynamic_);
      FB_COMMCHECKIGNORE(regElem->doDeregister());
      it = regHdlToElemMap.erase(it);
    }

    for (auto avlHdl : segmentsAvl->getAllElems()) {
      auto seg = reinterpret_cast<ctran::regcache::Segment*>(
          segmentsAvl->lookup(avlHdl));
      CLOGF_TRACE(
          ALLOC,
          "Remaining avlHdl {} range {} ncclManaged {} in segmentsAvl",
          (void*)avlHdl,
          seg->range.toString(),
          seg->ncclManaged);
      segmentsAvl->remove(avlHdl);
      delete seg;
    }
  }

  if (asyncRegThread_.joinable()) {
    AsyncRegCmd cmd;
    cmd.stopFlag = true;

    asyncRegQueue_.lock()->push(cmd);
    asyncRegCv_.notify_one();
    asyncRegThread_.join();
    // Clear the queue after thread terminates
    std::queue<AsyncRegCmd>().swap(*asyncRegQueue_.lock());
    // Reset thread object to default state so it can be restarted by init()
    asyncRegThread_ = std::thread{};
  }

  // Report snapshot at destroy if enabled
  if (NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT >= 0) {
    profiler.rlock()->reportSnapshot();
  }

  return commSuccess;
}

commResult_t ctran::RegCache::globalRegister(
    const void* buf,
    size_t len,
    bool forceReg,
    int deviceId) {
  if (buf == nullptr || len == 0) {
    return commSuccess;
  }

  int cudaDev = 0;
  if (deviceId != -1) {
    cudaDev = deviceId;
  } else {
    // Auto-detect cudaDev from buffer pointer.
    // For CPU tensors (malloc'd memory), getCudaDevFromPtr may fail.
    // In that case, fall back to current device like CtranMapper does.
    commResult_t devResult = getCudaDevFromPtr(buf, cudaDev);
    if (devResult != commSuccess) {
      // Fall back to current CUDA device for CPU memory
      FB_CUDACHECK(cudaGetDevice(&cudaDev));
    }
  }

  // Cache the segments first
  std::vector<ctran::regcache::Segment*> segments;
  std::vector<void*> segHdls;
  FB_COMMCHECK(cacheSegment(
      buf,
      len,
      cudaDev,
      false /* ncclManaged */,
      0 /* commHash - not used for global registration */,
      segments,
      segHdls));

  // Register if in eager mode or forced by caller
  if (NCCL_CTRAN_REGISTER == NCCL_CTRAN_REGISTER::eager || forceReg) {
    bool didRegister = false;
    ctran::regcache::RegElem* regHdl = nullptr;
    CommLogData globalLogData{};
    globalLogData.commDesc = "global";
    FB_COMMCHECK(regRange(
        buf,
        len,
        cudaDev,
        "eagerGlobalRegister",
        globalLogData,
        globalBackends_,
        didRegister,
        &regHdl,
        false /* ncclManaged */));
  }

  return commSuccess;
}

commResult_t
ctran::RegCache::globalDeregister(const void* buf, size_t len, int deviceId) {
  if (buf == nullptr || len == 0) {
    return commSuccess;
  }

  int cudaDev = 0;
  if (deviceId != -1) {
    cudaDev = deviceId;
  } else {
    // Auto-detect cudaDev from buffer pointer.
    // For CPU tensors (malloc'd memory), getCudaDevFromPtr may fail.
    // In that case, fall back to current device like globalRegister does.
    commResult_t devResult = getCudaDevFromPtr(buf, cudaDev);
    if (devResult != commSuccess) {
      // Fall back to current CUDA device for CPU memory
      FB_CUDACHECK(cudaGetDevice(&cudaDev));
    }
  }

  auto timerBegin = std::chrono::steady_clock::now();

  // Use lookupSegmentsForBuffer to discover all cached segments and regElems
  std::vector<void*> segHdls;
  std::vector<ctran::regcache::RegElem*> regElems;
  FB_COMMCHECK(lookupSegmentsForBuffer(buf, len, cudaDev, segHdls, regElems));

  // Call remReleaseMem on ipcRegElems before freeing segments.
  // This notifies remote peers to release their imported NVL memory.
  // Like the mapper path, we use fire-and-forget semantics - the async socket
  // sends will complete in the background. We don't wait for completion since
  // the mapper also clears postedCbCtrlReqs_ without waiting at destruction.
  // TODO: verify this is correct and that we shouldn't be waiting for the
  // requests to finish.
  auto ipcRegCache = ctran::IpcRegCache::getInstance();
  std::string localPeerId = ipcRegCache->getLocalPeerId();
  std::deque<std::unique_ptr<ctran::regcache::IpcReqCb>> postedReqs;
  for (auto& regElem : regElems) {
    auto ipcRegElem =
        reinterpret_cast<ctran::regcache::IpcRegElem*>(regElem->ipcRegElem);
    if (ipcRegElem != nullptr) {
      FB_COMMCHECK(
          ipcRegCache->remReleaseMem(localPeerId, ipcRegElem, postedReqs));
    }
  }

  // Free each segment
  size_t totalSegmentsFreed = 0;
  for (auto segHdl : segHdls) {
    bool freed = false;
    bool ncclManaged = false;
    std::vector<std::unique_ptr<ctran::regcache::RegElem>> regElemsFreed;
    FB_COMMCHECK(freeSegment(segHdl, freed, ncclManaged, regElemsFreed));

    if (freed) {
      totalSegmentsFreed++;
    }
  }

  // Log a single memory event for the entire deregistration
  if (totalSegmentsFreed > 0) {
    CommLogData globalLogData{};
    globalLogData.commDesc = "global";
    logMemoryEvent(
        globalLogData,
        "",
        "globalDeregister",
        reinterpret_cast<uintptr_t>(buf),
        len,
        totalSegmentsFreed,
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - timerBegin)
            .count(),
        true /* isRegMemEvent */);
  }

  return commSuccess;
}

void ctran::RegCache::asyncRegThreadFn(int cudaDev) {
  folly::setThreadName("CTranAsyncReg");
  commNamedThreadStart("CTranAsyncReg");

  FB_CUDACHECKTHROW_EX_NOCOMM(cudaSetDevice(cudaDev));

  while (true) {
    AsyncRegCmd cmd;

    {
      auto locked = asyncRegQueue_.lock();

      asyncRegCv_.wait(
          locked.as_lock(), [&locked] { return !locked->empty(); });

      cmd = locked->front();
      // Keep current cmd at frond of the queue to indicate ongoing
      // registration. Pop upon completion.
    }

    if (cmd.stopFlag) {
      CLOGF_SUBSYS(
          INFO, INIT, "CTranMapperRegCache asyncRegThreadFn: terminate");
      return;
    }

    FB_CHECKABORT(
        cmd.buf && cmd.len > 0 && cmd.cudaDev >= 0,
        "Invalid buffer registration request: buf {} len {} cudaDev {}",
        cmd.buf,
        cmd.len,
        cmd.cudaDev);

    bool didRegister = false;
    ctran::regcache::RegElem* regHdl = nullptr;

    // Expected behavior:
    // - If didRegister is true, meaning the buffer is registered by the
    //   asyncThread. Later GPE thread will lookup hit at
    //   searchRegHandle->regRange.
    // - If didRegister is false and regHdl is not nullptr, meaning the buffer
    //   has already been registered by a previous async request or GPE
    //   registration.
    // - If regHdl is nullptr, meaning the buffer is not cached by user; let
    //   dynamic registration handle it by GPE thread.
    //   NOTE: In rare case, freeSegment will be called before asyncReg thread
    //   executes the registration request, e.g., too slow asyncReg thread.
    //   Then, asyncReg thread will also tread it as dynamic registration and
    //   skip.
    FB_COMMCHECKTHROW_EX_NOCOMM(regRange(
        cmd.buf,
        cmd.len,
        cmd.cudaDev,
        "asyncRegMem",
        cmd.logMetaData,
        cmd.backends,
        didRegister,
        &regHdl));

    if (didRegister) {
      profiler.wlock()->record(ctran::regcache::EventType::kAsyncRegMemEvent);
    }

    // NOTE: regHdl may already be released by concurrent deregMem from main
    // thread; unsafe to read its content
    CLOGF_TRACE(
        ALLOC,
        "CTRAN-REGCACHE: async registered buf {} len {} didRegister {} regHdl {}",
        cmd.buf,
        cmd.len,
        didRegister,
        (void*)regHdl);

    // Completed the current cmd. Pop out from queue.
    {
      auto locked = asyncRegQueue_.lock();
      locked->pop();
    }
  }
  return;
}

commResult_t ctran::RegCache::asyncRegRange(
    const void* buf,
    const size_t len,
    const int cudaDev,
    const struct CommLogData& logMetaData,
    const std::vector<bool>& backend) {
  if (!asyncRegThread_.joinable()) {
    CLOGF(
        ERR,
        "AsyncReg thread is not running. Check whether NCCL_CTRAN_REGISTER=async is set.");
    return commInvalidUsage;
  }

  AsyncRegCmd cmd = AsyncRegCmd{
      .buf = const_cast<void*>(buf),
      .len = len,
      .cudaDev = cudaDev,
      .stopFlag = false,
      .logMetaData = logMetaData,
      .backends = backend};

  {
    auto locked = asyncRegQueue_.lock();
    locked->push(cmd);
  }
  asyncRegCv_.notify_one();
  return commSuccess;
}

void ctran::RegCache::waitAsyncRegComplete() {
  while (true) {
    auto locked = asyncRegQueue_.lock();
    if (locked->empty()) {
      break;
    }
  }
}

ctran::regcache::RegElem* ctran::RegCache::searchRegElem(
    const void* ptr,
    const size_t len) {
  ctran::regcache::RegElem* regHdl = nullptr;

  auto regElemsMaps = regElemsMaps_.rlock();
  auto& regHdlToElemMap = regElemsMaps->regHdlToElemMap;

  // Find range in regElemsMaps
  uintptr_t startAddr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t endAddr = startAddr + len;
  for (auto it = regHdlToElemMap.begin(); it != regHdlToElemMap.end(); it++) {
    // Not count any dynamic registeration as it will be released immediately
    // after the current collective that uses it
    auto& searchRegElem = it->second;
    if (searchRegElem->isDynamic_) {
      continue;
    }

    uintptr_t regStartAddr = reinterpret_cast<uintptr_t>(searchRegElem->buf);
    uintptr_t regEndAddr = regStartAddr + searchRegElem->len;
    if (regStartAddr <= startAddr && endAddr <= regEndAddr) {
      // Lookup hit
      regHdl = searchRegElem.get();
      searchRegElem->lookupHit_++;
      break;
    }
  }
  return regHdl;
}

bool ctran::RegCache::isRegistered(const void* ptr, const size_t len) {
  // Find range in regElemsMaps
  auto regHdl = searchRegElem(ptr, len);
  return regHdl != nullptr;
}

std::vector<void*> ctran::RegCache::getSegments() const {
  return segmentsAvl_.rlock()->getAllElems();
}

commResult_t ctran::RegCache::lookupSegmentsForBuffer(
    const void* buf,
    size_t len,
    int cudaDev,
    std::vector<void*>& segHdls,
    std::vector<ctran::regcache::RegElem*>& regElems) {
  if (buf == nullptr || len == 0) {
    return commSuccess;
  }

  SetCudaDevRAII setCudaDev(cudaDev);

  segHdls = segmentsAvl_.rlock()->searchRange(buf, len);
  regElems = getRegElems(segHdls);

  return commSuccess;
}

commResult_t ctran::regcache::SegmentRange::pinRange(
    const void* ptr,
    const int cudaDev,
    size_t len,
    std::vector<ctran::regcache::SegmentRange>& segRangs) {
  DevMemType memType{DevMemType::kCumem};
  FB_COMMCHECK(getDevMemType(ptr, cudaDev, memType));

  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-MAPPER pinRange: input ptr={} len={} cudaDev={} fbMemType={}",
      ptr,
      len,
      cudaDev,
      (int)memType);

  // Host unregistered memory or host pinned or cudaMalloc-ed buffer, return
  // entire range as a single segment
  if (memType != DevMemType::kCumem) {
    segRangs.emplace_back(ptr, len, memType);
    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "CTRAN-MAPPER pinRange: non-cumem single segment ptr={} len={}",
        ptr,
        len);
    return commSuccess;
  }

  size_t curRange = 0;
  CUdeviceptr curPbase = 0;
  CUdeviceptr ptr_ = reinterpret_cast<CUdeviceptr>(const_cast<void*>(ptr));
  // This is a cumem type which may contain multiple segment ranges
  // - Record the first found range
  FB_CUCHECK(cuMemGetAddressRange(&curPbase, &curRange, ptr_));
  segRangs.emplace_back(
      reinterpret_cast<const void*>(curPbase), curRange, memType);
  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-MAPPER pinRange: discovered segment[0] pbase={:#x} range={}",
      (uintptr_t)curPbase,
      curRange);

  // - Continue search the remaining ranges until reached the end of the buffer
  size_t cur_offset = (size_t)ctran::utils::subDevicePtr(
      ctran::utils::addDevicePtr(curPbase, curRange), (void*)ptr_);
  int segmentIdx = 0;
  while (cur_offset < len) {
    CUdeviceptr curPtr_ = ctran::utils::addDevicePtr(ptr_, cur_offset);
    FB_CUCHECK(
        cuMemGetAddressRange(&curPbase, &curRange, (CUdeviceptr)curPtr_));
    segRangs.emplace_back(
        reinterpret_cast<const void*>(curPbase), curRange, memType);
    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "CTRAN-MAPPER pinRange: discovered segment[{}] pbase={:#x} range={} (offset={})",
        segmentIdx,
        (uintptr_t)curPbase,
        curRange,
        cur_offset);

    cur_offset = (size_t)ctran::utils::subDevicePtr(
        ctran::utils::addDevicePtr(curPbase, curRange), (void*)ptr_);
    segmentIdx++;
  }

  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-MAPPER pinRange: total {} segments discovered for input len={}",
      segRangs.size(),
      len);

  // MIN_TODO: check properties

  return commSuccess;
}

commResult_t ctran::RegCache::cacheSegment(
    const void* ptr,
    const size_t len,
    const int cudaDev,
    const bool ncclManaged,
    uint64_t commHash,
    std::vector<ctran::regcache::Segment*>& segments,
    std::vector<void*>& segHdls) {
  SetCudaDevRAII setCudaDev(cudaDev);
  bool newSegmentCreated = false;

  // Discover all physical segments underlying this buffer via pinRange
  std::vector<ctran::regcache::SegmentRange> ranges;
  FB_COMMCHECK(
      ctran::regcache::SegmentRange::pinRange(ptr, cudaDev, len, ranges));

  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-MAPPER cacheSegment: ptr={} len={} discovered {} physical segments",
      ptr,
      len,
      ranges.size());

  {
    auto segmentsAvl = segmentsAvl_.wlock();

    // Cache each segment range
    for (size_t i = 0; i < ranges.size(); i++) {
      const auto& range = ranges.at(i);
      void* avlHdl = nullptr;

      // Check if this segment is already cached
      avlHdl = segmentsAvl->search(range.buf, range.len);
      if (avlHdl) {
        // Segment already cached, increase refcount
        auto foundSeg = reinterpret_cast<ctran::regcache::Segment*>(
            segmentsAvl->lookup(avlHdl));
        int64_t curRefCount;
        {
          auto segState = foundSeg->stateMnger.wlock();
          segState->refCount++;
          curRefCount = segState->refCount;
        }
        segments.push_back(foundSeg);
        segHdls.push_back(foundSeg->avlHdl_);

        CLOGF_TRACE(
            ALLOC,
            "CTRAN-MAPPER cacheSegment: segment[{}] already cached ptr={} len={} refCount={}",
            i,
            range.buf,
            range.len,
            curRefCount);
      } else {
        // Create new cache entry for this segment
        auto newSeg = new ctran::regcache::Segment(range, cudaDev, ncclManaged);
        avlHdl = segmentsAvl->insert(range.buf, range.len, newSeg);
        newSeg->avlHdl_ = avlHdl;
        segments.push_back(newSeg);
        segHdls.push_back(avlHdl);
        newSegmentCreated = true;

        const auto type = newSeg->getType();
        CLOGF_TRACE(
            ALLOC,
            "CTRAN-MAPPER cacheSegment: segment[{}] cached type={} ({}) segHdl={} ptr={} len={} ncclManaged={} cudaDev={}, cache size={}",
            i,
            (int)type,
            devMemTypeStr(type),
            (void*)avlHdl,
            range.buf,
            range.len,
            ncclManaged,
            cudaDev,
            segmentsAvl->size());
      }
    }
  }

  // Only record a cache event when at least one new segment was created
  if (newSegmentCreated) {
    profiler.wlock()->record(ctran::regcache::EventType::kCacheSegEvent);
  }
  return commSuccess;
}

// Helper function to perform backend registration for a set of segments.
// Creates a RegElem, registers with backends, and updates regElemsMaps.
// Caller must hold segmentsAvl lock (for thread safety with segment pointers).
// This function acquires regElemsMaps_ lock internally.
//
// Returns commSuccess on success, or error code on failure.
// On success, *regHdl is set to the created RegElem pointer.
commResult_t ctran::RegCache::registerSegmentsTogether(
    void* ptr,
    size_t len,
    int cudaDev,
    std::vector<ctran::regcache::Segment*>& segments,
    const std::vector<bool>& backends,
    bool ncclManaged,
    ctran::regcache::RegElem** regHdl) {
  // Create a new registration element for the segments
  auto newRegElem = std::make_unique<ctran::regcache::RegElem>(
      ptr, len, cudaDev, segments, ncclManaged);

  // Backend registration
  FB_COMMCHECK(newRegElem->doRegister(backends));

  auto regHdlPtr = newRegElem.get();

  // Acquire regElemsMaps_ lock to update maps
  auto regElemsMaps = regElemsMaps_.wlock();
  auto& regHdlToElemMap = regElemsMaps->regHdlToElemMap;
  auto& segToRegElemsMap = regElemsMaps->segToRegElemsMap;

  regHdlToElemMap.emplace(regHdlPtr, std::move(newRegElem));
  // Correlate the regElem with all associated segments to deregister it
  // when any segment is freed
  for (auto seg : segments) {
    segToRegElemsMap[seg].emplace_back(regHdlPtr);
  }

  *regHdl = regHdlPtr;
  return commSuccess;
}

commResult_t ctran::RegCache::regRange(
    const void* ptr,
    const size_t len,
    const int cudaDev,
    const std::string& useDesc,
    const struct CommLogData& logMetaData,
    const std::vector<bool>& backends,
    bool& didRegister,
    ctran::regcache::RegElem** regHdl,
    bool ncclManaged) {
  auto dur = CtranMapperTimer();
  SetCudaDevRAII setCudaDev(cudaDev);
  auto timerBegin = std::chrono::steady_clock::now();

  {
    // FAST PATH: find whether range has already been registered in
    // regElemsMaps. regElemsMaps should not be wlocked while performing
    // expensive registration/deregistration.
    *regHdl = searchRegElem(ptr, len);
    // Lookup hit
    if (*regHdl) {
      return commSuccess;
    }
  }

  // Copy state for scuba logging after releasing lock
  size_t lenToReg = 0;
  void* ptrToReg = nullptr;
  size_t numSegmentsToReg = 0;

  {
    // Global lock:
    // - Serialize concurrent registration updates, also with cache|free
    // segments.
    auto segmentsAvl = segmentsAvl_.wlock();

    // While holding the global lock, let's check again no one else has
    // registered the range.
    *regHdl = searchRegElem(ptr, len);
    if (*regHdl) {
      return commSuccess;
    }

    // - SLOW PATH: if the range is not yet registered, check if all
    // underlying segment ranges are cached. If found, let's register it
    std::vector<ctran::regcache::SegmentRange> ranges;
    FB_COMMCHECK(
        ctran::regcache::SegmentRange::pinRange(ptr, cudaDev, len, ranges));

    std::vector<ctran::regcache::Segment*> segments(ranges.size(), nullptr);
    bool foundAll = true;

    // - SLOW PATH: find the cached segments corresponding to each range.
    for (int i = 0; i < ranges.size(); i++) {
      auto& segRange = ranges.at(i);
      void* avlHdl = segmentsAvl->search(segRange.buf, segRange.len);
      if (!avlHdl) {
        CLOGF(
            WARN,
            "CTRAN-REGCACHE:[pbase {} range {}] associated with [ptr {} len {}] is not pre-registered by user",
            (void*)segRange.buf,
            segRange.len,
            (void*)ptr,
            len);
        foundAll = false;
        break;
      }
      segments[i] = reinterpret_cast<ctran::regcache::Segment*>(
          segmentsAvl->lookup(avlHdl));
      lenToReg += segments.at(i)->range.len;
    }

    if (foundAll) {
      // - SLOW PATH: found all cached segments, register the full segment
      // range.
      ptrToReg = const_cast<void*>(segments.at(0)->range.buf);
      numSegmentsToReg = segments.size();

      // Use helper to perform backend registration and update maps
      FB_COMMCHECK(registerSegmentsTogether(
          ptrToReg,
          lenToReg,
          cudaDev,
          segments,
          backends,
          ncclManaged,
          regHdl));
      didRegister = true;
    } else {
      // - WORST PATH: if any one is not found, return nullptr to trigger
      // dynamic registration
      *regHdl = nullptr;
      return commSuccess;
    }
  }

  // Log to scuba
  logMemoryEvent(
      logMetaData,
      "",
      useDesc,
      reinterpret_cast<uintptr_t>(ptrToReg),
      lenToReg,
      numSegmentsToReg,
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - timerBegin)
          .count(),
      true /* isRegMemEvent */);

  profiler.wlock()->record(ctran::regcache::EventType::kRegMemEvent, dur);
  return commSuccess;
}

bool ctran::regcache::Segment::askFree() {
  auto stat = stateMnger.wlock();
  stat->refCount--;
  FB_CHECKABORT(
      stat->refCount >= 0,
      "Unexpected negative refCount {} in segment {} [{}]",
      stat->refCount,
      (void*)this,
      toString(stat->refCount).c_str());

  return stat->refCount == 0;
}

ctran::regcache::Segment* ctran::RegCache::getSegment(void* segHdl) {
  return reinterpret_cast<ctran::regcache::Segment*>(
      segmentsAvl_.rlock()->lookup(segHdl));
}

std::vector<ctran::regcache::RegElem*> ctran::RegCache::getRegElems(
    const void* segHdl) const {
  return getRegElems(std::vector<void*>{const_cast<void*>(segHdl)});
}

std::vector<ctran::regcache::RegElem*> ctran::RegCache::getRegElems(
    const std::vector<void*>& segHdls) const {
  std::unordered_set<ctran::regcache::RegElem*> seen;
  std::vector<ctran::regcache::RegElem*> regElems;

  auto segmentsLock = segmentsAvl_.rlock();
  auto regElemsLock = regElemsMaps_.rlock();
  auto& segToRegElemsMap = regElemsLock->segToRegElemsMap;

  for (void* segHdl : segHdls) {
    const auto segment = reinterpret_cast<ctran::regcache::Segment*>(
        segmentsLock->lookup(segHdl));
    if (segment) {
      auto segIt = segToRegElemsMap.find(segment);
      if (segIt != segToRegElemsMap.end()) {
        for (auto* regElem : segIt->second) {
          if (regElem == nullptr) {
            continue;
          }
          auto [_, inserted] = seen.insert(regElem);
          if (inserted) {
            regElems.push_back(regElem);
          }
        }
      }
    }
  }

  return regElems;
}

commResult_t ctran::RegCache::freeSegment(
    void* segHdl,
    bool& freed,
    bool& ncclManaged,
    std::vector<std::unique_ptr<ctran::regcache::RegElem>>& regElems) {
  ctran::regcache::Segment* segment = nullptr;
  {
    // Global lock:
    // Lock both segmentsAvl and regElemsMaps since we may need remove segment
    // and all associated regElems.
    //
    // Perf impact to quick-lookup should be minimal as freeSegment happens
    // after majority of communication, and expensive deregElem happens after
    // releasing the lock.
    auto [segmentsAvl, regElemsMaps] =
        folly::acquireLocked(segmentsAvl_, regElemsMaps_);
    auto& segToRegElemsMap = regElemsMaps->segToRegElemsMap;
    auto& regHdlToElemMap = regElemsMaps->regHdlToElemMap;

    // MIN_TODO: check if segHdl is valid in lookup
    segment = reinterpret_cast<ctran::regcache::Segment*>(
        segmentsAvl->lookup(segHdl));

    // Invalid segment handle, likely double free
    if (!segment) {
      CLOGF(ERR, "Invalid segment handle {}", (void*)segHdl);
      return commInvalidUsage;
    }

    ncclManaged = segment->ncclManaged;

    // Ask for free. False if still in use, then no-op and return
    if (!segment->askFree()) {
      return commSuccess;
    }

    // Now the segment is ready to be freed
    // - Find all associated regElems of the segment
    auto segIt = segToRegElemsMap.find(segment);
    if (segIt != segToRegElemsMap.end()) {
      auto& regHdls = segIt->second;

      // - Find each regElem and remove from global cache
      for (auto regHdl : regHdls) {
        auto regIt = regHdlToElemMap.find(regHdl);

        // The regElem has already been deregistered likely when freeing
        // another associated segment
        if (regIt == regHdlToElemMap.end()) {
          continue;
        }

        // Remove regElem from global cache and to be deregistered
        regElems.push_back(std::move(regIt->second));
        regHdlToElemMap.erase(regIt);
      }

      segToRegElemsMap.erase(segIt);
    }

    // - Remove segment from cache
    FB_COMMCHECK(segmentsAvl->remove(segment->avlHdl_));
    CLOGF_TRACE(
        ALLOC,
        "Removed segment {} segHdl {} ptr {} len {} ncclManaged {} cudaDev {}, cache size {}",
        (void*)segment,
        (void*)segHdl,
        segment->range.buf,
        segment->range.len,
        segment->ncclManaged,
        segment->cudaDev,
        segmentsAvl->size());

    // Tell mapper the segment is no longer in cache
    freed = true;
  }

  // Deregister all regElems.
  // NOTE: not yet free the memory. Return the ownership to caller for any
  // remote registration release.
  for (auto& regElem : regElems) {
    FB_COMMCHECK(deregElem(regElem.get()));
  }

  // Free segment here, in case regElem accesses it during deregisteration
  delete segment;

  profiler.wlock()->record(ctran::regcache::EventType::kFreeSegEvent);
  return commSuccess;
}

commResult_t ctran::RegCache::deregElem(ctran::regcache::RegElem* regElem) {
  auto dur = CtranMapperTimer();
  FB_COMMCHECK(regElem->doDeregister());
  profiler.wlock()->record(ctran::regcache::EventType::kDeregMemEvent, dur);
  return commSuccess;
}

commResult_t ctran::RegCache::regDynamic(
    const void* ptr,
    const size_t len,
    int cudaDev,
    const std::vector<bool>& backends,
    ctran::regcache::RegElem** regElem) {
  auto dur = CtranMapperTimer();
  SetCudaDevRAII setCudaDev(cudaDev);

  std::vector<ctran::regcache::SegmentRange> ranges;
  FB_COMMCHECK(
      ctran::regcache::SegmentRange::pinRange(ptr, cudaDev, len, ranges));
  FB_CHECKABORT(
      ranges.size() > 0, "No range found for ptr {} len {}", ptr, len);

  // Raw ptr can be unaligned to host page size, so we need to register base
  // ranges instead of ptr.
  size_t lenToReg = 0;
  for (const auto& range : ranges) {
    lenToReg += range.len;
  }
  auto newRegElem_ = std::make_unique<ctran::regcache::RegElem>(
      ranges.at(0).buf,
      lenToReg,
      cudaDev,
      true /*isDynamic*/,
      ranges.at(0).type);

  // Registration (expensive)
  FB_COMMCHECK(newRegElem_->doRegister(backends));

  *regElem = newRegElem_.get();

  // Global lock to update regElemsMaps_.
  // Lock after registration, avoid long holding time of the lock.
  regElemsMaps_.wlock()->regHdlToElemMap.emplace(
      *regElem, std::move(newRegElem_));

  {
    auto profilerLk = profiler.wlock();
    profilerLk->record(ctran::regcache::EventType::kDynamicRegMemEvent);
    profilerLk->record(ctran::regcache::EventType::kRegMemEvent, dur);
  }

  return commSuccess;
}

commResult_t ctran::RegCache::deregDynamic(ctran::regcache::RegElem* regHdl) {
  std::unique_ptr<ctran::regcache::RegElem> regElem = nullptr;
  // Global lock to update regElemsMaps_.
  // Unlock before deregistration, avoid long holding time of the lock.
  {
    auto regElemsMaps = regElemsMaps_.wlock();
    auto& regHdlToElemMap = regElemsMaps->regHdlToElemMap;

    auto it = regHdlToElemMap.find(regHdl);
    if (it == regHdlToElemMap.end()) {
      CLOGF(ERR, "deregDynamic: regElem {} not found", (void*)regHdl);
      return commInvalidUsage;
    }
    // Remove regElem from global cache and return ownership to caller for any
    // remote registration release
    regElem = std::move(it->second);
    regHdlToElemMap.erase(it);
  }

  // Deregistration (expensive)
  FB_COMMCHECK(deregElem(regElem.get()));

  return commSuccess;
}

// Helper function to get all segments and group them into contiguous memory
// regions. Sorts segments by address and groups when one segment's end address
// equals the next segment's start address.
// Takes segments by value to allow move semantics and avoid extra copies.
namespace {
std::vector<std::vector<ctran::regcache::Segment*>> getContiguousRegions(
    std::vector<ctran::regcache::Segment*> segments) {
  std::vector<std::vector<ctran::regcache::Segment*>> contiguousRegions;

  if (segments.empty()) {
    return contiguousRegions;
  }

  // Sort segments by starting address (sort in-place)
  std::sort(
      segments.begin(),
      segments.end(),
      [](const ctran::regcache::Segment* a, const ctran::regcache::Segment* b) {
        return reinterpret_cast<uintptr_t>(a->range.buf) <
            reinterpret_cast<uintptr_t>(b->range.buf);
      });

  // Group segments into contiguous regions
  // A contiguous region is a set of segments where each segment's end address
  // equals the next segment's start address
  std::vector<ctran::regcache::Segment*> currentRegion;
  currentRegion.push_back(segments[0]);

  for (size_t i = 1; i < segments.size(); i++) {
    auto prevSeg = segments[i - 1];
    auto currSeg = segments[i];

    uintptr_t prevEndAddr =
        reinterpret_cast<uintptr_t>(prevSeg->range.buf) + prevSeg->range.len;
    uintptr_t currStartAddr = reinterpret_cast<uintptr_t>(currSeg->range.buf);

    if (prevEndAddr == currStartAddr) {
      // Contiguous with previous segment, add to current region
      currentRegion.push_back(currSeg);
    } else {
      // Gap detected, start a new region
      contiguousRegions.push_back(std::move(currentRegion));
      currentRegion.clear();
      currentRegion.push_back(currSeg);
    }
  }
  // Add the last region
  contiguousRegions.push_back(std::move(currentRegion));

  // Log summary of contiguous regions
  for (size_t i = 0; i < contiguousRegions.size(); i++) {
    auto& region = contiguousRegions[i];
    uintptr_t regionStart =
        reinterpret_cast<uintptr_t>(region.front()->range.buf);
    uintptr_t regionEnd =
        reinterpret_cast<uintptr_t>(region.back()->range.buf) +
        region.back()->range.len;
    size_t regionLen = regionEnd - regionStart;
    CLOGF_TRACE(
        ALLOC,
        "getContiguousRegions: region[{}] ptr=0x{:x} len={} ({} segments)",
        i,
        regionStart,
        regionLen,
        region.size());
  }

  return contiguousRegions;
}
} // namespace

// Global API: Static version that doesn't require communicator-specific params
commResult_t ctran::RegCache::regAll() {
  auto regCache = ctran::RegCache::getInstance();
  if (!regCache) {
    CLOGF(ERR, "regAll: RegCache instance not available");
    return commInternalError;
  }

  auto dur = CtranMapperTimer();
  auto timerBegin = std::chrono::steady_clock::now();

  // Track total stats for scuba logging
  size_t totalLenRegistered = 0;
  size_t totalSegmentsRegistered = 0;
  size_t numContiguousRegions = 0;

  {
    // Global lock:
    // - Serialize concurrent registration updates, also with cache|free
    // segments.
    auto segmentsAvl = regCache->segmentsAvl_.wlock();

    // Get all segment values directly from AVL tree
    auto allSegmentVals = segmentsAvl->getAllElemVals();
    if (allSegmentVals.empty()) {
      CLOGF(WARN, "regAll: no cached segments found");
      return commSuccess;
    }

    // Convert to typed segment pointers
    std::vector<ctran::regcache::Segment*> segments;
    segments.reserve(allSegmentVals.size());
    for (auto val : allSegmentVals) {
      segments.push_back(reinterpret_cast<ctran::regcache::Segment*>(val));
    }

    // Use helper to group segments into contiguous regions
    // Move segments since we no longer need them after this call
    auto contiguousRegions = getContiguousRegions(std::move(segments));

    if (contiguousRegions.empty()) {
      CLOGF(WARN, "regAll: no cached segments found");
      return commSuccess;
    }

    int cudaDev = contiguousRegions[0].front()->cudaDev;
    if (cudaDev < 0) {
      CLOGF(ERR, "regAll: could not determine cudaDev from cached segments");
      return commInternalError;
    }

    SetCudaDevRAII setCudaDev(cudaDev);

    numContiguousRegions = contiguousRegions.size();

    // Register each contiguous region separately using the helper
    for (size_t regionIdx = 0; regionIdx < contiguousRegions.size();
         regionIdx++) {
      auto& regionSegments = contiguousRegions[regionIdx];

      // Calculate the range for this contiguous region
      void* regionPtr = const_cast<void*>(regionSegments.front()->range.buf);
      uintptr_t regionStartAddr = reinterpret_cast<uintptr_t>(regionPtr);
      uintptr_t regionEndAddr =
          reinterpret_cast<uintptr_t>(regionSegments.back()->range.buf) +
          regionSegments.back()->range.len;
      size_t regionLen = regionEndAddr - regionStartAddr;

      CLOGF_TRACE(
          ALLOC,
          "regAll: registering region {} with {} segments, ptr {} len {}",
          regionIdx,
          regionSegments.size(),
          regionPtr,
          regionLen);

      ctran::regcache::RegElem* regHdl = nullptr;
      FB_COMMCHECK(regCache->registerSegmentsTogether(
          regionPtr,
          regionLen,
          cudaDev,
          regionSegments,
          regCache->globalBackends_,
          false /* ncclManaged */,
          &regHdl));

      totalLenRegistered += regionLen;
      totalSegmentsRegistered += regionSegments.size();

      // Record registration event for this region
      regCache->profiler.wlock()->record(
          ctran::regcache::EventType::kRegMemEvent, dur);
    }
  }

  // Log to scuba (aggregate stats for all regions)
  if (totalLenRegistered > 0) {
    struct CommLogData defaultLogMetaData = {
        0, // opCount
        0, // commHash
        "regAll", // commDesc
        0, // rank
        0 // nRanks
    };
    logMemoryEvent(
        defaultLogMetaData,
        "",
        "regAll",
        0, // No single pointer for multiple regions
        totalLenRegistered,
        totalSegmentsRegistered,
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - timerBegin)
            .count(),
        true /* isRegMemEvent */);
  }

  CLOGF_TRACE(
      ALLOC,
      "regAll: completed - registered {} contiguous regions, "
      "total {} segments, {} bytes",
      numContiguousRegions,
      totalSegmentsRegistered,
      totalLenRegistered);

  return commSuccess;
}

// Global API: Deregister all non-dynamic registration elements.
// This removes all registrations but keeps cached segments intact.
commResult_t ctran::RegCache::deregAll() {
  auto regCache = ctran::RegCache::getInstance();
  if (!regCache) {
    CLOGF(ERR, "deregAll: RegCache instance not available");
    return commInternalError;
  }

  size_t totalDeregistered = 0;
  size_t totalSkippedDynamic = 0;

  // Collect regElems to deregister outside the lock
  std::vector<std::unique_ptr<ctran::regcache::RegElem>> toDeregister;

  {
    auto regElemsMaps = regCache->regElemsMaps_.wlock();
    auto& regHdlToElemMap = regElemsMaps->regHdlToElemMap;
    auto& segToRegElemsMap = regElemsMaps->segToRegElemsMap;

    // Iterate through all regElems and collect non-dynamic ones for
    // deregistration
    for (auto it = regHdlToElemMap.begin(); it != regHdlToElemMap.end();) {
      auto& regElem = it->second;

      // Skip dynamic registrations
      if (regElem->isDynamic_) {
        totalSkippedDynamic++;
        ++it;
        continue;
      }

      // Remove from segToRegElemsMap
      for (auto seg : regElem->segments_) {
        auto segIt = segToRegElemsMap.find(seg);
        if (segIt != segToRegElemsMap.end()) {
          auto& regElemsVec = segIt->second;
          regElemsVec.erase(
              std::remove(
                  regElemsVec.begin(), regElemsVec.end(), regElem.get()),
              regElemsVec.end());
          if (regElemsVec.empty()) {
            segToRegElemsMap.erase(segIt);
          }
        }
      }

      // Transfer ownership to toDeregister vector and remove from map
      toDeregister.push_back(std::move(regElem));
      it = regHdlToElemMap.erase(it);
      totalDeregistered++;
    }
  }

  // Call remReleaseMem on ipcRegElems before deregistering.
  // This notifies remote peers to release their imported NVL memory.
  // Like the mapper path, we use fire-and-forget semantics - the async socket
  // sends will complete in the background.
  auto ipcRegCache = ctran::IpcRegCache::getInstance();
  std::string localPeerId = ipcRegCache->getLocalPeerId();
  std::deque<std::unique_ptr<ctran::regcache::IpcReqCb>> postedReqs;
  for (auto& regElem : toDeregister) {
    auto ipcRegElem =
        reinterpret_cast<ctran::regcache::IpcRegElem*>(regElem->ipcRegElem);
    if (ipcRegElem != nullptr) {
      FB_COMMCHECK(
          ipcRegCache->remReleaseMem(localPeerId, ipcRegElem, postedReqs));
    }
  }

  // Perform actual deregistration outside the lock
  for (auto& regElem : toDeregister) {
    auto res = regCache->deregElem(regElem.get());
    if (res != commSuccess) {
      CLOGF(
          ERR,
          "deregAll: failed to deregister regElem ptr {} len {}",
          regElem->buf,
          regElem->len);
    }
  }

  CLOGF(
      INFO,
      "deregAll: completed - deregistered {} registrations, "
      "skipped {} dynamic",
      totalDeregistered,
      totalSkippedDynamic);

  return commSuccess;
}

commResult_t ctran::regcache::RegElem::doRegister(
    const std::vector<bool>& backends) {
  auto stat = stateMnger.wlock();

  // Register to backends
  if (type_ != DevMemType::kHostUnregistered &&
      // TODO: TCPDM does not support NVL yet.
      backends[CommBackend::TCPDM] == false) {
    // Register to NVL backend if it is device accessible memory
    // TODO: add support for managed and host pinned memory
    FB_CHECKABORT(ipcRegElem == nullptr, "ipcRegElem is already registered");
    try {
      // Note: shouldSupportCudaMalloc is safely enabled by ncclManaged.
      // The callsite will guarantee that all ranks will perform safe-release of
      // the buffer, avoiding any premature deallocation issues.
      FB_COMMCHECK(
          ctran::IpcRegCache::regMem(
              buf, len, cudaDev_, &ipcRegElem, ncclManaged_));
    } catch ([[maybe_unused]] const std::bad_alloc& e) {
      CLOGF(
          WARN,
          "CTRAN-REGCACHE: NVL backend not enabled. Skip IPC registration for buf {} len {}",
          (void*)buf,
          len);
    }
  }

  FB_CHECKABORT(ibRegElem == nullptr, "ibRegElem is already registered");
  if (backends[CommBackend::IB]) {
    try {
      FB_COMMCHECK(CtranIb::regMem(buf, len, cudaDev_, &ibRegElem));
    } catch ([[maybe_unused]] const std::bad_alloc& e) {
      CLOGF(
          WARN,
          "CTRAN-REGCACHE: IB backend not enabled. Skip IB registration for buf {} len {}",
          (void*)buf,
          len);
    }
  }

  // Register with TCPDM backend unless already registered with IB.
  if (backends[CommBackend::TCPDM]) {
    FB_COMMCHECK(
        ctran::CtranTcpDm::regMem((void*)buf, len, cudaDev_, &tcpRegElem));
  }

  stat->state = ctran::regcache::RegElemState::REGISTERED;
  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-REGCACHE: registered RegElem {} [{}] ",
      (void*)this,
      toString(stat->state).c_str());

  return commSuccess;
}

commResult_t ctran::regcache::RegElem::doDeregister() {
  auto stat = stateMnger.wlock();

  FB_CHECKABORT(
      stat->state == ctran::regcache::RegElemState::REGISTERED,
      "Unexpected state {} in deregistration of RegElem {} [{}]",
      stat->state,
      (void*)this,
      toString(stat->state).c_str());

  // Deregister from backends
  if (ipcRegElem) {
    ctran::IpcRegCache::deregMem(ipcRegElem);
    ipcRegElem = nullptr;
  }
  if (ibRegElem) {
    FB_COMMCHECK(CtranIb::deregMem(ibRegElem));
    ibRegElem = nullptr;
  }
  if (tcpRegElem) {
    FB_COMMCHECK(ctran::CtranTcpDm::deregMem(tcpRegElem));
    tcpRegElem = nullptr;
  }

  stat->state = ctran::regcache::RegElemState::DEREGISTERED;
  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-REGCACHE: deregistered RegElem {} [{}] ",
      (void*)this,
      toString(stat->state).c_str());

  return commSuccess;
}
