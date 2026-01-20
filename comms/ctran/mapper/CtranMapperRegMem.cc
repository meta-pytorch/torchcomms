// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/mapper/CtranMapperRegMem.h"
#include <folly/Singleton.h>
#include <folly/system/ThreadName.h>

#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/alloc.h"

static folly::Singleton<CtranMapperRegCache> regCacheSingleton;
std::shared_ptr<CtranMapperRegCache> CtranMapperRegCache::getInstance() {
  return regCacheSingleton.try_get();
}

CtranMapperRegCache::CtranMapperRegCache(void) {
  init();
}

CtranMapperRegCache::~CtranMapperRegCache(void) {
  // Define separate destroy() function to allow test to call it explicitly
  FB_COMMCHECKIGNORE(destroy());
}

static const std::unordered_map<CtranRegCacheEventType, std::string>
    RegCacheEventNameMap = {
        {kCacheSegEvent, "CACHE"},
        {kRegMemEvent, "REG"},
        {kDeregMemEvent, "DEREG"},
        {kDynamicRegMemEvent, "DYNAMIC_REG"},
        {kAsyncRegMemEvent, "ASYNC_REG"},
};

// Currently cached and registered buffers.
// The number may change overtime if any buffer is deregistered.
static const std::vector<CtranRegCacheEventType> currentEvents = {
    CtranRegCacheEventType::kCacheSegEvent,
    CtranRegCacheEventType::kRegMemEvent,
};

static const std::vector<CtranRegCacheEventType> latencyEvents = {
    CtranRegCacheEventType::kRegMemEvent,
    CtranRegCacheEventType::kDeregMemEvent,
};

static const std::vector<CtranRegCacheEventType> totalEvents = {
    CtranRegCacheEventType::kCacheSegEvent,
    CtranRegCacheEventType::kRegMemEvent,
    CtranRegCacheEventType::kDeregMemEvent,
    CtranRegCacheEventType::kDynamicRegMemEvent,
    CtranRegCacheEventType::kAsyncRegMemEvent,
};

CtranMapperRegCacheProfiler::CtranMapperRegCacheProfiler() {
  reset();
}

void CtranMapperRegCacheProfiler::reset(void) {
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

void CtranMapperRegCacheProfiler::record(
    CtranRegCacheEventType type,
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

void CtranMapperRegCacheProfiler::record(CtranRegCacheEventType type) {
  if (NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT < 0) {
    return;
  }

  if (std::find(totalEvents.begin(), totalEvents.end(), type) !=
      totalEvents.end()) {
    totalCountMap.at(type)++;
  }

  if (type == CtranRegCacheEventType::kFreeSegEvent) {
    currentCountMap.at(CtranRegCacheEventType::kCacheSegEvent)--;
  } else if (type == CtranRegCacheEventType::kDeregMemEvent) {
    currentCountMap.at(CtranRegCacheEventType::kRegMemEvent)--;
  } else if (
      std::find(currentEvents.begin(), currentEvents.end(), type) !=
      currentEvents.end()) {
    currentCountMap.at(type)++;
  }

  // Allow periodical snapshot report during long job running
  if (type == CtranRegCacheEventType::kRegMemEvent &&
      NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT > 0 &&
      (totalCountMap.at(type) % NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT ==
       0)) {
    reportSnapshot();
  }
}

void CtranMapperRegCacheProfiler::reportSnapshot(void) const {
  const std::string prefix = "CTRAN-MAPPER RegCache Snapshot";
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

CtranMapperRegCacheSnapshot CtranMapperRegCacheProfiler::getSnapshot() const {
  CtranMapperRegCacheSnapshot snapshot;
  snapshot.currentNumCache =
      currentCountMap.at(CtranRegCacheEventType::kCacheSegEvent);
  snapshot.currentNumReg =
      currentCountMap.at(CtranRegCacheEventType::kRegMemEvent);
  snapshot.totalNumCache =
      totalCountMap.at(CtranRegCacheEventType::kCacheSegEvent);
  snapshot.totalNumReg = totalCountMap.at(CtranRegCacheEventType::kRegMemEvent);
  snapshot.totalNumDereg =
      totalCountMap.at(CtranRegCacheEventType::kDeregMemEvent);
  snapshot.totalNumDynamicReg =
      totalCountMap.at(CtranRegCacheEventType::kDynamicRegMemEvent);
  snapshot.totalNumAsyncReg =
      totalCountMap.at(CtranRegCacheEventType::kAsyncRegMemEvent);
  snapshot.regMemLatency = latencyMap.at(CtranRegCacheEventType::kRegMemEvent) /
      snapshot.totalNumReg;
  snapshot.deregMemLatency =
      latencyMap.at(CtranRegCacheEventType::kDeregMemEvent) /
      snapshot.totalNumDereg;
  return snapshot;
}

void CtranMapperRegCache::init() {
  if (NCCL_CTRAN_REGISTER == NCCL_CTRAN_REGISTER::async &&
      !asyncRegThread_.joinable()) {
    int cudaDev;
    FB_CUDACHECKTHROW_EX_NOCOMM(cudaGetDevice(&cudaDev));
    asyncRegThread_ =
        std::thread{&CtranMapperRegCache::asyncRegThreadFn, this, cudaDev};
  }
}

commResult_t CtranMapperRegCache::destroy() {
  {
    // Warn if user missed any buffer registration.
    // Skip deregistration to avoid unexpected error at destruction time for
    // now. We need revisit after sets CtranMapperRegCache's dependency to
    // CtranIbSingleton, otherwise ib singleton may be released before
    // deregisteration here.
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
      auto seg =
          reinterpret_cast<CtranMapperSegment*>(segmentsAvl->lookup(avlHdl));
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
  }

  // Report snapshot at destroy if enabled
  if (NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT >= 0) {
    profiler.rlock()->reportSnapshot();
  }

  return commSuccess;
}

void CtranMapperRegCache::asyncRegThreadFn(int cudaDev) {
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
    CtranMapperRegElem* regHdl = nullptr;

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
      profiler.wlock()->record(CtranRegCacheEventType::kAsyncRegMemEvent);
    }

    // NOTE: regHdl may already be released by concurrent deregMem from main
    // thread; unsafe to read its content
    CLOGF_TRACE(
        ALLOC,
        "CTRAN-MAPPER: async registered buf {} len {} didRegister {} regHdl {}",
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

commResult_t CtranMapperRegCache::asyncRegRange(
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

void CtranMapperRegCache::waitAsyncRegComplete() {
  while (true) {
    auto locked = asyncRegQueue_.lock();
    if (locked->empty()) {
      break;
    }
  }
}

CtranMapperRegElem* CtranMapperRegCache::searchRegElem(
    const void* ptr,
    const size_t len) {
  CtranMapperRegElem* regHdl = nullptr;

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

bool CtranMapperRegCache::isRegistered(const void* ptr, const size_t len) {
  // Find range in regElemsMaps
  auto regHdl = searchRegElem(ptr, len);
  return regHdl != nullptr;
}

std::vector<void*> CtranMapperRegCache::getSegments() const {
  return segmentsAvl_.rlock()->getAllElems();
}

commResult_t CtranMapperSegmentRange::pinRange(
    const void* ptr,
    const int cudaDev,
    size_t len,
    std::vector<CtranMapperSegmentRange>& segRangs) {
  DevMemType memType{DevMemType::kCumem};
  FB_COMMCHECK(getDevMemType(ptr, cudaDev, memType));

  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-MAPPER pinRange: input ptr={} len={} cudaDev={} memType={}",
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
  int segmentIdx = 1;
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

commResult_t CtranMapperRegCache::cacheSegment(
    const void* ptr,
    const size_t len,
    const int cudaDev,
    const bool ncclManaged,
    uint64_t commHash,
    std::vector<CtranMapperSegment*>& segments,
    std::vector<void*>& segHdls) {
  SetCudaDevRAII setCudaDev(cudaDev);
  {
    auto segmentsAvl = segmentsAvl_.wlock();

    // First, discover all physical segments underlying this buffer
    std::vector<CtranMapperSegmentRange> ranges;
    FB_COMMCHECK(CtranMapperSegmentRange::pinRange(ptr, cudaDev, len, ranges));

    CLOGF_SUBSYS(
        INFO,
        ALLOC,
        "CTRAN-MAPPER cacheSegment: ptr={} len={} discovered {} physical segments",
        ptr,
        len,
        ranges.size());

    // Cache each discovered physical segment chunk
    for (size_t i = 0; i < ranges.size(); i++) {
      const auto& range = ranges.at(i);
      void* avlHdl = nullptr;

      // Check if this segment is already cached
      avlHdl = segmentsAvl->search(range.buf, range.len);
      if (avlHdl) {
        // Segment already cached, increase refcount
        auto foundSeg =
            reinterpret_cast<CtranMapperSegment*>(segmentsAvl->lookup(avlHdl));
        {
          auto segState = foundSeg->stateMnger.wlock();
          segState->refCount++;
        }
        segments.push_back(foundSeg);
        segHdls.push_back(foundSeg->avlHdl_);

        CLOGF_TRACE(
            ALLOC,
            "CTRAN-MAPPER cacheSegment: segment[{}] already cached ptr={} len={} refCount++",
            i,
            range.buf,
            range.len);
      } else {
        // Create new cache entry for this segment
        auto newSeg = new CtranMapperSegment(range, cudaDev, ncclManaged);
        avlHdl = segmentsAvl->insert(range.buf, range.len, newSeg);
        newSeg->avlHdl_ = avlHdl;
        segments.push_back(newSeg);
        segHdls.push_back(avlHdl);

        auto type = newSeg->getType();
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

  profiler.wlock()->record(CtranRegCacheEventType::kCacheSegEvent);
  return commSuccess;
}

commResult_t CtranMapperRegCache::regRange(
    const void* ptr,
    const size_t len,
    const int cudaDev,
    const std::string& useDesc,
    const struct CommLogData& logMetaData,
    const std::vector<bool>& backends,
    bool& didRegister,
    CtranMapperRegElem** regHdl,
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
    std::vector<CtranMapperSegmentRange> ranges;
    FB_COMMCHECK(CtranMapperSegmentRange::pinRange(ptr, cudaDev, len, ranges));

    std::vector<CtranMapperSegment*> segments(ranges.size(), nullptr);
    bool foundAll = true;

    // - SLOW PATH: find the cached segments corresponding to each range.
    for (int i = 0; i < ranges.size(); i++) {
      auto& segRange = ranges.at(i);
      void* avlHdl = segmentsAvl->search(segRange.buf, segRange.len);
      if (!avlHdl) {
        CLOGF(
            WARN,
            "CTRAN-MAPPER:[pbase {} range {}] associated with [ptr {} len {}] is not pre-registered by user",
            (void*)segRange.buf,
            segRange.len,
            (void*)ptr,
            len);
        foundAll = false;
        break;
      }
      segments[i] =
          reinterpret_cast<CtranMapperSegment*>(segmentsAvl->lookup(avlHdl));
      lenToReg += segments.at(i)->range.len;
    }

    if (foundAll) {
      // - SLOW PATH: found all cached segments, register the full segment
      // range.
      ptrToReg = const_cast<void*>(segments.at(0)->range.buf);
      numSegmentsToReg = segments.size();
      auto newRegElem = std::make_unique<CtranMapperRegElem>(
          ptrToReg, lenToReg, cudaDev, segments, ncclManaged);

      // Backend registration
      FB_COMMCHECK(newRegElem->doRegister(backends));
      auto regHdl_ = newRegElem.get();

      // Global lock to update regElemsMaps.
      // We have to update regElemsMaps before releasing the global lock to
      // segmentsAvl. Otherwise, another thread may hold the global lock to
      // segmentsAvl before regElemsMaps updates, and duplicate the registration
      // for a given buffer.
      {
        auto regElemsMaps = regElemsMaps_.wlock();
        auto& regHdlToElemMap = regElemsMaps->regHdlToElemMap;
        auto& segToRegElemsMap = regElemsMaps->segToRegElemsMap;

        regHdlToElemMap.emplace(regHdl_, std::move(newRegElem));
        // Correlate the regElem with all associated segments to deregister it
        // when any segment is freed
        for (auto seg : segments) {
          segToRegElemsMap[seg].emplace_back(regHdl_);
        }
      }

      *regHdl = regHdl_;
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

  profiler.wlock()->record(CtranRegCacheEventType::kRegMemEvent, dur);
  return commSuccess;
}

bool CtranMapperSegment::askFree() {
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

CtranMapperSegment* CtranMapperRegCache::getSegment(void* segHdl) {
  return reinterpret_cast<CtranMapperSegment*>(
      segmentsAvl_.rlock()->lookup(segHdl));
}

std::vector<CtranMapperRegElem*> CtranMapperRegCache::getRegElems(
    const void* segHdl) const {
  std::vector<CtranMapperRegElem*> regElems;

  const auto segment = reinterpret_cast<CtranMapperSegment*>(
      segmentsAvl_.rlock()->lookup(const_cast<void*>(segHdl)));
  if (segment) {
    // Find all associated regElems of the segment
    auto locked = regElemsMaps_.rlock();
    auto& segToRegElemsMap = locked->segToRegElemsMap;
    auto segIt = segToRegElemsMap.find(segment);
    if (segIt != segToRegElemsMap.end()) {
      // Copy vector and return
      regElems = segIt->second;
    }
  }

  return regElems;
}

commResult_t CtranMapperRegCache::freeSegment(
    void* segHdl,
    bool& freed,
    bool& ncclManaged,
    std::vector<std::unique_ptr<CtranMapperRegElem>>& regElems) {
  CtranMapperSegment* segment = nullptr;
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
    segment =
        reinterpret_cast<CtranMapperSegment*>(segmentsAvl->lookup(segHdl));

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

  profiler.wlock()->record(CtranRegCacheEventType::kFreeSegEvent);
  return commSuccess;
}

commResult_t CtranMapperRegCache::deregElem(CtranMapperRegElem* regElem) {
  auto dur = CtranMapperTimer();
  FB_COMMCHECK(regElem->doDeregister());
  profiler.wlock()->record(CtranRegCacheEventType::kDeregMemEvent, dur);
  return commSuccess;
}

commResult_t CtranMapperRegCache::regDynamic(
    const void* ptr,
    const size_t len,
    int cudaDev,
    const std::vector<bool>& backends,
    CtranMapperRegElem** regElem) {
  auto dur = CtranMapperTimer();
  SetCudaDevRAII setCudaDev(cudaDev);

  std::vector<CtranMapperSegmentRange> ranges;
  FB_COMMCHECK(CtranMapperSegmentRange::pinRange(ptr, cudaDev, len, ranges));
  FB_CHECKABORT(
      ranges.size() > 0, "No range found for ptr {} len {}", ptr, len);

  // Raw ptr can be unaligned to host page size, so we need to register base
  // ranges instead of ptr.
  size_t lenToReg = 0;
  for (const auto& range : ranges) {
    lenToReg += range.len;
  }
  auto newRegElem_ = std::make_unique<CtranMapperRegElem>(
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
    profilerLk->record(CtranRegCacheEventType::kDynamicRegMemEvent);
    profilerLk->record(CtranRegCacheEventType::kRegMemEvent, dur);
  }

  return commSuccess;
}

commResult_t CtranMapperRegCache::deregDynamic(CtranMapperRegElem* regHdl) {
  std::unique_ptr<CtranMapperRegElem> regElem = nullptr;
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

commResult_t CtranMapperRegElem::doRegister(const std::vector<bool>& backends) {
  auto stat = stateMnger.wlock();

  // Register to backends
  if (type_ != DevMemType::kHostUnregistered &&
      // TODO: TCPDM does not support NVL yet.
      backends[CommBackend::TCPDM] == false) {
    // Register to NVL backend if it is device accessible memory
    // TODO: add support for managed and host pinned memory
    FB_CHECKABORT(nvlRegElem == nullptr, "nvlRegElem is already registered");
    try {
      // Note: shouldSupportCudaMalloc is safely enabled by ncclManaged.
      // The callsite will guarantee that all ranks will perform safe-release of
      // the buffer, avoiding any premature deallocation issues.
      FB_COMMCHECK(
          CtranNvl::regMem(buf, len, cudaDev_, &nvlRegElem, ncclManaged_));
    } catch (const std::bad_alloc& e) {
      CLOGF(
          WARN,
          "CTRAN-MAPPER: NVL backend not enabled. Skip NVL registration for buf {} len {}",
          (void*)buf,
          len);
    }
  }

  FB_CHECKABORT(ibRegElem == nullptr, "ibRegElem is already registered");
  if (backends[CommBackend::IB]) {
    try {
      FB_COMMCHECK(CtranIb::regMem(buf, len, cudaDev_, &ibRegElem));
    } catch (const std::bad_alloc& e) {
      CLOGF(
          WARN,
          "CTRAN-MAPPER: IB backend not enabled. Skip IB registration for buf {} len {}",
          (void*)buf,
          len);
    }
  }

  // Register with TCPDM backend unless already registered with IB.
  if (backends[CommBackend::TCPDM]) {
    FB_COMMCHECK(
        ctran::CtranTcpDm::regMem((void*)buf, len, cudaDev_, &tcpRegElem));
  }

  stat->state = CtranMapperRegElemState::REGISTERED;
  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-MAPPER: registered RegElem {} [{}] ",
      (void*)this,
      toString(stat->state).c_str());

  return commSuccess;
}

commResult_t CtranMapperRegElem::doDeregister() {
  auto stat = stateMnger.wlock();

  FB_CHECKABORT(
      stat->state == CtranMapperRegElemState::REGISTERED,
      "Unexpected state {} in deregistration of RegElem {} [{}]",
      stat->state,
      (void*)this,
      toString(stat->state).c_str());

  // Deregister from backends
  if (nvlRegElem) {
    FB_COMMCHECK(CtranNvl::deregMem(nvlRegElem));
    nvlRegElem = nullptr;
  }
  if (ibRegElem) {
    FB_COMMCHECK(CtranIb::deregMem(ibRegElem));
    ibRegElem = nullptr;
  }
  if (tcpRegElem) {
    FB_COMMCHECK(ctran::CtranTcpDm::deregMem(tcpRegElem));
    tcpRegElem = nullptr;
  }

  stat->state = CtranMapperRegElemState::DEREGISTERED;
  CLOGF_SUBSYS(
      INFO,
      ALLOC,
      "CTRAN-MAPPER: deregistered RegElem {} [{}] ",
      (void*)this,
      toString(stat->state).c_str());

  return commSuccess;
}
