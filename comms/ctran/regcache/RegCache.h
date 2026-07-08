// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <fmt/format.h>
#include <folly/Synchronized.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_map>

#include "comms/ctran/backends/ib/CtranIbSingleton.h"
#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranAvlTree.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/DevMemType.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/commSpecs.h"

using meta::comms::CommBackend;

namespace ctran {

class RegCache;

namespace regcache {

using ExternalRegMemFn = std::function<commResult_t(
    const void* buf,
    const size_t len,
    const int cudaDev,
    void** regElem)>;

using ExternalDeregMemFn = std::function<commResult_t(void* regElem)>;

struct SegmentRange {
  const void* buf{nullptr};
  const std::size_t len{0};
  DevMemType type{DevMemType::kCudaMalloc};

 public:
  SegmentRange(const void* buf, const std::size_t len, DevMemType type)
      : buf(buf), len(len), type(type) {};

  std::string toString() const {
    std::stringstream ss;
    ss << "buf: " << buf << ", len: " << len
       << ", type: " << devMemTypeStr(type);
    return ss.str();
  }

  // Pin the underlying segment ranges of a given memory range
  static commResult_t pinRange(
      const void* ptr,
      const int cudaDev,
      size_t len,
      std::vector<SegmentRange>& segRangs);
};

struct SegmentStateMnger {
  // The number of communicators having cached the segment.
  // The segment must be removed and any associated RegElem must be deregistered
  // when the refCount is 0
  int64_t refCount{1};
};

struct RegElem;

struct Segment {
  SegmentRange range;
  const int cudaDev;
  bool ncclManaged{false};

  folly::Synchronized<SegmentStateMnger> stateMnger;

 public:
  Segment(SegmentRange range, const int cudaDev, const bool ncclManaged)
      : range(range), cudaDev(cudaDev), ncclManaged(ncclManaged) {};

  DevMemType getType() const {
    return range.type;
  }

  friend struct RegElem;
  friend class ::ctran::RegCache;

 protected:
  void* avlHdl_{nullptr};

  bool askFree();

  std::string toString(const int64_t refCount) const {
    std::stringstream ss;
    ss << "range: " << range.toString() << ", cudaDev:" << cudaDev
       << ", ncclManaged: " << ncclManaged << ", refCount: " << refCount
       << ", avlHdl: " << avlHdl_;
    return ss.str();
  }
};

enum RegElemState {
  REGISTERED,
  DEREGISTERED,
};

struct RegElemStateMnger {
  RegElemState state{RegElemState::REGISTERED};
};

struct RegElem {
  const void* buf{nullptr};
  const std::size_t len{0};
  void* ibRegElem{nullptr};
  void* ipcRegElem{nullptr};
  void* tcpRegElem{nullptr};
  void* externalRegElem{nullptr};

  // The state of the segment to ensure thread-safe access.
  // Concurrent writes:
  //   - doRegister may be called by multiple GPE threads concurrently when they
  //     lookup in collective. Only one thread should handle the actual backend
  //     registration; others are no-op (lookup-hit)
  //   - doDeregister for cached segment shall be called only by a single thread
  //     at commDeregister time.
  //   - doDeregister for dynamic regsitered segment shall be called only by the
  //     GPE thread at collective completion.
  // Concurrent reads:
  //   - segment may be looked up and accessed the internal fields by multiple
  //     GPE threads at collective time
  folly::Synchronized<RegElemStateMnger> stateMnger;

 public:
  RegElem(
      const void* buf,
      std::size_t len,
      const int cudaDev,
      bool isDynamic,
      DevMemType type,
      bool ncclManaged = false)
      : buf(buf),
        len(len),
        cudaDev_(cudaDev),
        isDynamic_(isDynamic),
        type_(type),
        ncclManaged_(ncclManaged) {}
  RegElem(
      const void* buf,
      std::size_t len,
      const int cudaDev,
      std::vector<Segment*>& segments,
      bool ncclManaged = false)
      // Explicitly copy segments to avoid changing passed-in vector which
      // may be used after the call
      : buf(buf),
        len(len),
        cudaDev_(cudaDev),
        segments_(segments),
        ncclManaged_(ncclManaged) {
    type_ = segments.at(0)->getType();
  };

  // Reference count controlling this RegElem's lifetime. A RegElem exists only
  // once the buffer is actually registered, and that registration holds one
  // reference, so refCount starts at 1. Each live ScopedRegHdl adds one more.
  //
  // The preexisting RegElem consumers (searchRegHandle / searchIbRegHandle /
  // getRegHandle) do NOT rely on refCount. They rely on the allocator-hook
  // contract: any lookup issued after the buffer's allocation hook
  // (globalRegister) and before its free hook (globalDeregister) is safe.
  // Consequently a ScopedRegHdl reference layered on top never triggers an
  // actual deregistration, and a scoped release can never drop the last
  // reference. Deregistration happens only when the allocator's free hook
  // (globalDeregister) runs, or when the RegCache shuts down.
  //
  // Atomic so a scoped acquire can increment it under the shared read lock.
  std::atomic<int64_t> refCount{1};

  // Process-unique monotonic id assigned at construction. Used to validate a
  // ScopedRegHdl's identity so that a reused heap address (a new RegElem
  // occupying a force-freed RegElem's old address) is not mistaken for the
  // original registration. The counter is a function-local static (NOT a global
  // variable), so it satisfies facebook-avoid-non-const-global-variables.
  const uint64_t regId_{[] {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
  }()};

  std::size_t numSegments() const {
    return segments_.size();
  }

  DevMemType getType() const {
    return type_;
  }

  bool isNcclManaged() const {
    // ncclManaged buffer always uses a single segment
    return segments_.size() == 1 && segments_.at(0)->ncclManaged;
  }

  friend class ::ctran::RegCache;

 protected:
  int cudaDev_{-1};
  bool isDynamic_{false};
  // Number of times the registration is reused by collective
  size_t lookupHit_{0};
  std::vector<Segment*> segments_;
  DevMemType type_{DevMemType::kCudaMalloc};
  bool ncclManaged_{false};

  // Thread-safe function to register the segment.
  // It internally locks the stateMnger to ensure thread-safe access.
  // The segment should be registered only once by the first thread and reused
  // by all later calls before deregistration.
  commResult_t doRegister(
      const std::vector<bool>& backends,
      const ExternalRegMemFn& externalRegMemFn = nullptr);

  // Thread-safe function to deregister the segment.
  // It internally locks the stateMnger to ensure thread-safe access.
  commResult_t doDeregister(
      const ExternalDeregMemFn& externalDeregMemFn = nullptr);

  // Thread-unsafe function to print internal fields.
  // It is only used for debugging purpose. Caller should ensure thread-safety
  // by locking the stateMnger.
  std::string toString(const RegElemState state) const {
    std::stringstream ss;
    ss << "buf: " << buf << ", len: " << len << ", state: "
       << (state == RegElemState::REGISTERED ? "REGISTERED" : "DEREGISTERED")
       << ", isDynamic: " << isDynamic_ << ", lookupHit: " << lookupHit_;
    if (state == RegElemState::REGISTERED) {
      ss << ", ibRegElem: " << ibRegElem << ", ipcRegElem: " << ipcRegElem
         << ", tcpRegElem: " << tcpRegElem;
    }
    if (segments_.size()) {
      ss << ", segments: [";
      for (auto seg : segments_) {
        ss << "[" + seg->toString(seg->stateMnger.rlock()->refCount) << "], ";
      }
      ss << "]";
    }
    return ss.str();
  }
};

enum EventType {
  kCacheSegEvent,
  kFreeSegEvent,
  kRegMemEvent,
  kDeregMemEvent,
  kDynamicRegMemEvent,
  kAsyncRegMemEvent,
};

struct Snapshot {
  uint32_t currentNumCache{0};
  uint32_t currentNumReg{0};

  uint32_t totalNumCache{0};
  uint32_t totalNumReg{0};
  uint32_t totalNumDereg{0};
  uint32_t totalNumDynamicReg{0};
  uint32_t totalNumAsyncReg{0};

  double regMemLatency{0};
  double deregMemLatency{0};
};

class Profiler {
  std::unordered_map<EventType, double> latencyMap;
  std::unordered_map<EventType, uint32_t> totalCountMap;
  std::unordered_map<EventType, uint32_t> currentCountMap;

 public:
  Profiler();

  // Record an event with latency
  void record(EventType type, CtranMapperTimer& dur);

  // Record an event without latency
  void record(EventType type);

  // Report the snapshot of the profiler via NCCL log
  void reportSnapshot() const;

  // Get an snapshot copy of the profiler
  Snapshot getSnapshot() const;

  // Reset the profiler records; used for testing only
  void reset();
};

} // namespace regcache

// Move-only RAII owner of a scoped local registration acquired via
// RegCache::acquireScopedRegister(). It owns ONLY a RegElem reference (one live
// ScopedRegHdl ref counted in RegElem::refCount); it does not cache or
// ref-count segments. The registration itself holds a reference; a scoped
// handle adds one on top and never triggers a deregistration. The destructor
// performs a pure SW-only ref decrement (graph-destroy safe): it never touches
// CUDA and never deregisters. A scoped release can never drop the last
// reference, so the registration persists (cached and reusable) until the
// allocator frees the segment via freeSegment/deregRange.
class ScopedRegHdl {
 public:
  ScopedRegHdl() = default;
  ScopedRegHdl(ScopedRegHdl&& other) noexcept;
  ScopedRegHdl& operator=(ScopedRegHdl&& other) noexcept;
  ScopedRegHdl(const ScopedRegHdl&) = delete;
  ScopedRegHdl& operator=(const ScopedRegHdl&) = delete;
  ~ScopedRegHdl();

  regcache::RegElem* get() const;
  explicit operator bool() const;
  std::string toString() const;

 private:
  friend class RegCache;

  RegCache* regCache_{nullptr};
  regcache::RegElem* regHdl_{nullptr};
  uint64_t regId_{0};
  const void* buf_{nullptr};
  size_t len_{0};
};

/**
 * Singleton class to hold the IB network resources that are reused by all
 * communicators in the lifetime of program.
 */
class RegCache {
 public:
  RegCache();
  ~RegCache();

  static std::shared_ptr<RegCache> getInstance();

  // Implement actual init/destroy logic here and called by
  // constructor/destructor. This allows test to manually trigger if needed.
  void init();
  commResult_t destroy();

  // Global registration using the globally-set backends.
  // This allows registration without requiring a communicator.
  // Backends are initialized from NCCL_CTRAN_BACKENDS cvar in init().
  // If forceReg is true, registration happens even in async/lazy mode.
  // deviceId is optional: if not assigned, infer it from getCudaDevFromPtr()
  commResult_t globalRegister(
      const void* buf,
      size_t len,
      bool forceReg = false,
      bool ncclManaged = false,
      int deviceId = -1,
      std::optional<std::vector<bool>> backends = std::nullopt);

  // Global deregistration using pointer lookup.
  // Frees cached segments and their associated registrations.
  commResult_t globalDeregister(
      const void* buf,
      size_t len,
      bool skipRemRelease = false,
      int deviceId = -1);

  // Acquire a scoped local registration for [buf, buf + len). The buffer's
  // underlying segment MUST already be cached by the allocator (globalRegister
  // / CCA memory hook); this API does not cache segments. On success it takes
  // one RegElem ref (counted in RegElem::refCount) and the returned
  // ScopedRegHdl owns that ref, releasing it SW-only in its destructor. If the
  // buffer is not backed by a cached segment, commInvalidUsage is returned and
  // the ScopedRegHdl stays empty.
  commResult_t acquireScopedRegister(
      const void* buf,
      size_t len,
      int cudaDev,
      const std::vector<bool>& backends,
      ScopedRegHdl& scopedRegHdl);

  // Thread-safe functions to cache a buffer range into the global cache.
  // This function uses pinRange to discover all physical segments underlying
  // the given buffer and caches each one individually.
  // input:
  //   - buf: the buffer to be cached
  //   - len: the length of the buffer
  //   - cudaDev: the cuda device id of the buffer
  //   - ncclManaged: whether the buffer is managed by NCCL
  //   - commHash: the commHash of the communicator that caches the buffer
  //               (logging purpose only, since commHash may not be 100%
  //               unique).
  // output:
  //   - segments: vector of cached segments (one per physical segment chunk)
  //   - segHdls: vector of handles for the cached segments
  commResult_t cacheSegment(
      const void* buf,
      const std::size_t len,
      const int cudaDev,
      const bool ncclManaged,
      uint64_t commHash,
      std::vector<regcache::Segment*>& segments,
      std::vector<void*>& segHdls);

  // Thread-safe functions to register a given cached buffer range.
  // If the buffer is already registered and cached, the pre-existing handle is
  // returned. Otherwise, it will check if all underlying memory segments of
  // this buffer are cached by user, and register the full segment range. If the
  // buffer contains any unknown memory segment, it will return nullptr and
  // require mapper to dynamically register it.
  // The registration will be freed when any assoicated segment is freed via
  // freeSegment().
  // input:
  //   - ptr: the pointer to the buffer to be registered
  //   - len: the length of the buffer
  //   - cudaDev: the cuda device id of the buffer
  //   - useDesc: the description of the buffer usage
  //   - logMetaData: the metadata of the communicator that registers the buffer
  //                  (logging purpose only)
  // output:
  //   - didRegister: whether regRangeCached registered regHdl, or just found it
  //   - regHdl: the registration handle
  // If acquireRef is true, the returned RegElem's scoped refcount is
  // incremented while holding the map lock (typically for a ScopedRegHdl, which
  // releases it via its destructor).
  commResult_t regRangeCached(
      const void* ptr,
      const size_t len,
      const int cudaDev,
      const std::string& useDesc,
      const struct CommLogData& logMetaData,
      const std::vector<bool>& backends,
      bool& didRegister,
      regcache::RegElem** regHdl,
      bool ncclManaged = false,
      bool acquireRef = false);

  // Thread-safe function to directly register a buffer range without consulting
  // the reusable segment/regElem cache. It registers every pinned physical
  // range covering [ptr, ptr + len) as one dynamic RegElem, does not allow
  // lookup reuse, and must be deregistered via deregRange().
  // input:
  //   - ptr: the pointer to the buffer to be registered
  //   - len: the length of the buffer
  //   - cudaDev: the cuda device id of the buffer
  // output:
  //   - regHdl: the direct registration handle
  commResult_t regRange(
      const void* ptr,
      const size_t len,
      int cudaDev,
      const std::vector<bool>& backends,
      regcache::RegElem** regHdl,
      bool ncclManaged = false,
      const struct CommLogData* logMetaData = nullptr,
      const std::string& useDesc = "dynamicRegMem");

  // Thread-safe functions to dynamically register a segment. Compatibility
  // wrapper around regRange() for existing callers. Always records the
  // registration under "dynamicRegMem" so dynamic and window registrations stay
  // separate in scuba.
  commResult_t regDynamic(
      const void* ptr,
      const size_t len,
      int cudaDev,
      const std::vector<bool>& backends,
      regcache::RegElem** regHdl,
      const struct CommLogData* logMetaData = nullptr);

  // Thread-safe function to deregister a direct range registration.
  // Unlike freeSegment(), it always deregisters since only the calling
  // communicator uses it.
  // input:
  //   - regHdl: the direct registration handle
  //   - releaseRef: if true, decrement the RegElem's refcount first and only
  //                 proceed to backend deregistration when it reaches 0.
  commResult_t deregRange(regcache::RegElem* regHdl, bool releaseRef = false);

  // Thread-safe function to deregister a dynamic registration. Compatibility
  // wrapper around deregRange() for existing callers.
  // input:
  //   - regHdl: the dynamic registration handle
  commResult_t deregDynamic(regcache::RegElem* regHdl);

  // Thread-safe functions to free a cached segment from the global cache and
  // deregister any associated registration. If the segment is already freed
  // (e.g. by a prior globalDeregister), this is a no-op. If the segment is
  // still in use by any communicator (refCount > 0), this call is a no-op
  // unless forceFree is true.
  // input:
  //   - segHdl: the handle of the cached segment
  //   - forceFree: if true, skip the refCount check and always free the
  //                segment. Used by globalDeregister when the underlying
  //                physical memory is about to be freed.
  // output:
  //   - freed: whether or not the segment is freed from the global cache
  //   - ncclManaged: whether or not the segment is managed by NCCL
  //   - regElems: a vector of all associated regElems that have been
  //               deregistered. It transfers the ownership of the regElems to
  //               the caller for releasing any remote registration.
  commResult_t freeSegment(
      void* segHdl,
      bool& freed,
      bool& ncclManaged,
      std::vector<std::unique_ptr<regcache::RegElem>>& regElems,
      bool forceFree = false);

  regcache::Segment* getSegment(void* segHdl);

  // Get vector of regElem associated with the specified segHdl.
  // If no regElem is associated, empty vector is returned
  std::vector<regcache::RegElem*> getRegElems(const void* segHdl) const;

  // Get deduplicated regElems associated with multiple segHdls.
  // Deduplication is needed because a single regElem can span multiple
  // segments.
  std::vector<regcache::RegElem*> getRegElems(
      const std::vector<void*>& segHdls) const;

  // Thread-safe functions to get a list of all cached segments in the global
  // cache.
  std::vector<void*> getSegments() const;

  // Look up all cached segments underlying a buffer range.
  // Uses pinRange to discover physical segments and returns their handles
  // along with associated regElems for remote release handling.
  // input:
  //   - buf: the buffer to look up
  //   - len: the length of the buffer
  //   - cudaDev: the cuda device id
  // output:
  //   - segHdls: vector of segment handles (one per physical segment)
  //   - regElems: vector of regElems associated with the segments
  commResult_t lookupSegmentsForBuffer(
      const void* buf,
      size_t len,
      int cudaDev,
      std::vector<void*>& segHdls,
      std::vector<regcache::RegElem*>& regElems);

  // Submit an async registration request to the global cache.
  // The registration will be handled by the asyncRegThread_.
  commResult_t asyncRegRange(
      const void* buf,
      const size_t len,
      const int cudaDev,
      const struct CommLogData& logMetaData,
      const std::vector<bool>& backends);

  // Thread-safe function to check if a given <ptr, len> range is registered.
  bool isRegistered(const void* ptr, const size_t len);

  // Thread-safe function to get the registration handle for a given <ptr, len>
  // range. Returns RegElem* as void* if registered; returns nullptr if not.
  // The returned handle can be used directly with mapper functions like
  // iput/isendCtrl.
  void* getRegHandle(const void* ptr, const size_t len);

  // Thread-safe function to search for a RegElem containing [ptr, ptr+len)
  // and return its ibRegElem. If the buffer is cached but not yet registered,
  // it will perform registration via regRangeCached(). Returns nullptr if not
  // cached.
  void* searchIbRegHandle(const void* ptr, size_t len, int deviceId = -1);

  // Thread-safe function to search for a RegElem containing [ptr, ptr+len)
  // and return its externalRegElem. If the buffer is cached but not yet
  // registered, it will perform registration via regRangeCached(). Returns
  // nullptr if not cached.
  void* searchExternalRegHandle(const void* ptr, size_t len, int deviceId = -1);

  // Thread-safe function to wait on all async registration requests to finish.
  // Used by test only.
  void waitAsyncRegComplete();

  // Register external memory registration/deregistration callbacks.
  // Threading: must be called at init time before any concurrent
  // regRangeCached/deregElem calls. Not safe for concurrent mutation.
  void registerExternalRegMemFn(
      regcache::ExternalRegMemFn regMem,
      regcache::ExternalDeregMemFn deregMem);

  void resetExternalRegMemFn();

  // Global API to register all cached segments. This is useful in lazy
  // registration mode where segments are cached but not immediately registered.
  // Instead of registering each segment individually via
  // searchRegHandle/regRangeCached, this function discovers all contiguous
  // memory regions among the cached segments and registers each region
  // separately.
  //
  // This function does NOT assume all cached segments form a single
  // contiguous region. It finds ALL contiguous regions (which may be
  // non-adjacent in memory) and creates one registration per region.
  //
  // The function:
  // 1. Retrieves all cached segments from the AVL tree
  // 2. Sorts segments by starting address
  // 3. Groups adjacent segments into contiguous regions (where one segment's
  //    end address equals the next segment's start address)
  // 4. Registers each contiguous region separately
  //
  // Example: If segments are at addresses [0x1000-0x2000], [0x2000-0x3000],
  // [0x5000-0x6000], this creates TWO registrations:
  //   - Region 1: [0x1000-0x3000] (first two segments are contiguous)
  //   - Region 2: [0x5000-0x6000] (third segment is isolated)
  //
  // This function does NOT check for existing registrations.
  // Callers should call deregAll() before regAll() if they want to avoid
  // duplicate registrations.
  //
  // Returns commSuccess if successful, or error code otherwise.
  static commResult_t regAll();

  // Deregister all non-dynamic registration elements from the global cache.
  // This removes all registrations that were created via regAll() or
  // regRangeCached(), but does NOT remove the cached segments themselves (they
  // can be re-registered later). Dynamic registrations (created via regRange()
  // or regDynamic()) are not affected.
  //
  // Returns commSuccess if successful, or error code otherwise.
  static commResult_t deregAll();

  // Profiler to record the events of the global cache.
  // Check its APIs for more details.
  folly::Synchronized<regcache::Profiler> profiler;

 private:
  friend class ScopedRegHdl;

  // Hold a reference to CtranIbSingleton to ensure proper destruction order.
  // By holding this shared_ptr, we guarantee CtranIbSingleton stays alive
  // as long as RegCache exists, preventing use-after-free during
  // deregistration.
  std::shared_ptr<CtranIbSingleton> ibSingleton_;

  // Global backends configuration, initialized from NCCL_CTRAN_BACKENDS in
  // init().
  std::vector<bool> globalBackends_;

  // AVL tree based segment cache
  folly::Synchronized<CtranAvlTree> segmentsAvl_;
  class RegElemMaps {
   public:
    // Map holds the ownership of all regElems, with reg handle (regElem raw
    // pointer) as key.
    std::unordered_map<regcache::RegElem*, std::unique_ptr<regcache::RegElem>>
        regHdlToElemMap;
    // Correlate segment with all associated regElems, used in freeSegment() to
    // deregister all regElems when segment is freed
    std::unordered_map<regcache::Segment*, std::vector<regcache::RegElem*>>
        segToRegElemsMap;
  };
  folly::Synchronized<RegElemMaps> regElemsMaps_;

  regcache::ExternalRegMemFn externalRegMemFn_;
  regcache::ExternalDeregMemFn externalDeregMemFn_;

  // Thread and cmd queue to handle async registration requests
  struct AsyncRegCmd {
    void* buf{nullptr};
    size_t len{0};
    int cudaDev{-1};
    bool stopFlag{false};
    struct CommLogData logMetaData;
    std::vector<bool> backends;
  };
  std::thread asyncRegThread_;
  folly::Synchronized<std::queue<AsyncRegCmd>, std::mutex> asyncRegQueue_;
  std::condition_variable asyncRegCv_;
  void asyncRegThreadFn(int cudaDev);

  // Thread-safe function to search given <ptr, len> range in regElem cache.
  // If acquireRef is true, atomically increments the found RegElem's refCount
  // on a lookup hit (safe under the shared read lock because refCount is
  // atomic).
  regcache::RegElem*
  searchRegElem(const void* ptr, const size_t len, bool acquireRef = false);

  // Helper function to perform backend registration for a set of segments.
  // Creates a RegElem, registers with backends, and updates regElemsMaps.
  // Caller must hold segmentsAvl lock (for thread safety with segment
  // pointers). This function acquires regElemsMaps_ lock internally.
  //
  // Returns commSuccess on success, or error code on failure.
  // On success, *regHdl is set to the created RegElem pointer.
  commResult_t registerSegmentsTogether(
      void* ptr,
      size_t len,
      int cudaDev,
      std::vector<regcache::Segment*>& segments,
      const std::vector<bool>& backends,
      bool ncclManaged,
      regcache::RegElem** regHdl,
      bool acquireRef = false);

  commResult_t deregElem(regcache::RegElem* regElem);

  // SW-only scoped-ref release used by ~ScopedRegHdl; graph-destroy safe. The
  // regId identifies the RegElem captured at acquire time so a reused heap
  // address is rejected. See .cc for details.
  void releaseScopedRegHdl(regcache::RegElem* regHdl, const uint64_t regId);
};

static inline void CHECK_VALID_REGCACHE(std::shared_ptr<RegCache> regCache) {
  FB_CHECKABORT(regCache != nullptr, "Failed to get RegCache instance");
}

} // namespace ctran

template <>
struct fmt::formatter<ctran::regcache::RegElemState> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(ctran::regcache::RegElemState status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};
