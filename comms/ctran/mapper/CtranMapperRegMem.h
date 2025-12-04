// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <folly/Synchronized.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "comms/ctran/mapper/CtranMapperTypes.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CtranAvlTree.h"
#include "comms/ctran/utils/CtranIpc.h"
#include "comms/ctran/utils/DevMemType.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/commSpecs.h"

using meta::comms::CommBackend;

struct CtranMapperSegmentRange {
  const void* buf{nullptr};
  const std::size_t len{0};
  DevMemType type{DevMemType::kCudaMalloc};

 public:
  CtranMapperSegmentRange(
      const void* buf,
      const std::size_t len,
      DevMemType type)
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
      std::vector<CtranMapperSegmentRange>& segRangs);
};

struct CtranMapperSegmentStateMnger {
  // The number of communicators having cached the segment.
  // The segment must be removed and any associated RegElem must be deregistered
  // when the refCount is 0
  int64_t refCount{1};
};

struct CtranMapperRegElem;

struct CtranMapperSegment {
  CtranMapperSegmentRange range;
  const int cudaDev;
  bool ncclManaged{false};

  folly::Synchronized<CtranMapperSegmentStateMnger> stateMnger;

 public:
  CtranMapperSegment(
      CtranMapperSegmentRange range,
      const int cudaDev,
      const bool ncclManaged)
      : range(range), cudaDev(cudaDev), ncclManaged(ncclManaged) {};

  DevMemType getType() const {
    return range.type;
  }

  friend struct CtranMapperRegElem;
  friend class CtranMapperRegCache;

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

enum CtranMapperRegElemState {
  REGISTERED,
  DEREGISTERED,
};

template <>
struct fmt::formatter<CtranMapperRegElemState> : fmt::formatter<int> {
  template <typename FormatContext>
  auto format(CtranMapperRegElemState status, FormatContext& ctx) const {
    return fmt::formatter<int>::format(static_cast<int>(status), ctx);
  }
};

struct CtranMapperRegElemStateMnger {
  CtranMapperRegElemState state{CtranMapperRegElemState::REGISTERED};
};

struct CtranMapperRegElem {
  const void* buf{nullptr};
  const std::size_t len{0};
  void* ibRegElem{nullptr};
  void* nvlRegElem{nullptr};
  void* tcpRegElem{nullptr};

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
  folly::Synchronized<CtranMapperRegElemStateMnger> stateMnger;

 public:
  CtranMapperRegElem(
      const void* buf,
      std::size_t len,
      const int cudaDev,
      bool isDynamic,
      DevMemType type)
      : buf(buf),
        len(len),
        cudaDev_(cudaDev),
        isDynamic_(isDynamic),
        type_(type) {};
  CtranMapperRegElem(
      const void* buf,
      std::size_t len,
      const int cudaDev,
      std::vector<CtranMapperSegment*>& segments)
      // Explicitly copy segments to avoid changing passed-in vector which
      // may be used after the call
      : buf(buf), len(len), cudaDev_(cudaDev), segments_(segments) {
    type_ = segments.at(0)->getType();
  };

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

  friend class CtranMapperRegCache;

 protected:
  int cudaDev_{-1};
  bool isDynamic_{false};
  // Number of times the registration is reused by collective
  size_t lookupHit_{0};
  std::vector<CtranMapperSegment*> segments_;
  DevMemType type_{DevMemType::kCudaMalloc};

  // Thread-safe function to register the segment.
  // It internally locks the stateMnger to ensure thread-safe access.
  // The segment should be registered only once by the first thread and reused
  // by all later calls before deregistration.
  commResult_t doRegister(const std::vector<bool>& backends);

  // Thread-safe function to deregister the segment.
  // It internally locks the stateMnger to ensure thread-safe access.
  commResult_t doDeregister();

  // Thread-unsafe function to print internal fields.
  // It is only used for debugging purpose. Caller should ensure thread-safety
  // by locking the stateMnger.
  std::string toString(const CtranMapperRegElemState state) const {
    std::stringstream ss;
    ss << "buf: " << buf << ", len: " << len << ", state: "
       << (state == CtranMapperRegElemState::REGISTERED ? "REGISTERED"
                                                        : "DEREGISTERED")
       << ", isDynamic: " << isDynamic_ << ", lookupHit: " << lookupHit_;
    if (state == CtranMapperRegElemState::REGISTERED) {
      ss << ", ibRegElem: " << ibRegElem << ", nvlRegElem: " << nvlRegElem
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

enum CtranRegCacheEventType {
  kCacheSegEvent,
  kFreeSegEvent,
  kRegMemEvent,
  kDeregMemEvent,
  kDynamicRegMemEvent,
  kAsyncRegMemEvent,
};

struct CtranMapperRegCacheSnapshot {
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

class CtranMapperRegCacheProfiler {
  std::unordered_map<CtranRegCacheEventType, double> latencyMap;
  std::unordered_map<CtranRegCacheEventType, uint32_t> totalCountMap;
  std::unordered_map<CtranRegCacheEventType, uint32_t> currentCountMap;

 public:
  CtranMapperRegCacheProfiler();

  // Record an event with latency
  void record(CtranRegCacheEventType type, CtranMapperTimer& dur);

  // Record an event without latency
  void record(CtranRegCacheEventType type);

  // Report the snapshot of the profiler via NCCL log
  void reportSnapshot() const;

  // Get an snapshot copy of the profiler
  CtranMapperRegCacheSnapshot getSnapshot() const;

  // Reset the profiler records; used for testing only
  void reset();
};

/**
 * Singleton class to hold the IB network resources that are reused by all
 * communicators in the lifetime of program.
 */
class CtranMapperRegCache {
 public:
  CtranMapperRegCache();
  ~CtranMapperRegCache();

  static std::shared_ptr<CtranMapperRegCache> getInstance();

  // Implement actual init/destroy logic here and called by
  // constructor/destructor. This allows test to manually trigger if needed.
  void init();
  commResult_t destroy();

  // Thread-safe functions to cache a buffer range into the global cache.
  // input:
  //   - buf: the buffer to be cached
  //   - len: the length of the buffer
  //   - cudaDev: the cuda device id of the buffer
  //   - ncclManaged: whether the buffer is managed by NCCL
  //   - commHash: the commHash of the communicator that caches the buffer
  //               (logging purpose only, since commHash may not be 100%
  //               unique).
  // output:
  //   - segment: the cached segment
  //   - segHdl: the handle of the cached segment
  commResult_t cacheSegment(
      const void* buf,
      const std::size_t len,
      const int cudaDev,
      const bool ncclManaged,
      uint64_t commHash,
      CtranMapperSegment** segment,
      void** segHdl);

  // Thread-safe functions to register a given buffer range.
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
  //   - didRegister: whether regRange registered regHdl, or just found it
  //   - regHdl: the registration handle
  commResult_t regRange(
      const void* ptr,
      const size_t len,
      const int cudaDev,
      const std::string& useDesc,
      const struct CommLogData& logMetaData,
      const std::vector<bool>& backends,
      bool& didRegister,
      CtranMapperRegElem** regHdl);

  // Thread-safe functions to dynamically register a segment.
  // It is similar to regRange() but is not associated with a cached segment nor
  // allow reuse. It must be deregistered via deregDynamic().
  // input:
  //   - ptr: the pointer to the buffer to be registered
  //   - len: the length of the buffer
  //   - cudaDev: the cuda device id of the buffer
  // output:
  //   - regHdl: the dynamic registration handle
  commResult_t regDynamic(
      const void* ptr,
      const size_t len,
      int cudaDev,
      const std::vector<bool>& backends,
      CtranMapperRegElem** regHdl);

  // Thread-safe functions to deregister a dynamic registration.
  // Unlike freeSegment(), it always deregisters since only the calling
  // communicator uses it.
  // input:
  //   - regHdl: the dynamic registration handle
  commResult_t deregDynamic(CtranMapperRegElem* regHdl);

  // Thread-safe functions to free a cached segment from the global cache and
  // deregister any associated registration. If the segment is still in use by
  // any communicator, this call is a no-op.
  // input:
  //   - segHdl: the handle of the cached segment
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
      std::vector<std::unique_ptr<CtranMapperRegElem>>& regElems);

  CtranMapperSegment* getSegment(void* segHdl);

  // Get vector of regElem associated with the specified segHdl.
  // If no regElem is associated, empty vector is returned
  std::vector<CtranMapperRegElem*> getRegElems(const void* segHdl) const;

  // Thread-safe functions to get a list of all cached segments in the global
  // cache.
  std::vector<void*> getSegments() const;

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

  // Thread-safe function to wait on all async registration requests to finish.
  // Used by test only.
  void waitAsyncRegComplete();

  // Profiler to record the events of the global cache.
  // Check its APIs for more details.
  folly::Synchronized<CtranMapperRegCacheProfiler> profiler;

 private:
  // AVL tree based segment cache
  folly::Synchronized<CtranAvlTree> segmentsAvl_;
  class RegElemMaps {
   public:
    // Map holds the ownership of all regElems, with reg handle (regElem raw
    // pointer) as key.
    std::unordered_map<CtranMapperRegElem*, std::unique_ptr<CtranMapperRegElem>>
        regHdlToElemMap;
    // Correlate segment with all associated regElems, used in freeSegment() to
    // deregister all regElems when segment is freed
    std::unordered_map<CtranMapperSegment*, std::vector<CtranMapperRegElem*>>
        segToRegElemsMap;
  };
  folly::Synchronized<RegElemMaps> regElemsMaps_;

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
  CtranMapperRegElem* searchRegElem(const void* ptr, const size_t len);

  commResult_t deregElem(CtranMapperRegElem* regElem);
};

static inline void CHECK_VALID_REGCACHE(
    std::shared_ptr<CtranMapperRegCache> regCache) {
  FB_CHECKABORT(
      regCache != nullptr, "Failed to get CtranMapperRegCache instance");
}
