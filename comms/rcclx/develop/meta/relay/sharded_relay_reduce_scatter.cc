/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_reduce_scatter.h"
#include "comm.h"
#include "sharded_relay_allreduce_kernels.h"
#include "sharded_relay_graph_scratch.h"
#include "sharded_relay_lp.h"
#include "sharded_relay_lp_arena.h"
#include "sharded_relay_lp_kernels.h"
#include "sharded_relay_oneshot.h"
#include "sharded_relay_route.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

// GPU memory access alignment in elements for chunk size rounding.
// Distinct from the CPU CACHE_LINE_SIZE (64 bytes) defined in comm.h
// which is used for struct padding.
static constexpr size_t CHUNK_ALIGN_ELEMENTS =
    rcclx::relay::kRelayChunkAlignElements;

// Infra below (ScratchBufferCache, rank-config builder, DISPATCH macros) is a
// deliberate copy of the file-local helpers in sharded_relay_allreduce.cc.
// They are file-local (static / anonymous namespace) there, so they cannot be
// linked across translation units; the reduce-scatter TU re-declares its own
// copies in an anonymous namespace to keep them internal and ODR-safe. The
// GPU kernels themselves are NOT duplicated — they are reused via
// sharded_relay_allreduce_kernels.h (dtype-generic, collective-agnostic).
namespace {

/**
 * Scratch Buffer Cache Singleton
 *
 * Amortizes cudaMalloc/cudaFree costs by caching and reusing scratch buffers.
 * Thread-safe, with one buffer per (device, stream, key). See the allreduce
 * copy for a full description; this is an independent cache scoped to
 * reduce-scatter.
 */
class ScratchBufferCache {
 public:
  static ScratchBufferCache& getInstance() {
    static ScratchBufferCache instance;
    return instance;
  }

  void* get(int key, size_t requiredBytes, cudaStream_t stream) {
    if (requiredBytes == 0) {
      return nullptr;
    }

    // A capturing stream must not use this cache. hipMallocAsync would record
    // an allocation node whose address is only valid while the graph runs, and
    // a later growth would hipFreeAsync a pointer this graph has already baked
    // in. Captures get a graph-scoped buffer instead; see
    // sharded_relay_graph_scratch.h.
    //
    // Ahead of the lock on purpose: this is a HIP runtime call, and there is no
    // reason to hold a process-wide mutex across it while relay collectives run
    // concurrently on other streams. Measured either way it is in the noise, so
    // this is hygiene rather than a fix for anything.
    struct ncclCudaGraph graph;
    if (ncclCudaGetCapturingGraph(&graph, stream) != ncclSuccess) {
      return nullptr;
    }
    if (ncclCudaGraphValid(graph)) {
      return rcclx::relay::graphScratchGet(
          this, key, requiredBytes, stream, graph);
    }

    int device;
    cudaGetDevice(&device);

    std::lock_guard<std::mutex> lock(mutex_);

    // Keyed by (device, stream, key). The stream is part of the key because two
    // relay collectives can run concurrently on one device on different streams
    // (independent communicators do exactly this): sharing one staging buffer
    // between them corrupts both. It also makes the stream-ordered free below
    // safe -- an entry is only ever read or written by the stream that owns it.
    auto& entry = buffers_[std::make_tuple(
        device, static_cast<const void*>(stream), key)];

    if (entry.buffer == nullptr || entry.size < requiredBytes) {
      if (entry.buffer != nullptr) {
        cudaFreeAsync(entry.buffer, stream);
      }

      size_t allocSize = requiredBytes;
      if (allocSize >= 1024 * 1024) {
        allocSize =
            ((requiredBytes + 64 * 1024 * 1024 - 1) / (64 * 1024 * 1024)) *
            (64 * 1024 * 1024);
      }

      cudaError_t err = cudaMallocAsync(&entry.buffer, allocSize, stream);
      if (err != cudaSuccess) {
        entry.buffer = nullptr;
        entry.size = 0;
        return nullptr;
      }
      entry.size = allocSize;
    }

    return entry.buffer;
  }

  void clear(cudaStream_t stream = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Each buffer is freed on the stream that owns it (from the key), not on
    // the caller's stream: a stream-ordered free on an unrelated stream would
    // not be ordered against the owner's pending work.
    (void)stream;
    for (auto& pair : buffers_) {
      if (pair.second.buffer != nullptr) {
        cudaFreeAsync(
            pair.second.buffer,
            static_cast<cudaStream_t>(
                const_cast<void*>(std::get<1>(pair.first))));
        pair.second.buffer = nullptr;
        pair.second.size = 0;
      }
    }
    buffers_.clear();
  }

  ScratchBufferCache(const ScratchBufferCache&) = delete;
  ScratchBufferCache& operator=(const ScratchBufferCache&) = delete;

 private:
  ScratchBufferCache() = default;
  ~ScratchBufferCache() = default;

  struct BufferEntry {
    void* buffer = nullptr;
    size_t size = 0;
  };

  std::mutex mutex_;
  // (device, stream, group) -> grow-only staging buffer.
  std::map<std::tuple<int, const void*, int>, BufferEntry> buffers_;
};

/**
 * Side-Stream Cache Singleton
 *
 * The pipelined reduce-scatter's owner reduce is pure serialized tail: it runs
 * only once the last transfer has landed, and at 1 GB with 2 active ranks it is
 * 0.61 ms of a 2.64 ms call (measured by skipping it). Splitting it per
 * pipeline stage on the caller's stream buys nothing, because stream order is
 * total -- the T+1 smaller reduces just serialize in the same places. So the
 * per-stage reduces go on a side stream, gated by an event recorded at each
 * group boundary, and the caller's stream joins once at the end. Only that
 * closing join creates a side -> caller dependency, so group k+1 never waits on
 * the reduce of stage k.
 *
 * One stream and one event pool per (device, caller stream, graph id), for the
 * same reason ScratchBufferCache keys on the stream: two relay collectives can
 * run concurrently on one device on different streams. The graph id is part of
 * the key because a stream can only belong to one capture at a time.
 */
class ReduceOverlapCache {
 public:
  // One event per pipeline group: the depth is at most kRelayMaxPipelineTiles
  // and a depth-T pipeline runs T+1 groups. Callers MUST check T+1 against this
  // before indexing stageDone -- a deeper pipeline than the pool cannot be
  // overlapped, and reading past the array is a segfault, not a slow path.
  static constexpr int kStageEvents = rcclx::relay::kRelayMaxPipelineTiles + 1;

  struct Handle {
    cudaStream_t stream{};
    cudaEvent_t stageDone[kStageEvents]{};
    cudaEvent_t allDone{};
  };

  static ReduceOverlapCache& getInstance() {
    static ReduceOverlapCache instance;
    return instance;
  }

  // Returns nullptr if the resources could not be created, in which case the
  // caller must fall back to one reduce on its own stream.
  //
  // A capture gets its own stream and event pool, keyed on the graph id. The
  // handshake here (record on the caller's stream, wait on the side stream,
  // rejoin at the end) is exactly the fork/join shape capture understands, so
  // the side stream is pulled into the graph by the event dependency and its
  // reduces become a parallel branch. What is NOT safe is sharing one side
  // stream between a capture and anything else, since a stream can only belong
  // to one capture at a time -- hence one per graph.
  const Handle* get(cudaStream_t callerStream, struct ncclCudaGraph graph) {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) {
      return nullptr;
    }

    // Drains outside mutex_, so a blocking stream/event destroy cannot stall
    // every other relay collective waiting for this cache.
    reclaimDead();

    const Key key{
        device, static_cast<const void*>(callerStream), graph.graphId};

    Handle orphaned{};
    bool haveOrphan = false;
    const Handle* result = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      Entry& entry = handles_[key];
      if (!entry.tried) {
        entry.tried = true;
        {
          // create() is stream/event creation, which is exactly the kind of
          // potentially-unsafe runtime call that invalidates an in-progress
          // capture under the default thread-local mode. get() is now called
          // WHILE the caller's stream is capturing, so the exchange is
          // required; sharded_relay_graph_scratch.cc does the same around its
          // allocation.
          RelaxedCaptureMode relaxed;
          entry.valid = create(&entry.handle);
        }
        if (ncclCudaGraphValid(graph)) {
          // The stream and events only carry capture topology; the graph holds
          // its own copy of the resulting nodes and does not touch either at
          // replay. So they are dead weight the moment the graph is gone, and
          // tying them to it keeps a process that captures repeatedly from
          // accumulating a stream and 10 events per graph forever.
          //
          // Registered even when create() FAILED: the destructor is what erases
          // this entry, so without it a failed attempt would leave a node per
          // graph id behind for the life of the process. destroy() on an
          // all-null handle is a no-op, so reclaim handles both cases.
          auto* tag = new unsigned long long(graph.graphId);
          if (ncclCudaGraphAddDestructor(graph, graphDiedCallback, tag) !=
              ncclSuccess) {
            delete tag;
            // Nothing will ever reclaim this entry now, so do not leave it (or
            // its stream and 10 events) behind. Destroyed after the lock drops.
            orphaned = entry.handle;
            haveOrphan = true;
            handles_.erase(key);
          }
        }
      }
      if (!haveOrphan) {
        result = entry.valid ? &entry.handle : nullptr;
      }
    }

    if (haveOrphan) {
      RelaxedCaptureMode relaxed;
      destroy(&orphaned);
    }
    return result;
  }

  ReduceOverlapCache(const ReduceOverlapCache&) = delete;
  ReduceOverlapCache& operator=(const ReduceOverlapCache&) = delete;

 private:
  ReduceOverlapCache() = default;
  ~ReduceOverlapCache() = default;

  using Key = std::tuple<int, const void*, unsigned long long>;

  // Puts the calling thread in relaxed capture mode for its lifetime. Under the
  // default mode, a potentially-unsafe runtime call made on a thread with a
  // capture in progress invalidates that capture; relaxed mode is what makes
  // stream/event creation and destruction legal here.
  class RelaxedCaptureMode {
   public:
    RelaxedCaptureMode() {
      ok_ = cudaThreadExchangeStreamCaptureMode(&mode_) == cudaSuccess;
    }
    ~RelaxedCaptureMode() {
      if (ok_) {
        (void)cudaThreadExchangeStreamCaptureMode(&mode_);
      }
    }
    RelaxedCaptureMode(const RelaxedCaptureMode&) = delete;
    RelaxedCaptureMode& operator=(const RelaxedCaptureMode&) = delete;

   private:
    cudaStreamCaptureMode mode_{cudaStreamCaptureModeRelaxed};
    bool ok_{true};
  };

  // cudaStreamNonBlocking so the side stream does not implicitly synchronize
  // with the legacy default stream; all ordering here is explicit via events.
  // Timing is disabled on the events because nothing queries their elapsed
  // time, which keeps the record cheap.
  static bool create(Handle* h) {
    if (cudaStreamCreateWithFlags(&h->stream, cudaStreamNonBlocking) !=
        cudaSuccess) {
      h->stream = nullptr;
      return false;
    }
    for (int i = 0; i < kStageEvents; i++) {
      if (cudaEventCreateWithFlags(&h->stageDone[i], cudaEventDisableTiming) !=
          cudaSuccess) {
        destroy(h);
        return false;
      }
    }
    if (cudaEventCreateWithFlags(&h->allDone, cudaEventDisableTiming) !=
        cudaSuccess) {
      destroy(h);
      return false;
    }
    return true;
  }

  static void destroy(Handle* h) {
    for (int i = 0; i < kStageEvents; i++) {
      if (h->stageDone[i] != nullptr) {
        cudaEventDestroy(h->stageDone[i]);
        h->stageDone[i] = nullptr;
      }
    }
    if (h->allDone != nullptr) {
      cudaEventDestroy(h->allDone);
      h->allDone = nullptr;
    }
    if (h->stream != nullptr) {
      cudaStreamDestroy(h->stream);
      h->stream = nullptr;
    }
  }

  struct Entry {
    Handle handle;
    bool tried = false;
    bool valid = false;
  };

  // Runs when a graph is destroyed, possibly on a HIP-internal thread, so it
  // records the id and nothing more -- destroying a stream or an event from
  // here would be a HIP call in an unspecified context. reclaimDeadLocked does
  // the work on the next get(), on a user thread.
  static void graphDiedCallback(void* arg) {
    auto* graphId = static_cast<unsigned long long*>(arg);
    {
      std::lock_guard<std::mutex> lock(deadMutex());
      deadGraphs().push_back(*graphId);
    }
    delete graphId;
  }

  static std::mutex& deadMutex() {
    static std::mutex m;
    return m;
  }

  static std::vector<unsigned long long>& deadGraphs() {
    static std::vector<unsigned long long> v;
    return v;
  }

  // Erases the entries of every graph whose destructor has fired, then destroys
  // their streams and events OUTSIDE mutex_. Stream and event destruction can
  // block on in-flight work, and this runs at the start of every get(), so
  // doing it under the lock would serialize every other relay collective on the
  // device. sharded_relay_graph_scratch.cc's reclaimDeadGraphs() frees outside
  // its table lock for the same reason.
  void reclaimDead() {
    std::vector<unsigned long long> dead;
    {
      std::lock_guard<std::mutex> lock(deadMutex());
      if (deadGraphs().empty()) {
        return;
      }
      dead.swap(deadGraphs());
    }

    std::vector<Handle> stale;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = handles_.begin(); it != handles_.end();) {
        const unsigned long long id = std::get<2>(it->first);
        if (std::find(dead.begin(), dead.end(), id) != dead.end()) {
          stale.push_back(it->second.handle);
          it = handles_.erase(it);
        } else {
          ++it;
        }
      }
    }

    if (stale.empty()) {
      return;
    }
    RelaxedCaptureMode relaxed;
    for (Handle& h : stale) {
      destroy(&h);
    }
  }

  std::mutex mutex_;
  // (device, caller stream, graph id) -> side stream and its event pool. The
  // graph id is ULLONG_MAX for uncaptured calls, so they all share one entry
  // per stream exactly as before.
  std::map<Key, Entry> handles_;
};

// Maximum number of helper ranks supported per group.
constexpr int SHARDED_RELAY_MAX_HELPERS = 8;

// Maximum number of active ranks per group. The recursive-halving relay
// schedule (round-r partner = myActiveIndex XOR round) requires nActiveRanks to
// be a power of two; supported values are 2 and 4 (on an 8-GPU node this leaves
// 6 or 4 helpers respectively).
constexpr int SHARDED_RELAY_MAX_ACTIVE = 8;

// Returns true if v is a power of two (v >= 1).
inline bool isPowerOfTwo(int v) {
  return v > 0 && (v & (v - 1)) == 0;
}

// Reverse the low `bits` bits of x. The recursive-halving reduce-scatter leaves
// active rank mi owning the segment at bit-reversed position bitReverse(mi);
// the reduce-scatter gather places block[j] at bit-reversed position so owner j
// ends up holding the reduced block[j].
inline int bitReverse(int x, int bits) {
  int r = 0;
  for (int b = 0; b < bits; b++) {
    r = (r << 1) | ((x >> b) & 1);
  }
  return r;
}

/**
 * Rank Configuration for Sharded Relay Reduce-Scatter
 *
 * Holds parsed active and helper rank information for a single group.
 * Supports a power-of-two number of active ranks per group (2 or 4).
 */
struct ShardedRelayRankConfig {
  int activeRanks[SHARDED_RELAY_MAX_ACTIVE]; // Active rank IDs (power of two)
  int nActiveRanks; // Number of active ranks (2 or 4)
  int helperRanks[SHARDED_RELAY_MAX_HELPERS]; // Helper rank IDs
  int numHelpers; // Number of helper ranks
  bool isActiveRank; // Is current rank active?
  int myActiveIndex; // Index in activeRanks array (-1 if helper)
  int myHelperIndex; // Index in helperRanks array (-1 if active)
};

/**
 * Build rank configuration from provided active ranks array.
 *
 * Requires a power-of-two active-rank count in [2, SHARDED_RELAY_MAX_ACTIVE];
 * the XOR round schedule of the A>2 recursive path depends on it.
 */
bool buildShardedRelayRankConfig(
    int nRanks,
    int rank,
    const int* activeRanksInput,
    int nActiveRanksInput,
    ShardedRelayRankConfig& config) {
  config.nActiveRanks = 0;
  config.numHelpers = 0;
  config.isActiveRank = false;
  config.myActiveIndex = -1;
  config.myHelperIndex = -1;

  // Validate input - require a power-of-two active-rank count in
  // [2, SHARDED_RELAY_MAX_ACTIVE]. The XOR round schedule depends on it.
  if (activeRanksInput == nullptr || nActiveRanksInput < 2 ||
      nActiveRanksInput > SHARDED_RELAY_MAX_ACTIVE ||
      !isPowerOfTwo(nActiveRanksInput)) {
    return false;
  }

  // Copy active ranks and validate
  for (int i = 0; i < nActiveRanksInput; i++) {
    int rankId = activeRanksInput[i];
    if (rankId >= 0 && rankId < nRanks) {
      config.activeRanks[config.nActiveRanks++] = rankId;
    }
  }

  // Validate: need exactly nActiveRanksInput valid active ranks
  if (config.nActiveRanks != nActiveRanksInput) {
    return false;
  }

  // Build list of helper ranks (all ranks NOT in activeRanks).
  for (int r = 0; r < nRanks; r++) {
    bool isActive = false;
    for (int a = 0; a < config.nActiveRanks; a++) {
      if (r == config.activeRanks[a]) {
        isActive = true;
        break;
      }
    }
    if (!isActive) {
      if (config.numHelpers >= SHARDED_RELAY_MAX_HELPERS) {
        return false;
      }
      config.helperRanks[config.numHelpers++] = r;
    }
  }

  // Validate: need at least 1 helper
  if (config.numHelpers < 1) {
    return false;
  }

  // Determine if this rank is active
  for (int a = 0; a < config.nActiveRanks; a++) {
    if (rank == config.activeRanks[a]) {
      config.isActiveRank = true;
      config.myActiveIndex = a;
      break;
    }
  }

  // For helpers, determine which chunk index this rank handles
  if (!config.isActiveRank) {
    for (int i = 0; i < config.numHelpers; i++) {
      if (config.helperRanks[i] == rank) {
        config.myHelperIndex = i;
        break;
      }
    }
  }
  return true;
}

} // namespace

// Host-side dispatch macros for the reused generic kernels. These mirror the
// file-local DISPATCH_* macros in sharded_relay_allreduce.cc (the kernels they
// launch are declared in sharded_relay_allreduce_kernels.h).

#define LAUNCH_INCREMENTAL_ADD_KERNEL(TYPE, output, input, count, stream) \
  launchIncrementalAddKernel<TYPE>(output, input, count, stream)

#define DISPATCH_INCREMENTAL_ADD(datatype, output, input, count, stream)       \
  do {                                                                         \
    switch (datatype) {                                                        \
      case ncclInt8:                                                           \
        LAUNCH_INCREMENTAL_ADD_KERNEL(int8_t, output, input, count, stream);   \
        break;                                                                 \
      case ncclUint8:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(uint8_t, output, input, count, stream);  \
        break;                                                                 \
      case ncclInt32:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(int32_t, output, input, count, stream);  \
        break;                                                                 \
      case ncclUint32:                                                         \
        LAUNCH_INCREMENTAL_ADD_KERNEL(uint32_t, output, input, count, stream); \
        break;                                                                 \
      case ncclInt64:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(int64_t, output, input, count, stream);  \
        break;                                                                 \
      case ncclUint64:                                                         \
        LAUNCH_INCREMENTAL_ADD_KERNEL(uint64_t, output, input, count, stream); \
        break;                                                                 \
      case ncclFloat16:                                                        \
        LAUNCH_INCREMENTAL_ADD_KERNEL(__half, output, input, count, stream);   \
        break;                                                                 \
      case ncclFloat:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(float, output, input, count, stream);    \
        break;                                                                 \
      case ncclDouble:                                                         \
        LAUNCH_INCREMENTAL_ADD_KERNEL(double, output, input, count, stream);   \
        break;                                                                 \
      case ncclBfloat16:                                                       \
        LAUNCH_INCREMENTAL_ADD_KERNEL(                                         \
            __nv_bfloat16, output, input, count, stream);                      \
        break;                                                                 \
      default:                                                                 \
        break;                                                                 \
    }                                                                          \
  } while (0)

#define LAUNCH_SCALE_KERNEL(TYPE, data, count, divisor, stream) \
  launchScaleKernel<TYPE>(data, count, divisor, stream)

#define DISPATCH_SCALE(datatype, data, count, divisor, stream)            \
  do {                                                                    \
    switch (datatype) {                                                   \
      case ncclInt8:                                                      \
        LAUNCH_SCALE_KERNEL(int8_t, data, count, divisor, stream);        \
        break;                                                            \
      case ncclUint8:                                                     \
        LAUNCH_SCALE_KERNEL(uint8_t, data, count, divisor, stream);       \
        break;                                                            \
      case ncclInt32:                                                     \
        LAUNCH_SCALE_KERNEL(int32_t, data, count, divisor, stream);       \
        break;                                                            \
      case ncclUint32:                                                    \
        LAUNCH_SCALE_KERNEL(uint32_t, data, count, divisor, stream);      \
        break;                                                            \
      case ncclInt64:                                                     \
        LAUNCH_SCALE_KERNEL(int64_t, data, count, divisor, stream);       \
        break;                                                            \
      case ncclUint64:                                                    \
        LAUNCH_SCALE_KERNEL(uint64_t, data, count, divisor, stream);      \
        break;                                                            \
      case ncclFloat16:                                                   \
        LAUNCH_SCALE_KERNEL(__half, data, count, divisor, stream);        \
        break;                                                            \
      case ncclFloat:                                                     \
        LAUNCH_SCALE_KERNEL(float, data, count, divisor, stream);         \
        break;                                                            \
      case ncclDouble:                                                    \
        LAUNCH_SCALE_KERNEL(double, data, count, divisor, stream);        \
        break;                                                            \
      case ncclBfloat16:                                                  \
        LAUNCH_SCALE_KERNEL(__nv_bfloat16, data, count, divisor, stream); \
        break;                                                            \
      default:                                                            \
        break;                                                            \
    }                                                                     \
  } while (0)

#define LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL( \
    TYPE, output, input, count, divisor, stream) \
  launchIncrementalAddAndScaleKernel<TYPE>(      \
      output, input, count, divisor, stream)

#define DISPATCH_INCREMENTAL_ADD_AND_SCALE(                        \
    datatype, output, input, count, divisor, stream)               \
  do {                                                             \
    switch (datatype) {                                            \
      case ncclInt8:                                               \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            int8_t, output, input, count, divisor, stream);        \
        break;                                                     \
      case ncclUint8:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            uint8_t, output, input, count, divisor, stream);       \
        break;                                                     \
      case ncclInt32:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            int32_t, output, input, count, divisor, stream);       \
        break;                                                     \
      case ncclUint32:                                             \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            uint32_t, output, input, count, divisor, stream);      \
        break;                                                     \
      case ncclInt64:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            int64_t, output, input, count, divisor, stream);       \
        break;                                                     \
      case ncclUint64:                                             \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            uint64_t, output, input, count, divisor, stream);      \
        break;                                                     \
      case ncclFloat16:                                            \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            __half, output, input, count, divisor, stream);        \
        break;                                                     \
      case ncclFloat:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            float, output, input, count, divisor, stream);         \
        break;                                                     \
      case ncclDouble:                                             \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            double, output, input, count, divisor, stream);        \
        break;                                                     \
      case ncclBfloat16:                                           \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            __nv_bfloat16, output, input, count, divisor, stream); \
        break;                                                     \
      default:                                                     \
        break;                                                     \
    }                                                              \
  } while (0)

#define LAUNCH_FUSED_REDUCE_KERNEL(                       \
    TYPE, output, inputA, inputB, count, divisor, stream) \
  launchFusedReduceKernel<TYPE>(output, inputA, inputB, count, divisor, stream)

#define DISPATCH_FUSED_REDUCE(                                              \
    datatype, output, inputA, inputB, count, divisor, stream)               \
  do {                                                                      \
    switch (datatype) {                                                     \
      case ncclInt8:                                                        \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            int8_t, output, inputA, inputB, count, divisor, stream);        \
        break;                                                              \
      case ncclUint8:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            uint8_t, output, inputA, inputB, count, divisor, stream);       \
        break;                                                              \
      case ncclInt32:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            int32_t, output, inputA, inputB, count, divisor, stream);       \
        break;                                                              \
      case ncclUint32:                                                      \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            uint32_t, output, inputA, inputB, count, divisor, stream);      \
        break;                                                              \
      case ncclInt64:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            int64_t, output, inputA, inputB, count, divisor, stream);       \
        break;                                                              \
      case ncclUint64:                                                      \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            uint64_t, output, inputA, inputB, count, divisor, stream);      \
        break;                                                              \
      case ncclFloat16:                                                     \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            __half, output, inputA, inputB, count, divisor, stream);        \
        break;                                                              \
      case ncclFloat:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            float, output, inputA, inputB, count, divisor, stream);         \
        break;                                                              \
      case ncclDouble:                                                      \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            double, output, inputA, inputB, count, divisor, stream);        \
        break;                                                              \
      case ncclBfloat16:                                                    \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            __nv_bfloat16, output, inputA, inputB, count, divisor, stream); \
        break;                                                              \
      default:                                                              \
        break;                                                              \
    }                                                                       \
  } while (0)

#define LAUNCH_MULTI_REDUCE_KERNEL(                           \
    TYPE, dst, contribs, numContribs, count, divisor, stream) \
  launchMultiReduceKernel<TYPE>(                              \
      dst, contribs, numContribs, count, divisor, stream)

// Fused multi-input reduce: dst = (dst + sum of `numContribs` contiguous
// contribution blocks) [/ divisor], in one launch. Replaces a loop of
// per-contribution incremental adds plus a trailing scale.
#define DISPATCH_MULTI_REDUCE(                                             \
    datatype, dst, contribs, numContribs, count, divisor, stream)          \
  do {                                                                     \
    switch (datatype) {                                                    \
      case ncclInt8:                                                       \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            int8_t, dst, contribs, numContribs, count, divisor, stream);   \
        break;                                                             \
      case ncclUint8:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            uint8_t, dst, contribs, numContribs, count, divisor, stream);  \
        break;                                                             \
      case ncclInt32:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            int32_t, dst, contribs, numContribs, count, divisor, stream);  \
        break;                                                             \
      case ncclUint32:                                                     \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            uint32_t, dst, contribs, numContribs, count, divisor, stream); \
        break;                                                             \
      case ncclInt64:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            int64_t, dst, contribs, numContribs, count, divisor, stream);  \
        break;                                                             \
      case ncclUint64:                                                     \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            uint64_t, dst, contribs, numContribs, count, divisor, stream); \
        break;                                                             \
      case ncclFloat16:                                                    \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            __half, dst, contribs, numContribs, count, divisor, stream);   \
        break;                                                             \
      case ncclFloat:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            float, dst, contribs, numContribs, count, divisor, stream);    \
        break;                                                             \
      case ncclDouble:                                                     \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            double, dst, contribs, numContribs, count, divisor, stream);   \
        break;                                                             \
      case ncclBfloat16:                                                   \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            __nv_bfloat16,                                                 \
            dst,                                                           \
            contribs,                                                      \
            numContribs,                                                   \
            count,                                                         \
            divisor,                                                       \
            stream);                                                       \
        break;                                                             \
      default:                                                             \
        break;                                                             \
    }                                                                      \
  } while (0)

// Dtype dispatch for the one-shot reduce-scatter. Deliberately narrower than
// the other macros: only the types the small-message paths actually see are
// worth an instantiation, and an unsupported type must fall through to the
// ncclSend/ncclRecv schedule rather than silently do nothing, so the caller
// checks the bool this sets.
#define DISPATCH_ONESHOT_REDUCE_SCATTER(              \
    datatype,                                         \
    handled,                                          \
    out,                                              \
    sendBuff,                                         \
    table,                                            \
    ranks,                                            \
    nActive,                                          \
    myRank,                                           \
    mySlot,                                           \
    rc,                                               \
    srcStride,                                        \
    ownOffset,                                        \
    slotBytes,                                        \
    seq,                                              \
    divisor,                                          \
    stream)                                           \
  do {                                                \
    (handled) = true;                                 \
    switch (datatype) {                               \
      case ncclInt32:                                 \
        launchOneShotPushReduceKernel<int32_t>(       \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclUint32:                                \
        launchOneShotPushReduceKernel<uint32_t>(      \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclInt64:                                 \
        launchOneShotPushReduceKernel<int64_t>(       \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclUint64:                                \
        launchOneShotPushReduceKernel<uint64_t>(      \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclFloat16:                               \
        launchOneShotPushReduceKernel<__half>(        \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclFloat:                                 \
        launchOneShotPushReduceKernel<float>(         \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclDouble:                                \
        launchOneShotPushReduceKernel<double>(        \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclBfloat16:                              \
        launchOneShotPushReduceKernel<__nv_bfloat16>( \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      default:                                        \
        (handled) = false;                            \
        break;                                        \
    }                                                 \
  } while (0)

// Try the one-shot IPC kernel for a single-group A-active reduce-scatter, and
// report whether it ran. Shared by the A==2 and A>2 dispatches because the
// schedule is identical -- push, flag, spin, reduce -- and only A differs.
//
// Every predicate is derived from sizes and the communicator only, so all ranks
// reach the same decision. That matters more than usual: a rank that ran the
// one-shot kernel while a peer took the ncclSend path would spin forever rather
// than merely run slower. oneShotAcquire() is COLLECTIVE on first use, so it is
// called before any branch on myActiveGroup and it agrees its own success
// across ranks.
static bool tryOneShotReduceScatter(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int nActiveRanksPerGroup,
    int nGroups,
    size_t elementSize) {
  const size_t maxCount = rcclx::relay::relayMaxCount(recvCounts, nGroups);
  if (nGroups != 1 || maxCount == 0) {
    return false;
  }
  if (static_cast<size_t>(nActiveRanksPerGroup) * maxCount * elementSize >
      rcclx::relay::kRelayOneShotMaxBytes) {
    return false;
  }
  // Creating the region is not capturable: it does a bootstrap all-gather and a
  // synchronous hipMemset. Using one that already exists is fine, so under
  // capture take the path only if the region is already up.
  struct ncclCudaGraph graph;
  if (ncclCudaGetCapturingGraph(&graph, stream) != ncclSuccess) {
    return false;
  }
  if (ncclCudaGraphValid(graph) && !rcclx::relay::oneShotReady(comm)) {
    return false;
  }

  rcclx::relay::OneShotLaunch osl{};
  if (!rcclx::relay::oneShotAcquire(comm, &osl)) {
    return false;
  }

  // Helpers have nothing to do here, but they DID have to reach the acquire
  // above, which is the whole reason it sits before this branch.
  if (myActiveGroup < 0 || recvCounts[myActiveGroup] == 0) {
    return true;
  }

  const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
  rcclx::relay::OneShotRanks ranks{};
  for (int a = 0; a < nActiveRanksPerGroup; a++) {
    ranks.r[a] = cfg.activeRanks[a];
  }
  bool handled = false;
  DISPATCH_ONESHOT_REDUCE_SCATTER(
      datatype,
      handled,
      recvBuffs[myActiveGroup],
      sendBuffs[myActiveGroup],
      osl.table,
      ranks,
      nActiveRanksPerGroup,
      comm->rank,
      cfg.myActiveIndex,
      recvCounts[myActiveGroup],
      /*srcStride=*/recvCounts[myActiveGroup],
      /*ownOffset=*/static_cast<size_t>(cfg.myActiveIndex) *
          recvCounts[myActiveGroup],
      osl.slotBytes,
      osl.seq,
      reductionDivisor,
      stream);
  // An unsupported datatype must not silently produce nothing. Falling back is
  // safe only because the dtype is identical on every rank, so either all of
  // them fall back or none do -- and in pure-direct mode the helpers post
  // nothing anyway, so a helper that already returned true is equivalent.
  return handled;
}

#define LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                          \
    TYPE, dst, seed, contribs, numContribs, count, divisor, stream) \
  launchSeededMultiReduceKernel<TYPE>(                              \
      dst, seed, contribs, numContribs, count, divisor, stream)

#define DISPATCH_SEEDED_MULTI_REDUCE(                                          \
    datatype, dst, seed, contribs, numContribs, count, divisor, stream)        \
  do {                                                                         \
    switch (datatype) {                                                        \
      case ncclInt8:                                                           \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            int8_t, dst, seed, contribs, numContribs, count, divisor, stream); \
        break;                                                                 \
      case ncclUint8:                                                          \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            uint8_t,                                                           \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      case ncclInt32:                                                          \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            int32_t,                                                           \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      case ncclUint32:                                                         \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            uint32_t,                                                          \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      case ncclInt64:                                                          \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            int64_t,                                                           \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      case ncclUint64:                                                         \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            uint64_t,                                                          \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      case ncclFloat16:                                                        \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            __half, dst, seed, contribs, numContribs, count, divisor, stream); \
        break;                                                                 \
      case ncclFloat:                                                          \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            float, dst, seed, contribs, numContribs, count, divisor, stream);  \
        break;                                                                 \
      case ncclDouble:                                                         \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            double, dst, seed, contribs, numContribs, count, divisor, stream); \
        break;                                                                 \
      case ncclBfloat16:                                                       \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            __nv_bfloat16,                                                     \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      default:                                                                 \
        break;                                                                 \
    }                                                                          \
  } while (0)

// =========================================================================
// LOW-PRECISION DISPATCH
// =========================================================================
// Only bf16 and fp32 appear here, because lpDtypeSupported() admits only those
// and the gate declines everything else back to full precision. The `default:`
// arm is therefore unreachable rather than a silent fallthrough; it aborts the
// call instead of returning ncclSuccess having quantized nothing, which is how
// a gate bug surfaces as an error rather than as wrong numbers.
//
// Deliberately a copy of the allreduce TU's macros, for the same reason the
// rest of the infra in this file is: they are file-local there and cannot be
// linked across translation units.
#define DISPATCH_LP(datatype, CALL, ...)                                                                      \
  do {                                                                                                        \
    switch (datatype) {                                                                                       \
      case ncclFloat32:                                                                                       \
        CALL<float>(__VA_ARGS__);                                                                             \
        break;                                                                                                \
      case ncclBfloat16:                                                                                      \
        CALL<__nv_bfloat16>(__VA_ARGS__);                                                                     \
        break;                                                                                                \
      default:                                                                                                \
        WARN(                                                                                                 \
            "Sharded relay: low precision reached an unsupported datatype %d; the eligibility gate is wrong", \
            static_cast<int>(datatype));                                                                      \
        return ncclInternalError;                                                                             \
    }                                                                                                         \
  } while (0)

#define DISPATCH_LP_QUANTIZE(datatype, wireOut, in, count, stream) \
  DISPATCH_LP(datatype, launchLpQuantizeKernel, wireOut, in, count, stream)

#define DISPATCH_LP_DEQUANTIZE(datatype, out, wireIn, count, stream) \
  DISPATCH_LP(datatype, launchLpDequantizeKernel, out, wireIn, count, stream)

#define DISPATCH_LP_MULTI_REDUCE(                           \
    datatype, dst, wireContribs, n, count, divisor, stream) \
  DISPATCH_LP(                                              \
      datatype,                                             \
      launchLpMultiReduceKernel,                            \
      dst,                                                  \
      wireContribs,                                         \
      n,                                                    \
      count,                                                \
      divisor,                                              \
      stream)

#define DISPATCH_LP_SEEDED_MULTI_REDUCE(                          \
    datatype, dst, seed, wireContribs, n, count, divisor, stream) \
  DISPATCH_LP(                                                    \
      datatype,                                                   \
      launchLpSeededMultiReduceKernel,                            \
      dst,                                                        \
      seed,                                                       \
      wireContribs,                                               \
      n,                                                          \
      count,                                                      \
      divisor,                                                    \
      stream)

namespace {

// The DISPATCH_* macros above instantiate reduce kernels for exactly these
// types and fall through silently (default: break) for anything else, so an
// unsupported datatype would return ncclSuccess having never reduced anything.
//
// This is deliberately a supported-set test rather than the
// `datatype < 0 || datatype >= ncclNumTypes` range test used by upstream
// ArgsCheck: ncclFloat8e4m3 and ncclFloat8e5m2 are valid NCCL types that
// ncclTypeSize() sizes at 1 byte, so they pass a range test but have no reduce
// kernel here. Keep this list in sync with the DISPATCH_* macros above.
bool isSupportedRelayDataType(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclInt8:
    case ncclUint8:
    case ncclInt32:
    case ncclUint32:
    case ncclInt64:
    case ncclUint64:
    case ncclFloat16:
    case ncclFloat:
    case ncclDouble:
    case ncclBfloat16:
      return true;
    default:
      return false;
  }
}

/**
 * The wire buffers one 2-active reduce-scatter call needs, carved from the
 * communicator's low-precision arena.
 *
 * Only THREE regions, and no mid-call producer among them: unlike the flat
 * allreduce, every reduce-scatter send reads the block this rank ships to its
 * partner, which exists in full before the first ncclGroupStart(). So one
 * hoisted quantize over that block covers the relayed chunks and both direct
 * chunks, and nothing has to be re-quantized between groups.
 *
 * Every region is carved UNCONDITIONALLY and at the WORST CASE over all groups,
 * even though a rank is active for exactly one group and a helper for the rest.
 * That is deliberate on two counts. It makes the partition byte-identical on
 * every rank and on every call with the same geometry, which is what lets a
 * captured graph replay against the same addresses. And it makes the footprint
 * a function of the counts alone, so the capacity check below is
 * rank-independent -- a rank that sized its own roles would decline on a
 * different set of calls than its peers, and low precision has to be unanimous
 * or the two disagree on wire byte counts and the call hangs.
 */
struct RsA2LpPlan {
  // wire(recvCount): the whole block shipped to the other active rank.
  char* sendShadow{nullptr};
  // wire(recvCount): mirrors foreignScratch, which mirrors the output block.
  char* foreignRecv{nullptr};
  char* helper[SHARDED_RELAY_MAX_GROUPS]{};
  bool valid{false};
};

// The size-and-dtype half of the gate, in one place so the dispatcher cannot
// accidentally feed it a different size metric than the route selector uses.
rcclx::relay::LpGateInputs reduceScatterLpGate(
    ncclDataType_t datatype,
    const size_t* recvCounts,
    int nGroups,
    int nActiveRanksPerGroup,
    size_t elementSize,
    bool relayRouteSelected,
    size_t countAlignElems) {
  rcclx::relay::LpGateInputs in;
  in.coll = rcclx::relay::LpCollective::ReduceScatter;
  in.datatype = datatype;
  in.counts = recvCounts;
  in.nGroups = nGroups;
  in.nActiveRanksPerGroup = nActiveRanksPerGroup;
  // nActiveRanksPerGroup * max(recvCount) * elementSize is exactly
  // selectReduceScatterRoute()'s metric (the bench's per-rank input label), so
  // the low-precision threshold and the route threshold are directly
  // comparable.
  in.routeSizeBytes = static_cast<size_t>(nActiveRanksPerGroup) *
      rcclx::relay::relayMaxCount(recvCounts, nGroups) * elementSize;
  in.relayRouteSelected = relayRouteSelected;
  in.countAlignElems = countAlignElems;
  return in;
}

size_t rsLpAlign(size_t bytes) {
  return ((bytes + rcclx::relay::LpArenaCarver::kAlign - 1) /
          rcclx::relay::LpArenaCarver::kAlign) *
      rcclx::relay::LpArenaCarver::kAlign;
}

// Bytes one call needs, derived only from the counts and the chunk geometry
// (which is itself derived only from the counts), so every rank computes the
// same number.
size_t rsA2LpRequiredBytes(
    const size_t* recvCounts,
    const size_t* chunkSizes,
    int nGroups,
    int nActiveRanks) {
  const size_t maxCount = rcclx::relay::relayMaxCount(recvCounts, nGroups);
  const size_t maxChunk = rcclx::relay::relayMaxCount(chunkSizes, nGroups);
  size_t total = rsLpAlign(rcclx::relay::lpWireBytes(maxCount)) +
      rsLpAlign(rcclx::relay::lpWireBytes(maxCount));
  total += static_cast<size_t>(nGroups) *
      rsLpAlign(
               static_cast<size_t>(nActiveRanks) *
               rcclx::relay::lpWireBytes(maxChunk));
  return total;
}

RsA2LpPlan rsA2LpCarve(
    const rcclx::relay::LpArenaLease& lease,
    const size_t* recvCounts,
    const size_t* chunkSizes,
    int nGroups,
    int nActiveRanks) {
  const size_t maxCount = rcclx::relay::relayMaxCount(recvCounts, nGroups);
  const size_t maxChunk = rcclx::relay::relayMaxCount(chunkSizes, nGroups);

  RsA2LpPlan p{};
  rcclx::relay::LpArenaCarver carver(lease);
  p.sendShadow = carver.take(rcclx::relay::lpWireBytes(maxCount));
  p.foreignRecv = carver.take(rcclx::relay::lpWireBytes(maxCount));
  for (int g = 0; g < nGroups; g++) {
    p.helper[g] = carver.take(
        static_cast<size_t>(nActiveRanks) *
        rcclx::relay::lpWireBytes(maxChunk));
  }
  p.valid = carver.ok();
  return p;
}

/**
 * Every region boundary this schedule sends or receives at is a whole number of
 * wire blocks.
 *
 * 128-aligned per-group counts (what lpEligible() checks) make ownBlockOffset,
 * sendBlockOffset, relayTotal, dirA and dirB aligned -- but ONLY when the
 * aligned chunk size is non-zero. In the chunkSize == 0 degenerate case the
 * geometry falls back to splitting the block in half, and count / 2 is a
 * multiple of 64, not of 128. Today's crossover keeps low precision far above
 * that regime, so this never fires; it is checked rather than assumed so that
 * lowering lpMinBytes() during tuning cannot silently produce unaligned wire
 * offsets. Pure function of the counts, so every rank declines together.
 */
bool rsA2LpGeometryOk(
    const size_t* recvCounts,
    const size_t* chunkSizes,
    int nGroups) {
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] > 0 && chunkSizes[g] == 0) {
      return false;
    }
  }
  return true;
}

/**
 * Turn the caller's request into a decision, and carve the buffers if it holds.
 *
 * COLLECTIVE: lpArenaAcquire() runs a bootstrap unanimity vote on first use, so
 * every rank must reach this whenever the dispatcher's size-only gate said yes
 * -- including the helper ranks, which is why it is called before any role
 * branch. Every reason it can return false is derived from the counts or is
 * already agreed across the communicator, so all ranks decline together.
 */
bool rsA2LpPrepare(
    bool wantLp,
    ncclComm_t comm,
    cudaStream_t stream,
    const size_t* recvCounts,
    const size_t* chunkSizes,
    int nGroups,
    int nActiveRanks,
    RsA2LpPlan* out) {
  if (!wantLp) {
    return false;
  }

  if (!rsA2LpGeometryOk(recvCounts, chunkSizes, nGroups)) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::Alignment);
    return false;
  }

  // Creating the arena is not capturable: it runs a bootstrap all-gather. Using
  // one that already exists is fine, so under capture take the path only if the
  // arena is already up. Same precedent as the one-shot region.
  struct ncclCudaGraph graph;
  if (ncclCudaGetCapturingGraph(&graph, stream) != ncclSuccess) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::GraphCapture);
    return false;
  }
  if (ncclCudaGraphValid(graph) && !rcclx::relay::lpArenaReady(comm)) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::GraphCapture);
    return false;
  }

  rcclx::relay::LpArenaLease lease{};
  if (!rcclx::relay::lpArenaAcquire(comm, &lease)) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::Arena);
    return false;
  }
  if (rsA2LpRequiredBytes(recvCounts, chunkSizes, nGroups, nActiveRanks) >
      lease.bytes) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::Arena);
    return false;
  }

  *out = rsA2LpCarve(lease, recvCounts, chunkSizes, nGroups, nActiveRanks);
  if (!out->valid) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::Arena);
    return false;
  }
  rcclx::relay::lpRecordEngage();
  return true;
}

} // namespace

/**
 * Two-active sharded relay reduce-scatter (original, performant path).
 *
 * Each group has exactly 2 active ranks; the logical collective is a 2-rank
 * reduce-scatter between them, accelerated by passthrough helpers that relay
 * sharded chunks of a single block (recvCounts[g] elements). This is the
 * production BM-FM path and is byte-for-byte unchanged from the original
 * implementation; the A>2 path lives in shardedRelayReduceScatterFlat.
 *
 * Per active rank (index myActiveIndex), with recvcount = recvCounts[g]:
 *   - sendBuff holds 2 × recvcount elements; block[i] = sendBuff[i*recvcount].
 *   - ownBlockOffset  = myActiveIndex    × recvcount (local contribution)
 *   - sendBlockOffset = otherActiveIndex × recvcount (shipped to other rank)
 *   - recvBuff[0..recvcount) = block[myActiveIndex](self) +
 *                              block[myActiveIndex](other).
 *
 * The relay relays the sendBlockOffset block chunk-by-chunk; the output block
 * (recvBuff) is seeded with the ownBlockOffset contribution then accumulates
 * the relayed/direct-exchanged chunks from the other active rank.
 *
 * In-place is detected when recvBuff == sendBuff + ownBlockOffset (the NCCL
 * reduce-scatter in-place convention). In that case recvBuff already holds the
 * local contribution (no seeding copy) and the direct chunk is reduced via a
 * scratch buffer to avoid overwriting the local data before it is read.
 *
 * LOW PRECISION, when `wantLp` survives rsA2LpPrepare(), substitutes the wire
 * format at each of the twelve boundary-crossing transfers and changes nothing
 * else -- same groups, same chunk geometry, same op counts. It is the cheapest
 * of the four collectives to carry, because every send reads the ONE block
 * shipped to the partner (so a single hoisted quantize covers all of them) and
 * every arrival lands in a buffer that already mirrors the output block (so the
 * single fused closing reduce stays a single launch).
 */
static constexpr int kHelperScratchKeyBase = SHARDED_RELAY_MAX_GROUPS + 1;

static ncclResult_t shardedRelayReduceScatter2Active(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nGroups,
    size_t elementSize,
    bool wantLp) {
  // ==========================================================================
  // SIZE-ADAPTIVE PURE-DIRECT FAST PATH (A==2)
  // ==========================================================================
  // At small sizes the 2-hop helper relay (phases 1-2 = two group boundaries
  // plus a helper HBM round trip) costs far more than the bandwidth it buys.
  // Instead the two active ranks exchange their full foreign block directly in
  // a single group (helpers idle) and reduce locally -- minimal latency, the
  // same shape as all-to-all. The size -> route mapping lives in
  // selectReduceScatterRoute() so the tests assert the same definition this
  // dispatch uses. This function is the A==2 path, so the selector is asked
  // about A==2 (its metric, 2 * recvCount * elemSize, is the bench's per-rank
  // input label here).
  if (rcclx::relay::selectReduceScatterRoute(
          2, numHelpers, nGroups, recvCounts, elementSize) ==
      rcclx::relay::ReduceScatterRoute::PureDirect) {
    // ======================================================================
    // ONE-SHOT IPC FAST PATH
    // ======================================================================
    // The schedule below is already minimal for ncclSend/ncclRecv -- one group
    // plus one fused reduce -- and that is still one launch more than NCCL,
    // which fuses transfer and reduction into a single kernel. Measured, the
    // group ALONE costs more than NCCL's whole collective, so no amount of
    // trimming here reaches 1x. The one-shot kernel removes the group entirely.
    if (tryOneShotReduceScatter(
            sendBuffs,
            recvBuffs,
            recvCounts,
            datatype,
            reductionDivisor,
            comm,
            stream,
            configs,
            myActiveGroup,
            2,
            nGroups,
            elementSize)) {
      return ncclSuccess;
    }

    void* pdScratch = nullptr;
    size_t pdRecvcount = 0;
    size_t pdOwnOff = 0;
    bool pdInPlace = false;
    if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
      const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
      pdRecvcount = recvCounts[myActiveGroup];
      pdOwnOff = static_cast<size_t>(cfg.myActiveIndex) * pdRecvcount;
      const char* ownContrib =
          static_cast<const char*>(sendBuffs[myActiveGroup]) +
          pdOwnOff * elementSize;
      pdInPlace =
          (static_cast<const void*>(recvBuffs[myActiveGroup]) ==
           static_cast<const void*>(ownContrib));
      pdScratch = ScratchBufferCache::getInstance().get(
          SHARDED_RELAY_MAX_GROUPS, pdRecvcount * elementSize, stream);
      if (pdScratch == nullptr) {
        return ncclInternalError;
      }
    }

    NCCLCHECK(ncclGroupStart());
    for (int g = 0; g < nGroups; g++) {
      if (recvCounts[g] == 0) {
        continue;
      }
      const ShardedRelayRankConfig& cfg = configs[g];
      if (!cfg.isActiveRank) {
        continue; // helpers idle in pure-direct mode
      }
      size_t rc = recvCounts[g];
      int other = 1 - cfg.myActiveIndex;
      int partner = cfg.activeRanks[other];
      // Send my contribution to the partner's owned block; receive the
      // partner's contribution to my owned block into scratch.
      NCCLCHECK(ncclSend(
          static_cast<const char*>(sendBuffs[g]) +
              static_cast<size_t>(other) * rc * elementSize,
          rc,
          datatype,
          partner,
          comm,
          stream));
      NCCLCHECK(ncclRecv(
          static_cast<char*>(pdScratch), rc, datatype, partner, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());

    if (myActiveGroup >= 0 && pdRecvcount > 0) {
      void* out = recvBuffs[myActiveGroup];
      if (pdInPlace) {
        // recvBuff already aliases the local contribution; fold the partner's
        // exchanged block (scratch) in with a single fused add[/scale] kernel.
        if (reductionDivisor > 1) {
          DISPATCH_INCREMENTAL_ADD_AND_SCALE(
              datatype, out, pdScratch, pdRecvcount, reductionDivisor, stream);
        } else {
          DISPATCH_INCREMENTAL_ADD(
              datatype, out, pdScratch, pdRecvcount, stream);
        }
      } else {
        // Out-of-place: read the local contribution and the partner block in a
        // single fused kernel (out = (own + scratch)[/divisor]) instead of a
        // seeding memcpy followed by an add[/scale], removing one launch and
        // one HBM round trip from the small-message critical path.
        DISPATCH_FUSED_REDUCE(
            datatype,
            out,
            static_cast<const char*>(sendBuffs[myActiveGroup]) +
                pdOwnOff * elementSize,
            pdScratch,
            pdRecvcount,
            reductionDivisor,
            stream);
      }
    }
    return ncclSuccess;
  }

  // =========================================================================
  // CHUNK GEOMETRY: numHelpers relayed chunks + TWO direct chunks
  // =========================================================================
  // The active<->active link is idle while the relay scatter and forward run on
  // the cross links, so instead of a third comm group for a single direct
  // chunk, one direct chunk rides along with each relay group. With numChunks =
  // numHelpers + 2 every link carries exactly one chunk per direction per
  // group, making the critical path 2*recvCount/numChunks instead of the
  // 3*recvCount/(numHelpers+1) of a separate direct phase.
  //
  // Unlike allreduce, the helper CANNOT reduce here: its slot 0 holds a0's
  // contribution to a1's output and slot 1 holds a1's contribution to a0's
  // output — different outputs. Helpers stay pure passthrough and the active
  // rank reduces.
  const int numChunks = numHelpers + 2;

  size_t chunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t relayTotals[SHARDED_RELAY_MAX_GROUPS]; // == direct chunk A's offset
  size_t dirASizes[SHARDED_RELAY_MAX_GROUPS];
  size_t dirBOffsets[SHARDED_RELAY_MAX_GROUPS];
  size_t dirBSizes[SHARDED_RELAY_MAX_GROUPS]; // absorbs the remainder

  for (int g = 0; g < nGroups; g++) {
    size_t count = recvCounts[g];

    // Zero-count groups are skipped by every loop below.
    if (count == 0) {
      chunkSizes[g] = 0;
      relayTotals[g] = 0;
      dirASizes[g] = 0;
      dirBOffsets[g] = 0;
      dirBSizes[g] = 0;
      continue;
    }

    size_t chunkSize = count / numChunks;
    chunkSize = (chunkSize / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
    chunkSizes[g] = chunkSize;
    relayTotals[g] = static_cast<size_t>(numHelpers) * chunkSize;
    if (chunkSize == 0) {
      dirASizes[g] = count / 2;
      dirBOffsets[g] = dirASizes[g];
    } else {
      dirASizes[g] = chunkSize;
      dirBOffsets[g] = relayTotals[g] + chunkSize;
    }
    dirBSizes[g] = count - dirBOffsets[g];
  }

  // =========================================================================
  // LOW PRECISION: decide and acquire the arena
  // =========================================================================
  // Collective: every rank reaches rsA2LpPrepare() when the dispatcher's
  // size-only gate said yes, and every way it can decline is agreed across the
  // communicator, so all ranks run the same wire format or none do.
  RsA2LpPlan lpPlan{};
  const bool lp = rsA2LpPrepare(
      wantLp,
      comm,
      stream,
      recvCounts,
      chunkSizes,
      nGroups,
      configs[0].nActiveRanks,
      &lpPlan);
  const rcclx::relay::RelayWire wire =
      rcclx::relay::lpWireFor(datatype, elementSize, lp);

  // =========================================================================
  // SCRATCH: the whole foreign contribution to my output block, contiguous
  // =========================================================================
  // One recvCount-element buffer laid out exactly like the output block: the
  // relayed chunks land at [0, relayTotal) and the two direct chunks fill
  // [relayTotal, recvCount). Because it mirrors the output layout, the entire
  // reduction collapses to ONE fused kernel launch at the end.
  void* foreignScratch = nullptr;
  size_t ownBlockOffset = 0;
  size_t sendBlockOffset = 0;
  bool isInPlace = false;
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    size_t recvcount = recvCounts[myActiveGroup];
    ownBlockOffset = static_cast<size_t>(cfg.myActiveIndex) * recvcount;
    sendBlockOffset = static_cast<size_t>(1 - cfg.myActiveIndex) * recvcount;

    // In-place when recvBuff aliases the local contribution block of sendBuff.
    const char* ownBlock = static_cast<const char*>(sendBuffs[myActiveGroup]) +
        ownBlockOffset * elementSize;
    isInPlace =
        (static_cast<const void*>(recvBuffs[myActiveGroup]) ==
         static_cast<const void*>(ownBlock));

    // Under low precision the arrivals are wire bytes, so lpPlan.foreignRecv --
    // laid out to mirror this same output-block ordering -- stages them instead
    // and the closing reduce reads them from there.
    if (!lp) {
      foreignScratch = ScratchBufferCache::getInstance().get(
          SHARDED_RELAY_MAX_GROUPS, recvcount * elementSize, stream);
      if (foreignScratch == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // One quantize over the ENTIRE boundary-crossing send region, before the
  // first ncclGroupStart. That region is exactly the block shipped to the other
  // active rank: the relayed chunks cover [0, relayTotal) of it and the two
  // direct chunks cover [relayTotal, recvCount). Nothing produces a send source
  // mid-call here -- every send reads this one block -- so a single launch
  // suffices, unlike the flat allreduce.
  if (lp && myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    DISPATCH_LP_QUANTIZE(
        datatype,
        lpPlan.sendShadow,
        static_cast<const char*>(sendBuffs[myActiveGroup]) +
            sendBlockOffset * elementSize,
        recvCounts[myActiveGroup],
        stream);
  }

  // Where a boundary-crossing send reads from, and where its counterpart lands.
  // Offsets are in ELEMENTS of the caller's dtype and relative to the shipped
  // block / the output block on both paths, which is what keeps every call site
  // below unchanged in shape.
  auto sendFrom = [&](int g, size_t offsetElems) -> const char* {
    if (lp) {
      return lpPlan.sendShadow + wire.bytes(offsetElems);
    }
    return static_cast<const char*>(sendBuffs[g]) +
        (sendBlockOffset + offsetElems) * elementSize;
  };

  auto foreignAt = [&](size_t offsetElems) -> char* {
    if (lp) {
      return lpPlan.foreignRecv + wire.bytes(offsetElems);
    }
    return static_cast<char*>(foreignScratch) + offsetElems * elementSize;
  };

  // Helper staging is kernel-owned scratch: callers pass a placeholder buffer
  // for groups where they are a helper. Each helper group holds nActiveRanks
  // chunks (one per active source) to receive and forward.
  void* helperScratch[SHARDED_RELAY_MAX_GROUPS] = {nullptr};
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    if (!cfg.isActiveRank && recvCounts[g] > 0 && chunkSizes[g] > 0) {
      if (lp) {
        // Already carved, and in wire bytes: the helper forwards wire blocks
        // without ever learning the caller's dtype. It does not reduce them
        // either -- see the schedule comment.
        helperScratch[g] = lpPlan.helper[g];
        continue;
      }
      size_t needBytes =
          static_cast<size_t>(cfg.nActiveRanks) * chunkSizes[g] * elementSize;
      helperScratch[g] = ScratchBufferCache::getInstance().get(
          kHelperScratchKeyBase + g, needBytes, stream);
      if (helperScratch[g] == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // =========================================================================
  // GROUP 1: relay scatter (active->helpers) || direct chunk A
  // (active<->active)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      for (int h = 0; h < cfg.numHelpers && chunkSize > 0; h++) {
        NCCLCHECK(ncclSend(
            sendFrom(g, static_cast<size_t>(h) * chunkSize),
            wire.count(chunkSize),
            wire.dtype,
            cfg.helperRanks[h],
            comm,
            stream));
      }

      // Direct chunk A over the otherwise-idle active<->active link.
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      if (dirASizes[g] > 0) {
        NCCLCHECK(ncclSend(
            sendFrom(g, relayTotals[g]),
            wire.count(dirASizes[g]),
            wire.dtype,
            partner,
            comm,
            stream));
        NCCLCHECK(ncclRecv(
            foreignAt(relayTotals[g]),
            wire.count(dirASizes[g]),
            wire.dtype,
            partner,
            comm,
            stream));
      }
    } else if (chunkSize > 0) {
      // Helper: receive active rank a's chunk into slot a.
      char* helperBuf = static_cast<char*>(helperScratch[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclRecv(
            helperBuf + wire.bytes(static_cast<size_t>(a) * chunkSize),
            wire.count(chunkSize),
            wire.dtype,
            cfg.activeRanks[a],
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // =========================================================================
  // GROUP 2: relay forward (helpers->active) || direct chunk B
  // (active<->active)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      for (int h = 0; h < cfg.numHelpers && chunkSize > 0; h++) {
        NCCLCHECK(ncclRecv(
            foreignAt(static_cast<size_t>(h) * chunkSize),
            wire.count(chunkSize),
            wire.dtype,
            cfg.helperRanks[h],
            comm,
            stream));
      }

      // Direct chunk B, again over the idle active<->active link.
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      if (dirBSizes[g] > 0) {
        NCCLCHECK(ncclSend(
            sendFrom(g, dirBOffsets[g]),
            wire.count(dirBSizes[g]),
            wire.dtype,
            partner,
            comm,
            stream));
        NCCLCHECK(ncclRecv(
            foreignAt(dirBOffsets[g]),
            wire.count(dirBSizes[g]),
            wire.dtype,
            partner,
            comm,
            stream));
      }
    } else if (chunkSize > 0) {
      // Passthrough: slot a goes to the OTHER active rank, which owns it. Still
      // a pure byte forward under low precision -- the helper never reduces
      // here, so it never has to interpret what it is carrying.
      const char* helperBuf = static_cast<const char*>(helperScratch[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclSend(
            helperBuf + wire.bytes(static_cast<size_t>(a) * chunkSize),
            wire.count(chunkSize),
            wire.dtype,
            cfg.activeRanks[1 - a],
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // =========================================================================
  // REDUCE: one fused pass over the whole output block
  // =========================================================================
  // foreignScratch mirrors the output layout, so relayed and direct chunks are
  // reduced together in a single launch.
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    size_t recvcount = recvCounts[myActiveGroup];
    void* out = recvBuffs[myActiveGroup];
    if (lp) {
      // ONE wire contribution -- the partner's whole contribution to my output
      // block, relayed and direct chunks together -- folded against my own,
      // accumulated in fp32. The divisor lands HERE and nowhere else: the
      // helper is a pure relay for this collective, so this is the only reduce
      // in the schedule. Do NOT port the allreduce's divisor-at-the-helper
      // reasoning across; there the helper reduces and this one does not.
      if (isInPlace) {
        DISPATCH_LP_MULTI_REDUCE(
            datatype,
            out,
            lpPlan.foreignRecv,
            1,
            recvcount,
            reductionDivisor,
            stream);
      } else {
        DISPATCH_LP_SEEDED_MULTI_REDUCE(
            datatype,
            out,
            static_cast<const char*>(sendBuffs[myActiveGroup]) +
                ownBlockOffset * elementSize,
            lpPlan.foreignRecv,
            1,
            recvcount,
            reductionDivisor,
            stream);
      }
    } else if (isInPlace) {
      if (reductionDivisor > 1) {
        DISPATCH_INCREMENTAL_ADD_AND_SCALE(
            datatype, out, foreignScratch, recvcount, reductionDivisor, stream);
      } else {
        DISPATCH_INCREMENTAL_ADD(
            datatype, out, foreignScratch, recvcount, stream);
      }
    } else {
      DISPATCH_FUSED_REDUCE(
          datatype,
          out,
          static_cast<const char*>(sendBuffs[myActiveGroup]) +
              ownBlockOffset * elementSize,
          foreignScratch,
          recvcount,
          reductionDivisor,
          stream);
    }
  }

  return ncclSuccess;
}

/**
 * Software-pipelined single-group 2-active sharded relay reduce-scatter.
 *
 * Same logical collective and the same passthrough helpers as
 * shardedRelayReduceScatter2Active, but for nGroups == 1 -- where the active
 * ranks and the helpers are disjoint sets -- the relay is tiled and pipelined
 * so both directions of every cross link stay busy. See relayPipelineTiles()
 * for why the two-group schedule cannot do that and what it costs.
 *
 * Helpers stay pure passthrough here, unlike allreduce: slot 0 is a0's
 * contribution to a1's output and slot 1 is a1's contribution to a0's output --
 * different outputs, so there is nothing to sum at the helper. The active rank
 * reduces once at the end, and foreignScratch still mirrors the output layout
 * so that stays a single fused launch no matter how many tiles the relay used.
 *
 * With T tiles and unit u = align(recvCount / ((H+1)*T + 1)), the block shipped
 * to the other active rank splits into T+1 STAGE-major regions, so that
 * everything arriving in one group is contiguous:
 *   region 0        direct chunk 0 alone, size u
 *   region k >= 1   relay stage k-1's H pieces of u, then direct chunk k;
 *                   size (H+1)*u, with region T absorbing the
 *                   /((H+1)*T + 1) remainder and the alignment loss
 *
 * Region k is exactly the set of pieces that lands in group k, which is what
 * lets a single launch reduce it. The unit count is unchanged at (H+1)*T + 1,
 * so relayPipelineTiles() still describes the depth.
 *
 * Group k, for k in [0, T]: the active rank scatters relay stage k (k < T) to
 * every helper, receives stage k-1 (k > 0) of the partner's contribution from
 * every helper into the matching offset of foreignScratch, and exchanges direct
 * chunk k over the active<->active link; helper h receives tile k of each
 * active's chunk into ping-pong buffer k%2 and forwards buffer (k-1)%2 to the
 * active rank that owns it.
 *
 * The owner's reduce is then issued per region on a side stream as each group
 * completes, rather than as one pass over the whole block at the end -- that
 * tail was 0.61 ms of a 2.64 ms call at 1 GB. See ReduceOverlapCache.
 */
static ncclResult_t shardedRelayReduceScatter2ActivePipelined(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nTiles,
    size_t elementSize) {
  const ShardedRelayRankConfig& cfg = configs[0];
  const size_t recvcount = recvCounts[0];
  const int H = numHelpers;
  const int T = nTiles;
  const size_t u = ((recvcount / (static_cast<size_t>(H + 1) * T + 1)) /
                    CHUNK_ALIGN_ELEMENTS) *
      CHUNK_ALIGN_ELEMENTS;
  if (u == 0) {
    return ncclInvalidArgument;
  }
  // STAGE-major geometry: region k holds precisely what group k receives.
  const size_t stageSpan = static_cast<size_t>(H + 1) * u;
  auto regionOffset = [&](int k) -> size_t {
    return (k == 0) ? 0 : (u + static_cast<size_t>(k - 1) * stageSpan);
  };
  auto regionSize = [&](int k) -> size_t {
    if (k == 0) {
      return u;
    }
    return (k < T) ? stageSpan : (recvcount - regionOffset(T));
  };
  // Relay tile (h, t) is the h-th piece of region t+1; direct chunk k is the
  // piece that follows region k's relay pieces.
  auto relayOffset = [&](int h, int t) -> size_t {
    return regionOffset(t + 1) + static_cast<size_t>(h) * u;
  };
  auto directOffset = [&](int k) -> size_t {
    return (k == 0) ? 0 : (regionOffset(k) + static_cast<size_t>(H) * u);
  };
  auto directSize = [&](int k) -> size_t {
    return (k < T) ? u : (recvcount - directOffset(T));
  };

  // Scratch mirroring the output block, so a region's reduce is one launch.
  void* foreignScratch = nullptr;
  size_t ownBlockOffset = 0;
  size_t sendBlockOffset = 0;
  bool isInPlace = false;
  if (myActiveGroup == 0) {
    ownBlockOffset = static_cast<size_t>(cfg.myActiveIndex) * recvcount;
    sendBlockOffset = static_cast<size_t>(1 - cfg.myActiveIndex) * recvcount;
    const char* ownBlock =
        static_cast<const char*>(sendBuffs[0]) + ownBlockOffset * elementSize;
    isInPlace =
        (static_cast<const void*>(recvBuffs[0]) ==
         static_cast<const void*>(ownBlock));
    foreignScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS, recvcount * elementSize, stream);
    if (foreignScratch == nullptr) {
      return ncclInternalError;
    }
  }

  // Helper staging: two ping-pong units per active source.
  char* hbuff = nullptr;
  if (!cfg.isActiveRank) {
    hbuff = static_cast<char*>(ScratchBufferCache::getInstance().get(
        kHelperScratchKeyBase,
        static_cast<size_t>(cfg.nActiveRanks) * 2 * u * elementSize,
        stream));
    if (hbuff == nullptr) {
      return ncclInternalError;
    }
  }
  auto helperSlot = [&](int a, int k) -> char* {
    return hbuff +
        (static_cast<size_t>(a) * 2 + static_cast<size_t>(k % 2)) * u *
        elementSize;
  };

  // out[span] = (own[span] + scratch[span]) / divisor.
  auto reduceSpan = [&](size_t off, size_t sz, cudaStream_t s) {
    char* out = static_cast<char*>(recvBuffs[0]) + off * elementSize;
    const char* scratch =
        static_cast<const char*>(foreignScratch) + off * elementSize;
    if (isInPlace) {
      if (reductionDivisor > 1) {
        DISPATCH_INCREMENTAL_ADD_AND_SCALE(
            datatype, out, scratch, sz, reductionDivisor, s);
      } else {
        DISPATCH_INCREMENTAL_ADD(datatype, out, scratch, sz, s);
      }
    } else {
      DISPATCH_FUSED_REDUCE(
          datatype,
          out,
          static_cast<const char*>(sendBuffs[0]) +
              (ownBlockOffset + off) * elementSize,
          scratch,
          sz,
          reductionDivisor,
          s);
    }
  };

  // Per-region reduces run on a side stream so they overlap the transfers that
  // follow. Three things force the single trailing pass on the caller's stream
  // instead: a depth deeper than the event pool (a depth-T pipeline runs T+1
  // groups and each needs its own event, so this bounds the overlap to the pool
  // rather than letting a raised kRelayMaxPipelineTiles index past it); a
  // message too small for the overlap to pay for its event traffic (see
  // kRelayOverlapReduceMinBytes); and any failure to create the side stream or
  // its events.
  //
  // Graph capture used to be a fourth. It is not, because the record/wait pair
  // below is the fork/join shape capture is defined for: the side stream joins
  // the capture through the event dependency, its reduces are captured as a
  // parallel branch, and the allDone wait rejoins the origin before the capture
  // ends. Only the resources have to be per graph, which the cache key handles.
  const bool ownerReduces = (myActiveGroup == 0);
  const ReduceOverlapCache::Handle* ovl = nullptr;
  if (ownerReduces && T + 1 <= ReduceOverlapCache::kStageEvents &&
      recvcount * static_cast<size_t>(cfg.nActiveRanks) * elementSize >=
          rcclx::relay::kRelayOverlapReduceMinBytes) {
    struct ncclCudaGraph graph;
    NCCLCHECK(ncclCudaGetCapturingGraph(&graph, stream));
    ovl = ReduceOverlapCache::getInstance().get(stream, graph);
  }

  // Once the side stream has been attached to the caller's stream, EVERY exit
  // from here has to rejoin it, error paths included. A capture that ends with
  // a forked-but-unjoined branch fails at cudaStreamEndCapture -- so an error
  // mid-pipeline would cost the whole graph rather than just this call -- and
  // leaves the cached side stream in a capturing state, poisoning it for every
  // later call that keys onto the same handle. The pipeline therefore runs in a
  // lambda whose result is held until after the join below.
  bool forked = false;
  const ncclResult_t pipelineResult = [&]() -> ncclResult_t {
    for (int k = 0; k <= T; k++) {
      NCCLCHECK(ncclGroupStart());
      if (cfg.isActiveRank) {
        const char* sendBlock = static_cast<const char*>(sendBuffs[0]) +
            sendBlockOffset * elementSize;
        char* scratch = static_cast<char*>(foreignScratch);
        const int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
        const size_t dOff = directOffset(k);
        const size_t dSz = directSize(k);

        if (k < T) {
          for (int h = 0; h < H; h++) {
            NCCLCHECK(ncclSend(
                sendBlock + relayOffset(h, k) * elementSize,
                u,
                datatype,
                cfg.helperRanks[h],
                comm,
                stream));
          }
        }
        NCCLCHECK(ncclSend(
            sendBlock + dOff * elementSize,
            dSz,
            datatype,
            partner,
            comm,
            stream));
        if (k > 0) {
          for (int h = 0; h < H; h++) {
            NCCLCHECK(ncclRecv(
                scratch + relayOffset(h, k - 1) * elementSize,
                u,
                datatype,
                cfg.helperRanks[h],
                comm,
                stream));
          }
        }
        NCCLCHECK(ncclRecv(
            scratch + dOff * elementSize,
            dSz,
            datatype,
            partner,
            comm,
            stream));
      } else {
        if (k < T) {
          for (int a = 0; a < cfg.nActiveRanks; a++) {
            NCCLCHECK(ncclRecv(
                helperSlot(a, k),
                u,
                datatype,
                cfg.activeRanks[a],
                comm,
                stream));
          }
        }
        if (k > 0) {
          for (int a = 0; a < cfg.nActiveRanks; a++) {
            NCCLCHECK(ncclSend(
                helperSlot(a, k - 1),
                u,
                datatype,
                cfg.activeRanks[1 - a],
                comm,
                stream));
          }
        }
      }
      NCCLCHECK(ncclGroupEnd());

      // Region k has landed in full. Reduce it now, on the side stream, so it
      // overlaps groups k+1..T instead of waiting behind them.
      if (ovl != nullptr) {
        CUDACHECK(cudaEventRecord(ovl->stageDone[k], stream));
        CUDACHECK(cudaStreamWaitEvent(ovl->stream, ovl->stageDone[k], 0));
        // The wait is what attaches the side stream, so the join is owed from
        // here on -- not from the record, which on its own leaves it detached.
        forked = true;
        reduceSpan(regionOffset(k), regionSize(k), ovl->stream);
      }
    }
    return ncclSuccess;
  }();

  if (forked) {
    // The caller's stream must not observe the output before the last region's
    // reduce has retired. This is the only side -> caller dependency, and it is
    // issued even when the pipeline failed, to close the fork. Both calls are
    // attempted before any error is reported, and the pipeline's own failure
    // takes precedence, since that is the one that describes what went wrong.
    const cudaError_t joinRecord = cudaEventRecord(ovl->allDone, ovl->stream);
    const cudaError_t joinWait = cudaStreamWaitEvent(stream, ovl->allDone, 0);
    if (pipelineResult != ncclSuccess) {
      return pipelineResult;
    }
    CUDACHECK(joinRecord);
    CUDACHECK(joinWait);
  } else if (pipelineResult != ncclSuccess) {
    return pipelineResult;
  }
  if (ovl == nullptr && ownerReduces) {
    // Fallback: one pass over the whole output block on the caller's stream.
    reduceSpan(0, recvcount, stream);
  }

  return ncclSuccess;
}

/**
 * Reduce-scatter for > 2 active ranks (two-group flat relay with
 * reduce-at-helper).
 *
 * Each active rank's sendBuff holds A blocks of recvCount; block[j] is this
 * rank's contribution to owner j's output. Owner j's output is the sum over all
 * A sources of block[j].
 *
 * Every block is split into a DIRECT region, exchanged 1-hop over the intra
 * active<->active links, and an OFFLOAD region, routed 2-hop through the
 * otherwise-idle helpers. A helper owns one position slice of every block: it
 * collects that slice from the A-1 non-owner sources, SUMS them, and sends the
 * single reduced slice on to the owner, which folds in its own contribution.
 * Reducing at the helper is what keeps the return hop cheap -- (A-1) chunks in,
 * one chunk out.
 *
 * Link accounting, per direction, in units of the per-(owner, helper) chunk cs.
 * A rank's helpers are the active ranks of another group, so its scatter and
 * its helper duty are egress on the same cross links:
 *
 *   group 1:  cross = (A-1)*cs (my A-1 foreign blocks' slice h)   intra =
 * (A-1)*cs group 2:  cross = cs       (one reduced slice per owner)      intra
 * = cs
 *
 * Balancing intra against cross in each group puts (A-1)*cs of the direct
 * region in group 1 and cs in group 2, so the direct region is A*cs and the
 * offload region is H*cs: cs = recvCount/(A+H), eight equal units on the 8-GPU
 * node. The critical path is (A-1)*cs + cs = A*cs = recvCount/2, against
 * recvCount for NCCL's intra-only reduce-scatter -- a 2x ceiling. (The previous
 * recursive-halving path measured ~0.96x.)
 *
 * Block layout [0, recvCount): direct region [0, A*cs) whose first (A-1)*cs go
 * out in group 1 and whose last cs goes out in group 2, then the offload region
 * [A*cs, recvCount) as H slices of cs, slice h owned by helper h.
 *
 * Helper scratch = recvBuffs[g], holding one cs chunk per (owner, contributing
 * source) pair: A*(A-1)*cs elements. The reduction is done in place into each
 * owner's first slot.
 *
 * Below a size threshold the offload is disabled and this degenerates to a
 * single-group pure-direct all-to-all reduce-scatter with the helpers idle.
 */
static ncclResult_t shardedRelayReduceScatterFlat(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nActiveRanksPerGroup,
    int nGroups,
    size_t elementSize) {
  const int A = nActiveRanksPerGroup;
  const int H = numHelpers;

  // The size -> route mapping lives in selectReduceScatterRoute() so the tests
  // assert the same definition this dispatch uses: the offload's extra hop plus
  // the second group boundary only pay for themselves past ~48 MB; below that
  // the single-group pure-direct all-to-all reduce-scatter wins outright
  // (1.24-1.42x at <= 576 KB, vs 0.69x if the offload is forced on at 4.5 MB).
  const bool useOffload = rcclx::relay::selectReduceScatterRoute(
                              A, H, nGroups, recvCounts, elementSize) ==
      rcclx::relay::ReduceScatterRoute::FlatOffload;

  // Below the offload crossover this runs a pure-direct exchange plus a reduce:
  // two launches against NCCL's one. Same story as the A==2 path, same fix --
  // one kernel that pushes, handshakes, and reduces. Only attempted when the
  // offload is disabled, i.e. in the regime the one-shot gate covers.
  if (!useOffload &&
      tryOneShotReduceScatter(
          sendBuffs,
          recvBuffs,
          recvCounts,
          datatype,
          reductionDivisor,
          comm,
          stream,
          configs,
          myActiveGroup,
          A,
          nGroups,
          elementSize)) {
    return ncclSuccess;
  }

  // Per-group geometry. cs = recvCount/(A+H) aligned down; the direct region
  // absorbs the remainder so directSz + H*cs == recvCount.
  size_t csArr[SHARDED_RELAY_MAX_GROUPS];
  size_t directArr[SHARDED_RELAY_MAX_GROUPS];
  size_t d1Arr[SHARDED_RELAY_MAX_GROUPS];
  for (int g = 0; g < nGroups; g++) {
    size_t rc = recvCounts[g];
    if (rc == 0) {
      csArr[g] = 0;
      directArr[g] = 0;
      d1Arr[g] = 0;
      continue;
    }
    size_t cs = useOffload ? (rc / static_cast<size_t>(A + H)) : 0;
    cs = (cs / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
    csArr[g] = cs;
    directArr[g] = rc - static_cast<size_t>(H) * cs;
    d1Arr[g] = (cs > 0) ? (directArr[g] - cs) : directArr[g];
  }

  // Active-rank scratch. dScratch holds the A-1 peer contributions to my direct
  // region, one contiguous directSz block each, so the whole direct region
  // reduces in a single multi-input pass. oScratch mirrors my output's offload
  // region and receives the H helper-reduced slices straight into place.
  void* dScratch = nullptr;
  void* oScratch = nullptr;
  bool isInPlace = false;
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    const char* ownBlock = static_cast<const char*>(sendBuffs[myActiveGroup]) +
        static_cast<size_t>(cfg.myActiveIndex) * recvCounts[myActiveGroup] *
            elementSize;
    isInPlace =
        (static_cast<const void*>(recvBuffs[myActiveGroup]) ==
         static_cast<const void*>(ownBlock));

    dScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS,
        static_cast<size_t>(A - 1) * directArr[myActiveGroup] * elementSize,
        stream);
    if (dScratch == nullptr) {
      return ncclInternalError;
    }
    if (csArr[myActiveGroup] > 0) {
      oScratch = ScratchBufferCache::getInstance().get(
          myActiveGroup,
          static_cast<size_t>(H) * csArr[myActiveGroup] * elementSize,
          stream);
      if (oScratch == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // Helper staging is kernel-owned scratch: callers pass a placeholder buffer
  // for groups where they are a helper. Each helper group holds A*(A-1) offload
  // slices of cs (one per (owner, contributing source) pair).
  void* helperScratch[SHARDED_RELAY_MAX_GROUPS] = {nullptr};
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    if (!cfg.isActiveRank && recvCounts[g] > 0 && csArr[g] > 0) {
      size_t needBytes =
          static_cast<size_t>(A) * (A - 1) * csArr[g] * elementSize;
      helperScratch[g] = ScratchBufferCache::getInstance().get(
          kHelperScratchKeyBase + g, needBytes, stream);
      if (helperScratch[g] == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // =========================================================================
  // GROUP 1: direct part 1 (active<->active) || offload scatter
  // (active->helper)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t rc = recvCounts[g];
    size_t cs = csArr[g];
    size_t directSz = directArr[g];
    size_t d1 = d1Arr[g];

    if (cfg.isActiveRank) {
      const char* sendbuff = static_cast<const char*>(sendBuffs[g]);
      int m = cfg.myActiveIndex;

      if (d1 > 0) {
        for (int k = 0; k < A; k++) {
          if (k == m)
            continue;
          NCCLCHECK(ncclSend(
              sendbuff + static_cast<size_t>(k) * rc * elementSize,
              d1,
              datatype,
              cfg.activeRanks[k],
              comm,
              stream));
        }
        for (int s = 0; s < A; s++) {
          if (s == m)
            continue;
          int p = (s < m) ? s : s - 1;
          NCCLCHECK(ncclRecv(
              static_cast<char*>(dScratch) +
                  static_cast<size_t>(p) * directSz * elementSize,
              d1,
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }

      // Offload scatter: helper h gets slice h of each of my foreign blocks.
      for (int h = 0; h < cfg.numHelpers && cs > 0; h++) {
        for (int j = 0; j < A; j++) {
          if (j == m)
            continue;
          NCCLCHECK(ncclSend(
              sendbuff +
                  (static_cast<size_t>(j) * rc + directSz +
                   static_cast<size_t>(h) * cs) *
                      elementSize,
              cs,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
    } else if (cs > 0) {
      // Helper: collect owner j's slice from each contributing source s != j.
      char* hbuff = static_cast<char*>(helperScratch[g]);
      for (int s = 0; s < cfg.nActiveRanks; s++) {
        for (int j = 0; j < A; j++) {
          if (j == s)
            continue;
          int t = (s < j) ? s : s - 1;
          size_t slot = static_cast<size_t>(j) * (A - 1) + t;
          NCCLCHECK(ncclRecv(
              hbuff + slot * cs * elementSize,
              cs,
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());

  bool anyOffload = false;
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] > 0 && csArr[g] > 0) {
      anyOffload = true;
      break;
    }
  }

  // =========================================================================
  // HELPER REDUCE: sum each owner's A-1 contributions in place
  // =========================================================================
  // No divisor here: the owner applies the AVG divisor once when it folds in
  // its own contribution.
  if (anyOffload) {
    for (int g = 0; g < nGroups; g++) {
      if (recvCounts[g] == 0 || csArr[g] == 0 || configs[g].isActiveRank)
        continue;
      char* hbuff = static_cast<char*>(helperScratch[g]);
      size_t cs = csArr[g];
      for (int j = 0; j < A; j++) {
        char* dst = hbuff + static_cast<size_t>(j) * (A - 1) * cs * elementSize;
        DISPATCH_MULTI_REDUCE(
            datatype, dst, dst + cs * elementSize, A - 2, cs, 1, stream);
      }
    }
  }

  // =========================================================================
  // GROUP 2: direct part 2 (active<->active) || reduced offload
  // (helper->active)
  // =========================================================================
  if (anyOffload) {
    NCCLCHECK(ncclGroupStart());
    for (int g = 0; g < nGroups; g++) {
      if (recvCounts[g] == 0 || csArr[g] == 0)
        continue;
      const ShardedRelayRankConfig& cfg = configs[g];
      size_t rc = recvCounts[g];
      size_t cs = csArr[g];
      size_t directSz = directArr[g];
      size_t d1 = d1Arr[g];
      size_t d2 = directSz - d1;

      if (cfg.isActiveRank) {
        const char* sendbuff = static_cast<const char*>(sendBuffs[g]);
        int m = cfg.myActiveIndex;

        if (d2 > 0) {
          for (int k = 0; k < A; k++) {
            if (k == m)
              continue;
            NCCLCHECK(ncclSend(
                sendbuff + (static_cast<size_t>(k) * rc + d1) * elementSize,
                d2,
                datatype,
                cfg.activeRanks[k],
                comm,
                stream));
          }
          for (int s = 0; s < A; s++) {
            if (s == m)
              continue;
            int p = (s < m) ? s : s - 1;
            NCCLCHECK(ncclRecv(
                static_cast<char*>(dScratch) +
                    (static_cast<size_t>(p) * directSz + d1) * elementSize,
                d2,
                datatype,
                cfg.activeRanks[s],
                comm,
                stream));
          }
        }

        // Reduced offload slices land contiguously, mirroring my output.
        for (int h = 0; h < cfg.numHelpers; h++) {
          NCCLCHECK(ncclRecv(
              static_cast<char*>(oScratch) +
                  static_cast<size_t>(h) * cs * elementSize,
              cs,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      } else {
        // Helper: hand owner j its single reduced slice.
        const char* hbuff = static_cast<const char*>(helperScratch[g]);
        for (int j = 0; j < A; j++) {
          NCCLCHECK(ncclSend(
              hbuff + static_cast<size_t>(j) * (A - 1) * cs * elementSize,
              cs,
              datatype,
              cfg.activeRanks[j],
              comm,
              stream));
        }
      }
    }
    NCCLCHECK(ncclGroupEnd());
  }

  // =========================================================================
  // OWNER REDUCE: fold my own contribution into the direct and offload regions
  // =========================================================================
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    size_t rc = recvCounts[myActiveGroup];
    size_t cs = csArr[myActiveGroup];
    size_t directSz = directArr[myActiveGroup];
    char* out = static_cast<char*>(recvBuffs[myActiveGroup]);
    const char* ownBlock = static_cast<const char*>(sendBuffs[myActiveGroup]) +
        static_cast<size_t>(cfg.myActiveIndex) * rc * elementSize;

    // Direct region: own + the A-1 peer blocks, one fused multi-input pass.
    if (A == 4 && directSz > 0 && !isInPlace) {
      DISPATCH_SEEDED_MULTI_REDUCE(
          datatype,
          out,
          ownBlock,
          dScratch,
          A - 1,
          directSz,
          reductionDivisor,
          stream);
    } else {
      if (!isInPlace) {
        cudaMemcpyAsync(
            out,
            ownBlock,
            directSz * elementSize,
            cudaMemcpyDeviceToDevice,
            stream);
      }
      DISPATCH_MULTI_REDUCE(
          datatype, out, dScratch, A - 1, directSz, reductionDivisor, stream);
    }

    // Offload region: own + the H helper-reduced slices (each already a sum of
    // the A-1 other sources).
    if (cs > 0) {
      char* oOut = out + directSz * elementSize;
      size_t oCount = rc - directSz;
      if (isInPlace) {
        if (reductionDivisor > 1) {
          DISPATCH_INCREMENTAL_ADD_AND_SCALE(
              datatype, oOut, oScratch, oCount, reductionDivisor, stream);
        } else {
          DISPATCH_INCREMENTAL_ADD(datatype, oOut, oScratch, oCount, stream);
        }
      } else {
        DISPATCH_FUSED_REDUCE(
            datatype,
            oOut,
            ownBlock + directSz * elementSize,
            oScratch,
            oCount,
            reductionDivisor,
            stream);
      }
    }
  }

  return ncclSuccess;
}

/**
 * Software-pipelined single-group A>2 flat reduce-scatter.
 *
 * Same reduce-at-helper relay as shardedRelayReduceScatterFlat, but for
 * nGroups == 1 -- where the active ranks and the helpers are disjoint -- the
 * offload is tiled so the scatter and the reduced return share a group and each
 * cross link runs duplex. See relayPipelineTiles().
 *
 * This is the mirror of the pipelined all-gather and equally ASYMMETRIC: a
 * source scatters the slice of ALL A-1 of its foreign blocks, so a cross link
 * carries A-1 units UP for every 1 the reduced return brings DOWN. Merging is
 * bounded by the up direction. With w = align(recvCount / ((H + A - 1)*T + 1))
 * each of the first T groups carries (A-1)*w on the busiest link and the last
 * carries w, giving ((A-1)*T + 1)*w: at A = H = 4 that is recvCount/2 at T = 1
 * (identical to the two-group schedule) falling towards 3*recvCount/7.
 *
 * The divisor is derived from A and H rather than fixed at the A = H = 4 value
 * of 7 because the layout below needs H*T + (A-1)*T + 1 units to fit: any other
 * geometry that reaches this path (A = 4 with H = 5..8 on a 9-to-12-rank comm,
 * or A = 8 with H = 8 on a 16-rank one) would otherwise push offloadTotal past
 * rc and underflow directSz, so an enormous count would reach
 * ncclSend/ncclRecv. relayShapeFanout() is shared with the all-gather because
 * the two schedules have the same unit accounting.
 *
 * Block layout [0, recvCount):
 *   [0, directSz)          direct region, one chunk per group sized to that
 *                          group's cross load: (A-1)*w for the first T groups
 * and the remainder (>= w) for the last [directSz, recvCount)  offload region;
 * helper h owns T tiles of w
 *
 * Unlike allreduce, the helper's reduce here is per tile because it gates the
 * return hop: after the group that receives tile k it sums that tile's A-1
 * contributions per owner, and the next group ships one reduced tile per owner.
 * The OWNER's reduce stays a single fused pass over the whole block at the end
 * -- reduce-scatter has no gather phase to wait on it -- so dScratch keeps the
 * flat path's shard-major layout and the closing reduction is unchanged.
 */
static ncclResult_t shardedRelayReduceScatterFlatPipelined(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nActiveRanksPerGroup,
    int nTiles,
    size_t elementSize) {
  const ShardedRelayRankConfig& cfg = configs[0];
  const size_t rc = recvCounts[0];
  const int A = nActiveRanksPerGroup;
  const int H = numHelpers;
  const int T = nTiles;
  const size_t w = ((rc /
                     (static_cast<size_t>(
                          rcclx::relay::relayShapeFanout(A, H).totalPerTile) *
                          T +
                      1)) /
                    CHUNK_ALIGN_ELEMENTS) *
      CHUNK_ALIGN_ELEMENTS;
  if (w == 0) {
    return ncclInvalidArgument;
  }
  const size_t tileStride = static_cast<size_t>(T) * w;
  const size_t offloadTotal = static_cast<size_t>(H) * tileStride;
  const size_t directSz = rc - offloadTotal;
  const size_t heavy = static_cast<size_t>(A - 1) * w;
  auto directOffset = [&](int k) -> size_t {
    return static_cast<size_t>(k) * heavy;
  };
  auto directSize = [&](int k) -> size_t {
    return (k < T) ? heavy : (directSz - directOffset(T));
  };

  // Active-rank scratch, both mirroring the flat path so the closing reduction
  // is identical: dScratch holds the A-1 peer contributions to my direct
  // region, one contiguous directSz block each; oScratch mirrors my output's
  // offload region.
  void* dScratch = nullptr;
  void* oScratch = nullptr;
  bool isInPlace = false;
  if (myActiveGroup == 0) {
    const char* ownBlock = static_cast<const char*>(sendBuffs[0]) +
        static_cast<size_t>(cfg.myActiveIndex) * rc * elementSize;
    isInPlace =
        (static_cast<const void*>(recvBuffs[0]) ==
         static_cast<const void*>(ownBlock));
    dScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS,
        static_cast<size_t>(A - 1) * directSz * elementSize,
        stream);
    oScratch = ScratchBufferCache::getInstance().get(
        0, offloadTotal * elementSize, stream);
    if (dScratch == nullptr || oScratch == nullptr) {
      return ncclInternalError;
    }
  }

  // Helper staging: one slot per (owner, contributing source) per ping-pong
  // buffer. Laid out buffer-major so that for a fixed buffer and owner the A-1
  // contributions are contiguous with stride w, which is what the fused
  // multi-input reduce below requires.
  char* hbuff = nullptr;
  const size_t slotsPerBuf = static_cast<size_t>(A) * (A - 1);
  if (!cfg.isActiveRank) {
    hbuff = static_cast<char*>(ScratchBufferCache::getInstance().get(
        kHelperScratchKeyBase, 2 * slotsPerBuf * w * elementSize, stream));
    if (hbuff == nullptr) {
      return ncclInternalError;
    }
  }
  auto helperSlot = [&](int owner, int t, int k) -> char* {
    return hbuff +
        ((static_cast<size_t>(k % 2) * slotsPerBuf +
          static_cast<size_t>(owner) * (A - 1) + static_cast<size_t>(t)) *
         w) *
        elementSize;
  };

  for (int k = 0; k <= T; k++) {
    NCCLCHECK(ncclGroupStart());
    if (cfg.isActiveRank) {
      const char* sendbuff = static_cast<const char*>(sendBuffs[0]);
      const int m = cfg.myActiveIndex;
      const size_t dOff = directOffset(k);
      const size_t dSz = directSize(k);

      // Direct reduce-scatter: my chunk of block j goes to its owner j, issued
      // as w-sized pieces rather than one (A-1)*w op.
      //
      // Both directions of every link carry the same 3w per group, but they
      // were carrying it as DIFFERENT op counts: the cross links as three
      // w-sized offload sends, the intra links as a single 3w direct send. RCCL
      // budgets channels per operation, so the single-op link received a third
      // of the channels and became the bottleneck while the cross links
      // finished early. Splitting the direct region into w-sized pieces makes
      // every op in the group the same size, so all seven links are provisioned
      // alike.
      //
      // Measured at 1 GB, 4 active ranks: 4.239 -> 3.539 ms (1.30x -> 1.56x).
      // Splitting further is worse (six pieces read 4.304 ms), so uniformity is
      // the goal, not op count for its own sake. The 2-active shapes already
      // ship a direct chunk of exactly u, which is why they never showed this.
      const size_t dPiece = w;
      const int dN = (dSz >= dPiece) ? static_cast<int>(dSz / dPiece) : 1;
      auto dPieceOff = [&](int i) -> size_t {
        return dOff + static_cast<size_t>(i) * dPiece;
      };
      auto dPieceSz = [&](int i) -> size_t {
        return (i < dN - 1) ? dPiece
                            : (dSz - static_cast<size_t>(dN - 1) * dPiece);
      };
      for (int j = 0; j < A; j++) {
        if (j == m) {
          continue;
        }
        for (int i = 0; i < dN; i++) {
          NCCLCHECK(ncclSend(
              sendbuff +
                  (static_cast<size_t>(j) * rc + dPieceOff(i)) * elementSize,
              dPieceSz(i),
              datatype,
              cfg.activeRanks[j],
              comm,
              stream));
        }
      }
      if (k < T) {
        // Offload scatter: helper h gets tile k of each of my foreign blocks.
        for (int h = 0; h < H; h++) {
          for (int j = 0; j < A; j++) {
            if (j == m) {
              continue;
            }
            NCCLCHECK(ncclSend(
                sendbuff +
                    (static_cast<size_t>(j) * rc + directSz +
                     static_cast<size_t>(h) * tileStride +
                     static_cast<size_t>(k) * w) *
                        elementSize,
                w,
                datatype,
                cfg.helperRanks[h],
                comm,
                stream));
          }
        }
      }
      for (int s = 0; s < A; s++) {
        if (s == m) {
          continue;
        }
        const int p = (s < m) ? s : s - 1;
        for (int i = 0; i < dN; i++) {
          NCCLCHECK(ncclRecv(
              static_cast<char*>(dScratch) +
                  (static_cast<size_t>(p) * directSz + dPieceOff(i)) *
                      elementSize,
              dPieceSz(i),
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }
      if (k > 0) {
        // One already-reduced tile per helper, landing where my output wants
        // it.
        for (int h = 0; h < H; h++) {
          NCCLCHECK(ncclRecv(
              static_cast<char*>(oScratch) +
                  (static_cast<size_t>(h) * tileStride +
                   static_cast<size_t>(k - 1) * w) *
                      elementSize,
              w,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
    } else {
      if (k < T) {
        // Collect owner j's tile k from every contributing source s != j. The
        // per-peer order matches each source's send order above.
        for (int s = 0; s < A; s++) {
          for (int j = 0; j < A; j++) {
            if (j == s) {
              continue;
            }
            const int t = (s < j) ? s : s - 1;
            NCCLCHECK(ncclRecv(
                helperSlot(j, t, k),
                w,
                datatype,
                cfg.activeRanks[s],
                comm,
                stream));
          }
        }
      }
      if (k > 0) {
        for (int j = 0; j < A; j++) {
          NCCLCHECK(ncclSend(
              helperSlot(j, 0, k - 1),
              w,
              datatype,
              cfg.activeRanks[j],
              comm,
              stream));
        }
      }
    }
    NCCLCHECK(ncclGroupEnd());

    // Sum this tile's A-1 contributions per owner, before the next group ships
    // them. No divisor here: the owner applies it once at the end.
    if (!cfg.isActiveRank && k < T) {
      for (int j = 0; j < A; j++) {
        char* dst = helperSlot(j, 0, k);
        DISPATCH_MULTI_REDUCE(
            datatype, dst, dst + w * elementSize, A - 2, w, 1, stream);
      }
    }
  }

  // Owner reduce, identical to the flat path: one fused pass over the direct
  // region and one over the offload region.
  if (myActiveGroup == 0) {
    char* out = static_cast<char*>(recvBuffs[0]);
    const char* ownBlock = static_cast<const char*>(sendBuffs[0]) +
        static_cast<size_t>(cfg.myActiveIndex) * rc * elementSize;

    if (A == 4 && !isInPlace) {
      DISPATCH_SEEDED_MULTI_REDUCE(
          datatype,
          out,
          ownBlock,
          dScratch,
          A - 1,
          directSz,
          reductionDivisor,
          stream);
    } else {
      if (!isInPlace) {
        cudaMemcpyAsync(
            out,
            ownBlock,
            directSz * elementSize,
            cudaMemcpyDeviceToDevice,
            stream);
      }
      DISPATCH_MULTI_REDUCE(
          datatype, out, dScratch, A - 1, directSz, reductionDivisor, stream);
    }

    char* oOut = out + directSz * elementSize;
    if (isInPlace) {
      if (reductionDivisor > 1) {
        DISPATCH_INCREMENTAL_ADD_AND_SCALE(
            datatype, oOut, oScratch, offloadTotal, reductionDivisor, stream);
      } else {
        DISPATCH_INCREMENTAL_ADD(
            datatype, oOut, oScratch, offloadTotal, stream);
      }
    } else {
      DISPATCH_FUSED_REDUCE(
          datatype,
          oOut,
          ownBlock + directSz * elementSize,
          oScratch,
          offloadTotal,
          reductionDivisor,
          stream);
    }
  }

  return ncclSuccess;
}

/**
 * Fused Multi-Group Sharded Relay Reduce-Scatter.
 *
 * Executes multiple sharded relay reduce-scatters in one fused call,
 * phase-synced across all groups so XGMI links carry unidirectional traffic.
 * Helpers are pure passthrough; reductions happen on the active ranks. Each
 * rank is ACTIVE for exactly one group and a HELPER for the others.
 *
 * nActiveRanksPerGroup must be a power of two (2 or 4): A==2 uses the original
 * 2-active path; A>2 uses the bandwidth-optimal recursive path.
 */
HOT ncclResult_t ncclShardedRelayMultiGroupReduceScatterImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups,
    int lowPrecision) {
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  // Validate every argument before touching recvCounts: the all-zero scan
  // below indexes recvCounts[0..nGroups), so a null pointer or an out-of-range
  // nGroups has to be rejected first. Bounds-checking nGroups up here also
  // means nGroups <= 0 reports ncclInvalidArgument rather than skipping the
  // scan entirely and returning ncclSuccess.
  if (nGroups < 1 || nGroups > SHARDED_RELAY_MAX_GROUPS) {
    return ncclInvalidArgument;
  }

  if (recvBuffs == nullptr || allActiveRanks == nullptr ||
      recvCounts == nullptr || sendBuffs == nullptr) {
    return ncclInvalidArgument;
  }

  // Require a power-of-two active-rank count (>= 2) for the XOR schedule.
  if (nActiveRanksPerGroup < 2 || !isPowerOfTwo(nActiveRanksPerGroup)) {
    return ncclInvalidArgument;
  }

  // Validate operation - only SUM and AVG are supported
  if (op != ncclSum && op != ncclAvg) {
    return ncclInvalidArgument;
  }

  if (!isSupportedRelayDataType(datatype)) {
    return ncclInvalidArgument;
  }

  // Check if all recvCounts are zero
  bool allZero = true;
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] != 0) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    return ncclSuccess;
  }

  size_t elementSize = ncclTypeSize(datatype);

  // Compute divisor for reduction: 1 for SUM, nActiveRanksPerGroup for AVG
  int reductionDivisor = (op == ncclAvg) ? nActiveRanksPerGroup : 1;

  // =========================================================================
  // BUILD RANK CONFIGURATION FOR ALL GROUPS
  // =========================================================================
  ShardedRelayRankConfig configs[SHARDED_RELAY_MAX_GROUPS];
  int myActiveGroup = -1; // Which group is this rank active for?

  for (int g = 0; g < nGroups; g++) {
    if (!buildShardedRelayRankConfig(
            nRanks,
            rank,
            allActiveRanks[g],
            nActiveRanksPerGroup,
            configs[g])) {
      return ncclInvalidArgument;
    }
    if (configs[g].isActiveRank) {
      myActiveGroup = g;
    }
  }

  // All groups should have the same number of helpers (same chunk structure)
  int numHelpers = configs[0].numHelpers;

  if (nActiveRanksPerGroup == 2) {
    // 2-active path is unchanged internally; feed it per-group contiguous
    // buffers. Helper groups use their scratch (recvBuffs[g]); the active
    // group uses its caller input/output buffers directly.
    const void* sendBuffs2[SHARDED_RELAY_MAX_GROUPS];
    void* recvBuffs2[SHARDED_RELAY_MAX_GROUPS];
    for (int g = 0; g < nGroups; g++) {
      sendBuffs2[g] = sendBuffs[g];
      recvBuffs2[g] = recvBuffs[g];
    }
    // A single-group relay call has the helpers to itself, so the scatter and
    // the forward run on opposite directions of each cross link and can be
    // software-pipelined into one duplex stream. relayPipelineTiles() returns
    // 1 whenever that does not apply, and the small-message pure-direct route
    // (owned by shardedRelayReduceScatter2Active) never pipelines.
    const rcclx::relay::ReduceScatterRoute route =
        rcclx::relay::selectReduceScatterRoute(
            2, numHelpers, nGroups, recvCounts, elementSize);
    const int nTiles = (route == rcclx::relay::ReduceScatterRoute::A2Relay)
        ? rcclx::relay::relayPipelineTiles(
              nGroups,
              rcclx::relay::relayShapeA2(numHelpers),
              rcclx::relay::relayMaxCount(recvCounts, nGroups),
              elementSize)
        : 1;

    // The caller's request narrowed by the size-only gate, so every rank
    // reaches the same answer without communicating. Only the non-pipelined
    // 2-active schedule carries the wire format so far, so a pipelined call is
    // reported to the gate as "no LP-capable route selected" and lands in the
    // Route decline counter rather than declining silently. That is safe
    // precisely because the tile count is a pure function of the counts -- a
    // per-rank disagreement here would be a hang, not a slowdown.
    bool wantLp = false;
    if (lowPrecision != 0) {
      wantLp = rcclx::relay::lpEligible(reduceScatterLpGate(
          datatype,
          recvCounts,
          nGroups,
          nActiveRanksPerGroup,
          elementSize,
          route == rcclx::relay::ReduceScatterRoute::A2Relay && nTiles == 1,
          rcclx::relay::kLpBlockElems));
    }
    if (nTiles > 1) {
      return shardedRelayReduceScatter2ActivePipelined(
          sendBuffs2,
          recvBuffs2,
          recvCounts,
          datatype,
          reductionDivisor,
          comm,
          stream,
          configs,
          myActiveGroup,
          numHelpers,
          nTiles,
          elementSize);
    }
    return shardedRelayReduceScatter2Active(
        sendBuffs2,
        recvBuffs2,
        recvCounts,
        datatype,
        reductionDivisor,
        comm,
        stream,
        configs,
        myActiveGroup,
        numHelpers,
        nGroups,
        elementSize,
        wantLp);
  }
  // A>2: two-group flat relay with reduce-at-helper woven with a direct
  // all-to-all reduce-scatter over the intra links. A single-group call with
  // the offload enabled can additionally be software-pipelined so the scatter
  // and the reduced return share each cross link's two directions.
  //
  // Neither flat schedule carries the wire format yet, so low precision is
  // declined here -- identically on every rank, since the condition is
  // nActiveRanksPerGroup.
  const bool offload =
      rcclx::relay::selectReduceScatterRoute(
          nActiveRanksPerGroup, numHelpers, nGroups, recvCounts, elementSize) ==
      rcclx::relay::ReduceScatterRoute::FlatOffload;
  const int nTiles = offload
      ? rcclx::relay::relayPipelineTiles(
            nGroups,
            rcclx::relay::relayShapeFanout(nActiveRanksPerGroup, numHelpers),
            rcclx::relay::relayMaxCount(recvCounts, nGroups),
            elementSize)
      : 1;
  if (nTiles > 1) {
    return shardedRelayReduceScatterFlatPipelined(
        sendBuffs,
        recvBuffs,
        recvCounts,
        datatype,
        reductionDivisor,
        comm,
        stream,
        configs,
        myActiveGroup,
        numHelpers,
        nActiveRanksPerGroup,
        nTiles,
        elementSize);
  }
  return shardedRelayReduceScatterFlat(
      sendBuffs,
      recvBuffs,
      recvCounts,
      datatype,
      reductionDivisor,
      comm,
      stream,
      configs,
      myActiveGroup,
      numHelpers,
      nActiveRanksPerGroup,
      nGroups,
      elementSize);
}
