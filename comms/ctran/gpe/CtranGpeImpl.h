// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_GPE_IMPL_H_
#define CTRAN_GPE_IMPL_H_

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <queue>
#include <stack>
#include <thread>

#include <folly/Synchronized.h>
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/algos/common/GpeRing.h"
#include "comms/ctran/gpe/CtranChecksum.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/gpe/GpeDeviceRing.h"
#include "comms/ctran/profiler/GpeProfiler.h"
#include "comms/ctran/profiler/IGpeProfilerReporter.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/ctran/utils/PinnedHostPool.h"
#include "comms/utils/GraphCaptureSideStream.h"

struct CommLogData;

struct alignas(128) KernelFlagItem {
  using Self = KernelFlagItem;

  static const char* name() {
    return "KernelFlag";
  }

  void reset() {
    for (int i = 0; i < numGroups_; i++) {
      dev.flag_[i] = KERNEL_UNSET;
    }
    // Clear the full ring header so enabled==1 always implies ring/cmdId are
    // current; submit() re-arms it per-cmd on the ring path.
    dev.gpeHdr = {};
    // Same invariant for the colltrace header: enabled==1 must always imply a
    // current ring/collId; submit() re-arms it per-cmd for grouped kernels.
    dev.colltraceHdr = {};
  }

  bool inUse() {
    if (persistent_) {
      return true;
    }
    for (int i = 0; i < numGroups_; i++) {
      if (dev.flag_[i] != KERNEL_UNSET) {
        return true;
      }
    }
    return false;
  }

  void onPop() {
    for (int i = 0; i < CTRAN_ALGO_MAX_THREAD_BLOCKS; ++i) {
      dev.flag_[i] = KERNEL_SCHEDULED;
    }
    numGroups_ = 1;
    // Clear the full ring header (not just enabled) so a stale ring/cmdId can
    // never be published; submit() re-arms it per-cmd on the ring path.
    dev.gpeHdr = {};
    // Same for the colltrace header: a stale ring/collId must never be
    // published; submit() re-arms it per-cmd for grouped kernels.
    dev.colltraceHdr = {};
  }

  bool testFlagAllGroups(int flag) {
    for (int i = 0; i < numGroups_; ++i) {
      if (dev.flag_[i] != flag) {
        return false;
      }
    }
    return true;
  }

  void setFlagPerGroup(int flag) {
    for (int i = 0; i < numGroups_; ++i) {
      dev.flag_[i] = flag;
    }
  }

  // Prevent pool reclaim between graph replays. The kernel writes
  // KERNEL_UNSET after each replay, but persistent keeps inUse() true.
  void setPersistent() {
    persistent_ = true;
  }

  // Allow pool reclaim. Called at graph destruction to release the flag.
  void clearPersistent() {
    persistent_ = false;
  }

  // Device-facing flag object (per-block flags + ring header). Passed to the
  // kernel as &dev (a ctran::gpe::KernelFlagDev*); MUST be the first member so
  // that pointer is at offset 0 of the KernelFlagItem.
  ctran::gpe::KernelFlagDev dev;
  int numGroups_{1};
  // If true, inUse() always returns true — prevents reclaim() from stealing
  // the flag while a persistent cmd (graph capture) still owns it.
  // Not cleared by reset() — only cleared by clearPersistent().
  bool persistent_{false};

  void _() {
    // Make sure KernelFlagItem satisfies the PinnedHostItem concept
    static_assert(PinnedHostItem<Self>);
  }
};

// By default, KernelFlagPool uses 32KB of pinned memory
// It is NOT thread-safe, as one pool for every GPE and both pop and reclaim
// operations are expected to be called from main thread before submitting
// a new command to the GPE.
using KernelFlagPool = PinnedHostPool<KernelFlagItem>;

class CtranGpeCmd {
 public:
  CtranGpeCmd() = default;
  ~CtranGpeCmd();

  enum TypeEnum {
    GRAPH_ENQUEUE,
    TERMINATE,
  } type;

  struct {
    std::vector<std::unique_ptr<struct OpElem>> opGroup;
    opFunc func;
    std::shared_ptr<meta::comms::colltrace::ICollTraceHandle> collHandle;
    CtranComm* comm;
  } coll;

  // kernelFlag to assist device mem communication
  KernelFlagItem* kernelFlag{nullptr};
  // cpuFlag to track completion of host mem communication
  std::shared_ptr<std::atomic_flag> cpuFlag{nullptr};

  bool persistent{false};
  // Count of queued-but-not-yet-processed instances of this cmd. Used by
  // cmdDestroy to wait for the GPE to drain stale queue entries before
  // deleting the cmd.
  std::atomic_uint32_t inFlight{0};
  CtranGpe* gpe{nullptr};

  // Device-ring dispatch (NCCL_CTRAN_GPE_DEVICE_RING). cmdId is the comm-local
  // id the kernel publishes to the ring; inDeviceRingRegistry marks that this
  // cmd owns a registry entry cmdDestroy must erase. Set only on the ring path.
  ctran::gpe::GpeCmdId cmdId{0};
  bool inDeviceRingRegistry{false};

  std::optional<std::chrono::milliseconds> timeout{std::nullopt};

  // Unpack queue to teardown after the eager kernel completes, or when
  // a persistent CUDA graph command is destroyed (for TcpDM backend).
  void* unpackPool{nullptr};

  // Post-kernel cleanup callback. Called by GPE thread after kernel finishes.
  std::function<void()> postKernelCleanup{nullptr};
};

/**
 * Pool of KernelElem objects allocated from cudaHostAlloc. It is NOT
 * thread-safe, as one pool for every GPE and both pop and reclaim operations
 * are expected to be called from main thread before submitting a new command to
 * the GPE.
 */
class KernelElemPool {
 public:
  KernelElemPool(size_t capacity);
  ~KernelElemPool();

  // Pop a KernelElem from the free pool; enqueue to the in-use queue for
  // later reclaimant
  // Input arguments:
  //   - ngroups: number of thread block groups to use each p2pElem object; used
  //              to set inuse flag
  KernelElem* pop(int ngroups);

  // Reclaim any unused KernelElem objects back to the free pool.
  void reclaim();

  // Return the number of KernelElem objects in the free pool.
  size_t size();

  // Return the capacity of the pool.
  size_t capacity();

 private:
  void resetWorkElem(KernelElem* workElem);

  std::stack<KernelElem*> freeWorkElems_;
  std::list<KernelElem*> inuseWorkElems_;
  const size_t capacity_{0};
  void* memPtr_{nullptr};
};

bool checksumIsSampled(KernelConfig::KernelType kernelType, int opCount);

std::optional<ChecksumArgs> ctranFillChecksumArgs(
    KernelConfig& kernelConfig,
    ChecksumItem* checksumItem,
    const CtranComm* comm);

// GpeKernelSyncPool alias is now declared in CtranGpe.h (public).
commResult_t allocGpeKernelSyncs(
    GpeKernelSyncPool* gpeKernelSyncPool,
    size_t count,
    int nworkers,
    std::vector<ctran::algos::GpeKernelSync*>& gpeKernelSyncs);

class OrderedWorkStreamGuard {
 public:
  ~OrderedWorkStreamGuard();

  void init(const CommLogData& logMetaData);

  class Scope {
   public:
    Scope(
        OrderedWorkStreamGuard& guard,
        cudaStream_t userStream,
        const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo);
    ~Scope();

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope(Scope&& other) noexcept;
    Scope& operator=(Scope&& other) noexcept;

    commResult_t status() const {
      return status_;
    }
    cudaStream_t stream() const {
      return userStream_;
    }

   private:
    OrderedWorkStreamGuard* guard_;
    cudaStream_t userStream_;
    ctran::utils::cudagraph::StreamCaptureInfo captureInfo_;
    commResult_t status_;
  };

  Scope acquire(
      cudaStream_t userStream,
      const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo);

 private:
  commResult_t doAcquire(
      cudaStream_t userStream,
      const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo);
  commResult_t doRelease(
      cudaStream_t userStream,
      const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo);

  cudaEvent_t execModeSyncEvent_{};
  unsigned long long lastCaptureId_{0};
  bool everCaptured_{false};
  cudaStream_t lastUserStream_{nullptr};
  cudaGraphNode_t lastRecordNode_{};

  // Side stream used during capture to host the external cudaEventRecord
  // node for execModeSyncEvent_ off the user stream's critical path, so
  // its release fence doesn't stall compute between ctran submissions.
  // The next doAcquire still adds lastRecordNode_ (now on the side) as
  // an explicit capture dependency of userStream, preserving ordering.
  std::unique_ptr<meta::comms::GraphSideStream> sideStream_;

  const CommLogData* logMetaData_{nullptr};
};

class CtranGpe::Impl {
 public:
  Impl();
  ~Impl();

  // submit device mem communication
  commResult_t submit(
      CtranGpeCmd::TypeEnum type,
      std::vector<std::unique_ptr<struct OpElem>> opGroup,
      opFunc func,
      KernelConfig& kernelConfig,
      const void* ncclKernel,
      std::optional<std::chrono::milliseconds> timeout = std::nullopt);

  // submit host mem communication
  commResult_t submitHost(
      CtranGpeCmd::TypeEnum type,
      std::vector<std::unique_ptr<struct OpElem>> opGroup,
      opFunc func,
      KernelConfig& kernelConfig,
      std::shared_ptr<std::atomic_flag> cpuFlag);

  // start the GPE thread.
  void start();
  // terminate the GPE thread.
  void terminate();

  // Allocate the per-comm device dispatch ring + reader when
  // NCCL_CTRAN_GPE_DEVICE_RING is set (and supported). Called from the
  // CtranGpe constructor once comm/cudaDev are wired. No-op (leaves the ring
  // disabled) on HIP/pre-Hopper or if allocation fails, so callers silently
  // fall back to the host-node path.
  void initDeviceRing();

  // True when the device-ring dispatch path is active for this comm.
  bool deviceRingEnabled() const {
    return deviceRing_ != nullptr;
  }

  // A lightweight device handle to the ring, stored in a cmd's KernelFlagItem
  // header (GpeKernelFlagHeader) at submit. Only valid when
  // deviceRingEnabled().
  ctran::gpe::GpeRingHandle deviceRingHandle() const {
    return deviceRing_->deviceHandle();
  }

  // Registry accessor used by submit() and the cmdDestroy callback.
  GpeDeviceRingCmdRegistry& deviceRingCmdRegistry() {
    return deviceRingCmdRegistry_;
  }

  CtranComm* comm{nullptr};

  std::unique_ptr<KernelElemPool> kernelElemPool;
  std::unique_ptr<KernelFlagPool> kernelFlagPool;
  std::unique_ptr<ChecksumPool> checksumPool;
  std::unique_ptr<GpeKernelSyncPool> gpeKernelSyncPool;

  int cudaDev{-1};
  CtranGpe* gpe{nullptr};

  // Used only by the GPE thread.
  std::unique_ptr<ctran::GpeProfiler> gpeProfiler_;

 private:
  struct CmdQueue {
    std::queue<CtranGpeCmd*> queue;
  };
  folly::Synchronized<CmdQueue, std::mutex> cmdQueue_;
  std::condition_variable cmdQueueCv_;
  std::thread thread_;
  OrderedWorkStreamGuard ws_;

  // Device-ring dispatch state (NCCL_CTRAN_GPE_DEVICE_RING). Ring + reader are
  // null unless the ring path is enabled. The reader and ringPending_ are
  // owned exclusively by the GPE worker thread (single consumer).
  std::unique_ptr<ctran::gpe::GpeRing> deviceRing_;
  std::unique_ptr<ctran::gpe::GpeRingReader> deviceRingReader_;
  std::queue<CtranGpeCmd*> ringPending_;
  GpeDeviceRingCmdRegistry deviceRingCmdRegistry_;

  // In-kernel colltrace grouping: a multi-submit collective (e.g. AllGatherP's
  // PipeStart..PipeSync..PipeEnd) records one CollTrace event whose start is
  // written by the group's Begin kernel and end by its End kernel. Begin
  // stashes the device handle (ring + collId) here so the following End submit
  // arms its kernel with the same collId. A default (null-ring, !valid())
  // handle means no group is open. Written and read only on the submit
  // (capture) thread, and a group's submits are always contiguous, so a single
  // pending slot suffices.
  meta::comms::colltrace::ColltraceDeviceHandle pendingColltraceGroup_{};

  // Main function called by the GPE thread. It waits and handles any  commands
  // submitted to cmdQueue until the TERMINATE command is received.
  void gpeThreadFn();

  // Return the next command for the GPE worker to process. Without the device
  // ring this is exactly cmdDequeue() (blocking on the CPU FIFO). With the ring
  // enabled it polls the CPU FIFO (for TERMINATE, eager, and non-ring cmds)
  // and the device ring (for captured ring cmds, in GPU execution order),
  // preferring the FIFO so TERMINATE/abort is never starved by ring traffic.
  CtranGpeCmd* acquireNextCmd();

  // Publish a captured (graph) cmd for replay: arm the device ring (no host
  // node) when it is enabled, otherwise add an in-graph host node. Both retain
  // the cmd on the graph so it is freed at cudaGraphDestroy. Extracted from
  // submit() to keep that function short.
  commResult_t publishCapturedCmd(
      CtranGpeCmd* cmd,
      KernelFlagItem* kernelFlag,
      cudaStream_t stream,
      ctran::utils::cudagraph::StreamCaptureInfo& streamCaptureInfo);

  // Enqueue a command and notify the GPE thread to wake up.
  inline void cmdEnqueue(CtranGpeCmd* cmd) {
    {
      cmdQueue_.lock()->queue.push(cmd);
    }
    cmdQueueCv_.notify_one();
  }

  // Dequeue a command for the GPE thread.
  // If the queue is empty, the calling GPE thread will sleep until receive a
  // wakeup signal when a command is enqueued.
  inline CtranGpeCmd* cmdDequeue() {
    auto locked = cmdQueue_.lock();
    cmdQueueCv_.wait(
        locked.as_lock(), [&locked] { return !locked->queue.empty(); });

    auto cmd = locked->queue.front();
    locked->queue.pop();
    return cmd;
  }

  // Forwards opCount/opType from the dequeued command's leading op to
  // the GPE profiler, which uses opCount to compute this iter's
  // sampling verdict. No-op when opGroup is empty (TERMINATE cmd or
  // post-kernel cleanup-only cmd) since there is no meaningful op to
  // attribute metadata to. The TERMINATE iter still gets an explicit
  // TERMINATE_CMD tracepoint via the always-on path.
  void injectGpeProfilerMetadata(CtranGpeCmd* cmd);

  static void CUDART_CB cmdCb(void* data);
  static void CUDART_CB cmdDestroy(void* data);
};

#endif
