// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_GPE_IMPL_H_
#define CTRAN_GPE_IMPL_H_

#include <chrono>
#include <condition_variable>
#include <list>
#include <mutex>
#include <optional>
#include <queue>
#include <stack>
#include <thread>

#include <folly/Synchronized.h>
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/gpe/CtranChecksum.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/ctran/utils/ExtUtils.h"
#include "comms/ctran/utils/PinnedHostPool.h"

struct KernelFlagItem {
  using Self = KernelFlagItem;

  static const char* name() {
    return "KernelFlag";
  }

  void reset() {
    for (int i = 0; i < numGroups_; i++) {
      flag_[i] = KERNEL_UNSET;
    }
  }

  bool inUse() {
    for (int i = 0; i < numGroups_; i++) {
      if (flag_[i] != KERNEL_UNSET) {
        return true;
      }
    }
    return false;
  }

  void onPop() {
    for (int i = 0; i < CTRAN_ALGO_MAX_THREAD_BLOCKS; ++i) {
      flag_[i] = KERNEL_SCHEDULED;
    }
    numGroups_ = 1;
  }

  bool testFlagAllGroups(int flag) {
    for (int i = 0; i < numGroups_; ++i) {
      if (flag_[i] != flag) {
        return false;
      }
    }
    return true;
  }

  void setFlagPerGroup(int flag) {
    for (int i = 0; i < numGroups_; ++i) {
      flag_[i] = flag;
    }
  }

  volatile int flag_[CTRAN_ALGO_MAX_THREAD_BLOCKS];
  int numGroups_{1};
  // padding for different KernelFlagItems to be on different cache lines.
  static constexpr int kCacheLineSizeBytes = 128;
  int unused[(kCacheLineSizeBytes - sizeof(int)) / sizeof(int)];

  void _() {
    // Make sure KernelFlagItem satisfies the PinnedHostItem concept
    static_assert(PinnedHostItem<Self>);

    // The following compile-time check is a hint of the memory usage
    // KernelFlag uses 384 bytes of pinned memory
    static_assert(
        sizeof(Self) ==
        4 * CTRAN_ALGO_MAX_THREAD_BLOCKS + 4 + (kCacheLineSizeBytes - 4));

    // Ensure two KernelFlagItems are on different cache lines.
    //
    // TODO(T238727523): evaluate the impact of multiple flags in the same
    // KernelFlagItem on the same cache line. Each flag here is consumed by a
    // separate block for Ctran device code.
    //
    // Note that the device side CancellableWaits feature is disabled by default
    // for the device kernels, so these flags are used only for GPE <-> Kernel
    // start and stop-syncs in such cases.
    static_assert(sizeof(Self) % kCacheLineSizeBytes == 0);
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
  ~CtranGpeCmd() = default;

  enum TypeEnum {
    GRAPH_ENQUEUE,
    TERMINATE,
  } type;

  struct {
    std::vector<std::unique_ptr<struct OpElem>> opGroup;
    opFunc func;
    std::shared_ptr<meta::comms::colltrace::ICollTraceHandle> collHandle;
    CtranComm* comm;
    // If persistent is true for the cmd, storing unique_ptr to persistent
    // object (AlgoImpl) used by persistent / cudagraph-aware colls; otherwise
    // nothing.
    ctran::PersistentObj pObj;
  } coll;

  // kernelFlag to assist device mem communication
  KernelFlagItem* kernelFlag{nullptr};
  // cpuFlag to track completion of host mem communication
  std::shared_ptr<std::atomic_flag> cpuFlag{nullptr};

  bool persistent{false};

  std::optional<std::chrono::milliseconds> timeout{std::nullopt};
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

using GpeKernelSyncPool = PinnedHostPool<ctran::algos::GpeKernelSync>;
commResult_t allocGpeKernelSyncs(
    GpeKernelSyncPool* gpeKernelSyncPool,
    size_t count,
    int nworkers,
    std::vector<ctran::algos::GpeKernelSync*>& gpeKernelSyncs);

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
      std::optional<std::chrono::milliseconds> timeout = std::nullopt,
      ctran::PreLaunchGraphPrepareFn graphPrepareFn = nullptr);

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

  // Before enqueueing a command to the GPE, update the command if needed.
  // Currently, this is used for enabling cudagraph-aware AllToAll.
  inline commResult_t preLaunchGraphPrepare(
      CtranGpeCmd* cmd,
      ctran::PreLaunchGraphPrepareFn graphPrepareFn) {
    if (graphPrepareFn == nullptr) {
      return commSuccess;
    }
    auto op = cmd->coll.opGroup.front().get();
    // This is cudagraph-aware collective prepare function: transfer collective
    // to Persistent collective for perf optimization
    graphPrepareFn(cmd->coll.func, op, cmd->coll.pObj);
    return commSuccess;
  }

  // Maintain execution order between
  // user streams before launching Ctran kernel,
  commResult_t preKernelLaunch(cudaStream_t curStream);
  // Ensure future work waits on just launched Ctran kernel
  commResult_t postKernelLaunch(cudaStream_t curStream);

  CtranComm* comm{nullptr};

  std::unique_ptr<KernelElemPool> kernelElemPool;
  std::unique_ptr<KernelFlagPool> kernelFlagPool;
  std::unique_ptr<ChecksumPool> checksumPool;
  std::unique_ptr<GpeKernelSyncPool> gpeKernelSyncPool;

  int cudaDev{-1};
  CtranGpe* gpe{nullptr};

 private:
  struct CmdQueue {
    std::queue<CtranGpeCmd*> queue;
  };
  folly::Synchronized<CmdQueue, std::mutex> cmdQueue_;
  std::condition_variable cmdQueueCv_;
  std::thread thread_;
  // Internal event and stream used to manage execution order of Ctran kernels
  // if multiple streams are used to submit ops
  cudaEvent_t execEvent_;
  cudaStream_t execOrderStream_;
  // record the most recent user stream to allow a fast path bypass stream
  // ordering
  cudaStream_t lastUserStream_;

  // Main function called by the GPE thread. It waits and handles any  commands
  // submitted to cmdQueue until the TERMINATE command is received.
  void gpeThreadFn();

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

  static void CUDART_CB cmdCb(void* data);
  static void CUDART_CB cmdDestroy(void* data);
};

#endif
