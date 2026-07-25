// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <folly/Synchronized.h>
#include <folly/container/F14Set.h>
#include "comms/common/fault_tolerance/Abort.h"
#include "comms/ctran/algos/PersistentCleanup.h"
#include "comms/ctran/bootstrap/ICtranBootstrap.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/utils/AsyncError.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/ctran/window/WinCache.h"
#include "comms/utils/colltrace/AlgoStats.h"
#include "comms/utils/colltrace/CollTraceInterface.h"
#include "comms/utils/commSpecs.h"

namespace comms::prims {
class MultiPeerTransport;
class PipesTrace;
struct Transport;
} // namespace comms::prims

using meta::comms::CommBackend;

// Per-communicator Prims transport overrides.
// -1 means use CVAR default.
struct ctranPipesConfig {
  int64_t nvlChunkSize{-1};
  bool ibLazyConnect{false};
  int64_t ibgdaDataBufferSize{-1};

  bool operator==(const ctranPipesConfig& other) const {
    return nvlChunkSize == other.nvlChunkSize &&
        ibLazyConnect == other.ibLazyConnect &&
        ibgdaDataBufferSize == other.ibgdaDataBufferSize;
  }
};

struct ctranConfig {
  int blocking{-1};
  std::string commDesc;
  std::vector<enum CommBackend> backends = {};
  ctranPipesConfig pipesConfig;
  bool enableProfiler{NCCL_CTRAN_TRANSPORT_PROFILER};

  bool operator==(const ctranConfig& other) const {
    return (
        blocking == other.blocking && commDesc == other.commDesc &&
        backends == other.backends && pipesConfig == other.pipesConfig &&
        enableProfiler == other.enableProfiler);
  }
};

// Forward declaration to avoid circular dependency
struct ncclComm;
class CtranGpe;
namespace ncclx::memory {
class memCacheAllocator;
}
namespace comms::prims {
class MultiPeerTransport;
}
namespace ctran {
struct CtranWin;
}

using comms::fault_tolerance::Abort;
using ctran::utils::AsyncError;
using ctran::utils::Exception;

class CtranComm {
 public:
  // Make constructor public to allow dummy CtranComm to be created from UT.
  // For real communicationator we should use factory method to create.
  explicit CtranComm(
      std::shared_ptr<Abort> abort =
          comms::fault_tolerance::createAbort(/*enabled=*/false),
      ctranConfig commConfig = ctranConfig{});

  // The MemCache allocator is destroyed in a different time than all
  // other Ctran resources. To accommodate this, we split the CtranComm
  // destructor into two parts. In the first part, we destroy all
  // resources except for MemCache. The second part is moved to the
  // destructor, where it is safe to destroy MemCache and reset its
  // reference.
  void destroy();

  ~CtranComm();

  // Finalize any outstanding communication associated with the CtranComm
  // instance. Any resource release would be handled in later call to
  // destroy() at destruction time. It should be NOT be called in abort path,
  // to avoid unexpected hang due to absence of remote ranks.
  commResult_t finalize();

  inline Exception getAsyncException() const {
    return asyncErr_->getAsyncException();
  }

  inline void setAsyncException(const Exception& e) {
    asyncErr_->setAsyncException(e);
  }

  inline commResult_t getAsyncResult() const {
    return asyncErr_->getAsyncResult();
  }

  inline std::shared_ptr<AsyncError> getAsyncError() const {
    return asyncErr_;
  }

  inline std::shared_ptr<Abort> getAbort() const {
    return abort_;
  }

  inline bool abortEnabled() const {
    return abort_->Enabled();
  }

  inline void setAbort() {
    abort_->Set();
  }

  inline bool testAbort() const {
    return abort_->Test();
  }

  inline void setTimeout(const std::chrono::milliseconds& timeout) {
    return abort_->SetTimeout(timeout);
  }

  inline void cancelTimeout() {
    return abort_->CancelTimeout();
  }

  inline bool useNativeOpCount() const {
    return (opCount_ == &ctranOpCount_);
  }

  inline void updateCtranOpCount() {
    ctranOpCount_++;
  }

  inline uint64_t getCtranOpCount() const {
    return ctranOpCount_;
  }

  // Monotonic per-comm window id. Windows are registered collectively in the
  // same order on every rank, so a given window gets the same id on all ranks
  // -- used to check/log that all ranks pick the same window.
  inline uint64_t assignWindowId() {
    return nextWinId_++;
  }

  inline bool isSplitShare() const {
    return isSplitShare_;
  }

  inline CtranComm* resourceComm() {
    return resourceComm_ == nullptr ? this : resourceComm_;
  }

  inline const CtranComm* resourceComm() const {
    return resourceComm_ == nullptr ? this : resourceComm_;
  }

  // For splitShare children: child-rank -> parent-rank map. Empty otherwise.
  inline const std::vector<int>& parentRanks() const {
    return parentRanks_;
  }

  // Get a pointer to the Transport array from MultiPeerTransport,
  // indexed by global rank. Returns nullptr if MultiPeerTransport is not
  // initialized.
  comms::prims::Transport* getMultiPeerTransportsPtr() const;

  // Lazy-safe overload: materializes `peers` (via get_device_handle(peers))
  // and returns the Transport array pointer. Required in lazy-connect mode,
  // where the no-arg overload throws. Non-const because materialization
  // mutates transport state. An empty `peers` list materializes nothing and
  // still returns a valid pointer (for ranks that use no IB slots).
  comms::prims::Transport* getMultiPeerTransportsPtr(
      const std::vector<int>& peers);

  // Returns a snapshot of the algo stats, or std::nullopt if stats are
  // disabled.
  std::optional<meta::comms::colltrace::AlgoStatDump> dumpAlgoStats() const;

  void recordAlgoStats(
      const std::string& opName,
      const std::string& algoName,
      const size_t msgSize = 0);

  // Record a collective algorithm invocation. No-op if algoStats is disabled.
  inline void recordAlgoStat(
      const std::string& opName,
      const std::string& algoName,
      const size_t msgSize = 0) {
    recordAlgoStats(opName, algoName, msgSize);
  }

  // fields are public to allow access from external code and tests
  // TODO: remove config_, it's redundant
  ctranConfig config_;
  CommLogData logMetaData_;

  // opCount to be updated per kernel submit.
  // - Default points to the internal ctranOpCount_ field.
  // - When used with NCCL, will be updated to point to the NCCL opCount
  // field, so we keep the same counter when both Ctran and baseline
  // algorithms are used.
  uint64_t* opCount_{nullptr};

  // TODO: confirm with Ctran stakeholders if we need to keep this field
  // TODO: move to stateX?
  // Depending on this flag CtranAlgo initialized resources differently
  bool runtimeConn_{}; // if dynamic connection is supported

  // TODO: change shared_prt to unique_ptr after refactor all ctran code using
  // CtranComm
  std::shared_ptr<ICtran> ctran_;

  // Split-share comms define a rank group over a parent comm and share the
  // parent's Ctran resources. They must not be used directly for
  // collectives/RMA.
  // Persistent window internals can borrow resourceComm_ for registration.
  bool isSplitShare_{false};
  CtranComm* resourceComm_{nullptr};
  std::vector<int> parentRanks_;

  std::unique_ptr<meta::comms::ICtranBootstrap> bootstrap_;
  std::shared_ptr<meta::comms::colltrace::ICollTrace> colltraceNew_;
  std::shared_ptr<ncclx::memory::memCacheAllocator> memCache_;
  std::unique_ptr<ncclx::CommStateX> statex_;
  // AMD carve-out only: ENABLE_PRIMS is on for every non-AMD build (see
  // comms/ctran/def_build.bzl), so these members exist everywhere except AMD.
  // The guard changes CtranComm's layout, so consumers must compile with a
  // consistent macro (the OSS build propagates it via MCCL_ENABLE_PRIMS).
#if defined(ENABLE_PRIMS)
  std::unique_ptr<comms::prims::MultiPeerTransport> multiPeerTransport_;
  std::unique_ptr<comms::prims::PipesTrace> pipesTrace_;
#endif // defined(ENABLE_PRIMS)

  // Deferred cleanup for CUDA graph resources. CUDA user-object destructor
  // callbacks cannot call CUDA APIs, so cleanup is enqueued here and
  // executed at comm destruction where CUDA APIs are safe.
  class CudagraphDeferredCleanup {
   public:
    void add(std::function<void()> fn) {
      fns_.wlock()->push_back(std::move(fn));
    }
    void runAll() {
      auto fns = fns_.wlock();
      for (auto& fn : *fns) {
        fn();
      }
      fns->clear();
    }

   private:
    folly::Synchronized<std::vector<std::function<void()>>> fns_;
  };
  CudagraphDeferredCleanup cudagraphDeferredCleanup;

  // Registry of persistent-request cleanup tokens (see PersistentCleanup.h).
  // Each token releases one persistent request's pooled GpeKernelSync
  // (pipeSync) + scoped registration. destroy() drains this set (running each
  // token once) BEFORE ctran_.reset() -> CtranGpe::terminate(), guaranteeing
  // every pooled pipeSync is returned before terminate()'s pool-drain
  // spin-wait -- no matter which teardown path (eager free, graph-destroy
  // callback, comm cleanup) fires first.
  void registerPersistentCleanup(std::shared_ptr<PersistentCleanup> cleanup);
  void unregisterPersistentCleanup(
      const std::shared_ptr<PersistentCleanup>& cleanup);
  void drainPersistentCleanups();

  // Returns a cached window fully containing [addr, addr+bytes), or nullptr.
  // Only symmetric windows are cached and they are registered collectively in
  // the same order, so every rank resolves a buffer to the same window (needed
  // for symmetric-offset math). Non-owning: do not free a window that a
  // collective may still use.
  ctran::CtranWin* findWindowForBuffer(const void* addr, size_t bytes) const {
    return winCache_.find(addr, bytes);
  }

 private:
  friend class CtranGpe;
  friend struct ctran::CtranWin;
  friend commResult_t ctranInit(
      CtranComm* comm,
      std::unique_ptr<ctran::IProfilerReporter> reporter,
      std::unique_ptr<ctran::IGpeProfilerReporter> gpeReporter);
  std::shared_ptr<meta::comms::colltrace::AlgoStats> algoStats_;
  // TODO: define proper constructor to make CtranComm be independent of
  // ncclComm.
  // While doing refactoring we always create CtranComm from ncclComm and it
  // is the only valid way to initialize CtranComm. Therefore we make
  // constructor private and delete other constructors. After we finish
  // refactoring we will remove ncclx fields from ncclx and will initialize
  // them on CtranComm. Until then only factory method should be used to
  // initialize CtranComm.
  CtranComm(CtranComm&&);
  CtranComm& operator=(CtranComm&&);
  CtranComm(const CtranComm&) = delete;
  CtranComm& operator=(const CtranComm&) = delete;

  std::shared_ptr<AsyncError> asyncErr_;
  std::shared_ptr<Abort> abort_;
  uint64_t ctranOpCount_{0};
  uint64_t nextWinId_{0};

  folly::Synchronized<folly::F14FastSet<std::shared_ptr<PersistentCleanup>>>
      persistentCleanups_;

  // Per-comm window range cache backing findWindowForBuffer() above.
  ctran::WinCache winCache_;
};
