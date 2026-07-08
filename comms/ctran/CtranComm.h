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
#include "comms/ctran/bootstrap/ICtranBootstrap.h"
#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/utils/Abort.h"
#include "comms/ctran/utils/AsyncError.h"
#include "comms/ctran/utils/Exception.h"
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
  int useDualStateBuffer{-1}; // -1=cvar, 0=single, 1=dual
  bool ibLazyConnect{false};
  int64_t ibgdaDataBufferSize{-1};

  bool operator==(const ctranPipesConfig& other) const {
    return nvlChunkSize == other.nvlChunkSize &&
        useDualStateBuffer == other.useDualStateBuffer &&
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

using ctran::utils::Abort;
using ctran::utils::AsyncError;
using ctran::utils::Exception;

class CtranComm {
 public:
  // Make constructor public to allow dummy CtranComm to be created from UT.
  // For real communicationator we should use factory method to create.
  explicit CtranComm(
      std::shared_ptr<Abort> abort =
          ctran::utils::createAbort(/*enabled=*/false),
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

  // Persistent staging buffers for the small-message AllReduce-ring padding
  // path (opt-in via MCCL_FORCE_SMALL_MSG_AR_RING, driven by
  // ctranAllReduceRingSmallMsg). Lazily allocated on first use, grown on
  // demand, and reused across collectives; freed in destroy(). Remain nullptr
  // when the feature is unused.
  void* smallMsgStageSrc_{nullptr};
  void* smallMsgStageDst_{nullptr};
  size_t smallMsgStageBytes_{0};
#if defined(ENABLE_PRIMS)
  std::unique_ptr<comms::prims::MultiPeerTransport> multiPeerTransport_;
  uint64_t* hierarchicalAgReadyCounters_{nullptr};
  size_t hierarchicalAgReadyCounterCount_{0};
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

 private:
  friend class CtranGpe;
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
};
