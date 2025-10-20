// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include <folly/Synchronized.h>
#include "comms/ctran/commstate/CommStateX.h"
#include "comms/ctran/interfaces/IBootstrap.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/utils/Abort.h"
#include "comms/ctran/utils/AsyncError.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/utils/colltrace/CollTraceInterface.h"
#include "comms/utils/commSpecs.h"

struct ctranConfig {
  int blocking{-1};
  const char* commDesc{nullptr};
  const char* ncclAllGatherAlgo{nullptr};

  bool operator==(const ctranConfig& other) const {
    return (
        blocking == other.blocking && commDesc == other.commDesc &&
        commDesc == other.commDesc &&
        ncclAllGatherAlgo == other.ncclAllGatherAlgo);
  }
};

// Forward declaration to avoid circular dependency
class CollTrace;
struct ncclComm;
namespace ncclx::memory {
class memCacheAllocator;
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
          ctran::utils::createAbort(/*enabled=*/false))
      : abort_(abort) {
    asyncErr_ =
        std::make_shared<AsyncError>(NCCL_CTRAN_ABORT_ON_ERROR, "CtranComm");
    if (!abort_) {
      throw ctran::utils::Exception(
          "abort must not be empty", commInternalError);
    }
    // Default points to internal opCount
    opCount_ = &ctranOpCount_;
  }

  // The MemCache allocator is destroyed in a different time than all other
  // Ctran resources. To accommodate this, we split the CtranComm destructor
  // into two parts. In the first part, we destroy all resources except for
  // MemCache. The second part is moved to the destructor, where it is safe to
  // destroy MemCache and reset its reference.
  void destroy() {
    // All smart pointers are automatically de-initialized, but we want to
    // ensure they do so in a specific order. Therefore, we manually handle
    // their de-initialization here.
    ctran_.reset();
    bootstrap_.reset();
    collTrace_.reset();
    colltraceNew_.reset();
    statex_.reset();
    // NOTE: memCache needs to be destroyed after transportProxy_ to release all
    // buffers
    memCache_.reset();

    this->logMetaData_.commDesc.clear();
    this->logMetaData_.commDesc.shrink_to_fit();
  }

  ~CtranComm() {
    this->destroy();
  }

  // Finalize any outstanding communication associated with the CtranComm
  // instance. Any resource release would be handled in later call to
  // destroy() at desctruction time. It should be NOT be called in abort path,
  // to avoid unexpected hang due to absense of remote ranks.
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

  // TODO: after finish refactoring remove factory method and define proper
  // constructor
  friend commResult_t setCtranCommBase(ncclComm* comm);

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
  std::unique_ptr<ctran::bootstrap::IBootstrap> bootstrap_;
  std::shared_ptr<CollTrace> collTrace_;
  std::shared_ptr<meta::comms::colltrace::ICollTrace> colltraceNew_;
  std::shared_ptr<ncclx::memory::memCacheAllocator> memCache_;
  std::unique_ptr<ncclx::CommStateX> statex_;

 private:
  // TODO: define proper constructor to make CtranComm be independable of
  // ncclComm.
  // While doing refactoring we always create CtranComm from ncclComm and it
  // is the only valid way to initialize CtranComm. Therefore we make
  // constructor private and delete other constructors. After we finish
  // refactoring we will remove ncclx fields from ncclx and will initialize
  // them on CtranComm. Until then only factory method should be used to
  // initialize CtranComm.
  CtranComm(CtranComm&&) = default;
  CtranComm& operator=(CtranComm&&) = default;
  CtranComm(const CtranComm&) = delete;
  CtranComm& operator=(const CtranComm&) = delete;

  std::shared_ptr<AsyncError> asyncErr_;
  std::shared_ptr<Abort> abort_;
  uint64_t ctranOpCount_{0};
};
