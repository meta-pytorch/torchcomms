// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/common/OrderedWorkStreamGuard.h"

#include <utility>

#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::algos {

void OrderedWorkStreamGuard::init(
    const CommLogData& logMetaData,
    bool synchronizeEagerAfterCapturedWork) {
  FB_CHECKABORT(!initialized_, "OrderedWorkStreamGuard initialized twice");

  // Build resources locally so a failure leaves the guard uninitialized.
  auto sideStream = std::make_unique<meta::comms::GraphSideStream>();
  cudaEvent_t execModeSyncEvent{};
  FB_CUDACHECKTHROW_EX(
      cudaEventCreateWithFlags(&execModeSyncEvent, cudaEventDisableTiming),
      logMetaData);
  cudaEvent_t captureFenceEvent{};
  FB_CUDACHECKTHROW_EX(
      cudaEventCreateWithFlags(&captureFenceEvent, cudaEventDisableTiming),
      logMetaData);

  // Publish the initialized state only after every resource is ready.
  graphMixingSupport_ = (NCCL_CTRAN_GRAPH_MIXING_SUPPORT != 0);
  synchronizeEagerAfterCapturedWork_ = synchronizeEagerAfterCapturedWork;
  execModeSyncEvent_ = execModeSyncEvent;
  captureFenceEvent_ = captureFenceEvent;
  sideStream_ = std::move(sideStream);
  initialized_ = true;
}

OrderedWorkStreamGuard::~OrderedWorkStreamGuard() noexcept {
  if (!initialized_) {
    return;
  }
  FB_CUDACHECKIGNORE(cudaEventDestroy(execModeSyncEvent_));
  FB_CUDACHECKIGNORE(cudaEventDestroy(captureFenceEvent_));
}

OrderedWorkStreamGuard::Scope::Scope(
    OrderedWorkStreamGuard& guard,
    cudaStream_t userStream,
    const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo)
    : guard_(&guard),
      userStream_(userStream),
      captureInfo_(captureInfo),
      lock_(guard.submissionMutex_) {
  status_ = guard_->doAcquire(userStream_, captureInfo_);
  if (status_ != commSuccess) {
    guard_->error_ = status_;
  }
}

OrderedWorkStreamGuard::Scope::~Scope() {
  release();
}

OrderedWorkStreamGuard::Scope::Scope(Scope&& other) noexcept
    : guard_(other.guard_),
      userStream_(other.userStream_),
      captureInfo_(other.captureInfo_),
      status_(other.status_),
      lock_(std::move(other.lock_)) {
  other.guard_ = nullptr;
}

commResult_t OrderedWorkStreamGuard::Scope::release() {
  if (guard_ == nullptr) {
    return status_;
  }

  auto* guard = guard_;
  guard_ = nullptr;
  if (status_ == commSuccess) {
    status_ = guard->doRelease(userStream_, captureInfo_);
    if (status_ != commSuccess) {
      guard->error_ = status_;
    }
  }
  if (lock_.owns_lock()) {
    lock_.unlock();
  }
  return status_;
}

OrderedWorkStreamGuard::Scope OrderedWorkStreamGuard::acquire(
    cudaStream_t userStream,
    const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo) {
  FB_CHECKABORT(initialized_, "OrderedWorkStreamGuard used before init()");
  return Scope(*this, userStream, captureInfo);
}

commResult_t OrderedWorkStreamGuard::doAcquire(
    cudaStream_t userStream,
    const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo) {
  if (error_ != commSuccess) {
    return error_;
  }

  const bool isCapturing = captureInfo.status == cudaStreamCaptureStatusActive;

  bool isNewCapture = isCapturing && captureInfo.id != lastCaptureId_;
  if (isNewCapture) {
    lastCaptureId_ = captureInfo.id;
    everCaptured_ = true;
  }

  // Only an in-capture wait at mixing=0 targets the captured fence; every other
  // path waits on the event the eager side records. The two are never both
  // meaningful at once, so the choice follows from the capture state and mode
  // rather than needing a per-call-site argument.
  auto doWait = [&]() -> commResult_t {
    const bool inCaptureWithoutMixing = isCapturing && !graphMixingSupport_;
    FB_CUDACHECK(cudaStreamWaitEvent(
        userStream,
        inCaptureWithoutMixing ? captureFenceEvent_ : execModeSyncEvent_,
        (isCapturing && graphMixingSupport_) ? cudaEventWaitExternal
                                             : cudaEventWaitDefault));
    return commSuccess;
  };

  if (lastUserStream_ == nullptr) {
    if (isCapturing && graphMixingSupport_) {
      return doWait();
    }
    return commSuccess;
  }

  if (!isCapturing) {
    // GPE requires a host sync after graph work so its host-node command
    // reaches the single-threaded queue before a following eager command.
    // Other users retain fully asynchronous stream ordering.
    if (everCaptured_ && synchronizeEagerAfterCapturedWork_) {
      FB_CUDACHECK(cudaEventSynchronize(execModeSyncEvent_));
    } else if (everCaptured_ || userStream != lastUserStream_) {
      FB_COMMCHECK(doWait());
    }
    return commSuccess;
  }

  if (!isNewCapture) {
    if (!graphMixingSupport_) {
      // The fence is a captured plain record, so capture folds this wait into
      // the graph as a dependency edge; there is no node to re-link.
      return doWait();
    }
    // A later operation captured into the same graph must depend on the
    // prior record node even if CUDA cannot infer a dependency across streams:
    // cudaStreamWaitEvent cannot observe an EVENT_RECORD node.
#if defined(__HIP_PLATFORM_AMD__)
    FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
        userStream, &lastRecordNode_, 1, hipStreamAddCaptureDependencies));
#elif CUDART_VERSION >= 13000
    FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
        userStream,
        &lastRecordNode_,
        nullptr,
        1,
        cudaStreamAddCaptureDependencies));
#else
    FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
        userStream, &lastRecordNode_, 1, cudaStreamAddCaptureDependencies));
#endif
  }

  if (!graphMixingSupport_) {
    // New capture: the previous fence, if any, belongs to a different graph.
    // captureFenceEvent_ cannot order across graphs (a captured record is only
    // meaningful within its own capture) and execModeSyncEvent_ carries no
    // replay record in this mode, so there is nothing to wait on.
    return commSuccess;
  }

  return doWait();
}

commResult_t OrderedWorkStreamGuard::doRelease(
    cudaStream_t userStream,
    const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo) {
  const bool isCapturing = captureInfo.status == cudaStreamCaptureStatusActive;

  if (!isCapturing) {
    FB_CUDACHECK(cudaEventRecord(execModeSyncEvent_, userStream));
  } else if (!graphMixingSupport_) {
    // Capture absorbs this record rather than executing it, so no standalone
    // EVENT_RECORD node exists for cudaGraphInstantiate to place onto a busy
    // hardware channel, and the next doAcquire's wait folds into the graph as a
    // dependency edge.
    //
    // Use a dedicated event so that capture-bound state never lands on
    // execModeSyncEvent_, which the eager path consumes live
    // (cudaEventSynchronize / cudaStreamWaitEvent) and which a captured record
    // would leave unusable. The cost is that nothing records
    // execModeSyncEvent_ during capture, so the eager-after-capture GPE host
    // barrier has no replay completion to order against; mixing=0 therefore
    // does not support eager submissions after a capture. See
    // NCCL_CTRAN_GRAPH_MIXING_SUPPORT in nccl_cvars.yaml.
    FB_CUDACHECK(cudaEventRecord(captureFenceEvent_, userStream));
  } else {
    // Record from a forked capture stream so the external event can order
    // later eager work and operations captured by a different graph.
    commResult_t innerRes = commSuccess;
    FB_CUDACHECK(
        sideStream_->fork_from(userStream, [&](cudaStream_t sideStream) {
          innerRes = ctran::utils::cudagraph::addEventRecordNodeToCapture(
              sideStream, captureInfo.g, execModeSyncEvent_, &lastRecordNode_);
        }));
    if (innerRes != commSuccess) {
      return innerRes;
    }
  }

  lastUserStream_ = userStream;
  return commSuccess;
}

} // namespace ctran::algos
