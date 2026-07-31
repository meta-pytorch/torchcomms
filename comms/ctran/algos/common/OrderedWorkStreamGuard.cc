// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/common/OrderedWorkStreamGuard.h"

#include <utility>

#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/utils/logger/LogUtils.h"

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

  // Publish the initialized state only after every resource is ready.
  synchronizeEagerAfterCapturedWork_ = synchronizeEagerAfterCapturedWork;
  execModeSyncEvent_ = execModeSyncEvent;
  sideStream_ = std::move(sideStream);
  initialized_ = true;
}

OrderedWorkStreamGuard::~OrderedWorkStreamGuard() noexcept {
  if (!initialized_) {
    return;
  }
  FB_CUDACHECKIGNORE(cudaEventDestroy(execModeSyncEvent_));
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

  auto doWait = [&]() -> commResult_t {
    FB_CUDACHECK(cudaStreamWaitEvent(
        userStream,
        execModeSyncEvent_,
        isCapturing ? cudaEventWaitExternal : cudaEventWaitDefault));
    return commSuccess;
  };

  if (lastUserStream_ == nullptr) {
    if (isCapturing) {
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
    // A later operation captured into the same graph must depend on the
    // prior record node even if CUDA cannot infer a dependency across streams.
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

  return doWait();
}

commResult_t OrderedWorkStreamGuard::doRelease(
    cudaStream_t userStream,
    const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo) {
  const bool isCapturing = captureInfo.status == cudaStreamCaptureStatusActive;

  if (!isCapturing) {
    FB_CUDACHECK(cudaEventRecord(execModeSyncEvent_, userStream));
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
