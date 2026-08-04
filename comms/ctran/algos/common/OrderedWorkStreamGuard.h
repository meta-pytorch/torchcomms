// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <mutex>

#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/utils/GraphCaptureSideStream.h"
#include "comms/utils/commSpecs.h"

struct CommLogData;

namespace ctran::algos {

class OrderedWorkStreamGuard {
 public:
  ~OrderedWorkStreamGuard() noexcept;

  // Captured graphs that reference this guard must be destroyed first.
  void init(
      const CommLogData& logMetaData,
      bool synchronizeEagerAfterCapturedWork);

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
    Scope& operator=(Scope&& other) = delete;

    commResult_t status() const {
      return status_;
    }
    cudaStream_t stream() const {
      return userStream_;
    }
    commResult_t release();

   private:
    OrderedWorkStreamGuard* guard_;
    cudaStream_t userStream_;
    ctran::utils::cudagraph::StreamCaptureInfo captureInfo_;
    commResult_t status_{commSuccess};
    std::unique_lock<std::mutex> lock_;
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

  std::mutex submissionMutex_;
  cudaEvent_t execModeSyncEvent_{};
  // Fence recorded during capture. Separate from execModeSyncEvent_ so a
  // capture-absorbed record never taints the event the eager path consumes
  // live. Only ever waited on from within the same capture.
  cudaEvent_t captureFenceEvent_{};
  unsigned long long lastCaptureId_{0};
  bool everCaptured_{false};
  bool graphMixingSupport_{true};
  bool synchronizeEagerAfterCapturedWork_{false};
  bool initialized_{false};
  commResult_t error_{commSuccess};
  cudaStream_t lastUserStream_{nullptr};
  cudaGraphNode_t lastRecordNode_{};
  std::unique_ptr<meta::comms::GraphSideStream> sideStream_;
};

} // namespace ctran::algos
