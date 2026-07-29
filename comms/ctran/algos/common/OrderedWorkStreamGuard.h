// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_ALGOS_COMMON_ORDERED_WORK_STREAM_GUARD_H_
#define CTRAN_ALGOS_COMMON_ORDERED_WORK_STREAM_GUARD_H_

#include <cuda_runtime.h>

#include <memory>
#include <mutex>

#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/utils/GraphCaptureSideStream.h"
#include "comms/utils/commSpecs.h"

struct CommLogData;

namespace ctran::utils {

class OrderedWorkStreamGuard {
 public:
  ~OrderedWorkStreamGuard();

  // Captured graphs that reference this guard must be destroyed first.
  void init(
      const CommLogData& logMetaData,
      bool synchronizeEagerAfterCapturedWork);

  class Scope {
   public:
    Scope(
        OrderedWorkStreamGuard& guard,
        cudaStream_t userStream,
        const cudagraph::StreamCaptureInfo& captureInfo);
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
    cudagraph::StreamCaptureInfo captureInfo_;
    commResult_t status_{commSuccess};
    std::unique_lock<std::mutex> lock_;
  };

  Scope acquire(
      cudaStream_t userStream,
      const cudagraph::StreamCaptureInfo& captureInfo);

 private:
  commResult_t doAcquire(
      cudaStream_t userStream,
      const cudagraph::StreamCaptureInfo& captureInfo);
  commResult_t doRelease(
      cudaStream_t userStream,
      const cudagraph::StreamCaptureInfo& captureInfo);

  std::mutex submissionMutex_;
  cudaEvent_t execModeSyncEvent_{};
  unsigned long long lastCaptureId_{0};
  bool everCaptured_{false};
  bool synchronizeEagerAfterCapturedWork_{false};
  bool initialized_{false};
  commResult_t error_{commSuccess};
  cudaStream_t lastUserStream_{nullptr};
  cudaGraphNode_t lastRecordNode_{};
  std::unique_ptr<meta::comms::GraphSideStream> sideStream_;
  const CommLogData* logMetaData_{nullptr};
};

} // namespace ctran::utils

#endif
