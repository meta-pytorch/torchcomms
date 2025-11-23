// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include <folly/synchronization/Baton.h>

#include "comms/utils/CudaRAII.h"
#include "comms/utils/colltrace/CollWaitEvent.h"
#include "comms/utils/colltrace/CudaEventPool.h"

namespace meta::comms::colltrace {

// WARNING: The cudaEvent and cudaStream used by this class is intentionally
// NOT destroyed, this class is only intended to be used as global static
// variables. Creating too many of these objects might cause issue with cuda
class CudaReferencePoint {
  using system_clock_time_point = ICollWaitEvent::system_clock_time_point;

 public:
  CudaReferencePoint();

  CommsMaybe<system_clock_time_point> getTimeViaEvent(const CudaEvent& event);

 private:
  // We intentionally NOT using the RAII version to avoid segfault when calling
  // cudaEventDestory and cudaStreamDestroy during the program exit.
  cudaStream_t stream_{nullptr};
  cudaEvent_t event_{nullptr};
  system_clock_time_point time_;
};

class CudaWaitPoint {
 public:
  using system_clock_time_point = ICollWaitEvent::system_clock_time_point;

  enum class WaitPointType {
    start,
    end,
  };

  CudaWaitPoint(cudaStream_t stream, WaitPointType type);

  CommsMaybeVoid recordEvent() noexcept;

  CommsMaybe<bool> waitEventFinish(
      std::chrono::milliseconds sleepTimeMs) noexcept;

  CommsMaybe<system_clock_time_point> getEventFinishTime() noexcept;

  static CudaReferencePoint& getReferencePoint();

 private:
  // We do not own the stream, so we use raw pointer here.
  cudaStream_t recordStream_;
  WaitPointType type_;
  CachedCudaEvent event_;

  enum class CudaEventStatus {
    unrecorded = 0,
    recorded = 1,
    finished = 2,
  } eventStatus_{CudaEventStatus::unrecorded};

  std::string_view getWaitPointTypeString() const noexcept;

  static CommsMaybe<system_clock_time_point> getTimeViaReference(
      const CudaEvent& event);
};

class CudaWaitEvent : public ICollWaitEvent {
 public:
  CudaWaitEvent(cudaStream_t stream);

  ~CudaWaitEvent() = default;

  CommsMaybeVoid beforeCollKernelScheduled() noexcept override;

  CommsMaybeVoid afterCollKernelScheduled() noexcept override;

  CommsMaybe<bool> waitCollStart(
      std::chrono::milliseconds sleepTimeMs) noexcept override;

  CommsMaybe<bool> waitCollEnd(
      std::chrono::milliseconds sleepTimeMs) noexcept override;

  CommsMaybeVoid signalCollStart() noexcept override;

  CommsMaybeVoid signalCollEnd() noexcept override;

  CommsMaybe<system_clock_time_point> getCollEnqueueTime() noexcept override;

  CommsMaybe<system_clock_time_point> getCollStartTime() noexcept override;

  CommsMaybe<system_clock_time_point> getCollEndTime() noexcept override;

 private:
  system_clock_time_point enqueueTime_;

  CudaWaitPoint startWaitPoint_;
  CudaWaitPoint endWaitPoint_;
};

} // namespace meta::comms::colltrace
