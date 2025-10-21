// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/synchronization/SaturatingSemaphore.h>

#include "comms/utils/colltrace/CollWaitEvent.h"

namespace meta::comms::colltrace {

class CPUWaitEvent : public ICollWaitEvent {
 public:
  CPUWaitEvent();

  ~CPUWaitEvent() override = default;

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
  struct InternalWaitEvent {
    folly::SaturatingSemaphore<> waitSemaphore;
    std::atomic<system_clock_time_point> timePoint;
  };

  system_clock_time_point enqueueTime_;

  InternalWaitEvent startEvent_;
  InternalWaitEvent endEvent_;
};

} // namespace meta::comms::colltrace
