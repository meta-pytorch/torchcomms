// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>

#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

/**
 * An abstract class that represents an event that waits for a collective.
 * Could be implemented differently based on the type of collective.
 */
class ICollWaitEvent {
 public:
  using system_clock_time_point = decltype(std::chrono::system_clock::now());

  virtual ~ICollWaitEvent() = default;

  /**
   * beforeCollKernel should be called before the collective kernel scheduled
   * for the event to have an opportunity to add CUDA event/callback/etc
   * before the collective kernel
   * This function should not throw any exception
   * @return returns proper error code if there is an error. Otherwise
   * folly:Unit (i.e. void).
   */
  virtual CommsMaybeVoid beforeCollKernelScheduled() noexcept = 0;

  /**
   * Similarly, afterCollKernelScheduled should be called after the collective
   * kernel being scheduled, ideally immediately after the collective kernel
   * being scheduled.
   * This function should not throw any exception
   * @return returns proper error code if there is an error. Otherwise
   * folly:Unit (i.e. void).
   */
  virtual CommsMaybeVoid afterCollKernelScheduled() noexcept = 0;

  /**
   * Blocking wait for the collective to start.
   * This function should not throw any exception
   * @param sleepTimeMs Time in millisecond for how long the program shall wait
   * before return
   * @return Success:  Returns true if finished within timeout, otherwise return
   *                   false
   *         Failures: Returns error code if encountered any error during call
   */
  virtual CommsMaybe<bool> waitCollStart(
      std::chrono::milliseconds sleepTimeMs) noexcept = 0;

  /**
   * Blocking wait for the collective to end.
   * This function should not throw any exception
   * @param sleepTimeMs Time in millisecond for how long the program shall wait
   * before return
   * @return Success:  Returns true if finished within timeout, otherwise return
   *                   false
   *         Failures: Returns error code if encountered any error during call
   */
  virtual CommsMaybe<bool> waitCollEnd(
      std::chrono::milliseconds sleepTimeMs) noexcept = 0;

  /**
   * Should be called when a collective starts.
   * @return Success:  Returns folly:Unit if successful.
   *         Failures: Returns the error when signaling collective start.
   */
  virtual CommsMaybeVoid signalCollStart() noexcept = 0;

  /**
   * Should be called when a collective ends.
   * @return Success:  Returns folly:Unit if successful.
   *         Failures: Returns the error when signaling collective end.
   */
  virtual CommsMaybeVoid signalCollEnd() noexcept = 0;

  virtual CommsMaybe<system_clock_time_point> getCollEnqueueTime() noexcept = 0;

  virtual CommsMaybe<system_clock_time_point> getCollStartTime() noexcept = 0;

  virtual CommsMaybe<system_clock_time_point> getCollEndTime() noexcept = 0;
};

} // namespace meta::comms::colltrace
