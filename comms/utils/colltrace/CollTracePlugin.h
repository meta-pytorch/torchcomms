// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

// Abstract class for interfaces to implement for plugin of colltrace.
// How the plugin is used:
// 1. The plugin is registered in the colltrace library.
// 2. afterCollRecorded runs once when the record is created. Each execution
// then triggers the remaining callbacks in order:
//    beforeCollKernelScheduled -> afterCollKernelScheduled ->
//    afterCollKernelStart -> [collEventProgressing] -> afterCollKernelEnd
// Eager afterCollRecorded, beforeCollKernelScheduled, and
// afterCollKernelScheduled callbacks run in the calling thread. Graph replay
// scheduling callbacks and the remaining lifecycle callbacks run in the
// CollTrace thread.
class ICollTracePlugin {
 public:
  virtual ~ICollTracePlugin() = default;

  // CollTrace isolates each callback: returned errors and thrown exceptions
  // are counted and logged, and do not prevent subsequent plugins from
  // running. Plugin failures must never affect collective progress.

  // Get the name of the current plugin
  virtual std::string_view getName() const noexcept = 0;

  // ----- Callbacks below will be triggered in the calling (main) thread -----

  // Callback that will be called after a collective record has been created
  // and any capture identity is available. Graph collectives trigger this
  // during capture, before any replay occurs.
  virtual CommsMaybeVoid afterCollRecorded(
      const CollTraceEvent& /* curEvent */) {
    return folly::unit;
  }

  // Callback that will be called before a collective is scheduled. For cuda
  // event based tracking, this function will be called after the cuda event is
  // inserted into the stream.
  virtual CommsMaybeVoid beforeCollKernelScheduled(
      const CollTraceEvent& curEvent) = 0;
  // Callback that will be called after a collective is scheduled. For cuda
  // event based tracking, this function will be called before the cuda event is
  // inserted into the stream.
  virtual CommsMaybeVoid afterCollKernelScheduled(
      const CollTraceEvent& curEvent) = 0;

  // ----- Callbacks below will be triggered in the colltrace thread -----

  virtual CommsMaybeVoid afterCollKernelStart(
      const CollTraceEvent& curEvent) = 0;

  // Every maxCheckCancelInterval, this function will be called for each plugin
  // to give the plugin a chance to perform some checks like async error and
  // timeout. This will happen both when waiting for the collective to start and
  // when waiting for the collective to end.
  virtual CommsMaybeVoid collEventProgressing(
      const CollTraceEvent& curEvent) = 0;

  virtual CommsMaybeVoid afterCollKernelEnd(const CollTraceEvent& curEvent) = 0;

  /*
   * Called instead of afterCollKernelEnd when tracking cannot reach normal
   * completion. It may be the first and only callback for an event when
   * tracking fails before scheduled/start delivery. Queue rejection and
   * supersession invoke this callback on the calling thread; poll-time loss
   * invokes it on the CollTrace thread; final teardown invokes it after that
   * thread is joined. Implementations must synchronize shared state and
   * tolerate duplicate notification.
   */
  virtual CommsMaybeVoid afterCollTerminated(
      CollTraceEvent& /* curEvent */,
      CollTraceTerminalReason /* reason */) noexcept {
    return folly::unit;
  }

  // Return the maximum number of past events this plugin needs to retain.
  // CollTrace uses max(all plugins) to size the shared GPU ring buffer to
  // at least 2x this value, ensuring no data loss under normal operation.
  // Default: 0 (no retention requirement — ring buffer uses its default size).
  virtual int64_t maxEventRetention() const noexcept {
    return 0;
  }
};

} // namespace meta::comms::colltrace
