// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "comms/observatory/colltrace/LifecycleFeedTypes.h"
#include "comms/utils/colltrace/CollMetadata.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/CollTracePlugin.h"
#include "comms/utils/colltrace/CollWaitEvent.h"

namespace meta::comms::colltrace {

// Create an interface for colltrace for easy mocking and testing.
class ICollTrace {
 public:
  virtual ~ICollTrace() = default;

  // Describe a collective captured into a graph, by the id its replays
  // report. Empty when the id was not captured here, or its graph is gone.
  virtual std::optional<CapturedCollDescription> describeCapturedCollective(
      uint64_t /* capturedCollId */) noexcept {
    return std::nullopt;
  }

  // Record a collective event. If the waitEvent is a GraphCudaWaitEvent,
  // the collective is recorded for graph-mode polling. Otherwise it's
  // recorded as an eager collective.
  virtual CommsMaybe<std::shared_ptr<ICollTraceHandle>> recordCollective(
      std::unique_ptr<ICollMetadata> metadata,
      std::unique_ptr<ICollWaitEvent> waitEvent) noexcept = 0;

  virtual ICollTracePlugin* getPluginByName(std::string name) noexcept = 0;

  virtual CommsMaybeVoid triggerEventState(
      CollTraceEvent& collEvent,
      CollTraceHandleTriggerState state) noexcept = 0;

  virtual CommsMaybeVoid cancelEvent(CollTraceEvent& collEvent) noexcept = 0;

  // Request the poll thread to drain all pending events and return a
  // generation token. gen=0 is reserved as the no-op default; real
  // implementations start at 1 (via fetch_add(1)+1).
  virtual uint64_t requestFlush() noexcept {
    return 0;
  }

  // Block until the poll thread has completed the flush identified by gen.
  // Returns when completed >= gen OR the thread is cancelled. Callers
  // cannot distinguish the two — if the flush must have completed (e.g.
  // before dumping), check isThreadCancelled() separately.
  // MUST NOT be called from the poll thread (e.g. from a plugin callback).
  virtual void waitFlush(uint64_t /*gen*/) noexcept {}

  // The same wait, bounded. Returns whether the flush completed: false means
  // the deadline came first and whatever the poll thread had not yet published
  // is still unpublished.
  //
  // Separate from the unbounded overload rather than replacing it, because the
  // two have different callers. Something dumping state for an incident report
  // wants to block until it has everything. A telemetry drain runs at a step
  // boundary and during hang investigations, where a poll thread wedged in a
  // driver call would otherwise stall the trainer at the moment someone is
  // trying to find out why it is stuck -- and a late event is worth less there
  // than a prompt return.
  //
  // Defaults to reporting completion, matching the unbounded overload's default
  // of returning at once: an implementation with nothing to flush has nothing
  // to wait for.
  virtual bool waitFlush(
      uint64_t /*gen*/,
      std::chrono::nanoseconds /*timeout*/) noexcept {
    return true;
  }
};

} // namespace meta::comms::colltrace
