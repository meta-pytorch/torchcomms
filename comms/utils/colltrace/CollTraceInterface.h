// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/utils/colltrace/CollMetadata.h"
#include "comms/utils/colltrace/CollTraceHandle.h"
#include "comms/utils/colltrace/CollTracePlugin.h"
#include "comms/utils/colltrace/CollWaitEvent.h"

namespace meta::comms::colltrace {

// Create an interface for colltrace for easy mocking and testing.
class ICollTrace {
 public:
  virtual ~ICollTrace() = default;

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
};

} // namespace meta::comms::colltrace
