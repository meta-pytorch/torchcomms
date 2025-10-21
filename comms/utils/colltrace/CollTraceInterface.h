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

  virtual CommsMaybe<std::shared_ptr<CollTraceHandle>> recordCollective(
      std::unique_ptr<ICollMetadata> metadata,
      std::unique_ptr<ICollWaitEvent> waitEvent) noexcept = 0;

  virtual ICollTracePlugin* getPluginByName(std::string name) noexcept = 0;

  virtual CommsMaybeVoid triggerEventState(
      CollTraceEvent& collEvent,
      CollTraceHandleTriggerState state) noexcept = 0;
};

} // namespace meta::comms::colltrace
