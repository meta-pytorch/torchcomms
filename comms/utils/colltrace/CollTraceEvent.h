// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>

#include <folly/dynamic.h>

#include "comms/utils/colltrace/CollRecord.h"
#include "comms/utils/colltrace/CollWaitEvent.h"

namespace meta::comms::colltrace {

// Should be accessible by CollTrace and all the plugin callbacks
struct CollTraceEvent {
  // Other plugin might also want to keep a pointer to the event, so we use
  // shared pointer here
  std::shared_ptr<CollRecord> collRecord;
  std::unique_ptr<ICollWaitEvent> waitEvent; // How we wait the event
};

} // namespace meta::comms::colltrace
