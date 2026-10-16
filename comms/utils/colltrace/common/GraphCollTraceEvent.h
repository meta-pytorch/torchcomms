// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Typed event for the graph-initiated colltrace ring buffer.
// Replaces manual bit-packing into uint64_t.

#pragma once

#include <cstdint>

namespace meta::comms::colltrace {

enum class GraphCollTracePhase : uint8_t {
  kStart = 0,
  kEnd = 1,
};

struct GraphCollTraceEvent {
  uint32_t collId;
  GraphCollTracePhase phase;
};

} // namespace meta::comms::colltrace
