// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "comms/utils/colltrace/CollTraceEvent.h"

namespace meta::comms::colltrace {

class GraphCudaWaitEvent;
class ICollTraceHandle;

// Tracking entry for a graph-captured collective.
struct GraphCollectiveEntry {
  // Owned by the CollTraceEvent below (unique_ptr), which we also own.
  GraphCudaWaitEvent* graphWaitEvent;
  std::unique_ptr<CollTraceEvent> event;
  // Weak reference to the handle returned to the caller. Used to invalidate
  // the handle when the CUDA graph is destroyed, preventing use-after-free
  // of the raw GraphCudaWaitEvent pointer held by the handle.
  std::weak_ptr<ICollTraceHandle> handle;
};

// Per-graph coordinator for all graph-captured collectives. Manages CUDA
// resources for cleanup when the graph is destroyed.
//
// Timestamp streams and dependency events are per-collective (owned by
// GraphCudaWaitEvent) so concurrent collectives don't serialize.
//
// The ring buffer and write index are owned by the CollTrace instance and
// shared across ALL graphs. Each collective atomically claims a slot during
// replay so events are interleaved but never lost (as long as the poll thread
// keeps up within ringSize replays).
//
// Ref-counted via shared_ptr — the CUDA graph destruction callback holds a
// copy to keep this alive until it can set the released flag.
struct GraphCollTraceState {
  // Set by the CUDA graph destruction callback. The poll thread checks this
  // to detect when a graph has been destroyed and stop tracking it.
  std::atomic_bool graph_destructed{false};
  // All collectives captured in this graph, indexed by collId.
  // Populated during graph capture, read by the poll thread.
  std::unordered_map<uint32_t, GraphCollectiveEntry> collectives;

  GraphCollTraceState() = default;
  ~GraphCollTraceState() = default;
  GraphCollTraceState(const GraphCollTraceState&) = delete;
  GraphCollTraceState& operator=(const GraphCollTraceState&) = delete;
  GraphCollTraceState(GraphCollTraceState&&) = delete;
  GraphCollTraceState& operator=(GraphCollTraceState&&) = delete;
};

} // namespace meta::comms::colltrace
