// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <optional>

namespace meta::comms::colltrace {

struct CollTraceCapabilityStats {
  // A stop has been asked for -- not a liveness signal. The only writers of
  // the underlying flag are the destructor and a failed thread setup, so a
  // poller wedged in a plugin callback or blocked on the dump lock still
  // reports false here. Named for what it can actually observe.
  bool pollerStopRequested{false};
  bool graphTracingRequested{false};
  bool graphTracingSupported{false};
  bool graphRingAllocated{false};
  // Only says an anchor was taken. Nothing clears it, and the periodic
  // re-anchor is off by default, so it cannot report a drifted anchor -- it
  // distinguishes a degraded startup, not ongoing calibration quality.
  bool gpuClockCalibrationAvailable{false};
  bool lifecycleSubscriberAttached{false};
  bool commDumpSubscriberAttached{false};
  // Latching: set once a terminal transition has been lost, and never cleared
  // for the life of the process. A monitor bound to this is reporting history.
  bool commDumpEverDroppedTerminal{false};
  // Transient: this snapshot could not take the dump lock within the
  // configured timeout, so the commDump counters below are whatever could be
  // read without it rather than a current reading. Kept separate from the
  // latching bit above so one blip is not indistinguishable from a real loss.
  bool commDumpSnapshotStale{false};
};

struct CollTraceCoreStats {
  // These counters describe overlapping observations, not disjoint loss
  // buckets. For example, one overwritten ring entry can also produce a
  // start-without-end anomaly and a superseded terminal disposition.
  uint64_t graphRingOverwriteCount{0};
  uint64_t unmappedGraphEventCount{0};
  uint64_t graphStartWithoutEndCount{0};
  uint64_t graphEndWithoutStartCount{0};
  uint64_t supersededEnqueueCount{0};
  uint64_t pendingTraceQueueFullCount{0};
  uint64_t pluginErrorCount{0};
};

struct CollTraceLifecycleStats {
  uint64_t latestAssignedSequence{0};
  uint64_t highestDrainedSequence{0};
  uint64_t droppedEventCount{0};
  std::optional<uint64_t> lowestDroppedSequence;
  std::optional<uint64_t> highestDroppedSequence;
  uint64_t depth{0};
  uint64_t highWaterMark{0};
};

struct CollTraceCommDumpStats {
  uint64_t queueRejectedCount{0};
  uint64_t supersededBeforeScheduleCount{0};
  uint64_t trackingOverflowCount{0};
  uint64_t graphDestroyedCount{0};
  uint64_t traceDestroyedCount{0};
  uint64_t pluginContentionCount{0};
  uint64_t terminalTransitionDropCount{0};
  uint64_t pollLockTimeoutCount{0};
};

struct CollTraceStats {
  static constexpr uint32_t kSchemaVersion{1};

  uint32_t schemaVersion{kSchemaVersion};
  // Fields are sampled independently, so a live snapshot can span concurrent
  // updates. Quiescing the producers settles `core` and `lifecycle`, but not
  // `commDump`: collecting stats does not drain the plugin's deferred terminal
  // dispositions, and nothing else drains them once the comm goes idle, so
  // those counters can sit below the truth indefinitely. `pollLockTimeoutCount`
  // moving, or `commDumpSnapshotStale`, is the signal that this has happened.
  CollTraceCapabilityStats capabilities;
  CollTraceCoreStats core;
  CollTraceLifecycleStats lifecycle;
  CollTraceCommDumpStats commDump;
};

} // namespace meta::comms::colltrace
