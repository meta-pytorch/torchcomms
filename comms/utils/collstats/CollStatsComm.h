// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include "comms/utils/collstats/CollStatsDeviceBlock.h"
#include "comms/utils/collstats/CollStatsExporter.h"
#include "comms/utils/collstats/CollStatsHistogram.h"
#include "comms/utils/collstats/CollStatsIdentity.h"
#include "comms/utils/collstats/CollStatsKeyRegistry.h"
#include "comms/utils/collstats/CollStatsReadoutDriver.h"
#include "comms/utils/collstats/CollStatsTypes.h"

// Everything one communicator's collective-stats instrumentation owns: the
// device block, the key registry that indexes it, the export identity, the
// out-of-band exporter and the readout driver.
//
// There is exactly one instance per *logical* communicator, not one per
// collective implementation. Two would mean two device blocks, two epochs and
// two readout cadences reported under a single commHash, which no backend join
// could untangle -- so ownership sits on the communicator rather than on any
// object that runs collectives.
//
// Deliberately free of the cvars and of the logger: the owner reads
// configuration, chooses the sink and does the reporting, then hands the
// assembled pieces here. That keeps this directory light enough for a consumer
// that has neither to hold one.

namespace meta::comms::collstats {

class CollStatsComm {
 public:
  // Totals worth surfacing once at teardown, returned by shutdown(). A run that
  // dropped or failed every window otherwise looks exactly like one that had
  // nothing to write.
  struct Totals {
    uint64_t exported{0};
    uint64_t dropped{0};
    uint64_t failed{0};
    // Collectives that ran on a path instrumentation does not cover, so a run
    // with missing collectives can be explained rather than silently short.
    uint64_t uninstrumented{0};
    bool hasExporter{false};
  };

  // `block` must be non-empty and `keys` non-null. `exporter` is null when no
  // output directory is configured, in which case `driver`'s sink does its own
  // reporting. `driver` must already be wired to whichever sink the owner
  // chose.
  CollStatsComm(
      CollStatsDeviceBlockOwner block,
      std::unique_ptr<CollStatsKeyRegistry> keys,
      CollStatsExportIdentity identity,
      std::unique_ptr<CollStatsExporter> exporter,
      std::unique_ptr<CollStatsReadoutDriver> driver,
      CollStatSizeClasses sizeClasses = collStatDefaultSizeClasses());

  // Calls shutdown() if the owner did not.
  ~CollStatsComm();

  // Tears down in dependency order and returns the totals to report.
  //
  // The order is load-bearing, and is why this is not a plain accessor plus
  // member destruction: the driver's final flush submits the trailing window
  // into the exporter, so the counters must be read *after* the driver is gone
  // and *after* the exporter has drained and joined.
  //
  // Reading before the driver is released undercounts by the trailing window.
  // Reading before the drain undercounts by everything still queued -- which
  // includes that same trailing window, since the flush only just submitted it
  // -- and reports any window the drain deadline abandoned as zero.
  //
  // Idempotent; a second call returns zeroed totals.
  Totals shutdown();

  CollStatsComm(const CollStatsComm&) = delete;
  CollStatsComm& operator=(const CollStatsComm&) = delete;
  CollStatsComm(CollStatsComm&&) = delete;
  CollStatsComm& operator=(CollStatsComm&&) = delete;

  // The device-side handle the kernels reach through the per-comm device state.
  const CollStatsDeviceBlockHandle& handle() const {
    return block_.handle();
  }

  // A driver that failed to acquire its CUDA resources is born disabled: it
  // exists, but it will never flip an epoch, so every window accumulates into a
  // bank nobody reads. Reporting that as enabled makes the launch path
  // instrument collectives whose numbers can never be exported, and the only
  // surviving symptom is a zero export count at teardown.
  bool enabled() const {
    return driver_ != nullptr && !driver_->disabled();
  }

  const CollStatsExportIdentity& identity() const {
    return identity_;
  }

  // Resolve a collective's key to the dense value slot its kernel accumulates
  // into. Called on the enqueue thread before the launch, so the device
  // performs no lookup of its own.
  uint32_t resolveKeyId(const CollStatKey& key);

  /* The configured size-class edges, so the launch path buckets a message size
   * the same way the exported rows are labelled. */
  const CollStatSizeClasses& sizeClasses() const {
    return sizeClasses_;
  }

  // Tick the pipelined readout after an instrumented collective was launched on
  // `stream`. Never blocks.
  void onCollective(cudaStream_t stream);

  void noteUninstrumentedLaunch() {
    uninstrumented_.fetch_add(1, std::memory_order_relaxed);
  }

 private:
  CollStatsDeviceBlockOwner block_;
  std::unique_ptr<CollStatsKeyRegistry> keys_;
  CollStatsExportIdentity identity_;

  // Bumped from the launch path, which may be more than one thread, and read
  // once at teardown.
  std::atomic<uint64_t> uninstrumented_{0};

  std::unique_ptr<CollStatsExporter> exporter_;
  std::unique_ptr<CollStatsReadoutDriver> driver_;
  CollStatSizeClasses sizeClasses_;

  bool shutdownDone_{false};
};

} // namespace meta::comms::collstats
