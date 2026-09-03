// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsComm.h"

#include <utility>

namespace meta::comms::collstats {

CollStatsComm::CollStatsComm(
    CollStatsDeviceBlockOwner block,
    std::unique_ptr<CollStatsKeyRegistry> keys,
    CollStatsExportIdentity identity,
    std::unique_ptr<CollStatsExporter> exporter,
    std::unique_ptr<CollStatsReadoutDriver> driver,
    CollStatSizeClasses sizeClasses)
    : block_(std::move(block)),
      keys_(std::move(keys)),
      identity_(std::move(identity)),
      exporter_(std::move(exporter)),
      driver_(std::move(driver)),
      sizeClasses_(sizeClasses) {}

CollStatsComm::~CollStatsComm() {
  shutdown();
}

CollStatsComm::Totals CollStatsComm::shutdown() {
  Totals t;
  if (shutdownDone_) {
    return t;
  }
  shutdownDone_ = true;

  t.uninstrumented = uninstrumented_.load(std::memory_order_relaxed);
  // Release the driver first: its teardown flush issues and harvests the
  // trailing window, submitting it to the exporter. Then drain the exporter and
  // read its counters, so the totals describe every window it was handed --
  // including the ones the drain deadline abandoned. The block owner outlives
  // both and frees last.
  driver_.reset();
  if (exporter_) {
    t.hasExporter = true;
    // Explicitly, before the read: shutdown() charges abandoned windows to
    // dropped(), and leaving that to ~CollStatsExporter would bill them after
    // the only read of the counter, reporting every teardown loss as zero.
    exporter_->shutdown();
    t.exported = exporter_->exported();
    t.dropped = exporter_->dropped();
    t.failed = exporter_->failed();
    exporter_.reset();
  }
  return t;
}

uint32_t CollStatsComm::resolveKeyId(const CollStatKey& key) {
  if (!keys_) {
    return 0;
  }
  return keys_->resolve(key);
}

void CollStatsComm::onCollective(cudaStream_t stream) {
  if (driver_) {
    driver_->onCollective(stream);
  }
}

} // namespace meta::comms::collstats
