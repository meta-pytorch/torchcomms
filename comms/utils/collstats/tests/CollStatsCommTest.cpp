// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/collstats/CollStatsComm.h"
#include "comms/utils/collstats/CollStatsExporter.h"
#include "comms/utils/collstats/CollStatsIdentity.h"
#include "comms/utils/collstats/CollStatsKeyRegistry.h"

namespace meta::comms::collstats {
namespace {

CollStatsExportIdentity makeIdentity() {
  CollStatsExportIdentity id;
  id.rank = 3;
  return id;
}

std::string stubSerialize(
    const CollStatsExportIdentity& id,
    const CollStatSnapshot& snap) {
  return "rank=" + std::to_string(id.rank) +
      " epoch=" + std::to_string(snap.windowEpoch);
}

CollStatSnapshot makeSnapshot(uint64_t epoch) {
  CollStatSnapshot snap;
  snap.windowEpoch = epoch;
  return snap;
}

// No device block and no driver: shutdown() touches neither, and an empty
// owner frees nothing. This keeps the teardown accounting testable on a host
// without a GPU, which is where the ordering bug lived.
std::unique_ptr<CollStatsComm> makeComm(
    std::unique_ptr<CollStatsExporter> exporter) {
  return std::make_unique<CollStatsComm>(
      CollStatsDeviceBlockOwner{},
      std::make_unique<CollStatsKeyRegistry>(/*capacity=*/8),
      makeIdentity(),
      std::move(exporter),
      /*driver=*/nullptr);
}

} // namespace

// The exporter is drained before its counters are read, so the totals cover
// windows still queued at teardown. Reading first reports them as never
// exported -- and the driver's final flush makes the trailing window exactly
// such a window on every real teardown.
TEST(CollStatsCommTest, ShutdownDrainsTheExporterBeforeReadingCounters) {
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();

  auto exporter = std::make_unique<CollStatsExporter>(
      makeIdentity(), stubSerialize, [gate](const std::string&) {
        // Hold the background thread until the test lets go, so every window
        // is still queued when shutdown() is entered.
        gate.wait();
      });
  CollStatsExporter* raw = exporter.get();
  auto comm = makeComm(std::move(exporter));

  for (uint64_t e = 0; e < 4; ++e) {
    raw->submit(makeSnapshot(e));
  }
  release.set_value();

  const CollStatsComm::Totals totals = comm->shutdown();
  EXPECT_TRUE(totals.hasExporter);
  EXPECT_EQ(totals.exported, 4u);
  EXPECT_EQ(totals.dropped, 0u);
  EXPECT_EQ(totals.failed, 0u);
}

TEST(CollStatsCommTest, SecondShutdownReturnsZeroedTotals) {
  auto exporter = std::make_unique<CollStatsExporter>(
      makeIdentity(), stubSerialize, [](const std::string&) {});
  CollStatsExporter* raw = exporter.get();
  auto comm = makeComm(std::move(exporter));
  raw->submit(makeSnapshot(0));

  const CollStatsComm::Totals first = comm->shutdown();
  EXPECT_TRUE(first.hasExporter);
  EXPECT_EQ(first.exported, 1u);

  const CollStatsComm::Totals second = comm->shutdown();
  EXPECT_FALSE(second.hasExporter);
  EXPECT_EQ(second.exported, 0u);
}

// Uninstrumented launches are reported even with no exporter attached: a run
// that instrumented nothing must be distinguishable from one that exported
// nothing.
TEST(CollStatsCommTest, ReportsUninstrumentedLaunchesWithoutAnExporter) {
  auto comm = makeComm(/*exporter=*/nullptr);
  comm->noteUninstrumentedLaunch();
  comm->noteUninstrumentedLaunch();

  const CollStatsComm::Totals totals = comm->shutdown();
  EXPECT_FALSE(totals.hasExporter);
  EXPECT_EQ(totals.uninstrumented, 2u);
}

} // namespace meta::comms::collstats
