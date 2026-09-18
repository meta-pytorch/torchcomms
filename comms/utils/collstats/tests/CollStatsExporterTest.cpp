// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/collstats/CollStatsExporter.h"

#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/collstats/CollStatsIdentity.h"
#include "comms/utils/collstats/CollStatsSnapshot.h"
#include "comms/utils/collstats/CollStatsTypes.h"

// Covers the transport only: the queue bound, the loss accounting and the
// drain. What a window turns into is the serializer's business and is covered
// by CollStatsJsonTest, so these tests pass a stub that makes each window
// identifiable without pulling in a wire format.

namespace meta::comms::collstats {
namespace {

// Upper bound on any wait for the background thread. Long enough that a loaded
// CI host never trips it, short enough that a genuine hang fails the test
// rather than the suite's own timeout.
constexpr std::chrono::seconds kTestWait{30};

CollStatsExportIdentity makeIdentity() {
  CollStatsExportIdentity id;
  id.processGroup = "dp";
  id.commHash = 0xabc;
  id.rank = 3;
  id.host = "host-1";
  id.gpu = 5;
  return id;
}

// Renders both halves of the serializer's input, so a test can tell which
// window it is looking at and that the identity reached the background thread.
std::string stubSerialize(
    const CollStatsExportIdentity& id,
    const CollStatSnapshot& snap) {
  return "rank=" + std::to_string(id.rank) +
      " epoch=" + std::to_string(snap.windowEpoch);
}

// Blocks until `read` reports at least `want`, so a test can assert on the
// background thread's counters without racing it. Bounded so a regression
// fails the test instead of hanging the suite.
bool waitForCount(const std::function<uint64_t()>& read, uint64_t want) {
  for (int i = 0; i < 2000; ++i) {
    if (read() >= want) {
      return true;
    }
    // The bounded poll the guidance asks for, not a sleep standing in for
    // synchronization. The counters are atomics the exporter thread bumps with
    // nothing to wait on; the loop is capped so a regression fails rather than
    // hangs.
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

// A snapshot with `windowEpoch` set and one occupied key, so each window is
// distinguishable.
CollStatSnapshot makeSnapshot(uint64_t epoch) {
  CollStatSnapshot snap;
  snap.numKeys = 1;
  snap.windowEpoch = epoch;
  snap.keys.assign(
      1,
      CollStatKey{
          CollStatOp::AllReduce,
          CollStatAlgo::Direct,
          CollStatProto::Unknown,
          7u,
          3u});
  snap.values.assign(2, CollStatValue{});
  snap.values[0].count = 1;
  return snap;
}

TEST(CollStatsExporterTest, ExportsEverySubmittedWindowToSink) {
  std::vector<std::string> captured; // written only on the bg thread
  uint64_t exported = 0;
  uint64_t dropped = 0;
  uint64_t failed = 0;
  {
    CollStatsExporter exporter(
        makeIdentity(), stubSerialize, [&captured](const std::string& blob) {
          captured.push_back(blob);
        });
    for (uint64_t e = 0; e < 5; ++e) {
      exporter.submit(makeSnapshot(e));
    }
    // Drains and joins, so the counters are final and the sink is done writing.
    exporter.shutdown();
    exported = exporter.exported();
    dropped = exporter.dropped();
    failed = exporter.failed();
  }
  const std::vector<std::string> expected = {
      "rank=3 epoch=0",
      "rank=3 epoch=1",
      "rank=3 epoch=2",
      "rank=3 epoch=3",
      "rank=3 epoch=4"};
  EXPECT_EQ(captured, expected);
  // Asserted, not incidental: without this the exported_ increment can be
  // deleted with every test still green.
  EXPECT_EQ(exported, 5u);
  EXPECT_EQ(dropped, 0u);
  EXPECT_EQ(failed, 0u);
}

TEST(CollStatsExporterTest, DropsAndCountsWhenQueueFull) {
  std::promise<void> entered;
  std::promise<void> release;
  // Shared, so the sink can wait on it without consuming the promise's single
  // future, and bounded, so a regression in submit/run fails this test instead
  // of hanging the suite on a thread that never reaches the sink.
  std::shared_future<void> enteredFuture = entered.get_future().share();
  std::shared_future<void> releaseFuture = release.get_future().share();
  std::atomic<bool> firstDone{false};

  uint64_t exportedAtCheck = 0;
  {
    CollStatsExporter exporter(
        makeIdentity(),
        stubSerialize,
        [&](const std::string&) {
          // Block the background thread inside the first sink call so the queue
          // cannot drain, making the overflow deterministic.
          if (!firstDone.exchange(true)) {
            entered.set_value();
            releaseFuture.wait_for(kTestWait);
          }
        },
        /*queueCapacity=*/2);

    exporter.submit(makeSnapshot(0)); // popped by bg thread, blocks in sink
    ASSERT_EQ(enteredFuture.wait_for(kTestWait), std::future_status::ready)
        << "background thread never reached the sink";

    exporter.submit(makeSnapshot(1)); // queue: [1]
    exporter.submit(makeSnapshot(2)); // queue: [1,2] -> full
    exporter.submit(makeSnapshot(3)); // dropped
    exporter.submit(makeSnapshot(4)); // dropped
    exporter.submit(makeSnapshot(5)); // dropped

    EXPECT_EQ(exporter.dropped(), 3u);
    exportedAtCheck = exporter.exported(); // s0 still in sink, not yet counted

    release.set_value(); // let the bg thread finish and drain 1 and 2
  }
  EXPECT_EQ(exportedAtCheck, 0u);
}

// A serializer that throws must cost one window, not the process: the throw
// happens on the background thread, where an escape calls std::terminate.
TEST(CollStatsExporterTest, CountsFailedWhenSerializerThrows) {
  std::atomic<int> sinkCalls{0};
  CollStatsExporter exporter(
      makeIdentity(),
      // The lambda must match the Serializer signature and return std::string;
      // [[noreturn]] is not spellable on a lambda, and throwing is the whole
      // point here.
      // NOLINTNEXTLINE(clang-diagnostic-missing-noreturn)
      [](const CollStatsExportIdentity&, const CollStatSnapshot&)
          -> std::string { throw std::runtime_error("serializer failed"); },
      [&sinkCalls](const std::string&) { ++sinkCalls; });
  exporter.submit(makeSnapshot(0));
  exporter.submit(makeSnapshot(1));

  ASSERT_TRUE(waitForCount([&] { return exporter.failed(); }, 2));
  EXPECT_EQ(exporter.exported(), 0u);
  EXPECT_EQ(exporter.dropped(), 0u);
  EXPECT_EQ(sinkCalls.load(), 0);
}

// An exporter with no serializer writes nothing and says so, rather than
// looking like a run that had no windows.
TEST(CollStatsExporterTest, CountsFailedWithNoSerializer) {
  std::atomic<int> sinkCalls{0};
  CollStatsExporter exporter(
      makeIdentity(),
      CollStatsExporter::Serializer{},
      [&sinkCalls](const std::string&) { ++sinkCalls; });
  exporter.submit(makeSnapshot(0));

  ASSERT_TRUE(waitForCount([&] { return exporter.failed(); }, 1));
  EXPECT_EQ(exporter.exported(), 0u);
  EXPECT_EQ(sinkCalls.load(), 0);
}

TEST(CollStatsExporterTest, FileSinkWritesOneLinePerWindow) {
  const std::string path =
      "/tmp/collstats_exporter_test_" + std::to_string(::getpid()) + ".jsonl";
  std::remove(path.c_str());
  {
    CollStatsExporter exporter(
        makeIdentity(), stubSerialize, collStatsMakeFileSink(path));
    exporter.submit(makeSnapshot(0));
    exporter.submit(makeSnapshot(1));
  }
  std::ifstream in(path);
  ASSERT_TRUE(in.is_open());
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  in.close();
  std::remove(path.c_str());

  const std::vector<std::string> expected = {
      "rank=3 epoch=0", "rank=3 epoch=1"};
  EXPECT_EQ(lines, expected);
}

// A write that fails after a successful open must be counted, not swallowed.
// std::ofstream reports write errors through the stream state rather than by
// throwing, so an unchecked sink would keep "succeeding" on a full disk and the
// windows would be counted as exported while reaching nothing.
//
// /dev/full opens cleanly and fails every write with ENOSPC, which is the
// failure mode being guarded without needing to actually fill a disk.
TEST(CollStatsExporterTest, FileSinkCountsFailedWhenWriteFails) {
  std::ifstream probe("/dev/full");
  if (!probe.is_open()) {
    GTEST_SKIP() << "/dev/full not available";
  }
  probe.close();

  CollStatsExporter::JsonSink sink = collStatsMakeFileSink("/dev/full");
  ASSERT_TRUE(static_cast<bool>(sink)) << "/dev/full should open";

  CollStatsExporter exporter(makeIdentity(), stubSerialize, std::move(sink));
  exporter.submit(makeSnapshot(0));

  ASSERT_TRUE(waitForCount([&] { return exporter.failed(); }, 1));
  EXPECT_EQ(exporter.exported(), 0u);
}

// An exporter with no sink is the other half of the "produced no output" pair:
// the serializer runs or not, but nothing receives the blob either way.
TEST(CollStatsExporterTest, CountsFailedWithNoSink) {
  std::atomic<int> serializerCalls{0};
  CollStatsExporter exporter(
      makeIdentity(),
      [&serializerCalls](
          const CollStatsExportIdentity&, const CollStatSnapshot&) {
        ++serializerCalls;
        return std::string{};
      },
      CollStatsExporter::JsonSink{});
  exporter.submit(makeSnapshot(0));

  ASSERT_TRUE(waitForCount([&] { return exporter.failed(); }, 1));
  EXPECT_EQ(exporter.exported(), 0u);
  // The absent sink is detected before the serializer runs, so a window costs
  // nothing to discover it cannot be written.
  EXPECT_EQ(serializerCalls.load(), 0);
}

// A path that cannot be opened yields a falsy sink rather than one that
// silently discards, so the owner can report the bad path instead of shipping a
// run whose zero windows look like a quiet one.
TEST(CollStatsExporterTest, FileSinkIsEmptyWhenOpenFails) {
  const CollStatsExporter::JsonSink sink =
      collStatsMakeFileSink("/nonexistent-dir-for-collstats-test/out.jsonl");
  EXPECT_FALSE(static_cast<bool>(sink));
}

// Teardown is time-boxed, and what it abandons is counted rather than lost
// quietly. The first sink call outlasts the budget by construction, so by the
// time the background thread looks at the deadline it has already passed and
// every remaining window is abandoned in one go.
TEST(CollStatsExporterTest, ShutdownAbandonsAndCountsPastTheDrainBudget) {
  std::promise<void> entered;
  std::shared_future<void> enteredFuture = entered.get_future().share();
  std::atomic<bool> firstDone{false};

  CollStatsExporter exporter(
      makeIdentity(),
      stubSerialize,
      [&](const std::string&) {
        if (!firstDone.exchange(true)) {
          entered.set_value();
          // Outlasts the budget passed to shutdown() below. The deadline is
          // only consulted between windows, so this call is not interrupted --
          // it is what pushes the clock past the deadline.
          // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
      },
      /*queueCapacity=*/8);

  exporter.submit(makeSnapshot(0)); // popped, blocks in the slow sink
  ASSERT_EQ(enteredFuture.wait_for(kTestWait), std::future_status::ready);
  for (uint64_t e = 1; e < 5; ++e) {
    exporter.submit(makeSnapshot(e)); // queued behind it
  }

  exporter.shutdown(std::chrono::milliseconds(20));

  EXPECT_EQ(exporter.exported(), 1u); // only the in-flight window got through
  EXPECT_EQ(exporter.dropped(), 4u); // the queued remainder, counted
  EXPECT_EQ(exporter.failed(), 0u);
}

} // namespace
} // namespace meta::comms::collstats
