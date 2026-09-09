// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include <gtest/gtest.h>

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/colltrace/tests/MockTypes.h"

using namespace meta::comms::colltrace;

namespace {

CollTraceEvent createCollTraceEvent(uint64_t collId) {
  auto metadata = std::make_unique<MockCollMetadata>();
  auto collRecord = std::make_shared<CollRecord>(collId, std::move(metadata));

  CollTraceEvent event;
  event.collRecord = collRecord;
  return event;
}

} // namespace

// Verify that dump() returns an error when it cannot acquire the read lock
// within dumpLockAcquireTimeout, rather than dereferencing a null LockedPtr.
//
// Uses a 1ms dumpLockAcquireTimeout. One thread runs the colltrace lifecycle
// in a tight loop (acquiring wlock via afterCollKernelStart/End), while another
// thread calls dump() concurrently. Under contention with such a short timeout,
// the rlock in dump() will occasionally time out. Without the null check fix,
// this would dereference a null LockedPtr and crash.
TEST(CommDumpPluginConcurrencyTest, DumpReturnsErrorOnReadLockTimeout) {
  constexpr int kNumRuns = 3;

  for (int run = 0; run < kNumRuns; ++run) {
    CommDumpConfig config;
    config.dumpLockAcquireTimeout = std::chrono::milliseconds(1);
    auto plugin = std::make_unique<CommDumpPlugin>(config);

    std::atomic<bool> running{true};
    std::atomic<int> errorCount{0};
    std::atomic<uint64_t> collId{0};

    // Colltrace thread: runs lifecycle in a tight loop, holding wlock
    // frequently
    std::thread colltraceThread([&] {
      while (running.load(std::memory_order_relaxed)) {
        auto id = collId.fetch_add(1, std::memory_order_relaxed);
        auto ev = createCollTraceEvent(id);

        plugin->afterCollKernelScheduled(ev);
        plugin->afterCollKernelStart(ev);
        plugin->afterCollKernelEnd(ev);
      }
    });

    // Dump thread: calls dump() in a tight loop with the tiny timeout
    std::thread dumpThread([&] {
      while (running.load(std::memory_order_relaxed)) {
        auto result = plugin->dump();
        // Before the fix, a rlock timeout would dereference a null LockedPtr
        // and crash. After the fix, it returns an error.
        if (result.hasError()) {
          errorCount.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    running.store(false, std::memory_order_relaxed);

    colltraceThread.join();
    dumpThread.join();

    // If we get here without crashing, this run passed.
  }
}

TEST(CommDumpPluginConcurrencyTest, PollCallbackUsesBoundedLockWait) {
  CommDumpConfig config{
      .pollLockAcquireTimeout = std::chrono::milliseconds{10},
  };
  CommDumpPlugin plugin{config};
  auto event = createCollTraceEvent(1);
  ASSERT_TRUE(plugin.afterCollKernelScheduled(event).hasValue());

  std::promise<void> lockAcquired;
  auto lockAcquiredFuture = lockAcquired.get_future();
  std::promise<void> releaseLock;
  auto releaseLockFuture = releaseLock.get_future();
  std::thread reader([&] {
    plugin.testOnlyExecuteWithReadLock([&] {
      lockAcquired.set_value();
      releaseLockFuture.wait();
    });
  });
  if (lockAcquiredFuture.wait_for(std::chrono::seconds{1}) !=
      std::future_status::ready) {
    releaseLock.set_value();
    reader.join();
    ADD_FAILURE()
        << "Timed out waiting for the test reader to acquire the lock";
    return;
  }

  const auto start = std::chrono::steady_clock::now();
  auto result = plugin.afterCollKernelStart(event);
  const auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_TRUE(result.hasError());
  EXPECT_LT(elapsed, std::chrono::milliseconds{250});

  releaseLock.set_value();
  reader.join();

  auto dump = plugin.dump();
  ASSERT_TRUE(dump.hasValue());
  EXPECT_TRUE(dump.value().pendingColls.empty());
  EXPECT_TRUE(dump.value().currentColls.empty());
  ASSERT_EQ(dump.value().terminalColls.size(), 1);
  EXPECT_EQ(
      dump.value().terminalColls.front().reason,
      CollTraceTerminalReason::PluginContention);
  EXPECT_EQ(dump.value().pollLockTimeouts, 1);
}

TEST(CommDumpPluginConcurrencyTest, DeferredCleanupSaturationReconcilesState) {
  CommDumpPlugin plugin{CommDumpConfig{
      .pendingCollSize = 4,
      .currentCollSize = 4,
      .terminalCollSize = 1,
      .pollLockAcquireTimeout = std::chrono::milliseconds{10},
  }};
  auto event1 = createCollTraceEvent(1);
  auto event2 = createCollTraceEvent(2);
  auto event3 = createCollTraceEvent(3);
  for (auto* event : {&event1, &event2, &event3}) {
    ASSERT_TRUE(plugin.afterCollKernelScheduled(*event).hasValue());
  }

  std::promise<void> lockAcquired;
  auto lockAcquiredFuture = lockAcquired.get_future();
  std::promise<void> releaseLock;
  auto releaseLockFuture = releaseLock.get_future();
  std::thread reader([&] {
    plugin.testOnlyExecuteWithReadLock([&] {
      lockAcquired.set_value();
      releaseLockFuture.wait();
    });
  });
  if (lockAcquiredFuture.wait_for(std::chrono::seconds{1}) !=
      std::future_status::ready) {
    releaseLock.set_value();
    reader.join();
    ADD_FAILURE()
        << "Timed out waiting for the test reader to acquire the lock";
    return;
  }

  for (auto* event : {&event1, &event2, &event3}) {
    EXPECT_TRUE(plugin.afterCollKernelStart(*event).hasError());
  }
  releaseLock.set_value();
  reader.join();

  auto dump = plugin.dump();
  ASSERT_TRUE(dump.hasValue());
  EXPECT_TRUE(dump.value().pendingColls.empty());
  EXPECT_TRUE(dump.value().currentColls.empty());
  EXPECT_EQ(dump.value().terminalColls.size(), 1);
  EXPECT_EQ(
      dump.value().terminalReasonCounts[static_cast<std::size_t>(
          CollTraceTerminalReason::PluginContention)],
      3);
  EXPECT_EQ(dump.value().terminalTransitionDrops, 2);
  EXPECT_EQ(dump.value().pollLockTimeouts, 3);
}
