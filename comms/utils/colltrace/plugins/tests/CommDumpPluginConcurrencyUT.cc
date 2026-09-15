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

class ScopedReadLockHolder {
 public:
  explicit ScopedReadLockHolder(CommDumpPlugin& plugin)
      : lockAcquiredFuture_(lockAcquired_.get_future()),
        releaseLockFuture_(releaseLock_.get_future()),
        reader_([this, &plugin] {
          plugin.testOnlyExecuteWithReadLock([this] {
            lockAcquired_.set_value();
            releaseLockFuture_.wait();
          });
        }) {
    acquired_ = lockAcquiredFuture_.wait_for(std::chrono::seconds{1}) ==
        std::future_status::ready;
  }

  ~ScopedReadLockHolder() {
    release();
  }

  bool acquired() const {
    return acquired_;
  }

  void release() {
    if (!released_) {
      releaseLock_.set_value();
      released_ = true;
    }
    if (reader_.joinable()) {
      reader_.join();
    }
  }

 private:
  std::promise<void> lockAcquired_;
  std::future<void> lockAcquiredFuture_;
  std::promise<void> releaseLock_;
  std::future<void> releaseLockFuture_;
  std::thread reader_;
  bool acquired_{false};
  bool released_{false};
};

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

  ScopedReadLockHolder lockHolder{plugin};
  ASSERT_TRUE(lockHolder.acquired());

  const auto start = std::chrono::steady_clock::now();
  auto result = plugin.afterCollKernelStart(event);
  const auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_TRUE(result.hasError());
  EXPECT_LT(elapsed, std::chrono::seconds{1});

  lockHolder.release();

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
      .deferredTerminalSize = 1,
      .pollLockAcquireTimeout = std::chrono::milliseconds{10},
  }};
  auto event1 = createCollTraceEvent(1);
  auto event2 = createCollTraceEvent(2);
  auto event3 = createCollTraceEvent(3);
  for (auto* event : {&event1, &event2, &event3}) {
    ASSERT_TRUE(plugin.afterCollKernelScheduled(*event).hasValue());
  }

  ScopedReadLockHolder lockHolder{plugin};
  ASSERT_TRUE(lockHolder.acquired());

  for (auto* event : {&event1, &event2, &event3}) {
    EXPECT_TRUE(plugin.afterCollKernelStart(*event).hasError());
  }
  lockHolder.release();

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

TEST(CommDumpPluginConcurrencyTest, DeferredTerminalCannotReturnToPending) {
  CommDumpPlugin plugin{CommDumpConfig{
      .pendingCollSize = 4,
      .currentCollSize = 4,
      .terminalCollSize = 2,
      .pendingDrainBatchSize = 1,
      .pollLockAcquireTimeout = std::chrono::milliseconds{10},
  }};
  auto event1 = createCollTraceEvent(1);
  auto event2 = createCollTraceEvent(2);
  ASSERT_TRUE(plugin.afterCollKernelScheduled(event1).hasValue());
  ASSERT_TRUE(plugin.afterCollKernelScheduled(event2).hasValue());

  ScopedReadLockHolder lockHolder{plugin};
  ASSERT_TRUE(lockHolder.acquired());

  EXPECT_TRUE(
      plugin
          .afterCollTerminated(event2, CollTraceTerminalReason::TraceDestroyed)
          .hasError());
  lockHolder.release();

  EXPECT_TRUE(plugin.afterCollKernelStart(event1).hasValue());
  auto dump = plugin.dump();
  ASSERT_TRUE(dump.hasValue());
  EXPECT_TRUE(dump.value().pendingColls.empty());
  ASSERT_EQ(dump.value().currentColls.size(), 1);
  EXPECT_EQ(dump.value().currentColls.front().get(), event1.collRecord.get());
  ASSERT_EQ(dump.value().terminalColls.size(), 1);
  EXPECT_EQ(
      dump.value().terminalColls.front().collRecord.get(),
      event2.collRecord.get());
}

TEST(CommDumpPluginConcurrencyTest, BoundedDrainDoesNotDuplicateCurrentRecord) {
  CommDumpPlugin plugin{CommDumpConfig{
      .pendingCollSize = 4,
      .currentCollSize = 4,
      .pendingDrainBatchSize = 1,
  }};
  auto event1 = createCollTraceEvent(1);
  auto event2 = createCollTraceEvent(2);
  ASSERT_TRUE(plugin.afterCollKernelScheduled(event1).hasValue());
  ASSERT_TRUE(plugin.afterCollKernelScheduled(event2).hasValue());

  EXPECT_TRUE(plugin.afterCollKernelStart(event2).hasError());
  EXPECT_TRUE(plugin.afterCollKernelStart(event1).hasValue());

  auto dump = plugin.dump();
  ASSERT_TRUE(dump.hasValue());
  EXPECT_TRUE(dump.value().pendingColls.empty());
  ASSERT_EQ(dump.value().currentColls.size(), 2);
  EXPECT_EQ(dump.value().currentColls[0].get(), event2.collRecord.get());
  EXPECT_EQ(dump.value().currentColls[1].get(), event1.collRecord.get());
  EXPECT_EQ(dump.value().activeRecords.size(), 2);
}
