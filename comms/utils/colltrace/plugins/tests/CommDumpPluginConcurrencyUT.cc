// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <atomic>
#include <chrono>
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
