// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <folly/MPMCQueue.h>
#include <folly/Synchronized.h>
#include <folly/container/F14Set.h>

#include "comms/utils/colltrace/CollTracePlugin.h"

namespace meta::comms::logger {
class CommsSpdlogLogger;
}

namespace meta::comms::colltrace {

struct CommDumpConfig {
  // Default size of queue storing past collective communication operations.
  // The default value should be enough for debugging past collective. For slow
  // rank detection, we might need to increase the size.
  static constexpr int64_t kDefaultPastQueueSize{20};
  // 1024 elements should be sufficiently large to handle the number of
  // collective calls that is hanging when we dump the trace.
  static constexpr int kCommDumpQueueSize = 1024;
  static constexpr int kCurrentCollQueueSize = 1024;
  static constexpr int kTerminalQueueSize = 128;
  static constexpr int kTerminalIdQueueSize =
      2 * kCommDumpQueueSize + kTerminalQueueSize;
  // Default timeout for waiting for the lock to be acquired for dump. We don't
  // want to block the dump thread if there is any issue with the lock.
  static constexpr auto kDumpLockAcquireTimeout = std::chrono::seconds(1);

  std::string loggerName{"comms"};

  // Configures the size of the queue for past collective operations
  // (defaults to kDefaultPastQueueSize). They will be dumped as pastColls when
  // dump() is called.
  int64_t pastCollSize{kDefaultPastQueueSize};
  // Configures the size of the queue for pending collective operations
  // (defaults to kCommDumpQueueSize). Any further collective operations will be
  // dropped if the queue is full.
  int64_t pendingCollSize{kCommDumpQueueSize};
  /*
   * Active-operation capacity is independent of producer queue capacity.
   * Tuning enqueue backpressure must not discard already-started operations.
   */
  int64_t currentCollSize{kCurrentCollQueueSize};
  int64_t terminalCollSize{kTerminalQueueSize};
  /*
   * The tombstone window is finite by design: a callback delayed beyond this
   * window may be counted again, but cannot make retained plugin state
   * unbounded. Normalization keeps every retained terminal record covered.
   */
  int64_t terminalCollIdSize{kTerminalIdQueueSize};
  int64_t deferredTerminalSize{kTerminalQueueSize};
  int64_t pendingDrainBatchSize{0};
  std::chrono::milliseconds pollLockAcquireTimeout{
      std::chrono::milliseconds{1}};
  std::chrono::milliseconds dumpLockAcquireTimeout{kDumpLockAcquireTimeout};
};

struct CollRecordGreaterCollId {
  bool operator()(
      const std::shared_ptr<CollRecord>& a,
      const std::shared_ptr<CollRecord>& b) const {
    return a->getCollId() > b->getCollId();
  }
};

using PastCollsHeap = std::priority_queue<
    std::shared_ptr<CollRecord>,
    std::vector<std::shared_ptr<CollRecord>>,
    CollRecordGreaterCollId>;

struct TerminalCollRecord {
  std::shared_ptr<CollRecord> collRecord;
  CollTraceTerminalReason reason{CollTraceTerminalReason::Count};
};

constexpr auto kNumTerminalReasons =
    static_cast<std::size_t>(CollTraceTerminalReason::Count);

struct CollTraceDump {
  PastCollsHeap pastCollsHeap;
  std::deque<std::shared_ptr<CollRecord>> pastColls;
  std::deque<std::shared_ptr<CollRecord>> currentColls;
  std::deque<std::shared_ptr<CollRecord>> pendingColls;
  std::deque<TerminalCollRecord> terminalColls;
  std::deque<uint64_t> terminalCollIdOrder;
  folly::F14FastSet<uint64_t> terminalCollIds;
  folly::F14FastSet<const CollRecord*> activeRecords;
  std::array<uint64_t, kNumTerminalReasons> terminalReasonCounts{};
  uint64_t terminalTransitionDrops{0};
  uint64_t pollLockTimeouts{0};

  int64_t currentIteration{-1};
  int64_t currentIterationCommTimeUs{0};
  int64_t iterationCutoffUs{0};
};

struct IterationCommTime {
  int64_t iteration{-1};
  int64_t commTimeUs{0};
};

class CommDumpPlugin : public ICollTracePlugin {
 public:
  CommDumpPlugin(CommDumpConfig config = {});

  std::string_view getName() const noexcept override;

  CommsMaybeVoid beforeCollKernelScheduled(
      const CollTraceEvent& curEvent) override;

  CommsMaybeVoid afterCollKernelScheduled(
      const CollTraceEvent& curEvent) override;

  CommsMaybeVoid afterCollKernelStart(const CollTraceEvent& curEvent) override;

  CommsMaybeVoid collEventProgressing(const CollTraceEvent& curEvent) override;

  CommsMaybeVoid afterCollKernelEnd(const CollTraceEvent& curEvent) override;

  CommsMaybeVoid afterCollTerminated(
      CollTraceEvent& curEvent,
      CollTraceTerminalReason reason) noexcept override;
  void collectStats(CollTraceStats& stats) const override;

  int64_t maxEventRetention() const noexcept override;

  // CommDump specific API, supposed to be called by the dump (user) thread
  CommsMaybe<CollTraceDump> dump() noexcept;

  IterationCommTime getCurrentIterationCommTime() const noexcept;

  // For testing purpose only. This API is NOT thread safe! Clears all the
  // recorded colls. Please make sure all the previous colls are processed
  // before calling this API. Otherwise, the result might be unexpected.
  CommsMaybeVoid testOnlyClearColls() noexcept;
  void testOnlyExecuteWithReadLock(const std::function<void()>& fn) const;
  // Holds the dump lock exclusively, so a concurrent collectStats() takes the
  // acquire-timeout branch deterministically instead of waiting on a race.
  void testOnlyExecuteWithWriteLock(const std::function<void()>& fn);

  static constexpr std::string_view kCommDumpPluginName = "CommDumpPlugin";

 private:
  void evictPastColls(CollTraceDump& dump);
  CommsMaybeVoid drainPendingState(CollTraceDump& dump) noexcept;
  void applyTerminalDisposition(
      CollTraceDump& dump,
      std::shared_ptr<CollRecord> record,
      CollTraceTerminalReason reason,
      bool scrubTrackedState = true) noexcept;
  CommsMaybeVoid deferTerminalDisposition(
      const std::shared_ptr<CollRecord>& record,
      CollTraceTerminalReason reason) noexcept;
  bool enqueueDeferredTerminalDisposition(
      const std::shared_ptr<CollRecord>& record,
      CollTraceTerminalReason reason) noexcept;
  bool isTerminallyTracked(
      const CollTraceDump& dump,
      const std::shared_ptr<CollRecord>& record) const noexcept;
  void enforceStateBounds(CollTraceDump& dump) noexcept;

  CommDumpConfig config_;
  logger::CommsSpdlogLogger* logger_{nullptr};

  folly::Synchronized<CollTraceDump> collTraceDump_;

  // Using a MPMC queue to handle the enqueueing to ensure that we don't block
  // the scheduling thread.
  //
  // There is no guarantee that if multiple threads enqueing at the same time,
  // the order of the enqueuing is the same as the order colltrace observed.
  // But we don't support multiple threads enqueuing at the same time anyway.
  //
  // Producer: scheduling thread
  // Consumer: dump thread / colltrace thread (but not at the same time)
  // consuming APIs: dump/ afterCollKernelStart / afterCollKernelEnd /
  //                 whenCollKernelHang
  folly::MPMCQueue<std::shared_ptr<CollRecord>> newPendingColls_;
  folly::MPMCQueue<TerminalCollRecord> deferredTerminalColls_;
  std::atomic<uint64_t> terminalTransitionDrops_{0};
  std::atomic<uint64_t> pollLockTimeouts_{0};
  std::atomic_bool reconciliationRequired_{false};
};

// ------------------------------------------------------------------------
// Helper functions for CommDumpPlugin

std::unordered_map<std::string, std::string> commDumpToMap(
    const CollTraceDump& dump,
    const std::unordered_set<std::string>& requestFields = {});

} // namespace meta::comms::colltrace
