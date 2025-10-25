// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <deque>
#include <memory>

#include <folly/MPMCQueue.h>
#include <folly/Synchronized.h>

#include "comms/utils/colltrace/CollTracePlugin.h"

namespace meta::comms::colltrace {

struct CommDumpConfig {
  // Default size of queue storing past collective communication operations.
  // The default value should be enough for debugging past collective. For slow
  // rank detection, we might need to increase the size.
  static constexpr int64_t kDefaultPastQueueSize{20};
  // 1024 elements should be sufficiently large to handle the number of
  // collective calls that is hanging when we dump the trace.
  static constexpr int kCommDumpQueueSize = 1024;

  // Default timeout for waiting for the lock to be acquired for dump. We don't
  // want to block the dump thread if there is any issue with the lock.
  static constexpr auto kDumpLockAcquireTimeout = std::chrono::seconds(1);

  // Configures the size of the queue for past collective operations
  // (defaults to kDefaultPastQueueSize). They will be dumped as pastColls when
  // dump() is called.
  int64_t pastCollSize{kDefaultPastQueueSize};
  // Configures the size of the queue for pending collective operations
  // (defaults to kCommDumpQueueSize). Any further collective operations will be
  // dropped if the queue is full.
  int64_t pendingCollSize{kCommDumpQueueSize};
  std::chrono::milliseconds dumpLockAcquireTimeout{kDumpLockAcquireTimeout};
};

struct CollTraceDump {
  std::deque<std::shared_ptr<CollRecord>> pastColls;
  std::shared_ptr<CollRecord> currentColl;
  std::deque<std::shared_ptr<CollRecord>> pendingColls;
};

class CommDumpPlugin : public ICollTracePlugin {
 public:
  CommDumpPlugin(CommDumpConfig config = {});

  std::string_view getName() const noexcept override;

  CommsMaybeVoid beforeCollKernelScheduled(
      CollTraceEvent& curEvent) noexcept override;

  CommsMaybeVoid afterCollKernelScheduled(
      CollTraceEvent& curEvent) noexcept override;

  CommsMaybeVoid afterCollKernelStart(
      CollTraceEvent& curEvent) noexcept override;

  CommsMaybeVoid collEventProgressing(
      CollTraceEvent& curEvent) noexcept override;

  CommsMaybeVoid afterCollKernelEnd(CollTraceEvent& curEvent) noexcept override;

  // CommDump specific API, supposed to be called by the dump (user) thread
  CommsMaybe<CollTraceDump> dump() noexcept;

  // For testing purpose only. This API is NOT thread safe! Clears all the
  // recorded colls. Please make sure all the previous colls are processed
  // before calling this API. Otherwise, the result might be unexpected.
  CommsMaybeVoid testOnlyClearColls() noexcept;

  static constexpr std::string_view kCommDumpPluginName = "CommDumpPlugin";

 private:
  CommDumpConfig config_;

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
};

// ------------------------------------------------------------------------
// Helper functions for CommDumpPlugin

std::unordered_map<std::string, std::string> commDumpToMap(
    const CollTraceDump& dump);

} // namespace meta::comms::colltrace
