// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"

#include <folly/Unit.h>
#include <folly/json.h>
#include <folly/logging/xlog.h>

#include "comms/utils/CommsMaybeChecks.h"

namespace meta::comms::colltrace {

namespace {
CommsMaybeVoid enqueuePendingColls(
    folly::MPMCQueue<std::shared_ptr<CollRecord>>& mpmcQueue,
    std::deque<std::shared_ptr<CollRecord>>& pendingQueue,
    int64_t maxReadCount) noexcept {
  std::shared_ptr<CollRecord> nextEnqueue;
  int readCount{0};
  while (readCount < maxReadCount && mpmcQueue.read(nextEnqueue)) {
    pendingQueue.emplace_back(std::move(nextEnqueue));
    ++readCount;
  }
  if (readCount == maxReadCount) {
    XLOG_FIRST_N(
        ERR,
        2,
        "CommDumpPlugin: Read ",
        readCount,
        " pending colls, but queue is still not empty");
    return folly::makeUnexpected(CommsError(
        "CommDumpPlugin: Read " + std::to_string(readCount) +
            " pending colls, but queue is still not empty",
        commInternalError));
  }
  return folly::unit;
}
} // namespace

CommDumpPlugin::CommDumpPlugin(CommDumpConfig config)
    : config_(config), newPendingColls_(config_.pendingCollSize) {}

std::string_view CommDumpPlugin::getName() const noexcept {
  return kCommDumpPluginName;
}

CommsMaybeVoid CommDumpPlugin::beforeCollKernelScheduled(
    CollTraceEvent& curEvent) noexcept {
  // Dummy implementation - no-op
  return folly::unit;
}

CommsMaybeVoid CommDumpPlugin::afterCollKernelScheduled(
    CollTraceEvent& curEvent) noexcept {
  if (curEvent.collRecord == nullptr) [[unlikely]] {
    XLOG_FIRST_N(ERR, 2, "Got event with null collRecord in CommDumpPlugin");
    return folly::makeUnexpected(CommsError(
        "CollTraceEvent does not contain valid record", commInternalError));
  }

  // Try to enqueue, but don't block if queue is full
  auto success = newPendingColls_.write(curEvent.collRecord);

  if (!success) [[unlikely]] {
    XLOG_FIRST_N(ERR, 2, "Failed to enqueue event in CommDumpPlugin");
    return folly::makeUnexpected(CommsError(
        "Failed to enqueue event in CommDumpPlugin", commInternalError));
  }

  return folly::unit;
}

CommsMaybeVoid CommDumpPlugin::afterCollKernelStart(
    CollTraceEvent& curEvent) noexcept {
  if (curEvent.collRecord == nullptr) [[unlikely]] {
    XLOG_FIRST_N(ERR, 2, "Got event with null collRecord in CommDumpPlugin");
    return folly::makeUnexpected(CommsError(
        "CollTraceEvent does not contain valid record", commInternalError));
  }

  auto lockedCollTraceDump = collTraceDump_.wlock();

  EXPECT_CHECK_LOG_FIRST_N(
      2,
      enqueuePendingColls(
          newPendingColls_,
          lockedCollTraceDump->pendingColls,
          config_.pendingCollSize + 1));

  // ----- Get the first pending event -----
  if (lockedCollTraceDump->pendingColls.empty()) [[unlikely]] {
    XLOG_FIRST_N(ERR, 2, "Pending colls queue is empty in CommDumpPlugin");
    return folly::makeUnexpected(CommsError(
        "Pending colls queue is empty in CommDumpPlugin", commInternalError));
  }

  // ----- Check whether the pending event matches the current event -----
  if (lockedCollTraceDump->pendingColls.front().get() !=
      curEvent.collRecord.get()) [[unlikely]] {
    XLOG_FIRST_N(
        ERR, 2, "Got event with mismatched collRecord in CommDumpPlugin");
    return folly::makeUnexpected(CommsError(
        "Got event with mismatched collRecord in CommDumpPlugin",
        commInternalError));
  }

  // ----- Check whether the current event is not empty -----
  if (lockedCollTraceDump->currentColl != nullptr) [[unlikely]] {
    XLOG_FIRST_N(
        ERR, 2, "Got event with non-empty currentColl in CommDumpPlugin");
    return folly::makeUnexpected(CommsError(
        "Got event with non-empty currentColl in CommDumpPlugin",
        commInternalError));
  }

  // ----- Set the pending event and current event -----
  lockedCollTraceDump->currentColl =
      std::move(lockedCollTraceDump->pendingColls.front());
  lockedCollTraceDump->pendingColls.pop_front();

  return folly::unit;
}

CommsMaybeVoid CommDumpPlugin::collEventProgressing(
    CollTraceEvent& curEvent) noexcept {
  return folly::unit;
}

CommsMaybeVoid CommDumpPlugin::afterCollKernelEnd(
    CollTraceEvent& curEvent) noexcept {
  if (curEvent.collRecord == nullptr) [[unlikely]] {
    XLOG_FIRST_N(ERR, 2, "Got event with null collRecord in CommDumpPlugin");
    return folly::makeUnexpected(CommsError(
        "CollTraceEvent does not contain valid record", commInternalError));
  }

  auto lockedCollTraceDump = collTraceDump_.wlock();

  EXPECT_CHECK_LOG_FIRST_N(
      2,
      enqueuePendingColls(
          newPendingColls_,
          lockedCollTraceDump->pendingColls,
          config_.pendingCollSize + 1));

  // ----- Ensure CollRecord matches -----
  if (lockedCollTraceDump->currentColl.get() != curEvent.collRecord.get())
      [[unlikely]] {
    XLOG_FIRST_N(
        ERR,
        2,
        "Got event with mismatched collRecord in CommDumpPlugin during coll end");
    return folly::makeUnexpected(CommsError(
        "Got event with mismatched collRecord in CommDumpPlugin during coll end",
        commInternalError));
  }

  // ----- Move the coll to pastColls -----
  while (lockedCollTraceDump->pastColls.size() >= config_.pastCollSize) {
    lockedCollTraceDump->pastColls.pop_front();
  }
  lockedCollTraceDump->pastColls.emplace_back(
      std::move(lockedCollTraceDump->currentColl));

  return folly::unit;
}

CommsMaybe<CollTraceDump> CommDumpPlugin::dump() noexcept {
  if (!newPendingColls_.isEmpty()) {
    auto lockedCollTraceDump =
        collTraceDump_.wlock(config_.dumpLockAcquireTimeout);

    if (lockedCollTraceDump.isNull()) {
      XLOG_FIRST_N(
          ERR,
          2,
          "Failed to acquire lock for collTraceDump_ in CommDumpPlugin dump");
      return folly::makeUnexpected(CommsError(
          "Failed to acquire lock for collTraceDump_ in CommDumpPlugin dump",
          commInternalError));
    }

    EXPECT_CHECK_LOG_FIRST_N(
        2,
        enqueuePendingColls(
            newPendingColls_,
            lockedCollTraceDump->pendingColls,
            config_.pendingCollSize + 1));
  }

  auto readLockedCollTraceDump =
      collTraceDump_.rlock(config_.dumpLockAcquireTimeout);

  // Create a copy of the current state of collTraceDump_
  CollTraceDump dumpCopy = *readLockedCollTraceDump;

  // Temporary fix: Currently we use currentColl to also track the next
  // pending collective, this logic is being used in Analyzer to detect
  // dependencies between collectives. Without making the next pending
  // collective current, Analyzer will not work. For now we temporarily
  // track next pending collective as current, until we fully deprecate
  // old colltrace and change Analyzer logic
  if (dumpCopy.currentColl == nullptr && !dumpCopy.pendingColls.empty()) {
    dumpCopy.currentColl = std::move(dumpCopy.pendingColls.front());
    dumpCopy.pendingColls.pop_front();
  }

  return dumpCopy;
}

std::unordered_map<std::string, std::string> commDumpToMap(
    const CollTraceDump& dump) {
  std::unordered_map<std::string, std::string> map;

  auto pastColls = folly::dynamic::array();
  for (const auto& coll : dump.pastColls) {
    pastColls.push_back(coll->toDynamic());
  }
  map["CT_pastColls"] = folly::toJson(pastColls);

  auto pendingColls = folly::dynamic::array();
  for (const auto& coll : dump.pendingColls) {
    pendingColls.push_back(coll->toDynamic());
  }
  map["CT_pendingColls"] = folly::toJson(pendingColls);

  if (dump.currentColl != nullptr) {
    map["CT_currentColl"] = folly::toJson(dump.currentColl->toDynamic());
  } else {
    map["CT_currentColl"] = "null";
  }

  return map;
}

} // namespace meta::comms::colltrace
