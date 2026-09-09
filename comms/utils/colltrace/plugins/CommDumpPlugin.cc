// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"

#include <algorithm>
#include <utility>

#include <folly/Unit.h>
#include <folly/json.h>

#include "comms/utils/CommsMaybeChecks.h"
#include "comms/utils/logger/SpdlogLogger.h"
#include "comms/utils/trainer/TrainerContext.h"

namespace meta::comms::colltrace {

namespace {
CommDumpConfig normalizeCommDumpConfig(CommDumpConfig config) {
  config.pendingCollSize = std::max<int64_t>(1, config.pendingCollSize);
  config.currentCollSize = std::max<int64_t>(0, config.currentCollSize);
  config.terminalCollSize = std::max<int64_t>(0, config.terminalCollSize);
  if (config.pendingDrainBatchSize <= 0) {
    config.pendingDrainBatchSize = config.pendingCollSize + 1;
  }
  return config;
}

CommsMaybeVoid enqueuePendingColls(
    folly::MPMCQueue<std::shared_ptr<CollRecord>>& mpmcQueue,
    std::deque<std::shared_ptr<CollRecord>>& pendingQueue,
    int64_t maxReadCount,
    logger::CommsSpdlogLogger& logger) noexcept {
  std::shared_ptr<CollRecord> nextEnqueue;
  int readCount{0};
  while (readCount < maxReadCount && mpmcQueue.read(nextEnqueue)) {
    pendingQueue.emplace_back(std::move(nextEnqueue));
    ++readCount;
  }
  if (readCount == maxReadCount && !mpmcQueue.isEmpty()) {
    COMMS_LOGGER_STREAM_FIRST_N(logger, ERR, 2)
        << "CommDumpPlugin: Read " << readCount
        << " pending colls, but queue is still not empty";
    return folly::makeUnexpected(CommsError(
        "CommDumpPlugin: Read " + std::to_string(readCount) +
            " pending colls, but queue is still not empty",
        commInternalError));
  }
  return folly::unit;
}
} // namespace

CommDumpPlugin::CommDumpPlugin(CommDumpConfig config)
    : config_(normalizeCommDumpConfig(std::move(config))),
      logger_(&logger::getSpdlogLogger(config_.loggerName)),
      newPendingColls_(config_.pendingCollSize),
      deferredTerminalColls_(std::max<int64_t>(1, config_.terminalCollSize)) {}

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
    COMMS_LOGGER_STREAM_FIRST_N(*logger_, ERR, 2)
        << "Got event with null collRecord in CommDumpPlugin";
    return folly::makeUnexpected(CommsError(
        "CollTraceEvent does not contain valid record", commInternalError));
  }

  // Try to enqueue, but don't block if queue is full
  auto success = newPendingColls_.write(curEvent.collRecord);

  if (!success) [[unlikely]] {
    COMMS_LOGGER_STREAM_FIRST_N(*logger_, ERR, 2)
        << "Failed to enqueue event in CommDumpPlugin";
    return folly::makeUnexpected(CommsError(
        "Failed to enqueue event in CommDumpPlugin", commInternalError));
  }

  return folly::unit;
}

CommsMaybeVoid CommDumpPlugin::afterCollKernelStart(
    CollTraceEvent& curEvent) noexcept {
  if (curEvent.collRecord == nullptr) [[unlikely]] {
    COMMS_LOGGER_STREAM_FIRST_N(*logger_, ERR, 2)
        << "Got event with null collRecord in CommDumpPlugin";
    return folly::makeUnexpected(CommsError(
        "CollTraceEvent does not contain valid record", commInternalError));
  }

  auto lockedCollTraceDump =
      collTraceDump_.wlock(config_.pollLockAcquireTimeout);
  if (lockedCollTraceDump.isNull()) {
    return deferTerminalDisposition(
        curEvent.collRecord, CollTraceTerminalReason::PluginContention);
  }

  auto pendingDrainResult = drainPendingState(*lockedCollTraceDump);

  if (isTerminallyTracked(*lockedCollTraceDump, curEvent.collRecord)) {
    return pendingDrainResult;
  }

  // Find the matching pending collective.
  // With deferred graph polling, completions may arrive out of enqueue order
  auto it = std::find_if(
      lockedCollTraceDump->pendingColls.begin(),
      lockedCollTraceDump->pendingColls.end(),
      [&curEvent](const std::shared_ptr<CollRecord>& record) {
        return record.get() == curEvent.collRecord.get();
      });

  if (it == lockedCollTraceDump->pendingColls.end()) [[unlikely]] {
    COMMS_LOGGER_STREAM_FIRST_N(*logger_, ERR, 2)
        << "Could not find matching collRecord in pendingColls in CommDumpPlugin";
    lockedCollTraceDump->currentColls.push_back(curEvent.collRecord);
    enforceStateBounds(*lockedCollTraceDump);
    if (pendingDrainResult.hasError()) {
      return pendingDrainResult;
    }
    return folly::makeUnexpected(CommsError(
        "Recovered missing collRecord into currentColls in CommDumpPlugin",
        commInternalError));
  }

  // ----- Move to active collectives -----
  lockedCollTraceDump->currentColls.push_back(std::move(*it));
  lockedCollTraceDump->pendingColls.erase(it);
  enforceStateBounds(*lockedCollTraceDump);

  return pendingDrainResult;
}

CommsMaybeVoid CommDumpPlugin::collEventProgressing(
    CollTraceEvent& curEvent) noexcept {
  return folly::unit;
}

CommsMaybeVoid CommDumpPlugin::afterCollKernelEnd(
    CollTraceEvent& curEvent) noexcept {
  if (curEvent.collRecord == nullptr) [[unlikely]] {
    COMMS_LOGGER_STREAM_FIRST_N(*logger_, ERR, 2)
        << "Got event with null collRecord in CommDumpPlugin";
    return folly::makeUnexpected(CommsError(
        "CollTraceEvent does not contain valid record", commInternalError));
  }

  auto lockedCollTraceDump =
      collTraceDump_.wlock(config_.pollLockAcquireTimeout);
  if (lockedCollTraceDump.isNull()) {
    return deferTerminalDisposition(
        curEvent.collRecord, CollTraceTerminalReason::PluginContention);
  }

  auto pendingDrainResult = drainPendingState(*lockedCollTraceDump);

  if (isTerminallyTracked(*lockedCollTraceDump, curEvent.collRecord)) {
    return pendingDrainResult;
  }

  // ----- Find and move from currentColls to pastColls -----
  auto it = std::find_if(
      lockedCollTraceDump->currentColls.begin(),
      lockedCollTraceDump->currentColls.end(),
      [&curEvent](const std::shared_ptr<CollRecord>& record) {
        return record.get() == curEvent.collRecord.get();
      });

  if (it == lockedCollTraceDump->currentColls.end()) [[unlikely]] {
    COMMS_LOGGER_STREAM_FIRST_N(*logger_, ERR, 2)
        << "Could not find matching collRecord in currentColls during coll end";
    applyTerminalDisposition(
        *lockedCollTraceDump,
        curEvent.collRecord,
        CollTraceTerminalReason::TrackingOverflow);
    return folly::makeUnexpected(CommsError(
        "Could not find matching collRecord in currentColls during coll end",
        commInternalError));
  }

  lockedCollTraceDump->pastCollsHeap.push(std::move(*it));
  evictPastColls(*lockedCollTraceDump);
  lockedCollTraceDump->currentColls.erase(it);

  auto iterSnap = ncclxGetIterationSnapshot();
  if (iterSnap.iteration > lockedCollTraceDump->currentIteration) {
    lockedCollTraceDump->currentIteration = iterSnap.iteration;
    lockedCollTraceDump->currentIterationCommTimeUs = 0;
    lockedCollTraceDump->iterationCutoffUs = iterSnap.timestampUs;
  }

  auto collStartUs = std::chrono::duration_cast<std::chrono::microseconds>(
                         curEvent.collRecord->getTimingInfo()
                             .getCollStartTs()
                             .time_since_epoch())
                         .count();
  if (collStartUs >= lockedCollTraceDump->iterationCutoffUs) {
    auto latencyUs = std::chrono::duration_cast<std::chrono::microseconds>(
                         curEvent.collRecord->getTimingInfo().getCollEndTs() -
                         curEvent.collRecord->getTimingInfo().getCollStartTs())
                         .count();
    lockedCollTraceDump->currentIterationCommTimeUs +=
        std::max(latencyUs, int64_t{0});
  }

  return pendingDrainResult;
}

CommsMaybeVoid CommDumpPlugin::afterCollTerminated(
    CollTraceEvent& curEvent,
    CollTraceTerminalReason reason) noexcept {
  if (curEvent.collRecord == nullptr) [[unlikely]] {
    return folly::makeUnexpected(CommsError(
        "CollTraceEvent does not contain valid record", commInternalError));
  }

  auto lockedCollTraceDump =
      collTraceDump_.wlock(config_.pollLockAcquireTimeout);
  if (lockedCollTraceDump.isNull()) {
    return deferTerminalDisposition(curEvent.collRecord, reason);
  }

  auto pendingDrainResult = drainPendingState(*lockedCollTraceDump);
  applyTerminalDisposition(*lockedCollTraceDump, curEvent.collRecord, reason);
  return pendingDrainResult;
}

bool CommDumpPlugin::isTerminallyTracked(
    const CollTraceDump& dump,
    const std::shared_ptr<CollRecord>& record) const noexcept {
  return std::any_of(
      dump.terminalColls.begin(),
      dump.terminalColls.end(),
      [&record](const TerminalCollRecord& terminal) {
        return terminal.collRecord.get() == record.get();
      });
}

void CommDumpPlugin::applyTerminalDisposition(
    CollTraceDump& dump,
    std::shared_ptr<CollRecord> record,
    CollTraceTerminalReason reason) noexcept {
  if (record == nullptr || isTerminallyTracked(dump, record)) {
    return;
  }

  const auto matchesRecord =
      [&record](const std::shared_ptr<CollRecord>& item) {
        return item.get() == record.get();
      };
  std::erase_if(dump.pendingColls, matchesRecord);
  std::erase_if(dump.currentColls, matchesRecord);

  const auto reasonIndex = static_cast<std::size_t>(reason);
  if (reasonIndex < dump.terminalReasonCounts.size()) {
    ++dump.terminalReasonCounts[reasonIndex];
  }
  if (config_.terminalCollSize == 0) {
    return;
  }
  dump.terminalColls.push_back(TerminalCollRecord{std::move(record), reason});
  while (static_cast<int64_t>(dump.terminalColls.size()) >
         config_.terminalCollSize) {
    dump.terminalColls.pop_front();
  }
}

void CommDumpPlugin::enforceStateBounds(CollTraceDump& dump) noexcept {
  while (!dump.pendingColls.empty() &&
         static_cast<int64_t>(dump.pendingColls.size()) >
             config_.pendingCollSize) {
    auto record = std::move(dump.pendingColls.front());
    dump.pendingColls.pop_front();
    applyTerminalDisposition(
        dump, std::move(record), CollTraceTerminalReason::TrackingOverflow);
  }
  while (!dump.currentColls.empty() &&
         static_cast<int64_t>(dump.currentColls.size()) >
             config_.currentCollSize) {
    auto record = std::move(dump.currentColls.front());
    dump.currentColls.pop_front();
    applyTerminalDisposition(
        dump, std::move(record), CollTraceTerminalReason::TrackingOverflow);
  }
}

CommsMaybeVoid CommDumpPlugin::drainPendingState(CollTraceDump& dump) noexcept {
  auto result = enqueuePendingColls(
      newPendingColls_,
      dump.pendingColls,
      config_.pendingDrainBatchSize,
      *logger_);

  std::vector<TerminalCollRecord> terminalWork;
  const auto appendTerminal = [&terminalWork](TerminalCollRecord terminal) {
    const auto duplicate = std::any_of(
        terminalWork.begin(),
        terminalWork.end(),
        [&terminal](const TerminalCollRecord& existing) {
          return existing.collRecord.get() == terminal.collRecord.get();
        });
    if (!duplicate) {
      terminalWork.push_back(std::move(terminal));
    }
  };

  TerminalCollRecord terminal;
  while (deferredTerminalColls_.read(terminal)) {
    appendTerminal(std::move(terminal));
  }

  if (reconciliationRequired_.exchange(false)) {
    for (auto& record : dump.pendingColls) {
      appendTerminal(
          TerminalCollRecord{
              record, CollTraceTerminalReason::PluginContention});
    }
    for (auto& record : dump.currentColls) {
      appendTerminal(
          TerminalCollRecord{
              record, CollTraceTerminalReason::PluginContention});
    }
    dump.pendingColls.clear();
    dump.currentColls.clear();
  }

  for (auto& terminalRecord : terminalWork) {
    applyTerminalDisposition(
        dump, std::move(terminalRecord.collRecord), terminalRecord.reason);
  }
  enforceStateBounds(dump);
  return result;
}

CommsMaybeVoid CommDumpPlugin::deferTerminalDisposition(
    const std::shared_ptr<CollRecord>& record,
    CollTraceTerminalReason reason) noexcept {
  pollLockTimeouts_.fetch_add(1, std::memory_order_relaxed);
  if (!deferredTerminalColls_.write(TerminalCollRecord{record, reason})) {
    terminalTransitionDrops_.fetch_add(1, std::memory_order_relaxed);
    reconciliationRequired_.store(true, std::memory_order_release);
  }
  return folly::makeUnexpected(CommsError(
      "Timed out acquiring CollTrace dump state for plugin lifecycle callback",
      commInternalError));
}

void CommDumpPlugin::evictPastColls(CollTraceDump& dump) {
  while (config_.pastCollSize >= 0 &&
         static_cast<int64_t>(dump.pastCollsHeap.size()) >
             config_.pastCollSize) {
    dump.pastCollsHeap.pop();
  }
}

CommsMaybe<CollTraceDump> CommDumpPlugin::dump() noexcept {
  if (!newPendingColls_.isEmpty() || !deferredTerminalColls_.isEmpty() ||
      reconciliationRequired_.load(std::memory_order_acquire)) {
    auto lockedCollTraceDump =
        collTraceDump_.wlock(config_.dumpLockAcquireTimeout);

    if (lockedCollTraceDump.isNull()) {
      COMMS_LOGGER_STREAM_FIRST_N(*logger_, ERR, 2)
          << "Failed to acquire lock for collTraceDump_ in CommDumpPlugin dump";
      return folly::makeUnexpected(CommsError(
          "Failed to acquire lock for collTraceDump_ in CommDumpPlugin dump",
          commInternalError));
    }

    EXPECT_CHECK_LOG_FIRST_N(2, drainPendingState(*lockedCollTraceDump));
  }

  auto readLockedCollTraceDump =
      collTraceDump_.rlock(config_.dumpLockAcquireTimeout);

  if (readLockedCollTraceDump.isNull()) {
    COMMS_LOGGER_STREAM_FIRST_N(*logger_, ERR, 2)
        << "Failed to acquire read lock for collTraceDump_ in CommDumpPlugin dump";
    return folly::makeUnexpected(CommsError(
        "Failed to acquire read lock for collTraceDump_ in CommDumpPlugin dump",
        commInternalError));
  }

  // Create a copy of the current state of collTraceDump_
  CollTraceDump dumpCopy = *readLockedCollTraceDump;
  dumpCopy.terminalTransitionDrops =
      terminalTransitionDrops_.load(std::memory_order_relaxed);
  dumpCopy.pollLockTimeouts = pollLockTimeouts_.load(std::memory_order_relaxed);

  // Drain the min-heap into pastColls deque in ascending collId order
  while (!dumpCopy.pastCollsHeap.empty()) {
    dumpCopy.pastColls.push_back(dumpCopy.pastCollsHeap.top());
    dumpCopy.pastCollsHeap.pop();
  }

  // Temporary fix: Currently we use currentColls to also track the next
  // pending collective, this logic is being used in Analyzer to detect
  // dependencies between collectives. Without making the next pending
  // collective current, Analyzer will not work. For now we temporarily
  // track next pending collective as current, until we fully deprecate
  // old colltrace and change Analyzer logic
  if (dumpCopy.currentColls.empty() && !dumpCopy.pendingColls.empty()) {
    dumpCopy.currentColls.push_back(std::move(dumpCopy.pendingColls.front()));
    dumpCopy.pendingColls.pop_front();
  }

  return dumpCopy;
}

IterationCommTime CommDumpPlugin::getCurrentIterationCommTime() const noexcept {
  auto locked = collTraceDump_.rlock(config_.dumpLockAcquireTimeout);
  if (locked.isNull()) {
    return {};
  }
  return {locked->currentIteration, locked->currentIterationCommTimeUs};
}

namespace {
bool isKeyReq(
    const std::unordered_set<std::string>& fields,
    std::string_view key) {
  return fields.empty() || fields.contains(std::string{key});
}
} // namespace

std::unordered_map<std::string, std::string> commDumpToMap(
    const CollTraceDump& dump,
    const std::unordered_set<std::string>& requestFields) {
  std::unordered_map<std::string, std::string> map;

  if (isKeyReq(requestFields, "CT_pastColls")) {
    auto pastColls = folly::dynamic::array();
    for (const auto& coll : dump.pastColls) {
      pastColls.push_back(coll->toDynamic());
    }
    map["CT_pastColls"] = folly::toJson(pastColls);
  }

  if (isKeyReq(requestFields, "CT_pendingColls")) {
    auto pendingColls = folly::dynamic::array();
    for (const auto& coll : dump.pendingColls) {
      pendingColls.push_back(coll->toDynamic());
    }
    map["CT_pendingColls"] = folly::toJson(pendingColls);
  }

  if (isKeyReq(requestFields, "CT_currentColls")) {
    auto currentColls = folly::dynamic::array();
    for (const auto& coll : dump.currentColls) {
      currentColls.push_back(coll->toDynamic());
    }
    map["CT_currentColls"] = folly::toJson(currentColls);
  }

  if (isKeyReq(requestFields, "CT_terminalColls")) {
    auto terminalColls = folly::dynamic::array();
    for (const auto& terminal : dump.terminalColls) {
      auto record = terminal.collRecord->toDynamic();
      record["terminalReason"] =
          collTraceTerminalReasonToString(terminal.reason);
      terminalColls.push_back(std::move(record));
    }
    map["CT_terminalColls"] = folly::toJson(terminalColls);
  }

  if (isKeyReq(requestFields, "CT_terminalReasonCounts")) {
    folly::dynamic counts = folly::dynamic::object();
    for (std::size_t index = 0; index < dump.terminalReasonCounts.size();
         ++index) {
      const auto reason = static_cast<CollTraceTerminalReason>(index);
      counts[collTraceTerminalReasonToString(reason)] =
          dump.terminalReasonCounts[index];
    }
    map["CT_terminalReasonCounts"] = folly::toJson(counts);
  }

  if (isKeyReq(requestFields, "CT_terminalTransitionDrops")) {
    map["CT_terminalTransitionDrops"] =
        std::to_string(dump.terminalTransitionDrops);
  }

  if (isKeyReq(requestFields, "CT_pollLockTimeouts")) {
    map["CT_pollLockTimeouts"] = std::to_string(dump.pollLockTimeouts);
  }

  if (isKeyReq(requestFields, "CT_currentIteration")) {
    map["CT_currentIteration"] = std::to_string(dump.currentIteration);
  }

  if (isKeyReq(requestFields, "CT_currentIterationCommTimeUs")) {
    map["CT_currentIterationCommTimeUs"] =
        std::to_string(dump.currentIterationCommTimeUs);
  }

  return map;
}

int64_t CommDumpPlugin::maxEventRetention() const noexcept {
  return config_.pastCollSize;
}

CommsMaybeVoid CommDumpPlugin::testOnlyClearColls() noexcept {
  collTraceDump_.exchange(CollTraceDump{});
  newPendingColls_ =
      folly::MPMCQueue<std::shared_ptr<CollRecord>>(config_.pendingCollSize);
  deferredTerminalColls_ = folly::MPMCQueue<TerminalCollRecord>(
      std::max<int64_t>(1, config_.terminalCollSize));
  terminalTransitionDrops_.store(0, std::memory_order_relaxed);
  pollLockTimeouts_.store(0, std::memory_order_relaxed);
  reconciliationRequired_.store(false, std::memory_order_relaxed);
  return folly::unit;
}

void CommDumpPlugin::testOnlyExecuteWithReadLock(
    const std::function<void()>& fn) const {
  auto locked = collTraceDump_.rlock();
  fn();
}

} // namespace meta::comms::colltrace
