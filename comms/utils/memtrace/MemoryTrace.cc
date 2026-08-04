// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/memtrace/MemoryTrace.h"

#include <algorithm>

#include <folly/Synchronized.h>
#include <folly/json/dynamic.h>
#include <folly/json/json.h>

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/alloc.h"

namespace meta::comms::memtrace {

namespace {
// Maps commHash to per-communicator MemoryTrace instance.
// Dual ownership: this map holds a shared_ptr (created at first allocation,
// before CommsMonitor::registerComm), and NcclCommMonitorInfo adopts a
// shared_ptr later at registerComm time.
folly::Synchronized<std::unordered_map<uint64_t, std::shared_ptr<MemoryTrace>>>
    tracers;
} // namespace

std::shared_ptr<MemoryTrace> MemoryTrace::getOrCreate(uint64_t commHash) {
  auto locked = tracers.wlock();
  auto it = locked->find(commHash);
  if (it != locked->end()) {
    return it->second;
  }
  auto trace = std::make_shared<MemoryTrace>();
  locked->emplace(commHash, trace);
  return trace;
}

void MemoryTrace::recordAlloc(uintptr_t addr, int64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  allocMap_[addr] = bytes;
  stats_.totalAllocated += bytes;
  stats_.currentUsage += bytes;
  stats_.peakUsage = std::max(stats_.peakUsage, stats_.currentUsage);
}

void MemoryTrace::recordFree(uintptr_t addr, std::optional<int64_t> bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  int64_t freedBytes = 0;
  auto it = allocMap_.find(addr);
  if (it != allocMap_.end()) {
    freedBytes = bytes.value_or(it->second);
    allocMap_.erase(it);
  } else {
    freedBytes = bytes.value_or(0);
  }
  stats_.totalFreed += freedBytes;
  stats_.currentUsage -= freedBytes;
}

MemoryStats MemoryTrace::getStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

std::string MemoryTrace::dump() const {
  std::lock_guard<std::mutex> lock(mutex_);
  folly::dynamic obj = folly::dynamic::object(
      "totalAllocated", stats_.totalAllocated)("totalFreed", stats_.totalFreed)(
      "currentUsage", stats_.currentUsage)("peakUsage", stats_.peakUsage);
  return folly::toJson(obj);
}

// Free function implementations

void recordAlloc(
    const CommLogData& logMetaData,
    const MemCallsite& callsite,
    const std::string& use,
    uintptr_t addr,
    int64_t bytes,
    std::optional<int> numSegments,
    std::optional<int64_t> durationUs) {
  if (!NCCL_MEMTRACE_ENABLE) {
    return;
  }
  logMemoryEvent(
      logMetaData,
      callsite.function,
      use,
      addr,
      bytes,
      numSegments,
      durationUs,
      /*memType=*/std::nullopt,
      /*isRegMemEvent=*/false,
      callsite.scope);
  MemoryTrace::getOrCreate(logMetaData.commHash)->recordAlloc(addr, bytes);
}

void recordFree(
    const CommLogData& logMetaData,
    const MemCallsite& callsite,
    const std::string& use,
    uintptr_t addr,
    std::optional<int64_t> bytes) {
  if (!NCCL_MEMTRACE_ENABLE) {
    return;
  }
  // The free callsite supplies its own scope directly (symmetric with alloc),
  // so the scuba FREE row is tagged from callsite.scope.
  logMemoryEvent(
      logMetaData,
      callsite.function,
      use,
      addr,
      bytes,
      /*numSegments=*/std::nullopt,
      /*durationUs=*/std::nullopt,
      /*memType=*/std::nullopt,
      /*isRegMemEvent=*/false,
      callsite.scope);
  MemoryTrace::getOrCreate(logMetaData.commHash)->recordFree(addr, bytes);
}

void recordReg(
    const CommLogData& logMetaData,
    const MemCallsite& callsite,
    const std::string& use,
    uintptr_t addr,
    std::optional<int64_t> bytes,
    std::optional<int> numSegments,
    std::optional<int64_t> durationUs,
    const std::optional<std::string>& memType) {
  if (!NCCL_MEMTRACE_ENABLE) {
    return;
  }
  logMemoryEvent(
      logMetaData,
      callsite.function,
      use,
      addr,
      bytes,
      numSegments,
      durationUs,
      memType,
      /*isRegMemEvent=*/true,
      callsite.scope);
}

} // namespace meta::comms::memtrace
