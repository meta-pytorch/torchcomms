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
  if (bytes.has_value()) {
    freedBytes = bytes.value();
  } else {
    auto it = allocMap_.find(addr);
    if (it != allocMap_.end()) {
      freedBytes = it->second;
    }
  }
  allocMap_.erase(addr);
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
    const std::string& callsite,
    const std::string& use,
    uintptr_t addr,
    int64_t bytes,
    std::optional<int> numSegments,
    std::optional<int64_t> durationUs) {
  logMemoryEvent(
      logMetaData, callsite, use, addr, bytes, numSegments, durationUs);
}

void recordFree(
    const CommLogData& logMetaData,
    const std::string& callsite,
    const std::string& use,
    uintptr_t addr,
    std::optional<int64_t> bytes) {
  logMemoryEvent(logMetaData, callsite, use, addr, bytes);
}

void recordReg(
    const CommLogData& logMetaData,
    const std::string& callsite,
    const std::string& use,
    uintptr_t addr,
    std::optional<int64_t> bytes,
    std::optional<int> numSegments,
    std::optional<int64_t> durationUs) {
  logMemoryEvent(
      logMetaData,
      callsite,
      use,
      addr,
      bytes,
      numSegments,
      durationUs,
      /*isRegMemEvent=*/true);
}

} // namespace meta::comms::memtrace
