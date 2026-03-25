// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Single entry point for all NCCLX memory event recording.
// Each event is forwarded to Scuba (via logMemoryEvent) and recorded in a
// per-communicator MemoryTrace for local stats exposed through commDump.
//
// This header is safe to include from CUDA device-side code (stdlib only,
// no folly deps). All heavy implementation lives in MemoryTrace.cc.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "comms/utils/commSpecs.h"

namespace meta::comms::memtrace {

// --- Recording API ---

void recordAlloc(
    const CommLogData& logMetaData,
    const std::string& callsite,
    const std::string& use,
    uintptr_t addr,
    int64_t bytes,
    std::optional<int> numSegments = std::nullopt,
    std::optional<int64_t> durationUs = std::nullopt);

void recordFree(
    const CommLogData& logMetaData,
    const std::string& callsite,
    const std::string& use,
    uintptr_t addr,
    std::optional<int64_t> bytes = std::nullopt);

void recordReg(
    const CommLogData& logMetaData,
    const std::string& callsite,
    const std::string& use,
    uintptr_t addr,
    std::optional<int64_t> bytes = std::nullopt,
    std::optional<int> numSegments = std::nullopt,
    std::optional<int64_t> durationUs = std::nullopt);

// --- Per-communicator in-memory stats ---

struct MemoryStats {
  int64_t totalAllocated{0};
  int64_t totalFreed{0};
  int64_t currentUsage{0};
  int64_t peakUsage{0};
};

// Per-communicator memory tracker with aggregate stats.
//
// Dual ownership: a file-local static map in MemoryTrace.cc holds a shared_ptr
// (created at first allocation, before CommsMonitor::registerComm), and
// NcclCommMonitorInfo adopts a shared_ptr later at registerComm time.
// This is necessary because memory allocations happen during ncclCommInit
// before CommLogData/CommStateX fields are populated, so registerComm()
// cannot be called that early.
class MemoryTrace {
 public:
  static std::shared_ptr<MemoryTrace> getOrCreate(uint64_t commHash);

  void recordAlloc(uintptr_t addr, int64_t bytes);
  void recordFree(uintptr_t addr, std::optional<int64_t> bytes);
  MemoryStats getStats() const;
  std::string dump() const;

 private:
  mutable std::mutex mutex_;
  MemoryStats stats_;
  // Caches addr→bytes from recordAlloc; needed because some free paths
  // (e.g. ncclCudaFree) don't know the allocation size at free time.
  std::unordered_map<uintptr_t, int64_t> allocMap_;
};

} // namespace meta::comms::memtrace
