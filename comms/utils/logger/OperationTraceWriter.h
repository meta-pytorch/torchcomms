// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace comms::logger {

// Sample for operation trace logging to mccl_operation_trace scuba table.
// Defines the schema for all columns that ctran (and other lower-level
// components) can log through the DI writer, without depending on MCCL.
struct OperationTraceSample {
  std::string mcclop; // Operation type (e.g., "GPE_EXECUTION")
  std::string event; // Event name (e.g., "GPE_EXECUTION_END")
  int64_t timestampUs{0}; // Timestamp in microseconds
  int rank{-1};
  int64_t commHash{0};
  int64_t commId{0};
  int worldSize{-1};
  int64_t opCount{-1};
  std::optional<int64_t> durationUs;

  // === GPE Execution context (Optional) ===
  std::optional<std::string> gpeKernelType;
  std::optional<int> gpeNumBlocks;
  std::optional<int> gpeNumThreads;
  std::optional<bool> gpePersistent;
};

// Interface for operation trace writers.
// Implementations route samples to their respective scuba tables
// (e.g., mccl_operation_trace, ncclx_operation_trace).
class IOperationTraceWriter {
 public:
  virtual ~IOperationTraceWriter() = default;
  virtual bool isEnabled() const = 0;
  virtual void addSample(OperationTraceSample sample) = 0;
};

// Global registry for the operation trace writer.
// Set by the communication library (MCCL or NCCLX) during initialization;
// queried by lower-level components (e.g., ctran) at runtime.
class OperationTraceWriterRegistry {
 public:
  static void set(std::shared_ptr<IOperationTraceWriter> writer);
  static IOperationTraceWriter* get();

 private:
  OperationTraceWriterRegistry() = delete;
};

// Get current timestamp in microseconds (re-declared to avoid pulling in
// ScubaFileUtils.h — definition lives in OperationTraceWriter.cc).
int64_t getTraceTimestampUs();

// Lightweight logger that captures shared identity fields once and provides
// simple logEvent() calls. No RAII, no automatic START/END — the caller
// explicitly logs each event at the point they want.
//
// Usage:
//   OperationTraceLogger logger("GPE_EXECUTION", rank, commHash, commId,
//                               worldSize, opCount);
//   if (!logger.isActive()) return;  // writer not registered or disabled
//   logger.setGpeContext("AllGather", 4, 256, false);
//   logger.logEvent("GPE_EXECUTION_START");
//   // ... do work ...
//   logger.logEvent("GPE_EXECUTION_END", startUs, endUs - startUs);
class OperationTraceLogger {
 public:
  OperationTraceLogger(
      std::string mcclop,
      int rank,
      int64_t commHash,
      int64_t commId,
      int worldSize,
      int64_t opCount);

  bool isActive() const {
    return active_;
  }

  // Set GPE context fields on all subsequent logEvent() calls.
  void setGpeContext(
      std::string kernelType,
      int numBlocks,
      int numThreads,
      bool persistent);

  // Log an event with auto-generated timestamp, no duration.
  void logEvent(const std::string& event);

  // Log an event with explicit timestamp, no duration.
  void logEvent(const std::string& event, int64_t timestampUs);

  // Log an event with explicit timestamp and duration.
  void
  logEvent(const std::string& event, int64_t timestampUs, int64_t durationUs);

 private:
  OperationTraceSample buildSample(
      const std::string& event,
      int64_t timestampUs,
      std::optional<int64_t> durationUs) const;

  IOperationTraceWriter* writer_{nullptr};
  std::string mcclop_;
  int rank_;
  int64_t commHash_;
  int64_t commId_;
  int worldSize_;
  int64_t opCount_;
  bool active_{false};

  // GPE context (set once, applied to all samples)
  std::optional<std::string> gpeKernelType_;
  std::optional<int> gpeNumBlocks_;
  std::optional<int> gpeNumThreads_;
  std::optional<bool> gpePersistent_;
};

} // namespace comms::logger
