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
//
// Named fields document the scuba table schema. The writer implementation
// serializes all set fields into the scuba sample.
struct OperationTraceSample {
  // === Identity (maps to existing mccl_operation_trace columns) ===
  std::string mcclop; // Operation type (e.g., "GPE_EXECUTION")
  std::string event; // Event name (e.g., "GPE_EXECUTION_END")
  int64_t timestampUs{0}; // Timestamp in microseconds (scuba: "timestamp")
  int rank{-1};
  int64_t commHash{0};
  int64_t commId{0};
  int worldSize{-1};
  int64_t opCount{-1};

  // === Timing ===
  std::optional<int64_t> durationUs; // scuba: "duration_us"

  // === GPE Execution context ===
  std::optional<std::string> gpeKernelType; // scuba: "gpe_kernel_type"
  std::optional<int> gpeNumBlocks; // scuba: "gpe_num_blocks"
  std::optional<int> gpeNumThreads; // scuba: "gpe_num_threads"
  std::optional<bool> gpePersistent; // scuba: "gpe_persistent"
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

// RAII guard for operation-level trace logging.
// Logs {mcclop}_START on construction and {mcclop}_END on destruction
// with computed duration and context fields.
//
// Usage:
//   OperationTraceSample sample;
//   sample.mcclop = "GPE_EXECUTION";
//   sample.rank = rank;
//   // ... fill identity fields ...
//   OperationTraceGuard guard(std::move(sample));
//   guard.sample().gpeKernelType = "AllGather";
//   // ... do work ...
//   // GPE_EXECUTION_END logged automatically with duration on destruction
class OperationTraceGuard {
 public:
  // Construct with pre-filled sample (mcclop + identity fields).
  // If startUs > 0, uses it as the start time; otherwise captures now.
  // Logs {mcclop}_START immediately.
  explicit OperationTraceGuard(
      OperationTraceSample sample,
      int64_t startUs = 0);
  ~OperationTraceGuard();

  OperationTraceGuard(const OperationTraceGuard&) = delete;
  OperationTraceGuard& operator=(const OperationTraceGuard&) = delete;
  OperationTraceGuard(OperationTraceGuard&& other) noexcept;
  OperationTraceGuard& operator=(OperationTraceGuard&&) = delete;

  // Access the sample to set additional fields before destruction.
  OperationTraceSample& sample() {
    return sample_;
  }

  // Whether tracing is active (writer available and enabled).
  bool isActive() const {
    return active_;
  }

  // Prevent logging on destruction.
  void dismiss() {
    dismissed_ = true;
  }

 private:
  friend class EventLoggerGuard;
  OperationTraceSample sample_;
  int64_t startUs_{0};
  bool dismissed_{false};
  bool active_{false};
};

// RAII guard for event-level (sub-phase) trace logging within an operation.
// Logs {event}_START on construction and {event}_END on destruction
// with computed duration. Identity fields are copied from the parent
// OperationTraceGuard.
//
// Usage:
//   OperationTraceGuard opGuard(sample);
//   {
//     EventLoggerGuard evGuard(opGuard, "GPE_KERNEL_WAIT");
//     // ... wait for kernel ...
//   } // GPE_KERNEL_WAIT_END logged with duration
class EventLoggerGuard {
 public:
  // Construct with parent guard and event base name (e.g., "GPE_QUEUE_WAIT").
  // If startUs > 0, uses it as the start time; otherwise captures now.
  // Logs {event}_START immediately.
  EventLoggerGuard(
      const OperationTraceGuard& parent,
      std::string event,
      int64_t startUs = 0);
  ~EventLoggerGuard();

  EventLoggerGuard(const EventLoggerGuard&) = delete;
  EventLoggerGuard& operator=(const EventLoggerGuard&) = delete;

 private:
  const OperationTraceGuard* parent_;
  std::string event_;
  int64_t startUs_{0};
  bool active_{false};
};

} // namespace comms::logger
