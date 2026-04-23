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

} // namespace comms::logger
