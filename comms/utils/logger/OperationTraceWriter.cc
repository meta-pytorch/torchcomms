// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/logger/OperationTraceWriter.h"

namespace comms::logger {

namespace {
std::shared_ptr<IOperationTraceWriter>& writerInstance() {
  static std::shared_ptr<IOperationTraceWriter> instance;
  return instance;
}
} // namespace

void OperationTraceWriterRegistry::set(
    std::shared_ptr<IOperationTraceWriter> writer) {
  writerInstance() = std::move(writer);
}

IOperationTraceWriter* OperationTraceWriterRegistry::get() {
  return writerInstance().get();
}

} // namespace comms::logger
