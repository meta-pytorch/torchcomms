// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/logger/OperationTraceWriter.h"

#include "comms/utils/logger/ScubaFileUtils.h"

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

namespace {
// Helper to log a sample with the given event name.
void logTraceSample(
    IOperationTraceWriter* writer,
    const OperationTraceSample& base,
    const std::string& event,
    std::optional<int64_t> durationUs = std::nullopt) {
  OperationTraceSample s;
  s.mcclop = base.mcclop;
  s.event = event;
  s.timestampUs = getTimestampUs();
  s.rank = base.rank;
  s.commHash = base.commHash;
  s.commId = base.commId;
  s.worldSize = base.worldSize;
  s.opCount = base.opCount;
  s.durationUs = durationUs;
  writer->addSample(std::move(s));
}
} // namespace

// --- OperationTraceGuard ---

OperationTraceGuard::OperationTraceGuard(
    OperationTraceSample sample,
    int64_t startUs)
    : sample_(std::move(sample)) {
  auto* writer = OperationTraceWriterRegistry::get();
  if (writer && writer->isEnabled()) {
    active_ = true;
    startUs_ = startUs > 0 ? startUs : getTimestampUs();
    logTraceSample(writer, sample_, sample_.mcclop + "_START");
  }
}

OperationTraceGuard::~OperationTraceGuard() {
  if (dismissed_ || !active_) {
    return;
  }
  auto* writer = OperationTraceWriterRegistry::get();
  if (!writer) {
    return;
  }
  auto now = getTimestampUs();
  sample_.event = sample_.mcclop + "_END";
  sample_.timestampUs = now;
  if (!sample_.durationUs.has_value()) {
    sample_.durationUs = now - startUs_;
  }
  writer->addSample(std::move(sample_));
}

OperationTraceGuard::OperationTraceGuard(OperationTraceGuard&& other) noexcept
    : sample_(std::move(other.sample_)),
      startUs_(other.startUs_),
      dismissed_(other.dismissed_),
      active_(other.active_) {
  other.dismissed_ = true;
}

// --- EventLoggerGuard ---

EventLoggerGuard::EventLoggerGuard(
    const OperationTraceGuard& parent,
    std::string event,
    int64_t startUs)
    : parent_(&parent), event_(std::move(event)), active_(parent.isActive()) {
  if (active_) {
    startUs_ = startUs > 0 ? startUs : getTimestampUs();
    auto* writer = OperationTraceWriterRegistry::get();
    if (writer) {
      logTraceSample(writer, parent_->sample_, event_ + "_START");
    }
  }
}

EventLoggerGuard::~EventLoggerGuard() {
  if (!active_) {
    return;
  }
  auto* writer = OperationTraceWriterRegistry::get();
  if (!writer) {
    return;
  }
  auto now = getTimestampUs();
  logTraceSample(writer, parent_->sample_, event_ + "_END", now - startUs_);
}

} // namespace comms::logger
