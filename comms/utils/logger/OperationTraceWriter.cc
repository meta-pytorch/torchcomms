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

int64_t getTraceTimestampUs() {
  return getTimestampUs();
}

// --- OperationTraceLogger ---

OperationTraceLogger::OperationTraceLogger(
    std::string mcclop,
    int rank,
    int64_t commHash,
    int64_t commId,
    int worldSize,
    int64_t opCount)
    : mcclop_(std::move(mcclop)),
      rank_(rank),
      commHash_(commHash),
      commId_(commId),
      worldSize_(worldSize),
      opCount_(opCount) {
  writer_ = OperationTraceWriterRegistry::get();
  active_ = writer_ != nullptr && writer_->isEnabled();
}

void OperationTraceLogger::setGpeContext(
    std::string kernelType,
    int numBlocks,
    int numThreads,
    bool persistent) {
  gpeKernelType_ = std::move(kernelType);
  gpeNumBlocks_ = numBlocks;
  gpeNumThreads_ = numThreads;
  gpePersistent_ = persistent;
}

void OperationTraceLogger::logEvent(const std::string& event) {
  if (!active_) {
    return;
  }
  writer_->addSample(buildSample(event, getTimestampUs(), std::nullopt));
}

void OperationTraceLogger::logEvent(
    const std::string& event,
    int64_t timestampUs) {
  if (!active_) {
    return;
  }
  writer_->addSample(buildSample(event, timestampUs, std::nullopt));
}

void OperationTraceLogger::logEvent(
    const std::string& event,
    int64_t timestampUs,
    int64_t durationUs) {
  if (!active_) {
    return;
  }
  writer_->addSample(buildSample(event, timestampUs, durationUs));
}

OperationTraceSample OperationTraceLogger::buildSample(
    const std::string& event,
    int64_t timestampUs,
    std::optional<int64_t> durationUs) const {
  OperationTraceSample s;
  s.mcclop = mcclop_;
  s.event = event;
  s.timestampUs = timestampUs;
  s.rank = rank_;
  s.commHash = commHash_;
  s.commId = commId_;
  s.worldSize = worldSize_;
  s.opCount = opCount_;
  s.durationUs = durationUs;
  s.gpeKernelType = gpeKernelType_;
  s.gpeNumBlocks = gpeNumBlocks_;
  s.gpeNumThreads = gpeNumThreads_;
  s.gpePersistent = gpePersistent_;
  return s;
}

} // namespace comms::logger
