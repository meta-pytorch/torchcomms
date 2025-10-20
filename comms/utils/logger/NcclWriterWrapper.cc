// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/utils/logger/NcclWriterWrapper.h"

namespace meta::comms::logger {

NcclWriterWrapper::NcclWriterWrapper(
    std::shared_ptr<folly::LogWriter> mainWriter,
    std::shared_ptr<folly::LogWriter> warnWriter)
    : mainWriter_(std::move(mainWriter)), warnWriter_(std::move(warnWriter)){};

void NcclWriterWrapper::writeMessage(
    folly::StringPiece buffer,
    uint32_t flags) {
  // Hacky way to detect if the message is a warning, error, or fatal
  if (!buffer.empty() &&
      (buffer[0] == 'W' || buffer[0] == 'E' || buffer[0] == 'F')) {
    warnWriter_->writeMessage(buffer, flags);
    writeToWarnAfterFlush_ = true;
  }
  mainWriter_->writeMessage(buffer, flags);
}

void NcclWriterWrapper::flush() {
  mainWriter_->flush();
  if (writeToWarnAfterFlush_) {
    warnWriter_->flush();
    writeToWarnAfterFlush_ = false;
  }
}

} // namespace meta::comms::logger
