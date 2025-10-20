// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/logging/LogWriter.h>

namespace meta::comms::logger {

class NcclWriterWrapper : public folly::LogWriter {
 public:
  NcclWriterWrapper(
      std::shared_ptr<folly::LogWriter> mainWriter,
      std::shared_ptr<folly::LogWriter> warnWriter);
  void writeMessage(folly::StringPiece buffer, uint32_t flags = 0) override;

  void flush() override;

  bool ttyOutput() const override {
    // Most of the times we are logging to file, so return false here
    return false;
  }

 private:
  std::shared_ptr<folly::LogWriter> mainWriter_;
  std::shared_ptr<folly::LogWriter> warnWriter_;
  bool writeToWarnAfterFlush_{false};
};

} // namespace meta::comms::logger
