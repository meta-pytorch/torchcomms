// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef NCCL_LOGGER_H
#define NCCL_LOGGER_H

#include <memory>
#include <string>

#include <folly/File.h>
#include <folly/Synchronized.h>
#include <folly/logging/FileWriterFactory.h>
#include <folly/logging/LogConfig.h>
#include <folly/logging/LogFormatter.h>
#include <folly/logging/LogHandlerFactory.h>
#include <folly/logging/LogLevel.h>
#include <folly/logging/LogMessage.h>
#include <folly/logging/LogWriter.h>
#include <folly/logging/LoggerDB.h>
#include <folly/logging/StandardLogHandler.h>
#include <folly/logging/StandardLogHandlerFactory.h>

// dummy template to make sure deprecated functions are not used
// TODO: remove this once we have migrated completely.
template <typename T>
struct deprecated_function_check : std::false_type {};

struct NcclLoggerInitConfig {
  // Name of the context. This is used to identify which folder should the
  // logger being initialized for. If not provided, the default value is
  // the whole "comms" folder.
  std::string contextName{"comms"};
  std::string logPrefix{"NCCL"};
  std::string logFilePath;
  folly::LogLevel logLevel{folly::LogLevel::WARNING};
  std::function<int(void)> threadContextFn{[]() { return 0; }};
};

class NcclLogger {
 public:
  // This is not thread safe! Only provides bare minimal support for
  // simultanious calls to init.
  static void init(const NcclLoggerInitConfig& config = {});

  static void registerHandler(const NcclLoggerInitConfig& config);

  // close the logger singleton and its internal logger thread.
  static void close() noexcept;

  static std::atomic_flag firstInit_;

  NcclLogger(const NcclLogger&) = delete;
  NcclLogger& operator=(const NcclLogger&) = delete;
  ~NcclLogger();
};

class NcclLogFormatterFactory
    : public folly::StandardLogHandlerFactory::FormatterFactory {
 public:
  bool processOption(folly::StringPiece name, folly::StringPiece value)
      override;

  static void registerThreadContextFn(
      std::string_view name,
      std::function<int(void)> threadContextFn);

  std::shared_ptr<folly::LogFormatter> createFormatter(
      const std::shared_ptr<folly::LogWriter>& /* logWriter */) override;

 private:
  static std::unordered_map<std::string, std::function<int(void)>>&
  getPrefixToThreadContextFnMap();

  std::string curPrefix_{};
  std::function<int(void)> curThreadContextFn_{[]() { return 0; }};
};

// A custom log handler factory for nccl logging. This is mainly
// done for keeping the log message format same. Folly logging
// by default adds glog-style prefix to the log message. This
// can be removed if we are okay with the glog-style prefix.
class NcclLogHandlerFactory : public folly::LogHandlerFactory {
 public:
  static constexpr folly::StringPiece kLogger{"nccllogger"};
  folly::StringPiece getType() const override;

  std::shared_ptr<folly::LogHandler> createHandler(
      const Options& options) override;

  static void close();

  class WriterFactory : public folly::StandardLogHandlerFactory::WriterFactory {
   public:
    std::shared_ptr<folly::LogWriter> createWriter() override;
    bool processOption(folly::StringPiece name, folly::StringPiece value)
        override;

   private:
    std::string filePath_{};
    bool isStdout_{false};
    folly::FileWriterFactory fileWriterFactory_;
  };
};

class NcclWriterWrapper : public folly::LogWriter {
 public:
  void writeMessage(folly::StringPiece buffer, uint32_t flags = 0) override;

  void flush() override;

  bool ttyOutput() const override {
    // Most of the times we are logging to file, so return false here
    return false;
  }
};
#endif
