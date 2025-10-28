// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/Logger.h"

#include <memory>
#include <stdexcept>

#include <folly/logging/Init.h>
#include <folly/logging/LoggerDB.h>
#include <folly/logging/xlog.h>

#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/LoggingFormat.h"
#include "comms/utils/logger/NcclWriterWrapper.h"
#include "comms/utils/logger/ScubaLogger.h"

std::atomic_flag NcclLogger::firstInit_;

namespace {
std::string getHandlerNameWithCategory(
    std::string_view handlerfatoryName,
    std::string_view categoryName) {
  return fmt::format("{}:{}", handlerfatoryName, categoryName);
}
} // namespace

/* static */
void NcclLogger::init(const NcclLoggerInitConfig& config) {
  if (!firstInit_.test_and_set()) {
    folly::initLoggingOrDie();
    DataTableWrapper::init();
    NcclLogFormatterFactory::registerThreadContextFn(
        config.logPrefix, config.threadContextFn);
    NcclLogger::registerHandler(
        {.contextName = "comms.utils",
         .logPrefix = config.logPrefix,
         .logFilePath = config.logFilePath,
         .logLevel = config.logLevel,
         .threadContextFn = config.threadContextFn});
  }
  NcclLogFormatterFactory::registerThreadContextFn(
      config.logPrefix, config.threadContextFn);
  NcclLogger::registerHandler(config);
}

void NcclLogger::registerHandler(const NcclLoggerInitConfig& config) {
  folly::LoggerDB::get().registerHandlerFactory(
      std::make_unique<NcclLogHandlerFactory>(), true);
  const auto kLogger = NcclLogHandlerFactory::kLogger;
  auto fullConfig = folly::LoggerDB::get().getFullConfig();
  // Copy the default handler
  auto handlerConfig = fullConfig.getHandlerConfigs().at("default");
  handlerConfig.type = kLogger.str();
  if (NCCL_DEBUG_LOGGING_ASYNC) {
    handlerConfig.options["async"] = "true";
    // Only log fatal messages synchronously to ensure INFO/WARN/ERR messages
    // will not be out of order.
    handlerConfig.options["sync_level"] = "fatal";
  }
  if (config.logFilePath.empty()) {
    handlerConfig.options["stream"] = "stdout";
  } else {
    handlerConfig.options["path"] = config.logFilePath;
    handlerConfig.options.erase("stream");
  }
  handlerConfig.options["formatter_prefix"] = config.logPrefix;

  auto logHandlerName = getHandlerNameWithCategory(kLogger, config.contextName);

  folly::LogCategoryConfig currentCategoryConfig{
      config.logLevel,
      /* inheritParent */ false,
      std::vector<std::string>{logHandlerName}};
  currentCategoryConfig.propagateLevelMessagesToParent =
      folly::LogLevel::MAX_LEVEL; // No messages are propagated to parent

  // Setup the utils using the same config
  folly::LoggerDB::get().updateConfig(
      folly::LogConfig(
          {{logHandlerName, handlerConfig}},
          {{std::string{config.contextName}, currentCategoryConfig}}));
}

void NcclLogger::close() noexcept {
  NcclLogHandlerFactory::close();
  DataTableWrapper::shutdown();
  firstInit_.clear();
}

std::shared_ptr<folly::LogWriter>
NcclLogHandlerFactory::WriterFactory::createWriter() {
  if (isStdout_) {
    return fileWriterFactory_.createWriter(
        folly::File{STDOUT_FILENO, /* ownsFd */ false});
  }
  auto fileWriter = fileWriterFactory_.createWriter(
      folly::File{filePath_, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC});
  auto stderrWriter = fileWriterFactory_.createWriter(
      folly::File{STDERR_FILENO, /* ownsFd */ false});
  return std::make_shared<meta::comms::logger::NcclWriterWrapper>(
      std::move(fileWriter), std::move(stderrWriter));
}

bool NcclLogHandlerFactory::WriterFactory::processOption(
    folly::StringPiece name,
    folly::StringPiece value) {
  if (name == "path") {
    filePath_ = value.str();
    return true;
  }
  if (name == "stream") {
    if (value.str() == "stdout") {
      isStdout_ = true;
      return true;
    }
    throw std::runtime_error("Invalid stream option for nccl log handler");
  }
  return fileWriterFactory_.processOption(name, value);
}

folly::StringPiece NcclLogHandlerFactory::getType() const {
  return kLogger;
}

std::shared_ptr<folly::LogHandler> NcclLogHandlerFactory::createHandler(
    const Options& options) {
  WriterFactory writerFactory;
  NcclLogFormatterFactory formatterFactory;
  return folly::StandardLogHandlerFactory::createHandler(
      getType(), &writerFactory, &formatterFactory, options);
}

void NcclLogHandlerFactory::close() {
  folly::LoggerDB::get().unregisterHandlerFactory(kLogger);
}

std::unordered_map<std::string, std::function<int(void)>>&
NcclLogFormatterFactory::getPrefixToThreadContextFnMap() {
  static std::unordered_map<std::string, std::function<int(void)>> map{};
  return map;
}

void NcclLogFormatterFactory::registerThreadContextFn(
    std::string_view name,
    std::function<int(void)> threadContextFn) {
  if (getPrefixToThreadContextFnMap().contains(std::string{name})) {
    XLOGF(DBG1, "Prefix {} re-registering thread context fn", name);
  }
  getPrefixToThreadContextFnMap().insert_or_assign(
      std::string{name}, threadContextFn);
}

std::shared_ptr<folly::LogFormatter> NcclLogFormatterFactory::createFormatter(
    const std::shared_ptr<folly::LogWriter>& /* logWriter */) {
  return std::make_shared<meta::comms::logger::NcclLogFormatter>(
      curPrefix_, curThreadContextFn_);
}

bool NcclLogFormatterFactory::processOption(
    folly::StringPiece name,
    folly::StringPiece value) {
  if (name != "formatter_prefix") {
    return false;
  }
  curPrefix_ = value;
  if (!getPrefixToThreadContextFnMap().contains(std::string{value})) {
    XLOGF(ERR, "Thread Context Fn dict does not contain prefix {}", value);
  } else {
    curThreadContextFn_ =
        getPrefixToThreadContextFnMap().at(std::string{value});
  }
  return true;
}
