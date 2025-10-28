// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include <fmt/format.h>
#include <folly/Range.h>
#include <folly/logging/LogFormatter.h>
#include <folly/logging/LogLevel.h>

namespace meta::comms::logger {

enum class LogLevel {
  NONE = 0,
  VERSION = 1,
  ERROR = 2,
  WARN = 3,
  INFO = 4,
  ABORT = 5,
  TRACE = 6
};

enum SubSystem {
  INIT = 0x1,
  COLL = 0x2,
  P2P = 0x4,
  SHM = 0x8,
  NET = 0x10,
  GRAPH = 0x20,
  TUNING = 0x40,
  ENV = 0x80,
  ALLOC = 0x100,
  CALL = 0x200,
  PROXY = 0x400,
  NVLS = 0x800,
  BOOTSTRAP = 0x1000,
  REG = 0x2000,
  PROFILE = 0x4000,
  RAS = 0x8000,
  ALL = ~0
};

// TODO: Properly clean this up so that we directly parse NCCL logs to folly
// levels.
folly::LogLevel loggerLevelToFollyLogLevel(LogLevel level);

folly::StringPiece getCategoryNthParent(folly::StringPiece category, int n);

uint64_t parseDebugSubsysMask(const char* ncclDebugSubsysEnv);

std::string parseDebugFile(const char* ncclDebugFileEnv);

LogLevel getLoggerDebugLevel(std::string_view levelStr);

void initProcMetaData();

void initThreadMetaData(std::string_view threadName);

fmt::memory_buffer getLogPrefix(LogLevel level);

const char* getLastCommsError();

class NcclLogFormatter : public folly::LogFormatter {
 public:
  NcclLogFormatter(
      std::string prefix,
      std::function<int(void)> threadContextFn);

  std::string formatMessage(
      const folly::LogMessage& message,
      const folly::LogCategory* handlerCategory) override;

 private:
  std::string prefix_;
  std::function<int(void)> threadContextFn_;
};

} // namespace meta::comms::logger

#define COMMS_NAMED_THREAD_START_EXT(threadName, rank, commHash, commDesc)                \
  do {                                                                                    \
    meta::comms::logger::initThreadMetaData(threadName);                                  \
    XLOGF(                                                                                \
        INFO,                                                                             \
        "[COMMS THREAD] Starting {} thread for rank {} commHash {:#x} commDesc {} at {}", \
        threadName,                                                                       \
        rank,                                                                             \
        commHash,                                                                         \
        commDesc,                                                                         \
        __func__);                                                                        \
  } while (0);
