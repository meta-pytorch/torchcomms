// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/core.h>
#include <glog/logging.h>
#include <optional>

#define LOG_METADATA_WITH_FUNC_NAME(prefixBuilder) \
  fmt::format("[{}()]{}", __FUNCTION__, prefixBuilder.getPrefix())

#define LOG_METADATA(prefixBuilder) prefixBuilder.getPrefix()

// variadic arg definitions for TC_VLOG, TC_LOG, and TC_LOG_IF based on:
// https://stackoverflow.com/questions/3046889/optional-parameters-with-c-macros
#define TC_VLOG_WITH_PREFIX_BUILDER(vlevel, prefixBuilder) \
  VLOG(vlevel) << LOG_METADATA_WITH_FUNC_NAME(prefixBuilder)
#define TC_VLOG_PICKER(x, vlevel, prefixBuilder, FUNC, ...) FUNC
#define TC_VLOG(...)                            \
  TC_VLOG_PICKER(                               \
      ,                                         \
      ##__VA_ARGS__,                            \
      TC_VLOG_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_VLOG_WITH_PREFIX_BUILDER(__VA_ARGS__, getDefaultPrefixBuilder()))

#define TC_VLOG_EVERY_MS_PREFIX_BUILDER(vlevel, ms, prefixBuilder) \
  VLOG_EVERY_MS(vlevel, ms) << LOG_METADATA_WITH_FUNC_NAME(prefixBuilder)
#define TC_VLOG_EVERY_MS_PICKER(x, vlevel, ms, prefixBuilder, FUNC, ...) FUNC
#define TC_VLOG_EVERY_MS(...)                       \
  TC_VLOG_EVERY_MS_PICKER(                          \
      ,                                             \
      ##__VA_ARGS__,                                \
      TC_VLOG_EVERY_MS_PREFIX_BUILDER(__VA_ARGS__), \
      TC_VLOG_EVERY_MS_PREFIX_BUILDER(__VA_ARGS__, getDefaultPrefixBuilder()))

// level is one of the following: INFO, WARNING, ERROR, FATAL
#define TC_LOG_WITH_PREFIX_BUILDER(level, prefixBuilder) \
  LOG(level) << LOG_METADATA(prefixBuilder)
#define TC_LOG_PICKER(x, level, prefixBuilder, FUNC, ...) FUNC
#define TC_LOG(...)                            \
  TC_LOG_PICKER(                               \
      ,                                        \
      ##__VA_ARGS__,                           \
      TC_LOG_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_LOG_WITH_PREFIX_BUILDER(__VA_ARGS__, getDefaultPrefixBuilder()))

#define TC_LOG_IF_WITH_PREFIX_BUILDER(level, condition, prefixBuilder) \
  LOG_IF(level, condition) << LOG_METADATA_WITH_FUNC_NAME(prefixBuilder)
#define TC_LOG_IF_PICKER(x, level, condition, prefixBuilder, FUNC, ...) FUNC
#define TC_LOG_IF(...)                            \
  TC_LOG_IF_PICKER(                               \
      ,                                           \
      ##__VA_ARGS__,                              \
      TC_LOG_IF_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_LOG_IF_WITH_PREFIX_BUILDER(__VA_ARGS__, getDefaultPrefixBuilder()))

#define TC_LOG_EVERY_MS_PREFIX_BUILDER(level, ms, prefixBuilder) \
  LOG_EVERY_MS(level, ms) << LOG_METADATA_WITH_FUNC_NAME(prefixBuilder)
#define TC_LOG_EVERY_MS_PICKER(x, level, ms, prefixBuilder, FUNC, ...) FUNC
#define TC_LOG_EVERY_MS(...)                       \
  TC_LOG_EVERY_MS_PICKER(                          \
      ,                                            \
      ##__VA_ARGS__,                               \
      TC_LOG_EVERY_MS_PREFIX_BUILDER(__VA_ARGS__), \
      TC_LOG_EVERY_MS_PREFIX_BUILDER(__VA_ARGS__, getDefaultPrefixBuilder()))

// condition should evaluate to a bool, representing the condition to check
#define TC_CHECK_WITH_PREFIX_BUILDER(condition, prefixBuilder) \
  CHECK(condition) << LOG_METADATA(prefixBuilder)
#define TC_CHECK_PICKER(x, condition, prefixBuilder, FUNC, ...) FUNC
#define TC_CHECK(...)                            \
  TC_CHECK_PICKER(                               \
      ,                                          \
      ##__VA_ARGS__,                             \
      TC_CHECK_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_CHECK_WITH_PREFIX_BUILDER(__VA_ARGS__, getDefaultPrefixBuilder()))

#define TC_CHECK_NOTNULL_WITH_PREFIX_BUILDER(condition, prefixBuilder) \
  CHECK_NOTNULL(condition) << LOG_METADATA(prefixBuilder)
#define TC_CHECK_NOTNULL_PICKER(x, condition, prefixBuilder, FUNC, ...) FUNC
#define TC_CHECK_NOTNULL(...)                            \
  TC_CHECK_NOTNULL_PICKER(                               \
      ,                                                  \
      ##__VA_ARGS__,                                     \
      TC_CHECK_NOTNULL_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_CHECK_NOTNULL_WITH_PREFIX_BUILDER(              \
          __VA_ARGS__, getDefaultPrefixBuilder()))

namespace torch::comms {

const google::LogSeverity kMinLogSeverity = google::ERROR;

class LogPrefixBuilder {
 public:
  explicit LogPrefixBuilder(int commRank) : commRank_(commRank) {
    initializePrefix();
  }
  // Build the prefix and return it. If there was an existing prefix, it will be
  // overwritten based on recent values set.
  void build();
  const std::string& getPrefix() const {
    return prefix_;
  }

  LogPrefixBuilder& setRank(int commRank) {
    commRank_ = commRank;
    return *this;
  }

  LogPrefixBuilder& setCommName(std::string_view commName) {
    commName_ = commName;
    return *this;
  }

  LogPrefixBuilder& resetDefaultPrefix() {
    defaultPrefix_ = "";
    return *this;
  }

 private:
  int commRank_ = -1;
  std::optional<std::string> commName_ = std::nullopt;
  std::string defaultPrefix_;
  std::string prefix_;

  const std::string& getDefaultPrefix();

  void initializePrefix() {
    getDefaultPrefix();
    prefix_ = defaultPrefix_;
  }
};

void tryTorchCommLoggingInit(
    std::string_view name,
    int commRank,
    const std::string& commName);
// Helper functions to create and get log prefix builder instance

// Use this function when you just want to access the default prefix builder.
// You cannot add additional metadata to this object as this is shared.
LogPrefixBuilder& getDefaultPrefixBuilder();
} // namespace torch::comms
