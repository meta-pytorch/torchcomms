// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/core.h>
#include <glog/logging.h>

#include "comms/torchcomms/TorchCommBackend.hpp"

#define DEFAULT_RANK -1

inline std::string getCommNamePrefix(torch::comms::TorchCommBackend* comm) {
  return comm ? "[name=" + std::string(comm->getCommName()) + "]" : "";
}

inline std::string getRankPrefix(torch::comms::TorchCommBackend* comm) {
  try {
    return comm ? "[rank=" + std::to_string(comm->getRank()) + "]" : "";
  } catch (...) {
    return "";
  }
}

#define TC_LOG_METADATA_WITH_FUNC_NAME(comm)                \
  "[" << __FUNCTION__ << "()][TC]" << ::getRankPrefix(comm) \
      << ::getCommNamePrefix(comm) << " "

#define TC_LOG_METADATA(comm) \
  "[TC]" << ::getRankPrefix(comm) << ::getCommNamePrefix(comm) << " "

// variadic arg definitions for TC_VLOG, TC_LOG, and TC_LOG_IF based on:
// https://stackoverflow.com/questions/3046889/optional-parameters-with-c-macros
#define TC_VLOG_WITH_PREFIX_BUILDER(vlevel, comm) \
  VLOG(vlevel) << TC_LOG_METADATA_WITH_FUNC_NAME(comm)
#define TC_VLOG_PICKER(x, vlevel, comm, FUNC, ...) FUNC
#define TC_VLOG(...)                            \
  TC_VLOG_PICKER(                               \
      ,                                         \
      ##__VA_ARGS__,                            \
      TC_VLOG_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_VLOG_WITH_PREFIX_BUILDER(__VA_ARGS__, getDefaultCommunicator()))

#define TC_VLOG_EVERY_MS_PREFIX_BUILDER(vlevel, ms, comm) \
  VLOG_EVERY_MS(vlevel, ms) << TC_LOG_METADATA_WITH_FUNC_NAME(comm)
#define TC_VLOG_EVERY_MS_PICKER(x, vlevel, ms, comm, FUNC, ...) FUNC
#define TC_VLOG_EVERY_MS(...)                       \
  TC_VLOG_EVERY_MS_PICKER(                          \
      ,                                             \
      ##__VA_ARGS__,                                \
      TC_VLOG_EVERY_MS_PREFIX_BUILDER(__VA_ARGS__), \
      TC_VLOG_EVERY_MS_PREFIX_BUILDER(__VA_ARGS__, getDefaultCommunicator()))

// level is one of the following: INFO, WARNING, ERROR, FATAL
#define TC_LOG_WITH_PREFIX_BUILDER(level, comm) \
  LOG(level) << TC_LOG_METADATA(comm)
#define TC_LOG_PICKER(x, level, comm, FUNC, ...) FUNC
#define TC_LOG(...)                            \
  TC_LOG_PICKER(                               \
      ,                                        \
      ##__VA_ARGS__,                           \
      TC_LOG_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_LOG_WITH_PREFIX_BUILDER(__VA_ARGS__, getDefaultCommunicator()))

#define TC_LOG_IF_WITH_PREFIX_BUILDER(level, condition, comm) \
  LOG_IF(level, condition) << TC_LOG_METADATA_WITH_FUNC_NAME(comm)
#define TC_LOG_IF_PICKER(x, level, condition, comm, FUNC, ...) FUNC
#define TC_LOG_IF(...)                            \
  TC_LOG_IF_PICKER(                               \
      ,                                           \
      ##__VA_ARGS__,                              \
      TC_LOG_IF_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_LOG_IF_WITH_PREFIX_BUILDER(__VA_ARGS__, getDefaultCommunicator()))

#define TC_LOG_EVERY_MS_PREFIX_BUILDER(level, ms, comm) \
  LOG_EVERY_MS(level, ms) << TC_LOG_METADATA_WITH_FUNC_NAME(comm)
#define TC_LOG_EVERY_MS_PICKER(x, level, ms, comm, FUNC, ...) FUNC
#define TC_LOG_EVERY_MS(...)                       \
  TC_LOG_EVERY_MS_PICKER(                          \
      ,                                            \
      ##__VA_ARGS__,                               \
      TC_LOG_EVERY_MS_PREFIX_BUILDER(__VA_ARGS__), \
      TC_LOG_EVERY_MS_PREFIX_BUILDER(__VA_ARGS__, getDefaultCommunicator()))

// condition should evaluate to a bool, representing the condition to check
#define TC_CHECK_WITH_PREFIX_BUILDER(condition, comm) \
  CHECK(condition) << TC_LOG_METADATA(comm)
#define TC_CHECK_PICKER(x, condition, comm, FUNC, ...) FUNC
#define TC_CHECK(...)                            \
  TC_CHECK_PICKER(                               \
      ,                                          \
      ##__VA_ARGS__,                             \
      TC_CHECK_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_CHECK_WITH_PREFIX_BUILDER(__VA_ARGS__, getDefaultCommunicator()))

#define TC_CHECK_NOTNULL_WITH_PREFIX_BUILDER(condition, comm) \
  CHECK_NOTNULL(condition) << TC_LOG_METADATA(comm)
#define TC_CHECK_NOTNULL_PICKER(x, condition, comm, FUNC, ...) FUNC
#define TC_CHECK_NOTNULL(...)                            \
  TC_CHECK_NOTNULL_PICKER(                               \
      ,                                                  \
      ##__VA_ARGS__,                                     \
      TC_CHECK_NOTNULL_WITH_PREFIX_BUILDER(__VA_ARGS__), \
      TC_CHECK_NOTNULL_WITH_PREFIX_BUILDER(              \
          __VA_ARGS__, getDefaultCommunicator()))

// Google glog's api does not have an external function that allows one to check
// if glog is initialized or not. It does have an internal function - so we are
// declaring it here. This is a hack but has been used by a bunch of others too
// (e.g. Torch).
// Copied from https://fburl.com/code/tu9hg6gf
namespace google::glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
} // namespace google::glog_internal_namespace_

namespace {

void tryTorchCommLoggingInit(std::string_view name) {
  // This trick can only be used on UNIX platforms
  if (!::google::glog_internal_namespace_::IsGoogleLoggingInitialized()) {
    ::google::InitGoogleLogging(name.data());
    // This will trigger a kernel panic on GB200 NVIDIA driver
    // temporarily disable signal handler until NVIDIA releases the new driver
    // in late Jan.
#if !defined(__aarch64__)
    ::google::InstallFailureSignalHandler();
#endif
  }
}

torch::comms::TorchCommBackend* getDefaultCommunicator() {
  static torch::comms::TorchCommBackend* defaultCommunicator = nullptr;
  return defaultCommunicator;
}

} // namespace
