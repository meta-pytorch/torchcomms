// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/core.h>
#include <glog/logging.h>

#include "comms/torchcomms/TorchCommBackend.hpp"

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

#define TC_LOG_METADATA(comm) \
  "[TC]" << ::getRankPrefix(comm) << ::getCommNamePrefix(comm) << " "

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

// Google glog's api does not have an external function that allows one to check
// if glog is initialized or not. It does have an internal function - so we are
// declaring it here. This is a hack but has been used by a bunch of others too
// (e.g. Torch).
// Copied from https://fburl.com/code/tu9hg6gf
namespace google::glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
} // namespace google::glog_internal_namespace_

namespace {

[[maybe_unused]] void tryTorchCommLoggingInit(std::string_view name) {
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

[[maybe_unused]] torch::comms::TorchCommBackend* getDefaultCommunicator() {
  static torch::comms::TorchCommBackend* defaultCommunicator = nullptr;
  return defaultCommunicator;
}

} // namespace
