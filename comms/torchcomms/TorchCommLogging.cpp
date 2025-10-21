// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "TorchCommLogging.hpp"

// Google glog's api does not have an external function that allows one to check
// if glog is initialized or not. It does have an internal function - so we are
// declaring it here. This is a hack but has been used by a bunch of others too
// (e.g. Torch).
// Copied from https://fburl.com/code/tu9hg6gf
namespace google::glog_internal_namespace_ {
bool IsGoogleLoggingInitialized();
} // namespace google::glog_internal_namespace_

namespace torch::comms {

void LogPrefixBuilder::build() {
  prefix_.clear();
  prefix_.reserve(36);
  prefix_.append(getDefaultPrefix());
  prefix_.append(" ");
}

const std::string& LogPrefixBuilder::getDefaultPrefix() {
  if (!defaultPrefix_.empty()) {
    return defaultPrefix_;
  }
  defaultPrefix_.reserve(35);
  defaultPrefix_.append("[TC][rank=")
      .append(std::to_string(commRank_))
      .append("]");
  if (commName_.has_value()) {
    defaultPrefix_.append("[name=").append(commName_.value()).append("]");
  }
  return defaultPrefix_;
}

LogPrefixBuilder& getDefaultPrefixBuilder() {
  // if global rank is not unset, we use a dummy object until it's set.
  static LogPrefixBuilder dummyBuilder(-1);
  return dummyBuilder;
}

void tryTorchCommLoggingInit(
    std::string_view name,
    int commRank,
    const std::string& commName) {
  // This trick can only be used on UNIX platforms
  if (!::google::glog_internal_namespace_::IsGoogleLoggingInitialized()) {
    ::google::InitGoogleLogging(name.data());
    // This is never defined on Windows
    ::google::InstallFailureSignalHandler();
  }

  auto& builder = getDefaultPrefixBuilder().setRank(commRank);
  if (!commName.empty()) {
    builder = builder.setCommName(commName);
  }
  builder.resetDefaultPrefix().build();
}
} // namespace torch::comms
