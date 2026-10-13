// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/registry/RuntimeVersion.h"

namespace uniflow {

const void* runtimeIdentity() noexcept {
  static uint8_t identity{0};
  return &identity;
}

RuntimeVersion runtimeVersion() {
  return {};
}

Status validateRuntimeVersion(const RuntimeVersion& version) {
  if (version.packageVersion != kRuntimePackageVersion) {
    return Err(
        ErrCode::InvalidArgument,
        "UniFlow package version mismatch: component requires " +
            version.packageVersion + ", runtime is " +
            std::string(kRuntimePackageVersion));
  }
  if (version.abiVersion != kRuntimeAbiVersion) {
    return Err(
        ErrCode::InvalidArgument,
        "UniFlow runtime ABI mismatch: component requires " +
            std::to_string(version.abiVersion) + ", runtime is " +
            std::to_string(kRuntimeAbiVersion));
  }
  if (version.identity != runtimeIdentity()) {
    return Err(
        ErrCode::InvalidArgument,
        "native component uses a different UniFlow runtime identity");
  }
  return Ok();
}

} // namespace uniflow
