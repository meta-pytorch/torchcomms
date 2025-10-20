// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/utils/EnvUtils.h"

namespace meta::comms {

std::optional<std::string> getStrEnv(const std::string& name) {
  const char* val = std::getenv(name.c_str());
  if (val == nullptr) {
    return std::nullopt;
  }
  return std::string(val);
}

} // namespace meta::comms
