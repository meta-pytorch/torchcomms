// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/utils/EnvUtils.h"

#include <cctype>
#include <cstdlib>
#include <string_view>

namespace meta::comms {

std::optional<std::string> getStrEnv(const std::string& name) {
  const char* val = std::getenv(name.c_str());
  if (val == nullptr) {
    return std::nullopt;
  }
  return std::string(val);
}

namespace {

std::string toLowerAscii(std::string_view s) {
  std::string out(s);
  for (auto& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

} // namespace

bool getBoolEnv(const char* name, bool defaultValue) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return defaultValue;
  }
  auto v = toLowerAscii(raw);
  if (v == "1" || v == "true" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "0" || v == "false" || v == "no" || v == "off") {
    return false;
  }
  return defaultValue;
}

} // namespace meta::comms
