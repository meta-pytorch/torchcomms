// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

namespace meta::comms::colltracev2 {

// colltrace v2 reads its own environment: it is not tied to any communication
// library's configuration.
inline uint64_t envUint(const char* name, uint64_t defaultValue) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return defaultValue;
  }
  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  return (end == value || *end != '\0') ? defaultValue : parsed;
}

inline std::string envStr(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string{value};
}

} // namespace meta::comms::colltracev2
