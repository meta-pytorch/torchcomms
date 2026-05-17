// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <cstring>
#include <string>

namespace ncclx {

// Load DEVICE_RACK_SERIAL from a topology file into a caller-provided buffer.
// Returns true if found and fits within maxLen, false otherwise.
bool loadRackSerial(const std::string& filepath, char* out, size_t maxLen);

// Returns true only if both serials are non-empty and equal.
inline bool isSameRackSerial(const char* a, const char* b) {
  return a[0] != '\0' && b[0] != '\0' && std::strcmp(a, b) == 0;
}

} // namespace ncclx
