// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/DeviceRackSerial.h"

#include <cstring>
#include <fstream>
#include <string>

namespace ncclx {

namespace {
constexpr std::string_view kDeviceRackSerial = "DEVICE_RACK_SERIAL";
} // namespace

bool loadRackSerial(const std::string& filepath, char* out, size_t maxLen) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    size_t pos = line.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    const auto key = line.substr(0, pos);
    if (key != kDeviceRackSerial) {
      continue;
    }
    const auto value = line.substr(pos + 1);
    if (value.empty() || value.size() >= maxLen) {
      return false;
    }
    std::strncpy(out, value.c_str(), maxLen);
    out[maxLen - 1] = '\0';
    return true;
  }
  return false;
}

} // namespace ncclx
