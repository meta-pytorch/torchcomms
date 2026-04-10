// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/sysfs/SysfsApi.h"

#include <filesystem>
#include <fstream>

namespace uniflow {

Result<std::string> SysfsApi::resolvePath(const std::string& path) {
  char resolved[PATH_MAX];
  if (realpath(path.data(), resolved) == nullptr) {
    return Err(
        ErrCode::InvalidArgument,
        "realpath failed for " + path + ": " +
            std::system_category().message(errno));
  }
  return std::string(resolved);
}

std::string SysfsApi::readFile(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    return {};
  }
  std::string content;
  std::getline(ifs, content);
  return content;
}

std::vector<std::string> SysfsApi::listDir(
    std::string_view dir,
    std::string_view prefix) {
  namespace fs = std::filesystem;
  std::vector<std::string> result;
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    auto name = entry.path().filename().string();
    if (prefix.empty() || name.compare(0, prefix.size(), prefix) == 0) {
      result.push_back(name);
    }
  }
  return result;
}

} // namespace uniflow
