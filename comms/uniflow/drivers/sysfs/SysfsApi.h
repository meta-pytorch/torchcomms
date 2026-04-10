// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <vector>

#include "comms/uniflow/Result.h"

namespace uniflow {

/// Mockable interface for sysfs filesystem access.
/// All methods are virtual for testability via MockSysfsApi.
/// Default implementation reads from the real filesystem.
class SysfsApi {
 public:
  virtual ~SysfsApi() = default;

  /// Resolve symlinks in a path via realpath(3).
  virtual Result<std::string> resolvePath(const std::string& path);

  /// Read a sysfs file and return its first line (trimmed).
  /// Returns empty string on error.
  virtual std::string readFile(const std::string& path);

  /// List directory entries matching a prefix.
  /// Returns entry names (not full paths).
  virtual std::vector<std::string> listDir(
      std::string_view dir,
      std::string_view prefix = "");
};

} // namespace uniflow
