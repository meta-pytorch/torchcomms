// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>

#include "comms/uniflow/drivers/sysfs/SysfsApi.h"

namespace uniflow {

/// gmock-based mock for SysfsApi.
/// All virtual methods are mocked. Use ON_CALL / EXPECT_CALL to configure
/// behavior. Wrap with testing::NiceMock to suppress warnings for unconfigured
/// methods.
class MockSysfsApi : public SysfsApi {
 public:
  MOCK_METHOD(
      Result<std::string>,
      resolvePath,
      (const std::string& path),
      (override));
  MOCK_METHOD(std::string, readFile, (const std::string& path), (override));
  MOCK_METHOD(
      std::vector<std::string>,
      listDir,
      (std::string_view dir, std::string_view prefix),
      (override));
};

} // namespace uniflow
