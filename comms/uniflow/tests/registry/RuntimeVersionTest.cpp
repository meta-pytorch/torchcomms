// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/registry/RuntimeVersion.h"

#include <gtest/gtest.h>

using namespace uniflow;

TEST(RuntimeVersionTest, AcceptsTheCurrentComponentContract) {
  const auto version = runtimeVersion();
  EXPECT_TRUE(validateRuntimeVersion(version));
  EXPECT_TRUE(validateRuntimeVersion(RuntimeVersion{}));
}

TEST(RuntimeVersionTest, RejectsPackageAbiAndRuntimeIdentityMismatches) {
  const uint8_t foreignRuntime{0};
  for (int mismatch = 0; mismatch < 4; ++mismatch) {
    auto version = runtimeVersion();
    std::string expected;
    switch (mismatch) {
      case 0:
        version.packageVersion += ".different";
        expected = "package version";
        break;
      case 1:
        ++version.abiVersion;
        expected = "ABI";
        break;
      case 2:
        version.identity = &foreignRuntime;
        expected = "identity";
        break;
      default:
        version.identity = nullptr;
        expected = "identity";
        break;
    }

    auto result = validateRuntimeVersion(version);

    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
    EXPECT_NE(result.error().message().find(expected), std::string::npos);
  }
}
