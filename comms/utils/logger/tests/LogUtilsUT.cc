// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/LogUtils.h"

#include <gtest/gtest.h>

namespace meta::comms::logger {
namespace {

TEST(LogUtilsTest, ChecksConfiguredSubsystemMask) {
  setSubSystemMask(COLL);
  EXPECT_TRUE(CLOGF_ENABLED(COLL));
  EXPECT_FALSE(CLOGF_ENABLED(NET));

  setSubSystemMask(NET | P2P);
  EXPECT_TRUE(CLOGF_ENABLED(NET));
  EXPECT_TRUE(CLOGF_ENABLED(P2P));
  EXPECT_TRUE(CLOGF_ENABLED(NET | P2P));
  EXPECT_FALSE(CLOGF_ENABLED(COLL));

  setSubSystemMask(0);
}

} // namespace
} // namespace meta::comms::logger
