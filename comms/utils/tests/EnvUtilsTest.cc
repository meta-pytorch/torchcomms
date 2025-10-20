// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/utils/EnvUtils.h"

#include <gtest/gtest.h>

namespace meta::comms {

TEST(EnvUtilsTest, envDoesNotExist) {
  ASSERT_FALSE(getStrEnv("ENV_DOES_NOT_EXIST"));
}

TEST(EnvUtilsTest, envExists) {
  setenv("JOB_NAME", "BLAHBLAH", 0);

  auto jobName = getStrEnv("JOB_NAME");
  ASSERT_TRUE(jobName);
  ASSERT_EQ(*getStrEnv("JOB_NAME"), "BLAHBLAH");
}

} // namespace meta::comms
