// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "checks.h"
#include "param.h"

TEST(ChecksTest, CheckAbort) {
  initEnv();
  int a = 1;
  int b = 2;
  auto func = [&]() { CHECKABORT(a == b, "a %d != b %d", a, b); };
  EXPECT_DEATH(func(), "");
}
