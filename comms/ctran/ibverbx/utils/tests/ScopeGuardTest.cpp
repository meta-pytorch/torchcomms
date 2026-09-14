// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/ibverbx/utils/ScopeGuard.h"

#include <utility>

#include <gtest/gtest.h>

namespace ibverbx::utils {

TEST(ScopeGuardTest, RunsOnScopeExit) {
  int cleanups = 0;

  {
    auto guard = makeScopeGuard([&] { ++cleanups; });
    EXPECT_EQ(cleanups, 0);
  }

  EXPECT_EQ(cleanups, 1);
}

TEST(ScopeGuardTest, DismissSkipsCleanup) {
  int cleanups = 0;

  {
    auto guard = makeScopeGuard([&] { ++cleanups; });
    guard.dismiss();
  }

  EXPECT_EQ(cleanups, 0);
}

// The move constructor dismisses the source, so ownership transfers rather than
// duplicating: exactly one of the two guards may run the cleanup.
TEST(ScopeGuardTest, MoveRunsCleanupOnce) {
  int cleanups = 0;

  {
    auto source = makeScopeGuard([&] { ++cleanups; });
    {
      auto moved = std::move(source);
      EXPECT_EQ(cleanups, 0);
    }
    EXPECT_EQ(cleanups, 1);
  }

  EXPECT_EQ(cleanups, 1);
}

TEST(ScopeGuardTest, MovePreservesDismissal) {
  int cleanups = 0;

  {
    auto source = makeScopeGuard([&] { ++cleanups; });
    source.dismiss();
    auto moved = std::move(source);
  }

  EXPECT_EQ(cleanups, 0);
}

} // namespace ibverbx::utils
