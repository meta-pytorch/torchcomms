// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/utils/Abort.h"

namespace ctran::testing {

TEST(AbortCompatTest, oldNamespaceAliasesCanonicalAbort) {
  ctran::utils::Abort abort{/*enabled=*/true};

  EXPECT_TRUE(abort.Enabled());

  abort.Set();

  EXPECT_TRUE(abort.Test());
}

TEST(AbortCompatTest, oldCreateAbortBoolShimStillWorks) {
  auto abort = ctran::utils::createAbort(/*enabled=*/true);

  EXPECT_TRUE(abort->Enabled());

  abort->Set();

  EXPECT_TRUE(abort->Test());
}

TEST(AbortCompatTest, oldCreateAbortDisabledShimStillWorks) {
  auto abort = ctran::utils::createAbort(/*enabled=*/false);

  EXPECT_FALSE(abort->Enabled());

  abort->Set();

  EXPECT_FALSE(abort->Test());
}

} // namespace ctran::testing
