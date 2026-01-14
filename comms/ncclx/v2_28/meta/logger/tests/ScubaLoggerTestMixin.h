// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include <folly/testing/TestUtil.h>

// If your test needs to log to scuba, you can use this mixin to set up the
// necessary logging state. Example usage:
//
// class MyFooTest : public ::testing::Test, public ScubaLoggerTestMixin {
//  public:
//   void SetUp() override {
//     ScubaLoggerTestMixin::SetUp();
//   }
// };
//
// TEST_F(MyFooTest, Foo) { ... }
class ScubaLoggerTestMixin {
 public:
  void SetUp();

  const folly::test::TemporaryDirectory& scubaDir() const;

 private:
  std::unique_ptr<folly::test::TemporaryDirectory> tmpDir_;
};
