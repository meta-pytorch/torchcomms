// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/torchcomms/utils/Utils.hpp"

using ::testing::_;
using ::testing::DoAll;
using ::testing::InSequence;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::SetArgPointee;

namespace torch::comms::test {

TEST(TorchCommOptionsTest, EnvToValueImplBool) {
  const std::set<std::string> truthy_values = {"1", "true", "yes", "y"};
  const std::set<std::string> falsy_values = {"0", "false", "no", "n"};
  const char* env_key = "TEST_ENV_KEY";

  for (const auto& value : truthy_values) {
    setenv(env_key, value.c_str(), 1);
    bool result = torch::comms::env_to_value<bool>(env_key, false);
    ASSERT_TRUE(result);
    unsetenv(env_key);
  }
  for (const auto& value : falsy_values) {
    setenv(env_key, value.c_str(), 1);
    bool result = torch::comms::env_to_value<bool>(env_key, false);
    ASSERT_FALSE(result);
    unsetenv(env_key);
  }

  const char* env_value5 = "invalid";
  setenv(env_key, env_value5, 1);
  EXPECT_THROW(
      torch::comms::env_to_value<bool>(env_key, true), std::runtime_error);
  unsetenv(env_key);
}

TEST(TorchCommOptionsTest, StringToBool) {
  // Test true values with different case combinations
  EXPECT_TRUE(torch::comms::string_to_bool("1"));
  EXPECT_TRUE(torch::comms::string_to_bool("true"));
  EXPECT_TRUE(torch::comms::string_to_bool("TRUE"));
  EXPECT_TRUE(torch::comms::string_to_bool("True"));
  EXPECT_TRUE(torch::comms::string_to_bool("yes"));
  EXPECT_TRUE(torch::comms::string_to_bool("YES"));
  EXPECT_TRUE(torch::comms::string_to_bool("Yes"));
  EXPECT_TRUE(torch::comms::string_to_bool("y"));
  EXPECT_TRUE(torch::comms::string_to_bool("Y"));

  // Test false values with different case combinations
  EXPECT_FALSE(torch::comms::string_to_bool("0"));
  EXPECT_FALSE(torch::comms::string_to_bool("false"));
  EXPECT_FALSE(torch::comms::string_to_bool("FALSE"));
  EXPECT_FALSE(torch::comms::string_to_bool("False"));
  EXPECT_FALSE(torch::comms::string_to_bool("no"));
  EXPECT_FALSE(torch::comms::string_to_bool("NO"));
  EXPECT_FALSE(torch::comms::string_to_bool("No"));
  EXPECT_FALSE(torch::comms::string_to_bool("n"));
  EXPECT_FALSE(torch::comms::string_to_bool("N"));

  // Test invalid values that should throw exceptions
  EXPECT_THROW(torch::comms::string_to_bool(""), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("invalid"), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("2"), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("truee"), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("yess"), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("nope"), std::runtime_error);
  EXPECT_THROW(torch::comms::string_to_bool("falsey"), std::runtime_error);
}

} // namespace torch::comms::test
