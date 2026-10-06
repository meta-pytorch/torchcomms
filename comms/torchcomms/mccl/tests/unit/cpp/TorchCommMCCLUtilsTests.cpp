// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/torchcomms/TorchCommOptions.hpp"
#include "comms/torchcomms/mccl/TorchCommMCCLUtils.hpp"

namespace torch::comms::mccl::test {

TEST(ParseOptionsTest, EmptyOptionsReturnsDefaults) {
  CommOptions options;

  auto configs = parseOptions(options);

  EXPECT_FALSE(configs.initDynamicRegime_);
  EXPECT_EQ(
      configs.garbage_collect_interval_ms_, kDefaultGarbageCollectIntervalMs);
}

TEST(ParseOptionsTest, EnableReconfigureFieldTrue) {
  CommOptions options;
  options.enable_reconfigure = true;

  auto configs = parseOptions(options);
  EXPECT_TRUE(configs.initDynamicRegime_);
}

TEST(ParseOptionsTest, EnableReconfigureFieldFalse) {
  CommOptions options;
  options.enable_reconfigure = false;

  auto configs = parseOptions(options);
  EXPECT_FALSE(configs.initDynamicRegime_);
}

TEST(ParseOptionsTest, InitDynamicRegimeHintTrue) {
  CommOptions options;
  std::vector<std::string> trueSets = {
      "true", "yes", "True", "1", "y", "Y", "YES"};

  for (const auto& trueStr : trueSets) {
    options.hints["initDynamicRegime"] = trueStr;
    auto configs = parseOptions(options);
    EXPECT_TRUE(configs.initDynamicRegime_);
  }
}

TEST(ParseOptionsTest, InitDynamicRegimeHintFalse) {
  CommOptions options;
  std::vector<std::string> falseSets = {
      "false", "no", "False", "0", "n", "N", "NO"};

  for (const auto& falseStr : falseSets) {
    options.hints["initDynamicRegime"] = falseStr;
    auto configs = parseOptions(options);
    EXPECT_FALSE(configs.initDynamicRegime_);
  }
}

TEST(ParseOptionsTest, EnableReconfigureFieldTakesPrecedence) {
  // When enable_reconfigure field is true, it should take precedence
  // over hints even if hints say false
  CommOptions options;
  options.enable_reconfigure = true;
  options.hints["initDynamicRegime"] = "false";

  auto configs = parseOptions(options);
  EXPECT_TRUE(configs.initDynamicRegime_);
}

TEST(ParseOptionsTest, EmptyInitDynamicRegimeHintThrows) {
  CommOptions options;
  options.hints["initDynamicRegime"] = "";

  EXPECT_THROW(parseOptions(options), std::runtime_error);
}

TEST(ParseOptionsTest, GarbageCollectIntervalMs) {
  CommOptions options;
  options.hints[std::string(kHintGarbageCollectIntervalMs)] = "500";

  auto configs = parseOptions(options);

  EXPECT_EQ(configs.garbage_collect_interval_ms_, 500);
}
TEST(ParseOptionsTest, UnrelatedHintsAreIgnored) {
  CommOptions options;
  options.hints["some_other_hint"] = "value";
  options.hints["unrelated_hint_key"] = "999";

  auto configs = parseOptions(options);

  EXPECT_FALSE(configs.initDynamicRegime_);
  EXPECT_EQ(
      configs.garbage_collect_interval_ms_, kDefaultGarbageCollectIntervalMs);
}

TEST(ParseOptionsTest, InvalidInitDynamicRegimeThrows) {
  CommOptions options;
  options.hints["initDynamicRegime"] = "invalid";

  EXPECT_THROW(parseOptions(options), std::runtime_error);
}

TEST(ParseOptionsTest, InvalidGarbageCollectIntervalMsThrows) {
  CommOptions options;
  options.hints[std::string(kHintGarbageCollectIntervalMs)] = "not_a_number";

  EXPECT_THROW(parseOptions(options), std::invalid_argument);
}

TEST(ParseOptionsTest, GarbageCollectIntervalMsZeroIsValid) {
  CommOptions options;
  options.hints[std::string(kHintGarbageCollectIntervalMs)] = "0";

  auto configs = parseOptions(options);

  EXPECT_EQ(configs.garbage_collect_interval_ms_, 0);
}

TEST(ParseOptionsTest, BothOptionsSetSimultaneously) {
  CommOptions options;
  options.hints["initDynamicRegime"] = "true";
  options.hints[std::string(kHintGarbageCollectIntervalMs)] = "250";

  auto configs = parseOptions(options);

  EXPECT_TRUE(configs.initDynamicRegime_);
  EXPECT_EQ(configs.garbage_collect_interval_ms_, 250);
}

TEST(ParseOptionsTest, InitModeDefaultIsFullMesh) {
  CommOptions options;

  auto configs = parseOptions(options);

  EXPECT_EQ(configs.initMode_, "full_mesh");
}

TEST(ParseOptionsTest, InitModeValidValues) {
  CommOptions options;
  std::vector<std::string> validModes = {"full_mesh", "ring"};

  for (const auto& mode : validModes) {
    options.hints["initMode"] = mode;
    auto configs = parseOptions(options);
    EXPECT_EQ(configs.initMode_, mode);
  }
}

TEST(ParseOptionsTest, InitModeInvalidThrows) {
  CommOptions options;
  options.hints["initMode"] = "invalid";

  EXPECT_THROW(parseOptions(options), std::invalid_argument);
}

} // namespace torch::comms::mccl::test
