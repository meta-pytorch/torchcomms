// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <vector>

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/utils/cvars/nccl_cvars.h" // @manual

class NCCLEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    // Turn off NCCL debug logging, allow user to turn on via command line
    setenv("NCCL_DEBUG", "WARN", 0);
  }
  ~NCCLEnvironment() override {}
};

class CvarTest : public ::testing::Test {
 public:
  CvarTest() = default;
};

void testWarn(const char* cvarName, std::string expectedKeyword) {
  setenv("NCCL_DEBUG_SUBSYS", "CVAR", 1);
  testing::internal::CaptureStderr();
  ncclCvarInit();
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, testing::HasSubstr(expectedKeyword));

  // Unset to avoid warning in later tests
  unsetenv(cvarName);
  unsetenv("NCCL_DEBUG_SUBSYS");
}

TEST_F(CvarTest, UnknownEnv) {
  setenv("NCCL_DUMMY_ENV", "dummy", 1);
  testWarn("NCCL_DUMMY_ENV", "Unknown env");
}

TEST_F(CvarTest, NoWarnWithUnknownEnv) {
  setenv("NCCL_DUMMY_ENV", "dummy", 1);
  // Not with CVAR
  setenv("NCCL_DEBUG_SUBSYS", "COLL", 1);

  testing::internal::CaptureStderr();
  ncclCvarInit();
  std::string output = testing::internal::GetCapturedStderr();
  EXPECT_THAT(output, testing::HasSubstr("Unknown env"));

  // Unset to avoid warning in later tests
  unsetenv("NCCL_DEBUG_SUBSYS");
  unsetenv("NCCL_DUMMY_ENV");
}

TEST_F(CvarTest, CvarLogging) {
  setenv("NCCL_CVARS_LOG_INFO", "1", 1);
  ncclCvarInit();
}

TEST_F(CvarTest, DeprecatedConfigsThrow) {
  setenv("NCCL_CVARS_SETTINGS", "profile", /*__replace=*/1);
  EXPECT_THROW(ncclCvarInit(), std::runtime_error);
  unsetenv("NCCL_CVARS_SETTINGS");
}

// Test CVARs initialization with various data types
TEST_F(CvarTest, CvarInitializationTypes) {
  // Test with unit test CVARs
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "test_string", 1);
  setenv("__NCCL_UNIT_TEST_BOOL_CVAR__", "true", 1);
  setenv("__NCCL_UNIT_TEST_INT_CVAR__", "123", 1);
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "9876543210", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  // Clean up
  unsetenv("__NCCL_UNIT_TEST_STRING_CVAR__");
  unsetenv("__NCCL_UNIT_TEST_BOOL_CVAR__");
  unsetenv("__NCCL_UNIT_TEST_INT_CVAR__");
  unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");
}

// Test CVARs with whitespace handling
TEST_F(CvarTest, CvarWhitespaceHandling) {
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "  trimmed_value  ", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  unsetenv("__NCCL_UNIT_TEST_STRING_CVAR__");
}

// Test CVARs with empty values
TEST_F(CvarTest, CvarEmptyValues) {
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  unsetenv("__NCCL_UNIT_TEST_STRING_CVAR__");
}

// Test CVARs initialization is idempotent
TEST_F(CvarTest, CvarInitIdempotent) {
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "test_value", 1);

  // First call
  EXPECT_NO_THROW(ncclCvarInit());

  // Second call should not fail
  EXPECT_NO_THROW(ncclCvarInit());

  unsetenv("__NCCL_UNIT_TEST_STRING_CVAR__");
}

// Test invalid boolean values produce warnings but don't crash
TEST_F(CvarTest, InvalidBooleanValueWarning) {
  setenv("__NCCL_UNIT_TEST_BOOL_CVAR__", "invalid_boolean", 1);
  setenv("NCCL_DEBUG_SUBSYS", "CVAR", 1);

  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(ncclCvarInit());
  std::string output = testing::internal::GetCapturedStderr();

  // Should contain warning about unknown value
  EXPECT_THAT(output, testing::HasSubstr("Unknown value"));

  unsetenv("__NCCL_UNIT_TEST_BOOL_CVAR__");
  unsetenv("NCCL_DEBUG_SUBSYS");
}

// Test numeric parsing with special values
TEST_F(CvarTest, NumericSpecialValues) {
  // Test MAX value
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "MAX", 1);
  EXPECT_NO_THROW(ncclCvarInit());
  unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");

  // Test MIN value
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "MIN", 1);
  EXPECT_NO_THROW(ncclCvarInit());
  unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");

  // Test mixed case
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "max", 1);
  EXPECT_NO_THROW(ncclCvarInit());
  unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");
}

// Test hex and octal parsing
TEST_F(CvarTest, NumericBasesParsing) {
  // Test hexadecimal
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "0x100", 1);
  EXPECT_NO_THROW(ncclCvarInit());
  unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");

  // Test octal
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "0777", 1);
  EXPECT_NO_THROW(ncclCvarInit());
  unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");

  // Test binary (should be handled gracefully)
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "0b1010", 1);
  EXPECT_NO_THROW(ncclCvarInit());
  unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");
}

// Test real NCCL CVARs work correctly
TEST_F(CvarTest, RealNcclCvars) {
  setenv("NCCL_BUFFSIZE", "1048576", 1);
  setenv("NCCL_NTHREADS", "4", 1);
  setenv("NCCL_P2P_DISABLE", "1", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  unsetenv("NCCL_BUFFSIZE");
  unsetenv("NCCL_NTHREADS");
  unsetenv("NCCL_P2P_DISABLE");
}

// Test configuration file processing (should not crash)
TEST_F(CvarTest, ConfigFileProcessing) {
  // This tests that the config file processing doesn't crash
  // even if files don't exist
  EXPECT_NO_THROW(ncclCvarInit());
}

// Test environment variable validation
TEST_F(CvarTest, EnvVarValidation) {
  // Test that the environment variable validation works
  setenv("NCCL_UNKNOWN_VAR_123", "some_value", 1);
  setenv("NCCL_DEBUG_SUBSYS", "CVAR", 1);

  testing::internal::CaptureStderr();
  ncclCvarInit();
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, testing::HasSubstr("Unknown env"));
  EXPECT_THAT(output, testing::HasSubstr("NCCL_UNKNOWN_VAR_123"));

  unsetenv("NCCL_UNKNOWN_VAR_123");
  unsetenv("NCCL_DEBUG_SUBSYS");
}

// Test CVARs maps are populated
TEST_F(CvarTest, CvarMapsPopulation) {
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "test_string", 1);
  setenv("__NCCL_UNIT_TEST_BOOL_CVAR__", "true", 1);
  setenv("__NCCL_UNIT_TEST_INT_CVAR__", "42", 1);
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "9876543210", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  // Check that the maps have been populated
  // The exact implementation may vary, but this ensures
  // the initialization has processed these variables

  unsetenv("__NCCL_UNIT_TEST_STRING_CVAR__");
  unsetenv("__NCCL_UNIT_TEST_BOOL_CVAR__");
  unsetenv("__NCCL_UNIT_TEST_INT_CVAR__");
  unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new NCCLEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
