// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <future>

#include "comms/utils/cvars/nccl_baseline_adapter.h" // @manual
#include "comms/utils/cvars/nccl_cvars.h" // @manual

#include <string>
#include <vector>

class NCCLCvarInitEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    // Turn off NCCL debug logging by default
    setenv("NCCL_DEBUG", "WARN", 0);
  }
  ~NCCLCvarInitEnvironment() override {}
};

class CvarInitTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Clear any existing environment variables before each test
    clearTestEnvVars();
  }

  void TearDown() override {
    // Clean up environment after each test
    clearTestEnvVars();
  }

 private:
  void clearTestEnvVars() {
    // Clear test-specific environment variables
    unsetenv("__NCCL_UNIT_TEST_STRING_CVAR__");
    unsetenv("__NCCL_UNIT_TEST_BOOL_CVAR__");
    unsetenv("__NCCL_UNIT_TEST_INT_CVAR__");
    unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_CVARS_LOG_INFO");
    unsetenv("NCCL_CVARS_SETTINGS");
    unsetenv("NCCL_DUMMY_TEST_VAR");
    unsetenv("NCCL_BUFFSIZE");
    unsetenv("NCCL_NTHREADS");
    unsetenv("NCCL_P2P_DISABLE");
    unsetenv("CUDA_LAUNCH_BLOCKING");
    unsetenv("NCCL_MIN_CTAS");
  }
};

TEST_F(CvarInitTest, BasicInitialization) {
  // Test basic ncclCvarInit functionality
  EXPECT_NO_THROW(ncclCvarInit());
}

TEST_F(CvarInitTest, InitWithStringCvar) {
  // Test initialization with string CVARs
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "test_value", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  // Verify the string CVAR was set correctly
  const char* result =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");
  EXPECT_STREQ(result, "test_value");
}

TEST_F(CvarInitTest, InitWithBooleanCvars) {
  // Test initialization with boolean CVARs - various formats
  struct TestCase {
    const char* value;
    bool expected;
  };

  std::vector<TestCase> testCases = {
      {"true", true},   {"TRUE", true},   {"True", true}, {"false", false},
      {"FALSE", false}, {"False", false}, {"1", true},    {"0", false},
      {"yes", true},    {"YES", true},    {"no", false},  {"NO", false},
      {"y", true},      {"Y", true},      {"n", false},   {"N", false},
      {"t", true},      {"T", true},      {"f", false},   {"F", false}};

  for (const auto& testCase : testCases) {
    setenv("__NCCL_UNIT_TEST_BOOL_CVAR__", testCase.value, 1);

    EXPECT_NO_THROW(ncclCvarInit()) << "Failed for value: " << testCase.value;

    // Test via baseline adapter
    int64_t cache = -1; // Use -1 as uninitialized value
    int64_t defaultVal = 999;
    int64_t uninitializedVal = -1;

    nccl_baseline_adapter::ncclLoadParam(
        "__NCCL_UNIT_TEST_BOOL_CVAR__", defaultVal, uninitializedVal, &cache);

    EXPECT_EQ(cache, testCase.expected ? 1 : 0)
        << "Failed for value: " << testCase.value;

    unsetenv("__NCCL_UNIT_TEST_BOOL_CVAR__");
  }
}

TEST_F(CvarInitTest, InitWithIntegerCvars) {
  // Test initialization with integer CVARs
  struct TestCase {
    const char* value;
    int64_t expected;
  };

  std::vector<TestCase> testCases = {
      {"0", 0},
      {"1", 1},
      {"42", 42},
      {"12345", 12345},
      {"-1", -1},
      {"-999", -999},
      {"2147483647", 2147483647}, // INT_MAX
      {"-2147483648", -2147483648}, // INT_MIN
  };

  for (const auto& testCase : testCases) {
    setenv("__NCCL_UNIT_TEST_INT_CVAR__", testCase.value, 1);

    EXPECT_NO_THROW(ncclCvarInit()) << "Failed for value: " << testCase.value;

    // Test via baseline adapter
    int64_t cache = -999; // Use -999 as uninitialized value
    int64_t defaultVal = 555;
    int64_t uninitializedVal = -999;

    nccl_baseline_adapter::ncclLoadParam(
        "__NCCL_UNIT_TEST_INT_CVAR__", defaultVal, uninitializedVal, &cache);

    EXPECT_EQ(cache, testCase.expected)
        << "Failed for value: " << testCase.value;

    unsetenv("__NCCL_UNIT_TEST_INT_CVAR__");
  }
}

TEST_F(CvarInitTest, InitWithInt64Cvars) {
  // Test initialization with int64_t CVARs
  struct TestCase {
    const char* value;
    int64_t expected;
  };

  std::vector<TestCase> testCases = {
      {"0", 0LL},
      {"9223372036854775807", 9223372036854775807LL}, // LLONG_MAX
      {"-9223372036854775808", -9223372036854775807LL - 1}, // LLONG_MIN
      {"1000000000000", 1000000000000LL},
      {"-1000000000000", -1000000000000LL},
  };

  for (const auto& testCase : testCases) {
    setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", testCase.value, 1);

    EXPECT_NO_THROW(ncclCvarInit()) << "Failed for value: " << testCase.value;

    // Test via baseline adapter
    int64_t cache = -999; // Use -999 as uninitialized value
    int64_t defaultVal = 555;
    int64_t uninitializedVal = -999;

    nccl_baseline_adapter::ncclLoadParam(
        "__NCCL_UNIT_TEST_INT64_T_CVAR__",
        defaultVal,
        uninitializedVal,
        &cache);

    EXPECT_EQ(cache, testCase.expected)
        << "Failed for value: " << testCase.value;

    unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");
  }
}

TEST_F(CvarInitTest, InitWithSpecialNumericStrings) {
  // Test initialization with special numeric strings
  struct TestCase {
    const char* envVar;
    const char* value;
    bool shouldSucceed;
  };

  std::vector<TestCase> testCases = {
      {"__NCCL_UNIT_TEST_INT64_T_CVAR__", "MAX", true},
      {"__NCCL_UNIT_TEST_INT64_T_CVAR__", "MIN", true},
      {"__NCCL_UNIT_TEST_INT64_T_CVAR__", "max", true},
      {"__NCCL_UNIT_TEST_INT64_T_CVAR__", "min", true},
      {"__NCCL_UNIT_TEST_INT_CVAR__", "MAX", true},
      {"__NCCL_UNIT_TEST_INT_CVAR__", "MIN", true},
  };

  for (const auto& testCase : testCases) {
    setenv(testCase.envVar, testCase.value, 1);

    if (testCase.shouldSucceed) {
      EXPECT_NO_THROW(ncclCvarInit())
          << "Failed for " << testCase.envVar << "=" << testCase.value;
    } else {
      // For invalid values, init should still succeed but issue warnings
      EXPECT_NO_THROW(ncclCvarInit())
          << "Init should not throw for " << testCase.envVar << "="
          << testCase.value;
    }

    unsetenv(testCase.envVar);
  }
}

TEST_F(CvarInitTest, EmptyStringHandling) {
  // Test empty string handling
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  const char* result =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");
  EXPECT_STREQ(
      result, nullptr); // Current impl returns nullptr instead of empty string
}

TEST_F(CvarInitTest, WhitespaceTrimming) {
  // Test whitespace trimming
  struct TestCase {
    const char* input;
    const char* expected;
  };

  std::vector<TestCase> testCases = {
      {"  test_value  ", "test_value"},
      {"\ttest_value\t", "test_value"},
      {"\n test_value \n", "test_value"},
      {"   ", nullptr}, // Current impl returns nullptr instead of empty string
                        // (and whitespace is trimemd from env var values)
      {"\t\n  \r  \t",
       nullptr}, // Current impl returns nullptr instead of empty string
                 // (and whitespace is trimemd from env var values)
  };

  for (const auto& testCase : testCases) {
    setenv("__NCCL_UNIT_TEST_STRING_CVAR__", testCase.input, 1);

    EXPECT_NO_THROW(ncclCvarInit())
        << "Failed for input: '" << testCase.input << "'";

    const char* result =
        nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");
    EXPECT_STREQ(result, testCase.expected)
        << "Failed for input: '" << testCase.input << "'";

    unsetenv("__NCCL_UNIT_TEST_STRING_CVAR__");
  }
}

TEST_F(CvarInitTest, DeprecatedSettingsThrows) {
  // Test deprecated NCCL_CVARS_SETTINGS throws exception
  setenv("NCCL_CVARS_SETTINGS", "some_deprecated_setting", 1);

  EXPECT_THROW(ncclCvarInit(), std::runtime_error);

  unsetenv("NCCL_CVARS_SETTINGS");
}

TEST_F(CvarInitTest, UnknownEnvVariableWarning) {
  // Test unknown environment variable warning
  setenv("NCCL_UNKNOWN_TEST_VAR", "test_value", 1);
  setenv("NCCL_DEBUG_SUBSYS", "CVAR", 1);

  testing::internal::CaptureStderr();
  ncclCvarInit();
  std::string output = testing::internal::GetCapturedStderr();

  EXPECT_THAT(output, testing::HasSubstr("Unknown env"));
  EXPECT_THAT(output, testing::HasSubstr("NCCL_UNKNOWN_TEST_VAR"));

  unsetenv("NCCL_UNKNOWN_TEST_VAR");
  unsetenv("NCCL_DEBUG_SUBSYS");
}

TEST_F(CvarInitTest, MultipleCvarsInitialization) {
  // Test multiple CVARs initialization
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "string_value", 1);
  setenv("__NCCL_UNIT_TEST_BOOL_CVAR__", "true", 1);
  setenv("__NCCL_UNIT_TEST_INT_CVAR__", "42", 1);
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "9876543210", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  // Verify all CVARs were set correctly
  const char* stringResult =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");
  EXPECT_STREQ(stringResult, "string_value");

  const char* boolResult =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_BOOL_CVAR__");
  EXPECT_STREQ(boolResult, "1");

  const char* intResult =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_INT_CVAR__");
  EXPECT_STREQ(intResult, "42");

  const char* int64Result =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_INT64_T_CVAR__");
  EXPECT_STREQ(int64Result, "9876543210");
}

TEST_F(CvarInitTest, RealNcclCvarsInitialization) {
  // Test real NCCL CVARs initialization
  setenv("NCCL_BUFFSIZE", "1048576", 1);
  setenv("NCCL_NTHREADS", "8", 1);
  setenv("NCCL_P2P_DISABLE", "1", 1);
  setenv("CUDA_LAUNCH_BLOCKING", "0", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  // Verify real CVARs were set correctly
  const char* buffsizeResult =
      nccl_baseline_adapter::ncclGetEnvImpl("NCCL_BUFFSIZE");
  EXPECT_STREQ(buffsizeResult, "1048576");

  const char* nthreadsResult =
      nccl_baseline_adapter::ncclGetEnvImpl("NCCL_NTHREADS");
  EXPECT_STREQ(nthreadsResult, "8");

  const char* p2pResult =
      nccl_baseline_adapter::ncclGetEnvImpl("NCCL_P2P_DISABLE");
  EXPECT_STREQ(p2pResult, "1");

  const char* cudaResult =
      nccl_baseline_adapter::ncclGetEnvImpl("CUDA_LAUNCH_BLOCKING");
  EXPECT_STREQ(cudaResult, "0");

  unsetenv("NCCL_BUFFSIZE");
  unsetenv("NCCL_NTHREADS");
  unsetenv("NCCL_P2P_DISABLE");
  unsetenv("CUDA_LAUNCH_BLOCKING");
}

TEST_F(CvarInitTest, IdempotencyTest) {
  // Test idempotency - calling ncclCvarInit multiple times
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "test_value", 1);

  // First initialization
  EXPECT_NO_THROW(ncclCvarInit());
  const char* result1 =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");
  EXPECT_STREQ(result1, "test_value");

  // Second initialization - should not change anything
  EXPECT_NO_THROW(ncclCvarInit());
  const char* result2 =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");
  EXPECT_STREQ(result2, "test_value");

  // Results should be the same
  EXPECT_EQ(result1, result2); // Same pointer
}

TEST_F(CvarInitTest, CvarMapsPopulated) {
  // Test that CVARs maps are populated correctly
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "test_string", 1);
  setenv("__NCCL_UNIT_TEST_BOOL_CVAR__", "true", 1);
  setenv("__NCCL_UNIT_TEST_INT_CVAR__", "42", 1);
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "9876543210", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  // Check that the maps contain the expected entries
  EXPECT_NE(
      ncclx::env_string_values.find("__NCCL_UNIT_TEST_STRING_CVAR__"),
      ncclx::env_string_values.end());
  EXPECT_NE(
      ncclx::env_bool_values.find("__NCCL_UNIT_TEST_BOOL_CVAR__"),
      ncclx::env_bool_values.end());
  EXPECT_NE(
      ncclx::env_int_values.find("__NCCL_UNIT_TEST_INT_CVAR__"),
      ncclx::env_int_values.end());
  EXPECT_NE(
      ncclx::env_int64_values.find("__NCCL_UNIT_TEST_INT64_T_CVAR__"),
      ncclx::env_int64_values.end());

  // Verify the values in the maps are correct
  EXPECT_EQ(
      *ncclx::env_string_values["__NCCL_UNIT_TEST_STRING_CVAR__"],
      "test_string");
  EXPECT_EQ(*ncclx::env_bool_values["__NCCL_UNIT_TEST_BOOL_CVAR__"], true);
  EXPECT_EQ(*ncclx::env_int_values["__NCCL_UNIT_TEST_INT_CVAR__"], 42);
  EXPECT_EQ(
      *ncclx::env_int64_values["__NCCL_UNIT_TEST_INT64_T_CVAR__"],
      9876543210LL);
}

TEST_F(CvarInitTest, CvarLogging) {
  // Test logging functionality
  setenv("NCCL_CVARS_LOG_INFO", "1", 1);
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "logged_value", 1);

  // Should not throw and should handle logging
  EXPECT_NO_THROW(ncclCvarInit());

  unsetenv("NCCL_CVARS_LOG_INFO");
}

TEST_F(CvarInitTest, NcclMinCtasDefaultValue) {
  // Test NCCL_MIN_CTAS initialization with default INT_MIN when not set
  // Don't set NCCL_MIN_CTAS environment variable
  // Initialize CVARs
  EXPECT_NO_THROW(ncclCvarInit());

  // Test via baseline adapter that default value is used
  int64_t cache = 999; // Use non-INT_MIN as uninitialized value
  int64_t defaultVal =
      std::numeric_limits<int>::min(); // Expected default INT_MIN
  int64_t uninitializedVal = 999;

  nccl_baseline_adapter::ncclLoadParam(
      "NCCL_MIN_CTAS", defaultVal, uninitializedVal, &cache);

  // Should get INT_MIN as the default value
  EXPECT_EQ(cache, std::numeric_limits<int>::min());
  EXPECT_EQ(NCCL_MIN_CTAS, std::numeric_limits<int>::min());
}

TEST_F(CvarInitTest, NcclMinCtasUserProvidedValue) {
  // Test NCCL_MIN_CTAS initialization with user-provided value
  // Set NCCL_MIN_CTAS to a specific value
  setenv("NCCL_MIN_CTAS", "16", 1);

  // Initialize CVARs
  EXPECT_NO_THROW(ncclCvarInit());

  // Test via baseline adapter that user value is used
  int64_t cache = 999; // Use different value as uninitialized
  int64_t defaultVal =
      std::numeric_limits<int>::min(); // Default would be INT_MIN
  int64_t uninitializedVal = 999;

  nccl_baseline_adapter::ncclLoadParam(
      "NCCL_MIN_CTAS", defaultVal, uninitializedVal, &cache);

  // Should get the user-provided value (16)
  EXPECT_EQ(cache, 16);
  EXPECT_EQ(NCCL_MIN_CTAS, 16);

  // Also verify via ncclGetEnv
  const char* result = nccl_baseline_adapter::ncclGetEnvImpl("NCCL_MIN_CTAS");
  EXPECT_STREQ(result, "16");

  unsetenv("NCCL_MIN_CTAS");
}

TEST_F(CvarInitTest, NcclMinCtasVariousUserValues) {
  // Test NCCL_MIN_CTAS with various user values
  struct TestCase {
    const char* value;
    int expected;
  };

  std::vector<TestCase> testCases = {
      {"1", 1},
      {"8", 8},
      {"32", 32},
      {"128", 128},
      {"0", 0},
      {"-1", -1},
      {"2147483647", std::numeric_limits<int>::max()}, // INT_MAX
  };

  for (const auto& testCase : testCases) {
    setenv("NCCL_MIN_CTAS", testCase.value, 1);

    EXPECT_NO_THROW(ncclCvarInit()) << "Failed for value: " << testCase.value;

    // Test via baseline adapter
    int64_t cache = 999; // Use different value as uninitialized
    int64_t defaultVal = std::numeric_limits<int>::min();
    int64_t uninitializedVal = 999;

    nccl_baseline_adapter::ncclLoadParam(
        "NCCL_MIN_CTAS", defaultVal, uninitializedVal, &cache);

    EXPECT_EQ(cache, testCase.expected)
        << "Failed for value: " << testCase.value;

    unsetenv("NCCL_MIN_CTAS");
  }
}

TEST_F(CvarInitTest, NcclMinCtasMinSpecialValue) {
  // Test NCCL_MIN_CTAS with MIN special value
  // Set NCCL_MIN_CTAS to "MIN" (special value)
  setenv("NCCL_MIN_CTAS", "MIN", 1);

  // Initialize CVARs
  EXPECT_NO_THROW(ncclCvarInit());

  // Test via baseline adapter that INT_MIN is used for "MIN"
  int64_t cache = 999; // Use different value as uninitialized
  int64_t defaultVal = 100; // Different default to ensure special handling
  int64_t uninitializedVal = 999;

  nccl_baseline_adapter::ncclLoadParam(
      "NCCL_MIN_CTAS", defaultVal, uninitializedVal, &cache);

  // Should get INT_MIN for "MIN" special value
  EXPECT_EQ(cache, std::numeric_limits<int>::min());
  EXPECT_EQ(NCCL_MIN_CTAS, std::numeric_limits<int>::min());

  unsetenv("NCCL_MIN_CTAS");
}

class NCCLCvarEdgeCasesEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    // Turn off NCCL debug logging by default
    setenv("NCCL_DEBUG", "WARN", 0);
  }
  ~NCCLCvarEdgeCasesEnvironment() override {}
};

class CvarInitEdgeCasesTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Clear any existing environment variables before each test
    clearTestEnvVars();
  }

  void TearDown() override {
    // Clean up environment after each test
    clearTestEnvVars();
  }

 private:
  void clearTestEnvVars() {
    // Clear test-specific environment variables
    unsetenv("__NCCL_UNIT_TEST_STRING_CVAR__");
    unsetenv("__NCCL_UNIT_TEST_BOOL_CVAR__");
    unsetenv("__NCCL_UNIT_TEST_INT_CVAR__");
    unsetenv("__NCCL_UNIT_TEST_INT64_T_CVAR__");
    unsetenv("NCCL_DEBUG_SUBSYS");
    unsetenv("NCCL_CVARS_LOG_INFO");
    unsetenv("NCCL_CVARS_SETTINGS");
    unsetenv("NCCL_DUMMY_TEST_VAR");
    unsetenv("NCCL_BUFFSIZE");
    unsetenv("NCCL_NTHREADS");
    unsetenv("NCCL_P2P_DISABLE");
    unsetenv("CUDA_LAUNCH_BLOCKING");
    unsetenv("NCCL_MIN_CTAS");
    unsetenv("NCCL_MAX_CTAS");
    unsetenv("NCCL_ALGO");
    unsetenv("NCCL_ALLREDUCE_ALGO");
    unsetenv("NCCL_CTRAN_BACKENDS");
    unsetenv("NCCL_HPC_JOB_IDS");
    unsetenv("NCCL_IB_HCA");
    unsetenv("NCCL_COLLTRACE");
    unsetenv("NCCL_NET_GDR_LEVEL");
    unsetenv("NCCL_MASTER_PORT");
    unsetenv("NCCL_MEM_POOL_SIZE");
    unsetenv("NCCL_TCPSTORE_BACKOFF_MULTIPLIER");
    unsetenv("NCCL_TCPSTORE_BACKOFF_RANDOMIZATION_FACTOR");
  }
};

TEST_F(CvarInitEdgeCasesTest, FloatingPointForIntegerCvars) {
  // Test floating point values for integer CVARs
  struct TestCase {
    const char* envVar;
    const char* value;
    int64_t expectedApprox; // Approximate expected value (truncated)
  };

  std::vector<TestCase> testCases = {
      {"__NCCL_UNIT_TEST_INT_CVAR__", "42.7", 42},
      {"__NCCL_UNIT_TEST_INT64_T_CVAR__", "123.456", 123},
      {"__NCCL_UNIT_TEST_INT_CVAR__", "-17.9", -17},
      {"__NCCL_UNIT_TEST_INT64_T_CVAR__", "0.999", 0},
      {"__NCCL_UNIT_TEST_INT_CVAR__",
       "1e3",
       1}, // Will stop converting at the non-numeric character 'e'
  };

  for (const auto& testCase : testCases) {
    setenv(testCase.envVar, testCase.value, 1);

    EXPECT_NO_THROW(ncclCvarInit())
        << "Failed for " << testCase.envVar << "=" << testCase.value;

    // Test via baseline adapter
    int64_t cache = -999;
    int64_t defaultVal = 555;
    int64_t uninitializedVal = -999;

    nccl_baseline_adapter::ncclLoadParam(
        testCase.envVar, defaultVal, uninitializedVal, &cache);

    // Should be approximately the expected truncated value
    EXPECT_EQ(cache, testCase.expectedApprox)
        << "Failed for " << testCase.envVar << "=" << testCase.value;

    unsetenv(testCase.envVar);
  }
}

TEST_F(CvarInitEdgeCasesTest, InvalidBooleanValues) {
  // Test invalid boolean values
  struct TestCase {
    const char* value;
    bool expectedDefault; // Should fallback to default (true based on env2bool)
  };

  std::vector<TestCase> testCases = {
      {"maybe", true}, // Invalid, should warn and return true
      {"perhaps", true}, // Invalid, should warn and return true
      {"2", true}, // Invalid, should warn and return true
      {"-1", true}, // Invalid, should warn and return true
      {"true1", true}, // Invalid, should warn and return true
      {"false0", true}, // Invalid, should warn and return true
      {"TRUE FALSE", true}, // Invalid, should warn and return true
      {"", true}, // Empty, should warn and return true
      {"   ", true}, // Whitespace only, should warn and return true
  };

  for (const auto& testCase : testCases) {
    setenv("__NCCL_UNIT_TEST_BOOL_CVAR__", testCase.value, 1);

    EXPECT_NO_THROW(ncclCvarInit()) << "Failed for value: " << testCase.value;

    // Test via baseline adapter
    int64_t cache = -1;
    int64_t defaultVal = 999;
    int64_t uninitializedVal = -1;

    nccl_baseline_adapter::ncclLoadParam(
        "__NCCL_UNIT_TEST_BOOL_CVAR__", defaultVal, uninitializedVal, &cache);

    // Should get the expected default
    EXPECT_EQ(cache, testCase.expectedDefault ? 1 : 0)
        << "Failed for value: " << testCase.value;

    unsetenv("__NCCL_UNIT_TEST_BOOL_CVAR__");
  }
}

TEST_F(CvarInitEdgeCasesTest, EnumCvarTesting) {
  // Test enum-based CVARs with valid and invalid values

  // Test NCCL_ALLREDUCE_ALGO enum
  struct TestCase {
    const char* value;
    bool shouldSucceed;
  };

  std::vector<TestCase> testCases = {
      {"orig", true},
      {"dda", true},
      {"ctran", true},
      {"ctdirect", true},
      {"ctarg", true},
      {"ctring", true},
      {"ctmring", true},
      {"ORIG", true}, // Test case sensitivity
      {"invalid_algo", false}, // Invalid enum value
      {"", false}, // Empty string
      {"orig,dda", false}, // Multiple values (not valid for single enum)
  };

  for (const auto& testCase : testCases) {
    setenv("NCCL_ALLREDUCE_ALGO", testCase.value, 1);

    if (testCase.shouldSucceed) {
      EXPECT_NO_THROW(ncclCvarInit())
          << "Should succeed for NCCL_ALLREDUCE_ALGO=" << testCase.value;
    } else {
      // Invalid values should not throw but may warn
      EXPECT_NO_THROW(ncclCvarInit())
          << "Should not throw for NCCL_ALLREDUCE_ALGO=" << testCase.value;
    }

    unsetenv("NCCL_ALLREDUCE_ALGO");
  }
}

TEST_F(CvarInitEdgeCasesTest, VectorCvarTesting) {
  // Test vector-based CVARs (comma-separated lists)

  struct TestCase {
    const char* envVar;
    const char* value;
    size_t expectedCount; // Expected number of elements after parsing
  };

  std::vector<TestCase> testCases = {
      {"NCCL_HPC_JOB_IDS", "job1,job2,job3", 3},
      {"NCCL_HPC_JOB_IDS", "single_job", 1},
      {"NCCL_HPC_JOB_IDS", "", 0}, // Empty string
      {"NCCL_HPC_JOB_IDS", ",,", 0}, // Only commas (should be ignored)
      {"NCCL_HPC_JOB_IDS", "job1,,job2", 2}, // Empty element in middle
      {"NCCL_HPC_JOB_IDS",
       " job1 , job2 , job3 ",
       3}, // Whitespace around elements
      {"NCCL_HPC_JOB_IDS",
       "job1,job1,job2",
       3}, // Duplicates (may warn but should work)
      {"NCCL_IB_HCA", "mlx5_0,mlx5_1", 2},
      {"NCCL_IB_HCA", "mlx5_0", 1},
      {"NCCL_COLLTRACE", "allreduce,allgather", 2},
  };

  for (const auto& testCase : testCases) {
    setenv(testCase.envVar, testCase.value, 1);

    EXPECT_NO_THROW(ncclCvarInit())
        << "Failed for " << testCase.envVar << "=" << testCase.value;

    // We can't easily verify the count without accessing internal data
    // structures, but at least verify initialization doesn't crash

    unsetenv(testCase.envVar);
  }
}

TEST_F(CvarInitEdgeCasesTest, LongStringTesting) {
  // Test very long strings that might stress memory allocation

  // Generate a very long string (10KB)
  std::string longString(10240, 'x');

  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", longString.c_str(), 1);

  EXPECT_NO_THROW(ncclCvarInit());

  const char* result =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");
  EXPECT_STREQ(result, longString.c_str());

  // Test extremely long string (1MB)
  std::string veryLongString(1048576, 'y');
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", veryLongString.c_str(), 1);

  EXPECT_NO_THROW(ncclCvarInit());

  result =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");
  EXPECT_STREQ(result, veryLongString.c_str());
}

TEST_F(CvarInitEdgeCasesTest, SpecialCharacterTesting) {
  // Test special characters and unicode in string CVARs

  struct TestCase {
    const char* name;
    const char* value;
  };

  std::vector<TestCase> testCases = {
      {"newlines", "line1\nline2\nline3"},
      {"tabs", "col1\tcol2\tcol3"},
      {"mixed_whitespace", " \t\n  value  \n\t "},
      {"quotes", "\"quoted value\""},
      {"single_quotes", "'single quoted'"},
      {"backslashes", "path\\to\\file"},
      {"forward_slashes", "/path/to/file"},
      {"special_chars", "!@#$%^&*()_+-={}[]|:;\"'<>?,./"},
      {"unicode", "こんにちは世界"}, // "Hello World" in Japanese
      {"emoji", "🚀🎉💻"}, // Emojis
      {"null_char",
       std::string("before\0after", 12).c_str()}, // Contains null character
      {"high_ascii", "\x80\x90\xa0\xb0\xc0\xd0\xe0\xf0"}, // High ASCII values
  };

  for (const auto& testCase : testCases) {
    setenv("__NCCL_UNIT_TEST_STRING_CVAR__", testCase.value, 1);

    EXPECT_NO_THROW(ncclCvarInit()) << "Failed for " << testCase.name;

    const char* result =
        nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");

    // For most cases, expect exact match (note: trimWhitespace may affect some)
    if (strcmp(testCase.name, "mixed_whitespace") == 0) {
      // This one will be trimmed
      EXPECT_STREQ(result, "value") << "Failed for " << testCase.name;
    } else if (strcmp(testCase.name, "null_char") != 0) {
      // Skip null char test for string comparison
      EXPECT_STREQ(result, testCase.value) << "Failed for " << testCase.name;
    }

    unsetenv("__NCCL_UNIT_TEST_STRING_CVAR__");
  }
}

TEST_F(CvarInitEdgeCasesTest, TypedCvarTesting) {
  // Test typed CVARs with specific types

  // Test uint16_t CVAR (__NCCL_UNIT_TEST_UINT16_T_CVAR__)
  setenv("__NCCL_UNIT_TEST_UINT16_T_CVAR__", "65535", 1); // Max uint16_t value
  EXPECT_NO_THROW(ncclCvarInit());
  EXPECT_EQ(__NCCL_UNIT_TEST_UINT16_T_CVAR__, 65535);
  unsetenv("__NCCL_UNIT_TEST_UINT16_T_CVAR__");

  setenv("__NCCL_UNIT_TEST_UINT16_T_CVAR__", "0", 1); // Min uint16_t value
  EXPECT_NO_THROW(ncclCvarInit());
  EXPECT_EQ(__NCCL_UNIT_TEST_UINT16_T_CVAR__, 0);
  unsetenv("__NCCL_UNIT_TEST_UINT16_T_CVAR__");

  setenv("__NCCL_UNIT_TEST_UINT16_T_CVAR__", "65536", 1); // Overflow uint16_t
  EXPECT_NO_THROW(ncclCvarInit()); // Should not crash but may overflow
  unsetenv("__NCCL_UNIT_TEST_UINT16_T_CVAR__");

  // Test size_t CVAR (__NCCL_UNIT_TEST_SIZE_T_CVAR__)
  setenv("__NCCL_UNIT_TEST_SIZE_T_CVAR__", "1073741824", 1); // 1GB
  EXPECT_NO_THROW(ncclCvarInit());
  EXPECT_EQ(__NCCL_UNIT_TEST_SIZE_T_CVAR__, 1073741824);
  unsetenv("__NCCL_UNIT_TEST_SIZE_T_CVAR__");

  // Test double CVAR (__NCCL_UNIT_TEST_DOUBLE_CVAR__)
  setenv("__NCCL_UNIT_TEST_DOUBLE_CVAR__", "2.5", 1);
  EXPECT_NO_THROW(ncclCvarInit());
  EXPECT_EQ(__NCCL_UNIT_TEST_DOUBLE_CVAR__, 2.5);
  unsetenv("__NCCL_UNIT_TEST_DOUBLE_CVAR__");

  setenv("__NCCL_UNIT_TEST_DOUBLE_CVAR__", "0.15", 1);
  EXPECT_NO_THROW(ncclCvarInit());
  EXPECT_EQ(__NCCL_UNIT_TEST_DOUBLE_CVAR__, 0.15);
  unsetenv("__NCCL_UNIT_TEST_DOUBLE_CVAR__");
}

TEST_F(CvarInitEdgeCasesTest, EnvironmentVariableNameEdgeCases) {
  // Test edge cases with environment variable names themselves

  // Test environment variables that start with NCCL_ but aren't recognized
  setenv("NCCL_UNKNOWN_TEST_VAR_123", "test_value", 1);
  setenv("NCCL_DEBUG_SUBSYS", "CVAR", 1); // Enable debug to see warnings

  testing::internal::CaptureStderr();
  ncclCvarInit();
  std::string output = testing::internal::GetCapturedStderr();

  // Should warn about unknown variable
  EXPECT_THAT(output, testing::HasSubstr("Unknown env"));
  EXPECT_THAT(output, testing::HasSubstr("NCCL_UNKNOWN_TEST_VAR_123"));

  unsetenv("NCCL_UNKNOWN_TEST_VAR_123");
  unsetenv("NCCL_DEBUG_SUBSYS");
}

TEST_F(CvarInitEdgeCasesTest, MemoryStressTest) {
  // Stress test with many environment variables

  const int numVars = 100;
  std::vector<std::string> varNames;

  // Set many environment variables (not NCCL ones to avoid conflicts)
  for (int i = 0; i < numVars; ++i) {
    std::string varName = "TEST_VAR_" + std::to_string(i);
    std::string varValue = "value_" + std::to_string(i);
    setenv(varName.c_str(), varValue.c_str(), 1);
    varNames.push_back(varName);
  }

  // Also set our test CVARs
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "stress_test", 1);
  setenv("__NCCL_UNIT_TEST_INT_CVAR__", "12345", 1);

  EXPECT_NO_THROW(ncclCvarInit());

  // Verify our CVARs still work
  const char* result =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");
  EXPECT_STREQ(result, "stress_test");

  // Clean up
  for (const auto& varName : varNames) {
    unsetenv(varName.c_str());
  }
}

TEST_F(CvarInitEdgeCasesTest, RepeatedInitializationWithChanges) {
  // Test calling ncclCvarInit multiple times with environment changes

  // First initialization
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "first_value", 1);
  EXPECT_NO_THROW(ncclCvarInit());

  const char* result1 =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");
  EXPECT_STREQ(result1, "first_value");

  // Change environment variable and reinitialize
  setenv("__NCCL_UNIT_TEST_STRING_CVAR__", "second_value", 1);
  EXPECT_NO_THROW(ncclCvarInit());

  // After reinitialization, it should pick up the new value.
  const char* result2 =
      nccl_baseline_adapter::ncclGetEnvImpl("__NCCL_UNIT_TEST_STRING_CVAR__");

  EXPECT_STREQ(result2, "second_value");
}

TEST_F(CvarInitEdgeCasesTest, BoundaryValueTesting) {
  // Test boundary values for different data types

  struct TestCase {
    const char* envVar;
    const char* value;
    bool shouldWork;
  };

  std::vector<TestCase> testCases = {
      // INT_MAX and INT_MIN for int CVARs
      {"__NCCL_UNIT_TEST_INT_CVAR__", "2147483647", true}, // INT_MAX
      {"__NCCL_UNIT_TEST_INT_CVAR__", "-2147483648", true}, // INT_MIN

      // LLONG_MAX and LLONG_MIN for int64_t CVARs
      {"__NCCL_UNIT_TEST_INT64_T_CVAR__",
       "9223372036854775807",
       true}, // LLONG_MAX
      {"__NCCL_UNIT_TEST_INT64_T_CVAR__",
       "-9223372036854775808",
       true}, // LLONG_MIN

      // Zero values
      {"__NCCL_UNIT_TEST_INT_CVAR__", "0", true},
      {"__NCCL_UNIT_TEST_INT64_T_CVAR__", "0", true},

      // Values just beyond limits (should handle gracefully)
      {"__NCCL_UNIT_TEST_INT_CVAR__", "2147483648", true}, // INT_MAX + 1
      {"__NCCL_UNIT_TEST_INT_CVAR__", "-2147483649", true}, // INT_MIN - 1
  };

  for (const auto& testCase : testCases) {
    setenv(testCase.envVar, testCase.value, 1);

    if (testCase.shouldWork) {
      EXPECT_NO_THROW(ncclCvarInit())
          << "Should work for " << testCase.envVar << "=" << testCase.value;
    } else {
      // Even if it doesn't work as expected, it shouldn't crash
      EXPECT_NO_THROW(ncclCvarInit())
          << "Should not crash for " << testCase.envVar << "="
          << testCase.value;
    }

    unsetenv(testCase.envVar);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new NCCLCvarInitEnvironment);
  ::testing::AddGlobalTestEnvironment(new NCCLCvarEdgeCasesEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
