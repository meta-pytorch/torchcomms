// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <atomic>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "comms/utils/cvars/nccl_baseline_adapter.h"
#include "comms/utils/cvars/nccl_cvars.h"

class NcclBaselineAdapterTest : public ::testing::Test {
 public:
  void SetUp() override {}

  void TearDown() override {}
};

// Tests for nccl_baseline_adapter::ncclLoadParam function
class NcclLoadParamTest : public NcclBaselineAdapterTest {};

TEST_F(NcclLoadParamTest, LoadFromInt64Map) {
  int64_t value = 12345;
  ncclx::env_int64_values["TEST_INT64"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_INT64", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 12345);
}

TEST_F(NcclLoadParamTest, LoadFromIntMap) {
  int value = 42;
  ncclx::env_int_values["TEST_INT"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_INT", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 42);
}

TEST_F(NcclLoadParamTest, LoadFromBoolMapTrue) {
  bool value = true;
  ncclx::env_bool_values["TEST_BOOL"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_BOOL", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 1);
}

TEST_F(NcclLoadParamTest, LoadFromBoolMapFalse) {
  bool value = false;
  ncclx::env_bool_values["TEST_BOOL"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_BOOL", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 0);
}

TEST_F(NcclLoadParamTest, LoadFromStringMapValidNumber) {
  std::string value = "789";
  ncclx::env_string_values["TEST_STRING"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_STRING", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 789);
}

TEST_F(NcclLoadParamTest, LoadFromStringMapHexNumber) {
  std::string value = "0x100";
  ncclx::env_string_values["TEST_STRING"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_STRING", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 256); // 0x100 = 256
}

TEST_F(NcclLoadParamTest, LoadFromStringMapOctalNumber) {
  std::string value = "010";
  ncclx::env_string_values["TEST_STRING"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_STRING", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 8); // 010 octal = 8
}

TEST_F(NcclLoadParamTest, LoadFromStringMapInvalidNumber) {
  std::string value = "invalid123";
  ncclx::env_string_values["TEST_STRING"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_STRING", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, defaultVal); // Should use default value for invalid string
}

TEST_F(NcclLoadParamTest, LoadFromStringMapEmptyString) {
  std::string value;
  ncclx::env_string_values["TEST_STRING"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_STRING", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, defaultVal); // Should use default value for empty string
}

TEST_F(NcclLoadParamTest, LoadFromStringMapNegativeNumber) {
  std::string value = "-456";
  ncclx::env_string_values["TEST_STRING"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_STRING", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, -456);
}

TEST_F(NcclLoadParamTest, LoadNonExistentParam) {
  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "NON_EXISTENT", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, defaultVal);
}

TEST_F(NcclLoadParamTest, AlreadyInitializedParam) {
  int64_t value = 12345;
  ncclx::env_int64_values["TEST_INT64"] = &value;

  int64_t cache = 500; // already initialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_INT64", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 500); // Should not change already initialized value
}

TEST_F(NcclLoadParamTest, PriorityOrder) {
  // Prepare - Set up all maps with different values for same key
  int64_t int64Value = 1000;
  int intValue = 2000;
  bool boolValue = true;
  std::string stringValue = "3000";

  ncclx::env_int64_values["TEST_PRIORITY"] = &int64Value;
  ncclx::env_int_values["TEST_PRIORITY"] = &intValue;
  ncclx::env_bool_values["TEST_PRIORITY"] = &boolValue;
  ncclx::env_string_values["TEST_PRIORITY"] = &stringValue;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_PRIORITY", defaultVal, uninitializedVal, &cache);

  // Assert - Should use int64 value (highest priority)
  EXPECT_EQ(cache, 1000);
}

TEST_F(NcclLoadParamTest, ThreadSafety) {
  int64_t value = 12345;
  ncclx::env_int64_values["TEST_THREAD_SAFETY"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  std::atomic<int> completed_threads{0};
  const int num_threads = 10;
  std::vector<std::thread> threads;
  threads.reserve(num_threads); // Pre-allocate capacity

  // Start multiple threads calling nccl_baseline_adapter::ncclLoadParam
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      nccl_baseline_adapter::ncclLoadParam(
          "TEST_THREAD_SAFETY", defaultVal, uninitializedVal, &cache);
      completed_threads++;
    });
  }

  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(cache, 12345);
  EXPECT_EQ(completed_threads.load(), num_threads);
}

TEST_F(NcclLoadParamTest, LoadParamNonExistentCvar) {
  int64_t cache = 0;
  int64_t defaultVal = 999;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "NON_EXISTENT_CVAR_12345", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, defaultVal);
}

TEST_F(NcclLoadParamTest, LoadParamStringWithWhitespace) {
  // String parsing edge cases in nccl_baseline_adapter::ncclLoadParam
  std::string value = "  789  ";
  ncclx::env_string_values["TEST_WHITESPACE"] = &value;

  int64_t cache = 0;
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_WHITESPACE", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 789); // Should parse despite whitespace
}

TEST_F(NcclLoadParamTest, LoadParamStringPartialNumber) {
  std::string value = "123abc456";
  ncclx::env_string_values["TEST_PARTIAL"] = &value;

  int64_t cache = 0;
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_PARTIAL", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 123); // strtoll should parse up to first non-digit
}

TEST_F(NcclLoadParamTest, LoadParamStringLeadingNonDigits) {
  std::string value = "abc123";
  ncclx::env_string_values["TEST_LEADING_ALPHA"] = &value;

  int64_t cache = 0;
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_LEADING_ALPHA", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, defaultVal); // Should use default for invalid string
}

TEST_F(NcclLoadParamTest, LoadParamStringBinaryNumber) {
  std::string value = "0b1010";
  ncclx::env_string_values["TEST_BINARY"] = &value;

  int64_t cache = 0;
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_BINARY", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 0); // strtoll with base 0 should parse 0b as 0
}

TEST_F(NcclLoadParamTest, LoadParamStringOverflowValue) {
  // Test a value that exceeds int64_t max
  std::string value = "99999999999999999999999999999";
  ncclx::env_string_values["TEST_OVERFLOW"] = &value;

  int64_t cache = 0;
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_OVERFLOW", defaultVal, uninitializedVal, &cache);

  // strtoll will set errno on overflow and return LLONG_MAX or LLONG_MIN
  // Implementation should probably detect this and use default
  EXPECT_EQ(cache, defaultVal);
}

// Test boundary values
TEST_F(NcclLoadParamTest, LoadParamBoundaryValues) {
  // Test INT64_MAX
  int64_t maxValue = std::numeric_limits<int64_t>::max();
  ncclx::env_int64_values["TEST_INT64_MAX"] = &maxValue;

  int64_t cache = 0;
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_INT64_MAX", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, maxValue);

  // Test INT64_MIN
  cache = 0; // reset
  int64_t minValue = std::numeric_limits<int64_t>::min();
  ncclx::env_int64_values["TEST_INT64_MIN"] = &minValue;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_INT64_MIN", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, minValue);
}

TEST_F(NcclLoadParamTest, LoadParamZeroValues) {
  // Test zero values for all types
  // Test zero int64_t
  int64_t zeroInt64 = 0;
  ncclx::env_int64_values["TEST_ZERO_INT64"] = &zeroInt64;

  int64_t cache = -1; // non-zero uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = -1;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_ZERO_INT64", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 0);

  // Test zero int
  cache = -1; // reset
  int zeroInt = 0;
  ncclx::env_int_values["TEST_ZERO_INT"] = &zeroInt;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_ZERO_INT", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 0);

  // Test false bool
  cache = -1; // reset
  bool falseBool = false;
  ncclx::env_bool_values["TEST_FALSE_BOOL"] = &falseBool;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_FALSE_BOOL", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 0);

  // Test zero string
  cache = -1; // reset
  std::string zeroString = "0";
  ncclx::env_string_values["TEST_ZERO_STRING"] = &zeroString;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_ZERO_STRING", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 0);
}

TEST_F(NcclLoadParamTest, ConcurrentDifferentKeys) {
  // Test multiple concurrent readers with different keys
  // Set up multiple values
  int64_t value1 = 111;
  int64_t value2 = 222;
  int64_t value3 = 333;
  ncclx::env_int64_values["TEST_CONCURRENT_1"] = &value1;
  ncclx::env_int64_values["TEST_CONCURRENT_2"] = &value2;
  ncclx::env_int64_values["TEST_CONCURRENT_3"] = &value3;

  const size_t numCaches = 3;

  std::vector<std::thread> threads;
  threads.reserve(numCaches);

  std::vector<int64_t> caches(numCaches, 0);
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  // Start threads accessing different keys
  for (int i = 0; i < 3; ++i) {
    threads.emplace_back([&, i]() {
      std::string key = "TEST_CONCURRENT_" + std::to_string(i + 1);
      nccl_baseline_adapter::ncclLoadParam(
          key.c_str(), defaultVal, uninitializedVal, &caches[i]);
    });
  }

  // Wait for all threads
  for (auto& t : threads) {
    t.join();
  }

  // Verify results
  EXPECT_EQ(caches[0], 111);
  EXPECT_EQ(caches[1], 222);
  EXPECT_EQ(caches[2], 333);
}

TEST_F(NcclLoadParamTest, LoadParamStringOnlyWhitespace) {
  // Test string with only whitespace
  std::string value = "   \t\n  ";
  ncclx::env_string_values["TEST_ONLY_WHITESPACE"] = &value;

  int64_t cache = 0;
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_ONLY_WHITESPACE", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, defaultVal); // Should use default for whitespace-only string
}

TEST_F(NcclLoadParamTest, LoadParamStringScientificNotation) {
  // Test string with scientific notation (edge case)
  std::string value = "1e3";
  ncclx::env_string_values["TEST_SCIENTIFIC"] = &value;

  int64_t cache = 0;
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_SCIENTIFIC", defaultVal, uninitializedVal, &cache);

  EXPECT_EQ(cache, 1); // strtoll should parse up to 'e'
}

TEST_F(NcclLoadParamTest, LoadParamMultipleInitialization) {
  // Test multiple initialization attempts (idempotency)
  int64_t value = 555;
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "555", 1);
  ncclCvarInit();

  int64_t cache = 0;
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  // First call.
  nccl_baseline_adapter::ncclLoadParam(
      "__NCCL_UNIT_TEST_INT64_T_CVAR__", defaultVal, uninitializedVal, &cache);
  EXPECT_EQ(cache, value);

  // Change both the map value and the environment variable value.
  // Note that the environment variable should not be read, and
  // the updated map value should not be accessed as the cache should
  // already be initialized.
  int64_t newValue = 777;
  setenv("__NCCL_UNIT_TEST_INT64_T_CVAR__", "777", 1);
  ncclx::env_int64_values["__NCCL_UNIT_TEST_INT64_T_CVAR__"] = &newValue;

  // Second call should not change cache (already initialized)
  nccl_baseline_adapter::ncclLoadParam(
      "__NCCL_UNIT_TEST_INT64_T_CVAR__", defaultVal, uninitializedVal, &cache);
  EXPECT_EQ(cache, value); // Should still be original value
}

// Tests for nccl_baseline_adapter::ncclGetEnv function
class NcclGetEnvTest : public NcclBaselineAdapterTest {};

TEST_F(NcclGetEnvTest, GetFromStringMap) {
  std::string value = "test_string_value";
  ncclx::env_string_values["TEST_STRING"] = &value;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_STRING");

  EXPECT_STREQ(result, "test_string_value");
}

TEST_F(NcclGetEnvTest, GetFromInt64Map) {
  int64_t value = 12345;
  ncclx::env_int64_values["TEST_INT64"] = &value;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_INT64");

  EXPECT_STREQ(result, "12345");
}

TEST_F(NcclGetEnvTest, GetFromInt64MapNegative) {
  int64_t value = -9876;
  ncclx::env_int64_values["TEST_INT64_NEG"] = &value;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_INT64_NEG");

  EXPECT_STREQ(result, "-9876");
}

TEST_F(NcclGetEnvTest, GetFromBoolMapTrue) {
  bool value = true;
  ncclx::env_bool_values["TEST_BOOL_TRUE"] = &value;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_BOOL_TRUE");

  EXPECT_STREQ(result, "1");
}

TEST_F(NcclGetEnvTest, GetFromBoolMapFalse) {
  bool value = false;
  ncclx::env_bool_values["TEST_BOOL_FALSE"] = &value;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_BOOL_FALSE");

  EXPECT_STREQ(result, "0");
}

TEST_F(NcclGetEnvTest, GetFromIntMap) {
  int value = 42;
  ncclx::env_int_values["TEST_INT"] = &value;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_INT");

  EXPECT_STREQ(result, "42");
}

TEST_F(NcclGetEnvTest, GetFromIntMapNegative) {
  int value = -789;
  ncclx::env_int_values["TEST_INT_NEG"] = &value;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_INT_NEG");

  EXPECT_STREQ(result, "-789");
}

TEST_F(NcclGetEnvTest, GetNonExistentEnv) {
  EXPECT_THROW(
      nccl_baseline_adapter::ncclGetEnv("NON_EXISTENT"), std::runtime_error);
}

TEST_F(NcclGetEnvTest, GetNonExistentEnvMessage) {
  try {
    nccl_baseline_adapter::ncclGetEnv("NON_EXISTENT_VAR");
    FAIL() << "Expected std::runtime_error to be thrown";
  } catch (const std::runtime_error& e) {
    EXPECT_THAT(
        std::string(e.what()),
        testing::HasSubstr("Undefined NCCL environment variable"));
    EXPECT_THAT(std::string(e.what()), testing::HasSubstr("NON_EXISTENT_VAR"));
  }
}

TEST_F(NcclGetEnvTest, PriorityOrderStringFirst) {
  // Prepare - String should have highest priority
  std::string stringValue = "string_value";
  int64_t int64Value = 1000;
  bool boolValue = true;
  int intValue = 2000;

  ncclx::env_string_values["TEST_PRIORITY"] = &stringValue;
  ncclx::env_int64_values["TEST_PRIORITY"] = &int64Value;
  ncclx::env_bool_values["TEST_PRIORITY"] = &boolValue;
  ncclx::env_int_values["TEST_PRIORITY"] = &intValue;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_PRIORITY");

  // Assert - Should return string value (highest priority)
  EXPECT_STREQ(result, stringValue.c_str());
}

TEST_F(NcclGetEnvTest, PriorityOrderInt64Second) {
  // Prepare - Only non-string values
  int64_t int64Value = 1000;
  bool boolValue = true;
  int intValue = 2000;

  ncclx::env_int64_values["TEST_PRIORITY"] = &int64Value;
  ncclx::env_bool_values["TEST_PRIORITY"] = &boolValue;
  ncclx::env_int_values["TEST_PRIORITY"] = &intValue;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_PRIORITY");

  // Assert - Should return int64 value (second priority)
  EXPECT_STREQ(result, "1000");
}

TEST_F(NcclGetEnvTest, PriorityOrderBoolThird) {
  // Prepare - Only bool and int values
  bool boolValue = true;
  int intValue = 2000;

  ncclx::env_bool_values["TEST_PRIORITY"] = &boolValue;
  ncclx::env_int_values["TEST_PRIORITY"] = &intValue;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_PRIORITY");

  // Assert - Should return bool value (third priority)
  EXPECT_STREQ(result, "1");
}

TEST_F(NcclGetEnvTest, PriorityOrderIntLast) {
  // Prepare - Only int value
  int intValue = 2000;
  ncclx::env_int_values["TEST_PRIORITY"] = &intValue;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_PRIORITY");

  // Assert - Should return int value (last priority)
  EXPECT_STREQ(result, "2000");
}

TEST_F(NcclGetEnvTest, CachedConversion) {
  int64_t value = 12345;
  ncclx::env_int64_values["TEST_CACHED"] = &value;

  // Call multiple times
  const char* result1 = nccl_baseline_adapter::ncclGetEnv("TEST_CACHED");
  const char* result2 = nccl_baseline_adapter::ncclGetEnv("TEST_CACHED");

  // Assert - Should return same pointer (cached)
  EXPECT_EQ(result1, result2);
  EXPECT_STREQ(result1, "12345");
}

TEST_F(NcclGetEnvTest, ThreadLocalCaching) {
  int64_t value = 12345;
  ncclx::env_int64_values["TEST_THREAD_LOCAL"] = &value;

  std::string result_from_thread;
  const char* result_from_main =
      nccl_baseline_adapter::ncclGetEnv("TEST_THREAD_LOCAL");

  // Get value from another thread
  std::thread t([&]() {
    const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_THREAD_LOCAL");
    result_from_thread = std::string(result);
  });
  t.join();

  // Assert - Both threads should get the same value
  EXPECT_STREQ(result_from_main, "12345");
  EXPECT_EQ(result_from_thread, "12345");
}

TEST_F(NcclGetEnvTest, LargeNumbers) {
  int64_t maxValue = std::numeric_limits<int64_t>::max();
  int64_t minValue = std::numeric_limits<int64_t>::min();

  ncclx::env_int64_values["TEST_MAX"] = &maxValue;
  ncclx::env_int64_values["TEST_MIN"] = &minValue;

  const char* maxResult = nccl_baseline_adapter::ncclGetEnv("TEST_MAX");
  const char* minResult = nccl_baseline_adapter::ncclGetEnv("TEST_MIN");

  EXPECT_STREQ(maxResult, std::to_string(maxValue).c_str());
  EXPECT_STREQ(minResult, std::to_string(minValue).c_str());
}

TEST_F(NcclGetEnvTest, EmptyString) {
  std::string emptyValue;
  ncclx::env_string_values["TEST_EMPTY"] = &emptyValue;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_EMPTY");

  EXPECT_STREQ(result, nullptr);
}

TEST_F(NcclGetEnvTest, SpecialCharactersInString) {
  std::string specialValue = "test!@#$%^&*()_+-={}[]|\\:;\"'<>?,./ ";
  ncclx::env_string_values["TEST_SPECIAL"] = &specialValue;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_SPECIAL");

  EXPECT_STREQ(result, specialValue.c_str());
}

TEST_F(NcclGetEnvTest, GetEnvNonExistentVar) {
  EXPECT_THROW(
      nccl_baseline_adapter::ncclGetEnv("NON_EXISTENT_ENV_VAR_12345"),
      std::runtime_error);
}

TEST_F(NcclGetEnvTest, GetEnvNumericEdgeCases) {
  // Test ncclGetEnv with various numeric edge cases
  // Test with zero values
  int64_t zeroValue = 0;
  ncclx::env_int64_values["TEST_GET_ZERO"] = &zeroValue;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_GET_ZERO");
  EXPECT_STREQ(result, "0");

  // Test with negative values
  int64_t negValue = -12345;
  ncclx::env_int64_values["TEST_GET_NEG"] = &negValue;

  result = nccl_baseline_adapter::ncclGetEnv("TEST_GET_NEG");
  EXPECT_STREQ(result, "-12345");

  // Test with max/min values
  int64_t maxValue = std::numeric_limits<int64_t>::max();
  ncclx::env_int64_values["TEST_GET_MAX"] = &maxValue;

  result = nccl_baseline_adapter::ncclGetEnv("TEST_GET_MAX");
  EXPECT_STREQ(result, std::to_string(maxValue).c_str());
}

// Test concurrent access to ncclGetEnv with cache validation
TEST_F(NcclGetEnvTest, GetEnvConcurrentCacheTest) {
  int64_t value = 42;
  ncclx::env_int64_values["TEST_CACHE_CONCURRENT"] = &value;

  std::atomic<int> successful_calls{0};
  const int num_threads = 5;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Multiple threads calling ncclGetEnv on same key
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      const char* result =
          nccl_baseline_adapter::ncclGetEnv("TEST_CACHE_CONCURRENT");
      if (std::string(result) == "42") {
        successful_calls++;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(successful_calls.load(), num_threads);
}

TEST_F(NcclGetEnvTest, GetEnvVeryLongString) {
  // Test very long string value
  std::string longValue(10000, 'x'); // 10k character string
  ncclx::env_string_values["TEST_LONG_STRING"] = &longValue;

  const char* result = nccl_baseline_adapter::ncclGetEnv("TEST_LONG_STRING");
  EXPECT_EQ(strlen(result), 10000);
  EXPECT_EQ(result[0], 'x');
  EXPECT_EQ(result[9999], 'x');
}

TEST_F(NcclGetEnvTest, MixedTypeCleanupTest) {
  // Test with mixed type cleanup (ensure no memory issues)

  // Add values to all maps
  std::string strVal = "test_cleanup";
  int64_t int64Val = 123;
  int intVal = 456;
  bool boolVal = true;

  ncclx::env_string_values["CLEANUP_TEST"] = &strVal;
  ncclx::env_int64_values["CLEANUP_TEST"] = &int64Val;
  ncclx::env_int_values["CLEANUP_TEST"] = &intVal;
  ncclx::env_bool_values["CLEANUP_TEST"] = &boolVal;

  // Test both functions work correctly
  const char* envResult = nccl_baseline_adapter::ncclGetEnv("CLEANUP_TEST");
  EXPECT_STREQ(envResult, "test_cleanup"); // String has highest priority

  int64_t cache = 0;
  int64_t defaultVal = 999;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "CLEANUP_TEST", defaultVal, uninitializedVal, &cache);
  EXPECT_EQ(cache, int64Val); // int64 has highest priority in ncclLoadParam
}

// Integration tests combining both functions
class NcclBaselineAdapterIntegrationTest : public NcclBaselineAdapterTest {};

TEST_F(NcclBaselineAdapterIntegrationTest, LoadParamThenGetEnv) {
  int64_t value = 9999;
  ncclx::env_int64_values["TEST_INTEGRATION"] = &value;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_INTEGRATION", defaultVal, uninitializedVal, &cache);
  const char* envResult = nccl_baseline_adapter::ncclGetEnv("TEST_INTEGRATION");

  EXPECT_EQ(cache, 9999);
  EXPECT_STREQ(envResult, "9999");
}

TEST_F(NcclBaselineAdapterIntegrationTest, InitEnvThenLoadStringParam) {
  std::string value = "InitEnvThenLoadParam";

  const char* cvar = "__NCCL_UNIT_TEST_STRING_CVAR__";
  setenv(cvar, value.c_str(), 1);
  ncclCvarInit();

  const char* result = nccl_baseline_adapter::ncclGetEnv(cvar);
  EXPECT_STREQ(result, value.c_str());
}

TEST_F(NcclBaselineAdapterIntegrationTest, InitEnvThenLoadInt64Param) {
  int64_t value = 9999;

  std::string stringValue = "9999";
  const char* cvar = "__NCCL_UNIT_TEST_INT64_T_CVAR__";
  setenv(cvar, stringValue.c_str(), 1);

  ncclCvarInit();

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = -1;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      cvar, defaultVal, uninitializedVal, &cache);
  const char* envResult = nccl_baseline_adapter::ncclGetEnv(cvar);

  EXPECT_EQ(cache, value);
  EXPECT_STREQ(envResult, stringValue.c_str());
}

TEST_F(NcclBaselineAdapterIntegrationTest, InitEnvThenLoadIntParam) {
  int64_t value = 9999;

  std::string stringValue = "9999";
  const char* cvar = "__NCCL_UNIT_TEST_INT_CVAR__";
  setenv(cvar, stringValue.c_str(), 1);

  ncclCvarInit();

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = -1;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      cvar, defaultVal, uninitializedVal, &cache);
  const char* envResult = nccl_baseline_adapter::ncclGetEnv(cvar);

  EXPECT_EQ(cache, value);
  EXPECT_STREQ(envResult, stringValue.c_str());
}

TEST_F(NcclBaselineAdapterIntegrationTest, InitEnvThenLoadBoolParam) {
  bool value = true;

  std::string stringValue = "true";
  const char* cvar = "__NCCL_UNIT_TEST_BOOL_CVAR__";
  setenv(cvar, stringValue.c_str(), 1);

  ncclCvarInit();

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = -1;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      cvar, defaultVal, uninitializedVal, &cache);
  const char* envResult = nccl_baseline_adapter::ncclGetEnv(cvar);

  EXPECT_EQ(cache, value);
  EXPECT_STREQ(envResult, "1"); // bool is converted to int
}

TEST_F(NcclBaselineAdapterIntegrationTest, ConsistentBehaviorBetweenFunctions) {
  std::string stringValue = "12345";
  ncclx::env_string_values["TEST_CONSISTENT"] = &stringValue;

  int64_t cache = 0; // uninitialized value
  int64_t defaultVal = 100;
  int64_t uninitializedVal = 0;

  nccl_baseline_adapter::ncclLoadParam(
      "TEST_CONSISTENT", defaultVal, uninitializedVal, &cache);
  const char* envResult = nccl_baseline_adapter::ncclGetEnv("TEST_CONSISTENT");

  // Assert - Both should parse the string as a number
  EXPECT_EQ(cache, 12345);
  EXPECT_STREQ(
      envResult,
      "12345"); // nccl_baseline_adapter::ncclGetEnv returns the original string
}
