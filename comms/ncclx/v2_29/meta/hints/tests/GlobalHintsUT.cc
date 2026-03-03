// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <fmt/core.h>
#include <folly/Conv.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unordered_set>

#include "nccl.h" // @manual

#include "meta/hints/GlobalHints.h" // @manual

constexpr std::string_view testGlobalHintKey = "testGlobalHintKey";
constexpr std::string_view testGlobalHintVal = "testGlobalHintVal";

TEST(GlobalHintsUT, TestRegHint) {
  ncclx::testOnlyResetGlobalHints();

  auto hintMngr = ncclx::GlobalHints::getInstance();

  const auto keyStr = std::string(testGlobalHintKey);

  // Expect hint key is not available before registration
  ASSERT_FALSE(hintMngr->resetHint(keyStr));

  ASSERT_TRUE(hintMngr->regHintEntry(std::string(testGlobalHintKey)));

  // Expect found after registration
  ASSERT_TRUE(hintMngr->resetHint(keyStr));
}

TEST(GlobalHintsUT, TestBasicSet) {
  ncclx::testOnlyResetGlobalHints();

  auto hintMngr = ncclx::GlobalHints::getInstance();
  hintMngr->regHintEntry(std::string(testGlobalHintKey));

  EXPECT_EQ(
      ncclx::setGlobalHint(
          std::string{testGlobalHintKey}, std::string{testGlobalHintVal}),
      ncclSuccess);

  auto result = ncclx::getGlobalHint(std::string{testGlobalHintKey});
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), testGlobalHintVal);
}

TEST(GlobalHintsUT, TestBasicReset) {
  ncclx::testOnlyResetGlobalHints();

  auto hintMngr = ncclx::GlobalHints::getInstance();
  hintMngr->regHintEntry(std::string(testGlobalHintKey));

  EXPECT_EQ(
      ncclx::setGlobalHint(
          std::string{testGlobalHintKey}, std::string{testGlobalHintVal}),
      ncclSuccess);

  auto result = ncclx::getGlobalHint(std::string{testGlobalHintKey});
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), testGlobalHintVal);

  EXPECT_TRUE(ncclx::resetGlobalHint(std::string{testGlobalHintKey}));

  // Expect second reset is no-op
  EXPECT_TRUE(ncclx::resetGlobalHint(std::string{testGlobalHintKey}));

  // Expect hint is not found after reset
  result = ncclx::getGlobalHint(std::string{testGlobalHintKey});
  EXPECT_FALSE(result.has_value());
}

TEST(GlobalHintsUT, TestGetNonExistentKey) {
  ncclx::testOnlyResetGlobalHints();

  // Not registered key
  auto result1 = ncclx::getGlobalHint("nonexistent_key");
  EXPECT_FALSE(result1.has_value());

  // Registered but not set key
  auto hintMngr = ncclx::GlobalHints::getInstance();
  hintMngr->regHintEntry(std::string(testGlobalHintKey));

  auto result2 = ncclx::getGlobalHint(std::string(testGlobalHintKey));
  EXPECT_FALSE(result2.has_value());
}

TEST(GlobalHintsUT, TestOverwriteExistingKey) {
  ncclx::testOnlyResetGlobalHints();

  const std::string key = "overwrite_test_key";
  const std::string originalValue = "original_value";
  const std::string newValue = "new_value";

  auto hintMngr = ncclx::GlobalHints::getInstance();
  hintMngr->regHintEntry(std::string(key));

  // Set original value
  EXPECT_EQ(ncclx::setGlobalHint(key, originalValue), ncclSuccess);
  auto result = ncclx::getGlobalHint(key);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), originalValue);

  // Overwrite with new value
  EXPECT_EQ(ncclx::setGlobalHint(key, newValue), ncclSuccess);
  result = ncclx::getGlobalHint(key);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), newValue);
}

TEST(GlobalHintsUT, TestMultipleKeyValuePairs) {
  ncclx::testOnlyResetGlobalHints();

  const std::string key1 = "key1";
  const std::string value1 = "value1";
  const std::string key2 = "key2";
  const std::string value2 = "value2";
  const std::string key3 = "key3";
  const std::string value3 = "value3";

  auto hintMngr = ncclx::GlobalHints::getInstance();
  hintMngr->regHintEntry(std::string(key1));
  hintMngr->regHintEntry(std::string(key2));
  hintMngr->regHintEntry(std::string(key3));

  // Set multiple key-value pairs
  EXPECT_EQ(ncclx::setGlobalHint(key1, value1), ncclSuccess);
  EXPECT_EQ(ncclx::setGlobalHint(key2, value2), ncclSuccess);
  EXPECT_EQ(ncclx::setGlobalHint(key3, value3), ncclSuccess);

  // Verify all pairs are retrievable
  auto result1 = ncclx::getGlobalHint(key1);
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value(), value1);

  auto result2 = ncclx::getGlobalHint(key2);
  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2.value(), value2);

  auto result3 = ncclx::getGlobalHint(key3);
  EXPECT_TRUE(result3.has_value());
  EXPECT_EQ(result3.value(), value3);
}

TEST(GlobalHintsUT, TestEmptyKeyAndValue) {
  ncclx::testOnlyResetGlobalHints();

  const std::string emptyKey;
  const std::string emptyValue;
  const std::string normalKey = "normal_key";

  auto hintMngr = ncclx::GlobalHints::getInstance();
  hintMngr->regHintEntry(std::string(emptyKey));
  hintMngr->regHintEntry(std::string(normalKey));

  // Test empty key with normal value
  EXPECT_EQ(
      ncclx::setGlobalHint(emptyKey, std::string{testGlobalHintVal}),
      ncclSuccess);
  auto result = ncclx::getGlobalHint(emptyKey);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), testGlobalHintVal);

  // Test normal key with empty value
  EXPECT_EQ(ncclx::setGlobalHint(normalKey, emptyValue), ncclSuccess);
  result = ncclx::getGlobalHint(normalKey);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), emptyValue);

  // Test both empty key and empty value
  EXPECT_EQ(ncclx::setGlobalHint(emptyKey, emptyValue), ncclSuccess);
  result = ncclx::getGlobalHint(emptyKey);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), emptyValue);
}

TEST(GlobalHintsUT, TestResetFunctionality) {
  ncclx::testOnlyResetGlobalHints();

  const std::string key1 = "reset_test_key1";
  const std::string value1 = "reset_test_value1";
  const std::string key2 = "reset_test_key2";
  const std::string value2 = "reset_test_value2";

  auto hintMngr = ncclx::GlobalHints::getInstance();
  hintMngr->regHintEntry(std::string(key1));
  hintMngr->regHintEntry(std::string(key2));

  // Set multiple hints
  EXPECT_EQ(ncclx::setGlobalHint(key1, value1), ncclSuccess);
  EXPECT_EQ(ncclx::setGlobalHint(key2, value2), ncclSuccess);

  // Verify they exist
  auto result1 = ncclx::getGlobalHint(key1);
  EXPECT_TRUE(result1.has_value());
  EXPECT_EQ(result1.value(), value1);

  auto result2 = ncclx::getGlobalHint(key2);
  EXPECT_TRUE(result2.has_value());
  EXPECT_EQ(result2.value(), value2);

  // Reset all hints
  ncclx::testOnlyResetGlobalHints();

  // Verify they no longer exist
  result1 = ncclx::getGlobalHint(key1);
  EXPECT_FALSE(result1.has_value());

  result2 = ncclx::getGlobalHint(key2);
  EXPECT_FALSE(result2.has_value());
}

TEST(GlobalHintsUT, TestLargeValues) {
  ncclx::testOnlyResetGlobalHints();

  const std::string key = "large_value_key";
  const std::string largeValue = std::string(10000, 'A'); // 10KB string of 'A's

  auto hintMngr = ncclx::GlobalHints::getInstance();
  hintMngr->regHintEntry(std::string(key));

  EXPECT_EQ(ncclx::setGlobalHint(key, largeValue), ncclSuccess);
  auto result = ncclx::getGlobalHint(key);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), largeValue);
  EXPECT_EQ(result.value().size(), 10000);
}

TEST(GlobalHintsUT, TestSpecialCharacters) {
  ncclx::testOnlyResetGlobalHints();

  const std::string key = "special_chars_key";
  const std::string specialValue = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~\n\t\r";

  auto hintMngr = ncclx::GlobalHints::getInstance();
  hintMngr->regHintEntry(std::string(key));

  EXPECT_EQ(ncclx::setGlobalHint(key, specialValue), ncclSuccess);
  auto result = ncclx::getGlobalHint(key);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), specialValue);
}

TEST(GlobalHintsUT, TestNumericStringValues) {
  ncclx::testOnlyResetGlobalHints();

  const std::string intKey = "int_key";
  const std::string intValue = "12345";
  const std::string floatKey = "float_key";
  const std::string floatValue = "123.456";
  const std::string negativeKey = "negative_key";
  const std::string negativeValue = "-789";

  auto hintMngr = ncclx::GlobalHints::getInstance();
  hintMngr->regHintEntry(std::string(intKey));
  hintMngr->regHintEntry(std::string(floatKey));
  hintMngr->regHintEntry(std::string(negativeKey));

  EXPECT_EQ(ncclx::setGlobalHint(intKey, intValue), ncclSuccess);
  EXPECT_EQ(ncclx::setGlobalHint(floatKey, floatValue), ncclSuccess);
  EXPECT_EQ(ncclx::setGlobalHint(negativeKey, negativeValue), ncclSuccess);

  auto intResult = ncclx::getGlobalHint(intKey);
  EXPECT_TRUE(intResult.has_value());
  EXPECT_EQ(intResult.value(), intValue);

  auto floatResult = ncclx::getGlobalHint(floatKey);
  EXPECT_TRUE(floatResult.has_value());
  EXPECT_EQ(floatResult.value(), floatValue);

  auto negativeResult = ncclx::getGlobalHint(negativeKey);
  EXPECT_TRUE(negativeResult.has_value());
  EXPECT_EQ(negativeResult.value(), negativeValue);
}

namespace {
constexpr int testHintValDefault = 1;
int testHintVal = testHintValDefault;
const std::string testHintKey = "testHintsKey";

void testSetHook(
    const std::string& key __attribute__((unused)),
    const std::string& val) {
  // Allow only specified key and values
  testHintVal = folly::to<int>(val);
}

void testResetHook(const std::string& key __attribute__((unused))) {
  testHintVal = testHintValDefault;
}
} // namespace

TEST(GlobalHintsUT, TestHintHooks) {
  ncclx::testOnlyResetGlobalHints();

  auto hintMngr = ncclx::GlobalHints::getInstance();
  ncclx::GlobalHintEntry entry = {
      .setHook = testSetHook, .resetHook = testResetHook};
  hintMngr->regHintEntry(std::string(testHintKey), entry);

  EXPECT_EQ(ncclx::setGlobalHint(std::string{testHintKey}, "2"), ncclSuccess);
  // Expect testSetHook is called
  EXPECT_EQ(testHintVal, 2);

  auto result = ncclx::getGlobalHint(std::string{testHintKey});
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "2");

  EXPECT_TRUE(ncclx::resetGlobalHint(std::string{testHintKey}));
  // Expect testResetHook is called
  EXPECT_EQ(testHintVal, testHintValDefault);

  // Expect hint is not found after reset
  result = ncclx::getGlobalHint(std::string{testHintKey});
  EXPECT_FALSE(result.has_value());
}

TEST(GlobalHintsUT, TestHintHooksWithCharKeyVal) {
  ncclx::testOnlyResetGlobalHints();

  const char* testHintKeyChar = "testHintsKey";
  const char* testHintValChar = "2";

  // Expect hint key is not available before registration
  EXPECT_FALSE(ncclx::resetGlobalHint(testHintKeyChar));

  auto hintMngr = ncclx::GlobalHints::getInstance();
  ncclx::GlobalHintEntry entry = {
      .setHook = testSetHook, .resetHook = testResetHook};
  hintMngr->regHintEntry(std::string(testHintKey), entry);

  // Set and get
  EXPECT_TRUE(ncclx::setGlobalHint(testHintKeyChar, testHintValChar));
  // Expect testSetHook is called
  EXPECT_EQ(testHintVal, 2);

  EXPECT_TRUE(ncclx::resetGlobalHint(testHintKeyChar));
  // Expect testResetHook is called
  EXPECT_EQ(testHintVal, testHintValDefault);
}
