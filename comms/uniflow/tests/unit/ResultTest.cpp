// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/Result.h"

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>

using namespace uniflow;

// --- ErrCode and errorCodeToString tests ---

TEST(ErrCodeTest, ErrorCodeToStringReturnsCorrectNames) {
  EXPECT_STREQ(errorCodeToString(ErrCode::NotImplemented), "NotImplemented");
  EXPECT_STREQ(errorCodeToString(ErrCode::InvalidArgument), "InvalidArgument");
  EXPECT_STREQ(errorCodeToString(ErrCode::NotConnected), "NotConnected");
  EXPECT_STREQ(
      errorCodeToString(ErrCode::ConnectionFailed), "ConnectionFailed");
  EXPECT_STREQ(
      errorCodeToString(ErrCode::MemoryRegistrationError),
      "MemoryRegistrationError");
  EXPECT_STREQ(errorCodeToString(ErrCode::Timeout), "Timeout");
  EXPECT_STREQ(
      errorCodeToString(ErrCode::ResourceExhausted), "ResourceExhausted");
}

// --- Err tests ---

TEST(ErrTest, ConstructWithCodeOnly) {
  Err err(ErrCode::Timeout);
  EXPECT_EQ(err.code(), ErrCode::Timeout);
  EXPECT_TRUE(err.message().empty());
}

TEST(ErrTest, ConstructWithCodeAndMessage) {
  Err err(ErrCode::ConnectionFailed, "peer unreachable");
  EXPECT_EQ(err.code(), ErrCode::ConnectionFailed);
  EXPECT_EQ(err.message(), "peer unreachable");
}

TEST(ErrTest, ToStringWithoutMessage) {
  Err err(ErrCode::Timeout);
  EXPECT_EQ(err.toString(), "Timeout");
}

TEST(ErrTest, ToStringWithMessage) {
  Err err(ErrCode::Timeout, "after 30s");
  EXPECT_EQ(err.toString(), "Timeout: after 30s");
}

// --- Result<T> success tests ---

TEST(ResultTest, ConstructWithValue) {
  Result<int> r(42);
  EXPECT_TRUE(r.hasValue());
  EXPECT_FALSE(r.hasError());
  EXPECT_TRUE(static_cast<bool>(r));
  EXPECT_EQ(r.value(), 42);
}

TEST(ResultTest, ConstructWithStringValue) {
  Result<std::string> r(std::string("hello"));
  EXPECT_TRUE(r.hasValue());
  EXPECT_EQ(r.value(), "hello");
}

TEST(ResultTest, ImplicitConversionFromValue) {
  // Verify implicit construction works (no explicit keyword)
  Result<int> r = 10;
  EXPECT_TRUE(r.hasValue());
  EXPECT_EQ(r.value(), 10);
}

// --- Result<T> error tests ---

TEST(ResultTest, ConstructWithErr) {
  Result<int> r(Err(ErrCode::InvalidArgument, "bad value"));
  EXPECT_FALSE(r.hasValue());
  EXPECT_TRUE(r.hasError());
  EXPECT_FALSE(static_cast<bool>(r));
  EXPECT_EQ(r.error().code(), ErrCode::InvalidArgument);
  EXPECT_EQ(r.error().message(), "bad value");
}

TEST(ResultTest, ConstructWithErrCode) {
  Result<int> r(ErrCode::Timeout);
  EXPECT_TRUE(r.hasError());
  EXPECT_EQ(r.error().code(), ErrCode::Timeout);
}

TEST(ResultTest, ImplicitConversionFromErr) {
  Result<int> r = Err(ErrCode::ResourceExhausted);
  EXPECT_TRUE(r.hasError());
  EXPECT_EQ(r.error().code(), ErrCode::ResourceExhausted);
}

TEST(ResultTest, ImplicitConversionFromErrCode) {
  Result<int> r = ErrCode::NotConnected;
  EXPECT_TRUE(r.hasError());
  EXPECT_EQ(r.error().code(), ErrCode::NotConnected);
}

// --- Result<T> value access tests ---

TEST(ResultTest, ValueRef) {
  Result<std::string> r(std::string("test"));
  r.value() = "modified";
  EXPECT_EQ(r.value(), "modified");
}

TEST(ResultTest, ConstValueRef) {
  const Result<int> r(99);
  EXPECT_EQ(r.value(), 99);
}

TEST(ResultTest, MoveValue) {
  Result<std::string> r(std::string("moveme"));
  std::string moved = std::move(r).value();
  EXPECT_EQ(moved, "moveme");
}

TEST(ResultTest, ConstErrorRef) {
  const Result<int> r(Err(ErrCode::Timeout, "expired"));
  EXPECT_EQ(r.error().code(), ErrCode::Timeout);
  EXPECT_EQ(r.error().message(), "expired");
}

// --- Result<T> arrow operator tests ---

TEST(ResultTest, ArrowOperator) {
  Result<std::string> r(std::string("hello"));
  EXPECT_EQ(r->size(), 5u);
  EXPECT_EQ(r->substr(1), "ello");
}

TEST(ResultTest, ConstArrowOperator) {
  const Result<std::string> r(std::string("world"));
  EXPECT_EQ(r->size(), 5u);
  EXPECT_TRUE(r->starts_with("wor"));
}

// --- Result<T> valueOr tests ---

TEST(ResultTest, ValueOrReturnsValueWhenPresent) {
  Result<int> r(42);
  EXPECT_EQ(r.valueOr(0), 42);
}

TEST(ResultTest, ValueOrReturnsDefaultWhenError) {
  Result<int> r(ErrCode::Timeout);
  EXPECT_EQ(r.valueOr(99), 99);
}

TEST(ResultTest, ValueOrMoveReturnsValueWhenPresent) {
  Result<std::string> r(std::string("value"));
  std::string result = std::move(r).valueOr("default");
  EXPECT_EQ(result, "value");
}

TEST(ResultTest, ValueOrMoveReturnsDefaultWhenError) {
  Result<std::string> r(ErrCode::InvalidArgument);
  std::string result = std::move(r).valueOr("default");
  EXPECT_EQ(result, "default");
}

// --- Result<T> with convertible types ---

TEST(ResultTest, ImplicitConversionFromConvertibleType) {
  // unique_ptr<Derived> -> Result<unique_ptr<Base>>
  struct Base {
    virtual ~Base() = default;
  };
  struct Derived : Base {};

  Result<std::unique_ptr<Base>> r(std::make_unique<Derived>());
  EXPECT_TRUE(r.hasValue());
  EXPECT_NE(r.value(), nullptr);
}

// --- Result<T> as function return type ---

namespace {
Result<int> divide(int a, int b) {
  if (b == 0) {
    return Err(ErrCode::InvalidArgument, "division by zero");
  }
  return a / b;
}
} // namespace

TEST(ResultTest, FunctionReturningValue) {
  auto r = divide(10, 2);
  EXPECT_TRUE(r.hasValue());
  EXPECT_EQ(r.value(), 5);
}

TEST(ResultTest, FunctionReturningError) {
  auto r = divide(10, 0);
  EXPECT_TRUE(r.hasError());
  EXPECT_EQ(r.error().code(), ErrCode::InvalidArgument);
  EXPECT_EQ(r.error().message(), "division by zero");
}

TEST(StatusTest, OkHelperIsSuccess) {
  Status s = Ok();
  EXPECT_TRUE(s.hasValue());
  EXPECT_FALSE(s.hasError());
  EXPECT_TRUE(static_cast<bool>(s));
}
