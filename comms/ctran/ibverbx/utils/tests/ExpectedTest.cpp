// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/ibverbx/utils/Expected.h"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace ibverbx::utils {

TEST(ExpectedTest, HoldsValue) {
  Expected<int, std::string> result(17);

  ASSERT_TRUE(result.hasValue());
  EXPECT_FALSE(result.hasError());
  EXPECT_EQ(result.value(), 17);
}

TEST(ExpectedTest, HoldsError) {
  Expected<int, std::string> result = makeUnexpected(std::string("failed"));

  ASSERT_TRUE(result.hasError());
  EXPECT_FALSE(result.hasValue());
  EXPECT_EQ(result.error(), "failed");
}

TEST(ExpectedTest, SupportsImplicitReturnConversions) {
  auto makeValue = []() -> Expected<int, std::string> { return 23; };
  auto makeError = []() -> Expected<int, std::string> {
    return makeUnexpected(std::string("failed"));
  };

  EXPECT_EQ(makeValue().value(), 23);
  EXPECT_EQ(makeError().error(), "failed");
}

TEST(ExpectedTest, MovesValueOut) {
  Expected<std::unique_ptr<int>, std::string> result(std::make_unique<int>(42));

  std::unique_ptr<int> value = std::move(result).value();
  if (value == nullptr) {
    ADD_FAILURE() << "Expected moved value";
    return;
  }
  EXPECT_EQ(*value, 42);
}

TEST(ExpectedTest, DereferencesValue) {
  Expected<int, std::string> result(17);
  const Expected<int, std::string> constResult(29);

  EXPECT_EQ(*result, 17);
  EXPECT_EQ(*constResult, 29);
}

TEST(ExpectedTest, ArrowReachesMember) {
  Expected<std::string, int> result(std::string("payload"));
  const Expected<std::string, int> constResult(std::string("payload"));

  EXPECT_EQ(result->size(), 7u);
  EXPECT_EQ(constResult->size(), 7u);
}

TEST(ExpectedTest, DereferenceMovesValueOut) {
  static_assert(
      std::is_same_v<
          decltype(*std::declval<Expected<std::string, int>&&>()),
          std::string&&>,
      "operator* on an rvalue Expected must yield an rvalue reference");

  Expected<std::unique_ptr<int>, std::string> result(std::make_unique<int>(42));

  // A copy would not compile, so reaching a non-null value proves the move.
  std::unique_ptr<int> value = *std::move(result);
  if (value == nullptr) {
    ADD_FAILURE() << "Expected moved value";
    return;
  }
  EXPECT_EQ(*value, 42);
}

TEST(ExpectedTest, MovesErrorOut) {
  Unexpected<std::unique_ptr<int>> unexpected(std::make_unique<int>(9));

  std::unique_ptr<int> error = std::move(unexpected).error();
  if (error == nullptr) {
    ADD_FAILURE() << "Expected moved error";
    return;
  }
  EXPECT_EQ(*error, 9);
}

TEST(ExpectedTest, MovesExpectedErrorOut) {
  Expected<int, std::unique_ptr<int>> result =
      makeUnexpected(std::make_unique<int>(11));

  std::unique_ptr<int> error = std::move(result).error();
  if (error == nullptr) {
    ADD_FAILURE() << "Expected moved Expected error";
    return;
  }
  EXPECT_EQ(*error, 11);
}

// Reading the wrong alternative aborts in every build. std::get would throw
// std::bad_variant_access, which an opt build would surface as an unannounced
// exception.
TEST(ExpectedDeathTest, ValueOnAnErrorAborts) {
  const Expected<int, std::string> failed = makeUnexpected(std::string("boom"));
  EXPECT_DEATH((void)failed.value(), "value\\(\\) read on an error");
}

TEST(ExpectedDeathTest, ErrorOnAValueAborts) {
  const Expected<int, std::string> ok = 7;
  EXPECT_DEATH((void)ok.error(), "error\\(\\) read on a value");
}

TEST(ExpectedTest, ConvertsToBoolByState) {
  const Expected<int, std::string> ok = 7;
  const Expected<int, std::string> failed = makeUnexpected(std::string("boom"));

  EXPECT_TRUE(static_cast<bool>(ok));
  EXPECT_FALSE(static_cast<bool>(failed));
}

// operator bool is explicit, so it must not enable implicit conversions that
// would let an Expected silently participate in arithmetic or comparisons.
TEST(ExpectedTest, DoesNotConvertToBoolImplicitly) {
  static_assert(!std::is_convertible_v<Expected<int, std::string>, bool>);
}

// A reassignable Expected requires an assignable error type; ibverbx::Error
// originally had const members, which silently deleted assignment.
TEST(ExpectedTest, ReassignsFromValueToError) {
  Expected<int, std::string> result = 7;

  result = Expected<int, std::string>(makeUnexpected(std::string("boom")));

  EXPECT_FALSE(static_cast<bool>(result));
  EXPECT_EQ(result.error(), "boom");
}

} // namespace ibverbx::utils
