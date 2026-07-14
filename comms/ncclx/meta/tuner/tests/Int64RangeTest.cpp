// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>

#include "meta/tuner/Int64Range.h"

namespace {

using ncclx::tuner::Int64Range;
using ncclx::tuner::parseInt64Range;

// Convenience: parse and assert success, returning the Int64Range.
Int64Range parseOk(const std::string& value) {
  const std::optional<Int64Range> range = parseInt64Range(value);
  EXPECT_TRUE(range.has_value()) << "expected '" << value << "' to parse";
  return range.value_or(Int64Range{});
}

// "*" is the wildcard: fully unbounded, matches every value.
TEST(Int64RangeTest, WildcardMatchesEverything) {
  const Int64Range range = parseOk("*");
  EXPECT_FALSE(range.loBounded);
  EXPECT_FALSE(range.hiBounded);
  EXPECT_TRUE(range.matches(INT64_MIN));
  EXPECT_TRUE(range.matches(0));
  EXPECT_TRUE(range.matches(INT64_MAX));
}

// "-1" is no longer special: it parses as the ordinary exact value [-1, -1].
TEST(Int64RangeTest, NegativeOneIsExactNotWildcard) {
  const Int64Range range = parseOk("-1");
  EXPECT_TRUE(range.loBounded);
  EXPECT_TRUE(range.hiBounded);
  EXPECT_TRUE(range.matches(-1));
  EXPECT_FALSE(range.matches(0));
  EXPECT_FALSE(range.matches(5));
}

// A bare integer N parses to the exact closed interval [N, N].
TEST(Int64RangeTest, ExactValue) {
  const Int64Range zero = parseOk("0");
  EXPECT_TRUE(zero.matches(0));
  EXPECT_FALSE(zero.matches(1));
  EXPECT_FALSE(zero.matches(-1));

  const Int64Range five = parseOk("5");
  EXPECT_TRUE(five.loBounded);
  EXPECT_TRUE(five.hiBounded);
  EXPECT_TRUE(five.matches(5));
  EXPECT_FALSE(five.matches(4));
  EXPECT_FALSE(five.matches(6));
}

// Closed interval [0,2]: both endpoints included.
TEST(Int64RangeTest, ClosedInterval) {
  const Int64Range range = parseOk("[0,2]");
  EXPECT_FALSE(range.matches(-1));
  EXPECT_TRUE(range.matches(0));
  EXPECT_TRUE(range.matches(1));
  EXPECT_TRUE(range.matches(2));
  EXPECT_FALSE(range.matches(3));
}

// Open interval (0,2): both endpoints excluded.
TEST(Int64RangeTest, OpenInterval) {
  const Int64Range range = parseOk("(0,2)");
  EXPECT_FALSE(range.matches(0));
  EXPECT_TRUE(range.matches(1));
  EXPECT_FALSE(range.matches(2));
}

// Half-open (0,2]: lower excluded, upper included.
TEST(Int64RangeTest, HalfOpenLowerExclusive) {
  const Int64Range range = parseOk("(0,2]");
  EXPECT_FALSE(range.matches(0));
  EXPECT_TRUE(range.matches(1));
  EXPECT_TRUE(range.matches(2));
  EXPECT_FALSE(range.matches(3));
}

// Half-open [1,3): lower included, upper excluded.
TEST(Int64RangeTest, HalfOpenUpperExclusive) {
  const Int64Range range = parseOk("[1,3)");
  EXPECT_FALSE(range.matches(0));
  EXPECT_TRUE(range.matches(1));
  EXPECT_TRUE(range.matches(2));
  EXPECT_FALSE(range.matches(3));
}

// (1,): lower-bounded exclusive, upper unbounded -> n > 1.
TEST(Int64RangeTest, OpenEndedLowerExclusive) {
  const Int64Range range = parseOk("(1,)");
  EXPECT_TRUE(range.loBounded);
  EXPECT_FALSE(range.hiBounded);
  EXPECT_FALSE(range.matches(1));
  EXPECT_TRUE(range.matches(2));
  EXPECT_TRUE(range.matches(INT64_MAX));
}

// [2,): lower-bounded inclusive, upper unbounded -> n >= 2.
TEST(Int64RangeTest, OpenEndedLowerInclusive) {
  const Int64Range range = parseOk("[2,)");
  EXPECT_FALSE(range.matches(1));
  EXPECT_TRUE(range.matches(2));
  EXPECT_TRUE(range.matches(3));
}

// (,4]: lower unbounded, upper-bounded inclusive -> n <= 4.
TEST(Int64RangeTest, OpenEndedUpperInclusive) {
  const Int64Range range = parseOk("(,4]");
  EXPECT_FALSE(range.loBounded);
  EXPECT_TRUE(range.hiBounded);
  EXPECT_TRUE(range.matches(INT64_MIN));
  EXPECT_TRUE(range.matches(4));
  EXPECT_FALSE(range.matches(5));
}

// (,4): lower unbounded, upper-bounded exclusive -> n < 4.
TEST(Int64RangeTest, OpenEndedUpperExclusive) {
  const Int64Range range = parseOk("(,4)");
  EXPECT_TRUE(range.matches(3));
  EXPECT_FALSE(range.matches(4));
  EXPECT_FALSE(range.matches(5));
}

// 64-bit GB-scale values match (1 GiB .. 16 GiB inclusive).
TEST(Int64RangeTest, SixtyFourBitValues) {
  constexpr int64_t kOneGiB = 1073741824;
  constexpr int64_t kSixteenGiB = 17179869184;
  const Int64Range range = parseOk("[1073741824,17179869184]");
  EXPECT_EQ(range.lo, kOneGiB);
  EXPECT_EQ(range.hi, kSixteenGiB);
  EXPECT_FALSE(range.matches(kOneGiB - 1));
  EXPECT_TRUE(range.matches(kOneGiB));
  EXPECT_TRUE(range.matches(4294967296)); // 4 GiB, inside
  EXPECT_TRUE(range.matches(kSixteenGiB));
  EXPECT_FALSE(range.matches(kSixteenGiB + 1));
}

// Surrounding and internal whitespace is tolerated.
TEST(Int64RangeTest, Whitespace) {
  const Int64Range range = parseOk("  (1, 4]  ");
  EXPECT_FALSE(range.matches(1));
  EXPECT_TRUE(range.matches(2));
  EXPECT_TRUE(range.matches(4));
  EXPECT_FALSE(range.matches(5));
}

// Invalid inputs all return nullopt.
TEST(Int64RangeTest, InvalidInputsReturnNullopt) {
  EXPECT_FALSE(parseInt64Range("").has_value());
  EXPECT_FALSE(parseInt64Range("   ").has_value());
  EXPECT_FALSE(parseInt64Range("garbage").has_value());
  EXPECT_FALSE(parseInt64Range("[a,b]").has_value());
  EXPECT_FALSE(parseInt64Range("[1,x]").has_value());
  EXPECT_FALSE(parseInt64Range("[1,2").has_value()); // missing closing bracket
  EXPECT_FALSE(parseInt64Range("1,2]").has_value()); // missing opening bracket
  EXPECT_FALSE(parseInt64Range("[12]").has_value()); // no comma in interval
  EXPECT_FALSE(parseInt64Range("[1,2,3]").has_value()); // extra comma
  EXPECT_FALSE(parseInt64Range("(2,1]").has_value()); // lo > hi
}

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
