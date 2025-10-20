// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "comms/ctran/utils/Utils.h"

namespace ctran {

TEST(UtilxTest, GenerateCommHash) {
  std::vector<int> worldRanks = {0, 1, 2, 3, 4, 5, 6, 7};
  auto hash0 = utils::generateCommHash(worldRanks);
  // it should produce the same hash for every run
  EXPECT_EQ(hash0, 1637454487954707092);

  std::vector<int> childRanks = {0, 2, 4, 6};
  auto hash1 = utils::generateCommHash(childRanks);
  auto hash2 = utils::generateCommHash(childRanks);

  EXPECT_NE(hash1, hash2);
  EXPECT_NE(hash0, hash1);
  EXPECT_NE(hash0, hash2);
}

TEST(UtilxTest, GetRanges) {
  {
    auto sortedNums = std::vector<int>{0, 1, 2, 3, 4, 5};
    auto ranges = utils::getRanges(sortedNums);
    std::vector<std::pair<int, int>> expected = {{0, 5}};
    EXPECT_EQ(ranges, expected);
  }

  {
    auto sortedNums = std::vector<int>{0, 1, 2, 5, 6, 7};
    auto ranges = utils::getRanges(sortedNums);
    std::vector<std::pair<int, int>> expected = {{0, 2}, {5, 7}};
    EXPECT_EQ(ranges, expected);
  }

  {
    auto sortedNums = std::vector<int>{0, 1, 2, 3, 7};
    auto ranges = utils::getRanges(sortedNums);
    std::vector<std::pair<int, int>> expected = {{0, 3}, {7, 7}};
    EXPECT_EQ(ranges, expected);
  }

  {
    auto sortedNums = std::vector<int>{0, 3, 7};
    auto ranges = utils::getRanges(sortedNums);
    std::vector<std::pair<int, int>> expected = {{0, 0}, {3, 3}, {7, 7}};
    EXPECT_EQ(ranges, expected);
  }

  {
    auto sortedNums = std::vector<int>{};
    auto ranges = utils::getRanges(sortedNums);
    EXPECT_EQ(ranges.size(), 0);
  }
}

TEST(UtilxTest, RangesToStr) {
  {
    std::vector<std::pair<int, int>> ranges = {{0, 5}};
    auto str = utils::rangesToStr(ranges);
    EXPECT_EQ(str, "[0, 5]");
  }

  {
    std::vector<std::pair<int, int>> ranges = {{0, 1}, {3, 5}};
    auto str = utils::rangesToStr(ranges);
    EXPECT_EQ(str, "[0, 1], [3, 5]");
  }

  {
    std::vector<std::pair<int, int>> ranges = {};
    auto str = utils::rangesToStr(ranges);
    EXPECT_EQ(str, "none");
  }
}

TEST(UtilxTest, array2DToStr) {
  const std::vector<int> vec1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  // Print 2D array with different shapes
  const std::array<size_t, 2> shape1 = {4 /* dimY */, 3 /* dimX */};
  auto str1 = utils::array2DToStr(vec1.data(), shape1[0], shape1[1]);
  EXPECT_EQ(str1, "1 2 3, 4 5 6, 7 8 9, 10 11 12");

  const std::array<size_t, 2> shape2 = {6 /* dimY */, 2 /* dimX */};
  auto str2 = utils::array2DToStr(vec1.data(), shape2[0], shape2[1]);
  EXPECT_EQ(str2, "1 2, 3 4, 5 6, 7 8, 9 10, 11 12");

  // Print 2D array with subset of the data
  const std::array<size_t, 2> shape3 = {3 /* dimY */, 2 /* dimX */};
  auto str3 = utils::array2DToStr(vec1.data(), shape3[0], shape3[1]);
  EXPECT_EQ(str3, "1 2, 3 4, 5 6");

  // Print 2D array with omited Y dimension
  const std::array<size_t, 4> shape4 = {
      6 /* dimY */, 2 /* dimX */, 3 /* maxReportDimY */, 2 /* maxReportDimX */};
  auto str4 = utils::array2DToStr(
      vec1.data(), shape4[0], shape4[1], shape4[2], shape4[3]);
  EXPECT_EQ(str4, "1 2, 3 4, 5 6, ...");

  // Print 2D array with omited X dimension
  const std::array<size_t, 4> shape5 = {
      2 /* dimY */, 6 /* dimX */, 3 /* maxReportDimY */, 2 /* maxReportDimX */};
  auto str5 = utils::array2DToStr(
      vec1.data(), shape5[0], shape5[1], shape5[2], shape5[3]);
  EXPECT_EQ(str5, "1 2..., 7 8...");

  // Print 2D array with omited Y dimension and show last row
  auto str6 = utils::array2DToStr(
      vec1.data(), shape4[0], shape4[1], shape4[2], shape4[3], true);
  EXPECT_EQ(str6, "1 2, 3 4, 5 6, ..., 11 12");

  // Print 2D array with omited X dimension and show last col
  auto str7 = utils::array2DToStr(
      vec1.data(), shape5[0], shape5[1], shape5[2], shape5[3], true);
  EXPECT_EQ(str7, "1 2...6, 7 8...12");

  // Print 2D array with omited X dimension and Y dimension
  const std::array<size_t, 4> shape8 = {
      4 /* dimY */, 3 /* dimX */, 2 /* maxReportDimY */, 1 /* maxReportDimX */};
  auto str8 = utils::array2DToStr(
      vec1.data(), shape8[0], shape8[1], shape8[2], shape8[3]);
  EXPECT_EQ(str8, "1..., 4..., ...");

  // Print 2D array with omited X dimension and Y dimension and show last col
  // and row
  auto str9 = utils::array2DToStr(
      vec1.data(), shape8[0], shape8[1], shape8[2], shape8[3], true);
  EXPECT_EQ(str9, "1...3, 4...6, ..., 10...12");
}

TEST(UtilxTest, align) {
  const size_t size0 = 4091, alignment0 = 4096;
  ASSERT_EQ(utils::align(size0, alignment0), 4096);

  const size_t size1 = 8199, alignment1 = 4096;
  ASSERT_EQ(utils::align(size1, alignment1), 12288);

  const int size2 = 89, alignment2 = 16;
  ASSERT_EQ(utils::align(size2, alignment2), 96);
}

TEST(HostnameTest, GetFullHostnameReturnsNonEmpty) {
  std::string full = utils::getFullHostname();
  EXPECT_FALSE(full.empty());

  char hostname[1024];
  gethostname(hostname, 1024);
  EXPECT_EQ(full, std::string(hostname));
}
TEST(HostnameTest, GetHostnameReturnsFirstComponent) {
  std::string full = utils::getFullHostname();
  std::vector<std::string> parts;
  folly::split('.', full, parts);
  std::string expected = parts.empty() ? full : parts[0];
  EXPECT_EQ(utils::getHostname(), expected);
}

} // namespace ctran
