// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <unordered_map>

#include "comms/ctran/utils/Utils.h"

namespace ctran::utils {

uint64_t generateCommHash(const std::vector<int>& sortedRanks) {
  // static map to track rankStr occurances
  // e.g map<"0,1,2,": 2>: "communicator 0,1,2 has been created twice"
  static std::unordered_map<std::string, int> rankStrToCount{};

  std::string sortedRankStr;
  for (const int& rank : sortedRanks) {
    sortedRankStr += std::to_string(rank) + ",";
  }
  auto it = rankStrToCount.find(sortedRankStr);
  if (it == rankStrToCount.end()) {
    rankStrToCount[sortedRankStr] = 0;
  }
  rankStrToCount.at(sortedRankStr)++;

  const std::string commStr =
      sortedRankStr + std::to_string(rankStrToCount.at(sortedRankStr));
  return getHash(commStr.c_str(), commStr.size());
}

uint64_t generateCommHash(const int nRanks) {
  std::vector<int> ranks(nRanks);
  for (int i = 0; i < nRanks; ++i) {
    ranks.at(i) = i;
  }
  return generateCommHash(ranks);
}

std::vector<std::pair<int, int>> getRanges(const std::vector<int>& sortedNums) {
  std::vector<std::pair<int, int>> ranges{};
  if (sortedNums.empty()) {
    return ranges;
  }

  int start = 0;
  for (int i = 1; i < sortedNums.size(); ++i) {
    if (sortedNums[i] != sortedNums[i - 1] + 1) {
      ranges.emplace_back(sortedNums.at(start), sortedNums.at(i - 1));
      start = i;
    }
  }
  ranges.emplace_back(sortedNums.at(start), sortedNums.back());
  return ranges;
}

std::string rangesToStr(const std::vector<std::pair<int, int>>& ranges) {
  std::string str{};
  if (ranges.empty()) {
    return "none";
  }

  for (int i = 0; i < ranges.size(); ++i) {
    const auto& range = ranges.at(i);
    str += "[" + std::to_string(range.first) + ", " +
        std::to_string(range.second) + "]";
    if (i != ranges.size() - 1) {
      str += ", ";
    }
  }
  return str;
}

const char* parseCommDesc(const char* commDesc) {
  static const char kUndefined[] = "undefined";
  return commDesc != nullptr ? commDesc : kUndefined;
}

} // namespace ctran::utils
