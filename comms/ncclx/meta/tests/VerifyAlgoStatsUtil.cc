// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "VerifyAlgoStatsUtil.h"

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <unordered_map>
#include "meta/colltrace/AlgoStats.h"

namespace ncclx::test {

namespace {

using AlgoStatsMap = std::unordered_map<std::string, int64_t>;

// Retrieve per-algorithm stats for a collective.
// Returns the algo name -> call count map; empty if collective not found.
AlgoStatsMap getAlgoStats(ncclComm_t comm, const std::string& collective) {
  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>
      stats;
  ncclx::colltrace::dumpAlgoStat(comm, stats);

  auto it = stats.find(collective);
  EXPECT_NE(it, stats.end())
      << collective << " not found in AlgoStats. Stats may not be enabled.";
  if (it == stats.end()) {
    return {};
  }
  return std::move(it->second);
}

// Format algo stats map as "algo1(count1), algo2(count2), ...".
std::string formatAlgoStats(const AlgoStatsMap& algoStats) {
  std::string result;
  for (const auto& [algoName, callCount] : algoStats) {
    if (!result.empty()) {
      result += ", ";
    }
    result += fmt::format("{}({})", algoName, callCount);
  }
  return result;
}

// Check if any algorithm matching the substring was used (callCount > 0).
bool findAlgoWithCalls(
    const AlgoStatsMap& stats,
    const std::string& algoSubstr) {
  for (const auto& [algoName, callCount] : stats) {
    if (algoName.find(algoSubstr) != std::string::npos && callCount > 0) {
      return true;
    }
  }
  return false;
}

} // namespace

void VerifyAlgoStatsHelper::enable() {
  colltraceGuard_.emplace(NCCL_COLLTRACE, std::vector<std::string>{"algostat"});
}

void VerifyAlgoStatsHelper::dump(ncclComm_t comm, const std::string& collective)
    const {
  auto stats = getAlgoStats(comm, collective);
  fmt::print(
      stderr, "AlgoStats[{}]: [{}]\n", collective, formatAlgoStats(stats));
}

void VerifyAlgoStatsHelper::verify(
    ncclComm_t comm,
    const std::string& collective,
    const std::string& expectedAlgoSubstr) const {
  auto stats = getAlgoStats(comm, collective);
  EXPECT_TRUE(findAlgoWithCalls(stats, expectedAlgoSubstr))
      << "Expected algorithm containing '" << expectedAlgoSubstr
      << "' not found in " << collective << ". Found algorithms: ["
      << formatAlgoStats(stats) << "]";
}

void VerifyAlgoStatsHelper::verifyNot(
    ncclComm_t comm,
    const std::string& collective,
    const std::string& unexpectedAlgoSubstr) const {
  auto stats = getAlgoStats(comm, collective);
  EXPECT_FALSE(findAlgoWithCalls(stats, unexpectedAlgoSubstr))
      << "Unexpected algorithm containing '" << unexpectedAlgoSubstr
      << "' was used in " << collective << ". Found algorithms: ["
      << formatAlgoStats(stats) << "]";
}

} // namespace ncclx::test
