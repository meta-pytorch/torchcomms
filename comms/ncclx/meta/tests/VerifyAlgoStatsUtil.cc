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

// Check algorithm usage (callCount > 0) against a substring.
// When matchAll is false (default), returns true if ANY used algorithm name
// contains algoSubstr. When matchAll is true, returns true IFF there is at
// least one used algorithm AND every used algorithm name contains algoSubstr.
bool findAlgoWithCalls(
    const AlgoStatsMap& stats,
    const std::string& algoSubstr,
    const bool matchAll = false) {
  int numValid = 0;
  for (const auto& [algoName, callCount] : stats) {
    if (callCount <= 0) {
      continue;
    }
    numValid++;
    const bool matches = algoName.find(algoSubstr) != std::string::npos;
    if (matches) {
      // early return if match any
      if (!matchAll) {
        return true;
      }
    } else if (matchAll) {
      // early return if any mismatch found but match all is required
      return false;
    }
  }

  if (matchAll) {
    // any mismatch should have returned false above, so true if valid records
    // exists
    return numValid > 0;
  } else {
    // match any, any match should have returned true above.
    return false;
  }
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

void VerifyAlgoStatsHelper::verifyExact(
    ncclComm_t comm,
    const std::string& collective,
    const std::string& expectedSubstr) const {
  const auto stats = getAlgoStats(comm, collective);
  EXPECT_TRUE(findAlgoWithCalls(stats, expectedSubstr, /*matchAll=*/true))
      << "Not all " << collective << " algorithms contain '" << expectedSubstr
      << "'. Recorded: [" << formatAlgoStats(stats) << "]";
}

void VerifyAlgoStatsHelper::verifyEqual(
    ncclComm_t expected,
    ncclComm_t actual,
    const std::string& collective) const {
  const auto expectedStats = getAlgoStats(expected, collective);
  const auto actualStats = getAlgoStats(actual, collective);
  EXPECT_EQ(expectedStats, actualStats)
      << "Algo stats for " << collective << " differ: expected ["
      << formatAlgoStats(expectedStats) << "], actual ["
      << formatAlgoStats(actualStats) << "]";
}

} // namespace ncclx::test
