// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "VerifyAlgoStatsUtil.h"

#include <fmt/core.h>
#include <gtest/gtest.h>
#include "comms/utils/cvars/nccl_cvars.h"

namespace ctran::test {

VerifyAlgoStatsHelper::~VerifyAlgoStatsHelper() {
  if (enabled_) {
    NCCL_COLLTRACE = oldColltrace_;
  }
}

void VerifyAlgoStatsHelper::enable() {
  oldColltrace_ = NCCL_COLLTRACE;
  NCCL_COLLTRACE.push_back("algostat");
  enabled_ = true;
}

namespace {

std::string formatStats(const std::unordered_map<std::string, int64_t>& stats) {
  std::string result;
  for (const auto& [name, count] : stats) {
    if (!result.empty()) {
      result += ", ";
    }
    result += fmt::format("{}({})", name, count);
  }
  return result;
}

} // namespace

void VerifyAlgoStatsHelper::verify(
    CtranComm* comm,
    const std::string& collective,
    const std::string& expectedAlgoSubstr) const {
  auto statDumpOpt = comm->dumpAlgoStats();
  ASSERT_TRUE(statDumpOpt.has_value());
  auto statDump = std::move(*statDumpOpt);
  auto it = statDump.counts.find(collective);
  ASSERT_NE(it, statDump.counts.end())
      << collective << " not found in AlgoStats";
  bool found = false;
  for (const auto& [algoName, callCount] : it->second) {
    if (algoName.find(expectedAlgoSubstr) != std::string::npos &&
        callCount > 0) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected algorithm containing '" << expectedAlgoSubstr
                     << "' not found in " << collective
                     << ". Found: " << formatStats(it->second);
}

void VerifyAlgoStatsHelper::verifyNot(
    CtranComm* comm,
    const std::string& collective,
    const std::string& unexpectedAlgoSubstr) const {
  auto statDumpOpt = comm->dumpAlgoStats();
  ASSERT_TRUE(statDumpOpt.has_value());
  auto statDump = std::move(*statDumpOpt);
  auto it = statDump.counts.find(collective);
  if (it == statDump.counts.end()) {
    return;
  }
  for (const auto& [algoName, callCount] : it->second) {
    EXPECT_FALSE(
        algoName.find(unexpectedAlgoSubstr) != std::string::npos &&
        callCount > 0)
        << "Unexpected algorithm '" << algoName << "' with count " << callCount
        << " found in " << collective;
  }
}

void VerifyAlgoStatsHelper::dump(CtranComm* comm, const std::string& collective)
    const {
  auto statDumpOpt = comm->dumpAlgoStats();
  if (!statDumpOpt) {
    return;
  }
  auto statDump = std::move(*statDumpOpt);
  auto it = statDump.counts.find(collective);
  if (it != statDump.counts.end()) {
    fmt::print(
        stderr, "AlgoStats[{}]: [{}]\n", collective, formatStats(it->second));
  }
}

} // namespace ctran::test
