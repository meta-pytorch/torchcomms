// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "VerifyAlgoStatsUtil.h"

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <unordered_map>
#include "meta/colltrace/AlgoStats.h"

namespace ncclx {
namespace test {

void VerifyAlgoStatsHelper::enable() {
  guard_.emplace(NCCL_COLLTRACE, std::vector<std::string>{"algostat"});
}

void VerifyAlgoStatsHelper::verify(
    ncclComm_t comm,
    const std::string& collective,
    const std::string& expectedAlgoSubstr) {
  std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>
      stats;
  ncclx::colltrace::dumpAlgoStat(comm, stats);

  auto it = stats.find(collective);
  ASSERT_NE(it, stats.end())
      << collective << " not found in AlgoStats. Stats may not be enabled.";

  bool foundExpectedAlgo = false;
  std::string foundAlgos;
  for (const auto& [algoName, callCount] : it->second) {
    if (!foundAlgos.empty()) {
      foundAlgos += ", ";
    }
    foundAlgos += fmt::format("{}({})", algoName, callCount);
    if (algoName.find(expectedAlgoSubstr) != std::string::npos &&
        callCount > 0) {
      foundExpectedAlgo = true;
    }
  }
  EXPECT_TRUE(foundExpectedAlgo)
      << "Expected algorithm containing '" << expectedAlgoSubstr
      << "' not found in " << collective << ". Found algorithms: ["
      << foundAlgos << "]";
}

} // namespace test
} // namespace ncclx
