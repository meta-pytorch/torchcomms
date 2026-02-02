// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/AlgoStats.h"

namespace meta::comms::colltrace {

void AlgoStats::record(const std::string& opName, const std::string& algoName) {
  algoCounters_.withWLock(
      [&](auto& counters) { ++counters[AlgoKey{opName, algoName}]; });
}

AlgoStatDump AlgoStats::dump() const {
  AlgoStatDump result;
  result.commHash = commHash_;
  result.commDesc = commDesc_;

  algoCounters_.withRLock([&](const auto& counters) {
    for (const auto& [key, count] : counters) {
      result.counts[key.first][key.second] = count;
    }
  });

  return result;
}

void AlgoStats::reset() {
  algoCounters_.withWLock([](auto& counters) { counters.clear(); });
}

} // namespace meta::comms::colltrace
