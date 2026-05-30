// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/AlgoStats.h"

#include <folly/Synchronized.h>

namespace meta::comms::colltrace {

namespace {
folly::Synchronized<std::unordered_map<uint64_t, std::shared_ptr<AlgoStats>>>
    algoStatsInstances;
} // namespace

std::shared_ptr<AlgoStats> AlgoStats::getOrCreate(
    uint64_t commHash,
    const std::string& commDesc) {
  auto locked = algoStatsInstances.wlock();
  auto it = locked->find(commHash);
  if (it != locked->end()) {
    return it->second;
  }
  auto stats = std::make_shared<AlgoStats>(commHash, commDesc);
  locked->emplace(commHash, stats);
  return stats;
}

void AlgoStats::record(
    const std::string& opName,
    const std::string& algoName,
    const std::size_t msgSize) {
  algoCounters_.withWLock(
      [&](auto& counters) { ++counters[AlgoKey{opName, algoName, msgSize}]; });
}

AlgoStatDump AlgoStats::dump() const {
  AlgoStatDump result;
  result.commHash = commHash_;
  result.commDesc = commDesc_;

  algoCounters_.withRLock([&](const auto& counters) {
    for (const auto& [key, count] : counters) {
      result.entries[std::get<0>(key)][std::get<1>(key)][std::get<2>(key)] =
          count;
    }
  });

  return result;
}

void AlgoStats::reset() {
  algoCounters_.withWLock([](auto& counters) { counters.clear(); });
}

} // namespace meta::comms::colltrace
