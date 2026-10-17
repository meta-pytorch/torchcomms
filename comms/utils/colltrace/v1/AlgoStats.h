// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>

#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <folly/hash/Hash.h>

namespace meta::comms::colltrace {

// Result of algorithm statistics dump, per communicator.
// Map: op -> algo -> msgSize -> count
struct AlgoStatDump {
  uint64_t commHash{0};
  std::string commDesc;
  std::unordered_map<
      std::string,
      std::unordered_map<std::string, std::map<std::size_t, int64_t>>>
      entries;
};

// Thread-safe algorithm statistics tracker.
// Used in "stats" mode of colltrace to count collective calls by algorithm.
// Each communicator has its own AlgoStats instance.
class AlgoStats {
 public:
  AlgoStats() = default;
  AlgoStats(uint64_t commHash, const std::string& commDesc)
      : commHash_(commHash), commDesc_(commDesc) {}

  // Get or create a shared AlgoStats instance for a communicator.
  // Both baseline and ctran record into the same instance keyed by commHash.
  static std::shared_ptr<AlgoStats> getOrCreate(
      uint64_t commHash,
      const std::string& commDesc);

  // Record a collective execution with the given algorithm and message size.
  // Thread-safe: can be called concurrently from multiple threads.
  void record(
      const std::string& opName,
      const std::string& algoName,
      const std::size_t msgSize = 0);

  // Get aggregated counts with communicator info.
  AlgoStatDump dump() const;

  // Reset all counters to zero.
  void reset();

 private:
  uint64_t commHash_{0};
  std::string commDesc_;

  using AlgoKey = std::tuple<std::string, std::string, std::size_t>;
  struct AlgoKeyHash {
    std::size_t operator()(const AlgoKey& key) const noexcept {
      return folly::hash::hash_combine(
          std::get<0>(key), std::get<1>(key), std::get<2>(key));
    }
  };

  folly::Synchronized<folly::F14FastMap<AlgoKey, int64_t, AlgoKeyHash>>
      algoCounters_;
};

} // namespace meta::comms::colltrace
