// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <map>
#include <string>
#include <utility>

#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <folly/hash/Hash.h>

namespace meta::comms::colltrace {

// Result of algorithm statistics dump, per communicator.
struct AlgoStatDump {
  uint64_t commHash{0};
  std::string commDesc;
  // Map: collective name -> algorithm name -> call count
  std::map<std::string, std::map<std::string, int64_t>> counts;
};

// Thread-safe algorithm statistics tracker.
// Used in "stats" mode of colltrace to count collective calls by algorithm.
// Each communicator has its own AlgoStats instance.
class AlgoStats {
 public:
  AlgoStats() = default;

  // Set communicator metadata for this stats instance.
  void setCommInfo(uint64_t commHash, const std::string& commDesc);

  // Record a collective execution with the given algorithm.
  // Thread-safe: can be called concurrently from multiple threads.
  void record(const std::string& opName, const std::string& algoName);

  // Get aggregated counts with communicator info.
  AlgoStatDump dump() const;

  // Reset all counters to zero.
  void reset();

 private:
  uint64_t commHash_{0};
  std::string commDesc_;

  // Key: (opName, algoName), Value: count
  using AlgoKey = std::pair<std::string, std::string>;
  struct AlgoKeyHash {
    std::size_t operator()(const AlgoKey& key) const noexcept {
      return folly::hash::hash_combine(key.first, key.second);
    }
  };

  folly::Synchronized<folly::F14FastMap<AlgoKey, int64_t, AlgoKeyHash>>
      algoCounters_;
};

} // namespace meta::comms::colltrace
