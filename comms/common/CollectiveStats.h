// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>

#include <fmt/core.h>
#include <folly/Synchronized.h>
#include <folly/Utility.h>
#include <folly/container/F14Map.h>
#include <folly/hash/Hash.h>

namespace comms {

// Host-drive timing for one bucket of collectives, in microseconds.
struct CollectiveStat {
  uint64_t count{0};
  uint64_t total_us{0};
  uint64_t min_us{0};
  uint64_t max_us{0};

  void add(uint64_t durationUs) {
    if (count == 0) {
      min_us = max_us = durationUs;
    } else {
      min_us = std::min(min_us, durationUs);
      max_us = std::max(max_us, durationUs);
    }
    total_us += durationUs;
    ++count;
  }

  void merge(const CollectiveStat& other) {
    if (other.count == 0) {
      return;
    }
    if (count == 0) {
      *this = other;
      return;
    }
    min_us = std::min(min_us, other.min_us);
    max_us = std::max(max_us, other.max_us);
    total_us += other.total_us;
    count += other.count;
  }

  bool operator==(const CollectiveStat&) const = default;
};

// std::unordered_map, not F14: this crosses the mccl public interface and the
// pybind boundary, which has no F14 type caster.
using CollectiveStatsMap = std::unordered_map<std::string, CollectiveStat>;

// Reserved: getAndClear() publishes roll-ups under "<collective>.all" and a
// bare "all", so record() keys must not collide with either.
inline constexpr std::string_view kCollectiveStatsAllKey = "all";

// Shared by the comms backends (ctran, prims, mccl). Writers are the
// collective-issuing threads; the reader is a single consumer draining once
// per training step.
class CollectiveStats {
 public:
  // `collective` is the bare op name ("allreduce"); `key` is the fully
  // qualified bucket ("allreduce.ctring.1048576"). Roll-ups are derived in
  // getAndClear() to keep this to one lookup with no key allocation.
  void record(
      std::string_view collective,
      std::string_view key,
      uint64_t durationUs) {
    stats_.withWLock([&](auto& m) {
      auto it = m.find(key);
      if (it == m.end()) {
        it = m.emplace(std::string(key), Entry{std::string(collective), {}})
                 .first;
      }
      it->second.stat.add(durationUs);
    });
  }

  CollectiveStatsMap getAndClear() {
    EntryMap drained;
    stats_.withWLock([&](auto& m) { drained.swap(m); });

    CollectiveStatsMap out;
    out.reserve(2 * drained.size() + 1);

    CollectiveStat overall;
    for (const auto& [key, entry] : drained) {
      overall.merge(entry.stat);
      if (!entry.collective.empty()) {
        out[fmt::format("{}.{}", entry.collective, kCollectiveStatsAllKey)]
            .merge(entry.stat);
      }
      out[key].merge(entry.stat);
    }
    if (overall.count > 0) {
      out[std::string(kCollectiveStatsAllKey)].merge(overall);
    }
    return out;
  }

 private:
  struct Entry {
    std::string collective;
    CollectiveStat stat;
  };

  // Transparent hash so record() can look up by string_view without allocating.
  using EntryMap = folly::F14FastMap<
      std::string,
      Entry,
      folly::transparent<folly::hasher<std::string_view>>,
      folly::transparent<std::equal_to<std::string_view>>>;

  folly::Synchronized<EntryMap> stats_;
};

inline std::ostream& operator<<(std::ostream& os, const CollectiveStat& s) {
  return os << fmt::format(
             "count={} total_us={} min_us={} max_us={}",
             s.count,
             s.total_us,
             s.min_us,
             s.max_us);
}

} // namespace comms
