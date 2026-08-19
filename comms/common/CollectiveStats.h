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

  // Launch geometry; 0 means unknown. Per-SM utilisation is
  // blocks_per_sm * block_size / maxThreadsPerSM.
  uint32_t num_blocks{0};
  uint32_t block_size{0};
  uint32_t blocks_per_sm{0};

  // SM residency: sum of ceil(num_blocks / blocks_per_sm) * duration. Survives
  // roll-ups where the geometry above collapses to unknown.
  uint64_t total_sm_us{0};

  void add(
      uint64_t durationUs,
      uint32_t numBlocks = 0,
      uint32_t blockSize = 0,
      uint32_t blocksPerSm = 0) {
    if (count == 0) {
      min_us = max_us = durationUs;
    } else {
      min_us = std::min(min_us, durationUs);
      max_us = std::max(max_us, durationUs);
    }
    total_us += durationUs;
    ++count;

    if (blockSize == 0) {
      return; // no geometry reported; keep whatever another backend recorded
    }
    if (count > 1 &&
        (num_blocks != numBlocks || block_size != blockSize ||
         blocks_per_sm != blocksPerSm)) {
      // Not every key is geometry-constant: variable-size ops share a
      // "<op>.<algo>.0" bucket, and grouped submits are keyed off the first op.
      num_blocks = block_size = blocks_per_sm = 0;
    } else {
      num_blocks = numBlocks;
      block_size = blockSize;
      blocks_per_sm = blocksPerSm;
    }
    total_sm_us += smUs(numBlocks, blocksPerSm, durationUs);
  }

  // Falls back to the grid size when occupancy is unknown.
  static uint64_t
  smUs(uint32_t numBlocks, uint32_t blocksPerSm, uint64_t durationUs) {
    if (numBlocks == 0) {
      return 0;
    }
    const uint32_t perSm = blocksPerSm > 0 ? blocksPerSm : 1;
    const uint64_t sms = (numBlocks + perSm - 1) / perSm;
    return sms * durationUs;
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
    total_sm_us += other.total_sm_us;

    // Roll-ups span buckets whose geometry differs.
    if (num_blocks != other.num_blocks || block_size != other.block_size ||
        blocks_per_sm != other.blocks_per_sm) {
      num_blocks = block_size = blocks_per_sm = 0;
    }
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
      uint64_t durationUs,
      uint32_t numBlocks = 0,
      uint32_t blockSize = 0,
      uint32_t blocksPerSm = 0) {
    stats_.withWLock([&](auto& m) {
      auto it = m.find(key);
      if (it == m.end()) {
        it = m.emplace(std::string(key), Entry{std::string(collective), {}})
                 .first;
      }
      it->second.stat.add(durationUs, numBlocks, blockSize, blocksPerSm);
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
  os << fmt::format(
      "count={} total_us={} min_us={} max_us={} total_sm_us={}",
      s.count,
      s.total_us,
      s.min_us,
      s.max_us,
      s.total_sm_us);
  if (s.block_size != 0) {
    os << fmt::format(
        " num_blocks={} block_size={} blocks_per_sm={}",
        s.num_blocks,
        s.block_size,
        s.blocks_per_sm);
  }
  return os;
}

} // namespace comms
