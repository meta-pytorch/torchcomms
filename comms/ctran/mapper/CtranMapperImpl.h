// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once
#include <unordered_map>
#include "comms/ctran/regcache/RegCache.h"

namespace ctran {
// Class to cache exported regElem for each mapper instance.
// Tracks how many times each regElem has been exported to each peer rank,
// so that release notifications can decrement the import-side refcount by the
// correct amount.
// FIXME(alvinyc): after enabling AutoSwitch, we should remove the explicit NVL
// export API using CtrlMsg, and the exportCount tracking
class ExportRegCache {
 private:
  // regElem -> (rank -> export count)
  std::unordered_map<regcache::RegElem*, std::unordered_map<int, int>> map_;

 public:
  // Record an export of regElem to the given rank. Increments the export count
  // for this (regElem, rank) pair.
  inline void record(const regcache::RegElem* regElem, const int rank) {
    auto& rankMap = map_[const_cast<regcache::RegElem*>(regElem)];
    auto [it, inserted] = rankMap.emplace(rank, 0);
    it->second++;
  }

  // Remove the specified regElem from cache. Return the exported ranks map
  // (rank -> export count) if the regElem has been exported. Otherwise, empty
  // map is returned.
  std::unordered_map<int, int> remove(const regcache::RegElem* regElem);

  // Dump a full copy of the cache map;
  // used for testing only
  std::unordered_map<regcache::RegElem*, std::unordered_map<int, int>> dump()
      const;
};
} // namespace ctran
