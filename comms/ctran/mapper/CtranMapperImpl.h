// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once
#include <unordered_map>
#include <unordered_set>
#include "comms/ctran/mapper/CtranMapperRegMem.h"

namespace ctran {
// Class to cache exported regElem for each mapper instance
class ExportRegCache {
 private:
  std::unordered_map<CtranMapperRegElem*, std::unordered_set<int>> map_;

 public:
  // Record the regElem and the rank to export to.
  inline void record(const CtranMapperRegElem* regElem, const int rank) {
    map_[const_cast<CtranMapperRegElem*>(regElem)].insert(rank);
  }

  // Remove the specified regElem from cache. Return the exported ranks set if
  // the regElem has been exported. Otherwise, empty set is returned.
  std::unordered_set<int> remove(const CtranMapperRegElem* regElem);

  // Dump a full copy of the cache map;
  // used for testing only
  std::unordered_map<CtranMapperRegElem*, std::unordered_set<int>> dump() const;
};
} // namespace ctran
