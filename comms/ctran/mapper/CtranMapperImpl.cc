// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/mapper/CtranMapperImpl.h"

namespace ctran {
std::unordered_set<int> ExportRegCache::remove(
    const CtranMapperRegElem* regElem) {
  std::unordered_set<int> nvlRanks;

  auto it = map_.find(const_cast<CtranMapperRegElem*>(regElem));
  if (it != map_.end()) {
    nvlRanks = it->second;
    map_.erase(it);
  }

  return nvlRanks;
}

std::unordered_map<CtranMapperRegElem*, std::unordered_set<int>>
ExportRegCache::dump() const {
  return map_;
}
} // namespace ctran
