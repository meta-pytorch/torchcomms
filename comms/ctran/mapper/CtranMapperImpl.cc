// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/mapper/CtranMapperImpl.h"

namespace ctran {
std::unordered_map<int, int> ExportRegCache::remove(
    const regcache::RegElem* regElem) {
  std::unordered_map<int, int> nvlRanks;

  auto it = map_.find(const_cast<regcache::RegElem*>(regElem));
  if (it != map_.end()) {
    nvlRanks = std::move(it->second);
    map_.erase(it);
  }

  return nvlRanks;
}

std::unordered_map<regcache::RegElem*, std::unordered_map<int, int>>
ExportRegCache::dump() const {
  return map_;
}
} // namespace ctran
