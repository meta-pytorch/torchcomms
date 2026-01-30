// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/colltrace/AlgoStats.h"

#include "comm.h"

namespace ncclx::colltrace {

__attribute__((visibility("default"))) void dumpAlgoStat(
    ncclComm_t comm,
    std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>&
        map) {
  map.clear();
  if (comm == nullptr || comm->algoStats == nullptr) {
    return;
  }
  // Convert from internal std::map to public std::unordered_map
  auto internal = comm->algoStats->dump().counts;
  for (auto& [opName, algoMap] : internal) {
    for (auto& [algoName, count] : algoMap) {
      map[opName][algoName] = count;
    }
  }
}

} // namespace ncclx::colltrace
