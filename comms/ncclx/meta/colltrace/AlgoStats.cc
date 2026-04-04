// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "nccl.h"

// TODO: Remove this guard once v2_27 is deprecated — v2_27 does not have the
// comms/utils/colltrace:algostats dependency
#if NCCL_MINOR >= 28

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
  auto dump = comm->algoStats->dump();
  map.swap(dump.counts);
}

} // namespace ncclx::colltrace

#endif // NCCL_MINOR >= 28
