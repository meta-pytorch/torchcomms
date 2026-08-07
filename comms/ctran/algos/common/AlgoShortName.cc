// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/algos/common/AlgoShortName.h"

#include <string>

#include <folly/Utility.h>
#include <folly/container/F14Map.h>
#include <folly/hash/Hash.h>

#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvImpl.h"
#include "comms/ctran/algos/Broadcast/BroadcastImpl.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"

namespace ctran {
namespace {

using ShortNameMap = folly::F14FastMap<
    std::string,
    std::string,
    folly::transparent<folly::hasher<std::string_view>>,
    folly::transparent<std::equal_to<std::string_view>>>;

// Built from the *AlgoName() functions rather than literals so display-name
// edits stay in one place. Names shared across collectives (CtranAuto,
// Baseline) map to the same token, so first-wins insertion is safe.
//
// CTRAN algorithms use their NCCL_*_ALGO cvar token; the MCCL-hosted ones
// (ctree/cthierarchical_ring/ctmdirect) use their MCCL names instead, so
// "ring" always means MCCL Ring and never CTRAN's ctring. Those three run
// natively via Prims and do not reach the GPE today, so they are reserved
// rather than currently emitted.
ShortNameMap buildShortNames() {
  ShortNameMap m;
  const auto add = [&m](const std::string& display, const char* token) {
    m.emplace(display, token);
  };

  add(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::orig), "orig");
  add(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::ctran), "ctran");
  add(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::ctdirect), "ctdirect");
  add(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::ctring), "ctring");
  add(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::ctree), "tree");
  add(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::cthierarchical_ring), "ring");
  add(allReduceAlgoName(NCCL_ALLREDUCE_ALGO::ctmdirect), "direct");

  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::orig), "orig");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctran), "ctran");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctdirect), "ctdirect");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctring), "ctring");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctsrd), "ctsrd");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctbrucks), "ctbrucks");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::cthierarchical_ring), "ring");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctgraph), "ctgraph");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctgraph_pipeline),
      "ctgraph_pipeline");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctgraph_rdpipeline),
      "ctgraph_rdpipeline");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctgraph_ring), "ctgraph_ring");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctgraph_rd), "ctgraph_rd");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctwin), "ctwin");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctwin_ring), "ctwin_ring");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctwin_srd), "ctwin_srd");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctwin_pipeline), "ctwin_pipeline");
  add(allGatherAlgoName(NCCL_ALLGATHER_ALGO::ctwin_rdpipeline),
      "ctwin_rdpipeline");

  add(reduceScatterAlgoName(NCCL_REDUCESCATTER_ALGO::orig), "orig");
  add(reduceScatterAlgoName(NCCL_REDUCESCATTER_ALGO::ctran), "ctran");
  add(reduceScatterAlgoName(NCCL_REDUCESCATTER_ALGO::ctdirect), "ctdirect");
  add(reduceScatterAlgoName(NCCL_REDUCESCATTER_ALGO::ctring), "ctring");
  add(reduceScatterAlgoName(NCCL_REDUCESCATTER_ALGO::ctrhd), "ctrhd");
  add(reduceScatterAlgoName(NCCL_REDUCESCATTER_ALGO::ctdirect_ib),
      "ctdirect_ib");

  add(broadcastAlgoName(NCCL_BROADCAST_ALGO::ctran), "ctran");
  add(broadcastAlgoName(NCCL_BROADCAST_ALGO::ctdirect), "ctdirect");
  add(broadcastAlgoName(NCCL_BROADCAST_ALGO::ctbtree), "ctbtree");

  add(allToAllAlgoName(NCCL_ALLTOALL_ALGO::orig), "orig");
  add(allToAllAlgoName(NCCL_ALLTOALL_ALGO::ctran), "ctran");
  add(allToAllAlgoName(NCCL_ALLTOALL_ALGO::ctgraph), "ctgraph");
  add(allToAllAlgoName(NCCL_ALLTOALL_ALGO::ctwin), "ctwin");

  add(allToAllvAlgoName(NCCL_ALLTOALLV_ALGO::orig), "orig");
  add(allToAllvAlgoName(NCCL_ALLTOALLV_ALGO::ctran), "ctran");
  add(allToAllvAlgoName(NCCL_ALLTOALLV_ALGO::compCtran), "compCtran");
  add(allToAllvAlgoName(NCCL_ALLTOALLV_ALGO::bsCompCtran), "bsCompCtran");

  return m;
}

} // namespace

std::string_view algoShortName(std::string_view displayName) {
  static const ShortNameMap kShortNames = buildShortNames();
  const auto it = kShortNames.find(displayName);
  return it == kShortNames.end() ? displayName : it->second;
}

} // namespace ctran
