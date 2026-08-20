// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include "meta/algoconf/InfoExt.h"
#include "nccl.h"

// Forward decls to avoid pulling comm.h/info.h into every includer.
struct ncclComm;
struct ncclInfo;
struct ncclTaskColl;

namespace ncclx::algoconf {

// ---------------------------------------------------------------------------
// Versioned overrides plan (2.29/2.30 vs 2.31):
//  - 2.29 / 2.30 : overrides via InfoExt (ncclInfo::ext, ncclTaskColl::ext)
//    retained as-is behind facade maybeInfoExtOverride().
//  - 2.31       : overrides via ncclCollConfig_t configured collectives.
//    Use helpers below to build a collConfig from hints or from an InfoExt
//    shim for compat callers.
// This file is version-agnostic and compiles on all versions; 2.31-only
// APIs are guarded by NCCL_VERSION_CODE.
// ---------------------------------------------------------------------------

// Map (algorithm, protocol) integer pair to a 2.31 algSelection string.
// Uses the upstream algorithm registry names (e.g. "PAT_SIMPLE",
// "TREE_LL", "RING_SIMPLE"). Returns empty string if the pair has no
// registered kernel (caller should then leave algSelection automatic).
std::string algoProtoToString(int algo, int proto);

// Quantized ReduceScatter on the InfoExt path requires a doubled chunk size.
// On 2.31 quantization runs through the CTran direct path instead, where no
// ext carries a seed and this is a no-op.
inline int adjustChunkSizeForExt(
    const std::optional<ncclInfoExt>& ext,
    int chunkSize) {
  if (ext.has_value() && ext->quantizeRandomSeedPtr != nullptr) {
    return chunkSize * 2;
  }
  return chunkSize;
}

// Facade for 2.29/2.30 InfoExt path. Returns an override if the comm/task
// warrants one (e.g. PAT AVG, quantized path, or generic hints). Returns
// nullopt when no override applies. This is the single call-site that
// v2_30/src/collectives.cc should use:
//
//   info.ext = ncclx::algoconf::maybeInfoExtOverride(comm, info);
//
std::optional<ncclInfoExt> maybeInfoExtOverride(
    struct ncclComm* comm,
    struct ncclTaskColl* task);

// Overload taking ncclInfo directly (preferred for collectives.cc).
std::optional<ncclInfoExt> maybeInfoExtOverride(
    struct ncclComm* comm,
    const struct ncclInfo& info);

// --- 2.31 configured-collective helpers ------------------------------------

#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)

// Build a collConfig from an InfoExt shim. Used to keep compat callers
// that still produce an InfoExt (e.g. old PAT AVG helpers) working on
// 2.31 without propagating InfoExt through NCCL internals.
// Mapping:
//   algorithm+protocol -> algSelection string
//   nMaxChannels       -> maxCTAs (when >0)
//   nWarps             -> dropped (upstream tuning derives warps)
//   opDev              -> not representable in collConfig; if set, the
//                        caller must use an NCCLX-owned launch instead of
//                        a configured collective (this helper WARNs and
//                        returns a fully default-initialized config, so no
//                        field carries an override the caller could apply).
//   quantSeedPtr       -> not in collConfig; retained via CTran direct IB
//                        path, not via collConfig.
ncclCollConfig_t makeCollConfigFromExt(const std::optional<ncclInfoExt>& ext);

// Overload taking a concrete ext value.
ncclCollConfig_t makeCollConfigFromExt(const ncclInfoExt& ext);

// Build a collConfig directly from per-collective hints. Preferred path
// for new code on 2.31: hints already carry the same information that
// InfoExt used to carry, but without needing InfoExt at all.
// Supported hint keys (all optional):
//   "algSelection"  -> collConfig.algSelection (e.g. "PAT_SIMPLE")
//   "maxCTAs"       -> collConfig.maxCTAs
//   "minCTAs"       -> collConfig.minCTAs
//   "nvlsCTAs"      -> collConfig.nvlsCTAs
//   "cgaClusterSize"-> collConfig.cgaClusterSize
//   "CTAPolicy"     -> collConfig.CTAPolicy
//   "userProfilerTag" -> collConfig.userProfilerTag
// Unknown keys are silently ignored (invalid values WARN). The returned
// config owns any strdup'd algSelection string; caller must free via
// freeCollConfig() after the collective call (same contract as torchcomms
// sibling).
ncclCollConfig_t makeCollConfigFromHints(
    const std::unordered_map<std::string, std::string>& hints);

// Facade that mirrors maybeInfoExtOverride but returns a collConfig for
// 2.31 enqueue path:
//
//   #if NCCL_VERSION_CODE >= NCCL_VERSION(2,31,0)
//     auto collCfg = ncclx::algoconf::maybeCollConfig(comm, task, hints);
//     info.collConfig = collCfg;
//   #else
//     info.ext = ncclx::algoconf::maybeInfoExtOverride(comm, task);
//   #endif
//
ncclCollConfig_t maybeCollConfig(
    struct ncclComm* comm,
    struct ncclTaskColl* task,
    const std::unordered_map<std::string, std::string>& hints);

// Helper to free a collConfig's owned algSelection string if set.
inline void freeCollConfig(ncclCollConfig_t& cfg) {
  if (cfg.algSelection != nullptr &&
      cfg.algSelection != (const char*)NCCL_CONFIG_UNDEF_PTR) {
    free((void*)cfg.algSelection);
    cfg.algSelection = (const char*)NCCL_CONFIG_UNDEF_PTR;
  }
}

#endif // NCCL_VERSION_CODE >= 2.31

} // namespace ncclx::algoconf
