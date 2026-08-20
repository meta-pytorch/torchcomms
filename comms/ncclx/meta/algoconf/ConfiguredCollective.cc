// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/algoconf/ConfiguredCollective.h"

#include <cstdlib>

#include "comm.h"
#include "info.h"
#include "meta/collectives/PatAvgHelper.h"

namespace ncclx::algoconf {

// ---------------------------------------------------------------------------
// Shared helpers (version-agnostic)
// ---------------------------------------------------------------------------

std::string algoProtoToString(int algo, int proto) {
  // Map using upstream registry names. Keep this table in sync with
  // src/config/algorithm_registry.cc in NCCL 2.31. For unknown pairs,
  // return "" so the caller leaves algSelection automatic.
  if (algo == NCCL_ALGO_RING) {
    if (proto == NCCL_PROTO_SIMPLE) {
      return "RING_SIMPLE";
    }
    if (proto == NCCL_PROTO_LL) {
      return "RING_LL";
    }
    if (proto == NCCL_PROTO_LL128) {
      return "RING_LL128";
    }
  } else if (algo == NCCL_ALGO_TREE) {
    if (proto == NCCL_PROTO_SIMPLE) {
      return "TREE_SIMPLE";
    }
    if (proto == NCCL_PROTO_LL) {
      return "TREE_LL";
    }
    if (proto == NCCL_PROTO_LL128) {
      return "TREE_LL128";
    }
  } else if (algo == NCCL_ALGO_COLLNET_DIRECT) {
    if (proto == NCCL_PROTO_SIMPLE) {
      return "COLLNET_DIRECT_SIMPLE";
    }
  } else if (algo == NCCL_ALGO_COLLNET_CHAIN) {
    if (proto == NCCL_PROTO_SIMPLE) {
      return "COLLNET_CHAIN_SIMPLE";
    }
  } else if (algo == NCCL_ALGO_NVLS) {
    if (proto == NCCL_PROTO_SIMPLE) {
      return "NVLS_SIMPLE";
    }
  } else if (algo == NCCL_ALGO_NVLS_TREE) {
    if (proto == NCCL_PROTO_SIMPLE) {
      return "NVLSTREE_SIMPLE";
    }
  } else if (algo == NCCL_ALGO_PAT) {
    if (proto == NCCL_PROTO_SIMPLE) {
      return "PAT_SIMPLE";
    }
  }
  return "";
}

std::optional<ncclInfoExt> maybeInfoExtOverride(
    struct ncclComm* comm,
    struct ncclTaskColl* task) {
  // TODO: task-based overrides need hints threaded through the enqueue path.
  // Until then this overload has nothing the ncclInfo one does not cover.
  (void)task;
  (void)comm;
  return std::nullopt;
}

std::optional<ncclInfoExt> maybeInfoExtOverride(
    struct ncclComm* comm,
    const struct ncclInfo& info) {
  if (comm == nullptr) {
    return std::nullopt;
  }
#if defined(IS_NCCLX)
  // IS_NCCLX is defined in v2_29/v2_30 forked headers; pristine 2.31
  // does not have usePatAvg_ on ncclComm (PAT AVG there is via
  // configured collConfig PAT_SIMPLE). Guard so this file compiles on
  // both.
  if (comm->usePatAvg_ && info.op == ncclAvg &&
      ncclx::isPatAvgSupportedType(info.datatype)) {
    size_t nBytes = info.count * ncclTypeSize(info.datatype) * comm->nRanks;
    return ncclx::setupPatAvgInfoExt(comm, nBytes, info.datatype);
  }
#else
  (void)info;
#endif
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// 2.31 configured-collective path
// ---------------------------------------------------------------------------

#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)

ncclCollConfig_t makeCollConfigFromExt(const ncclInfoExt& ext) {
  ncclCollConfig_t cfg = NCCL_COLLCONFIG_INITIALIZER;

  std::string sel = algoProtoToString(ext.algorithm, ext.protocol);
  if (!sel.empty()) {
    // NCCL borrows the pointer rather than copying the string, so the config
    // owns heap storage that the caller releases via freeCollConfig().
    char* dup = strdup(sel.c_str());
    cfg.algSelection = dup;
  }

  if (ext.nMaxChannels > 0) {
    cfg.maxCTAs = ext.nMaxChannels;
  }

  if (ext.nWarps != 0) {
    INFO(
        NCCL_TUNING,
        "ConfiguredCollective: nWarps=%d dropped (upstream derives warps)",
        ext.nWarps);
  }

  if (ext.opDev.has_value()) {
    // A custom device reduction op has no collConfig representation: PAT AVG
    // with SumPostDiv belongs on a configured PAT_SIMPLE collective with the
    // standard AVG op, other custom ops need an NCCLX-owned launch. Every
    // field is reset, not just algSelection, so a caller that inspects
    // maxCTAs cannot pick up a stray override from an ext we rejected.
    WARN(
        "ConfiguredCollective: opDev override not representable in "
        "ncclCollConfig_t; use NCCLX-owned launch or PAT_SIMPLE AVG");
    freeCollConfig(cfg);
    cfg = NCCL_COLLCONFIG_INITIALIZER;
    return cfg;
  }

  if (ext.quantizeRandomSeedPtr != nullptr) {
    INFO(
        NCCL_TUNING,
        "ConfiguredCollective: quantizeRandomSeedPtr not in collConfig; "
        "use CTran direct path");
  }

  return cfg;
}

ncclCollConfig_t makeCollConfigFromExt(const std::optional<ncclInfoExt>& ext) {
  if (!ext.has_value()) {
    return NCCL_COLLCONFIG_INITIALIZER;
  }
  return makeCollConfigFromExt(*ext);
}

ncclCollConfig_t makeCollConfigFromHints(
    const std::unordered_map<std::string, std::string>& hints) {
  ncclCollConfig_t cfg = NCCL_COLLCONFIG_INITIALIZER;

  auto it = hints.find("algSelection");
  if (it != hints.end() && !it->second.empty()) {
    // Caller owns strdup'd string, must free via freeCollConfig().
    char* dup = strdup(it->second.c_str());
    cfg.algSelection = dup;
  }

  // NCCL_COLLCONFIG_INITIALIZER sets algSelection to NCCL_CONFIG_UNDEF_PTR, a
  // non-null sentinel, so it must be excluded before indexing the string.
  if (cfg.algSelection == nullptr ||
      cfg.algSelection == (const char*)NCCL_CONFIG_UNDEF_PTR ||
      cfg.algSelection[0] == '\0') {
    auto ia = hints.find("algo");
    auto ip = hints.find("protocol");
    if (ia != hints.end() && ip != hints.end()) {
      try {
        int algo = std::stoi(ia->second);
        int proto = std::stoi(ip->second);
        std::string sel = algoProtoToString(algo, proto);
        if (!sel.empty()) {
          // Caller owns strdup'd string, must free via freeCollConfig().
          char* dup = strdup(sel.c_str());
          cfg.algSelection = dup;
        }
      } catch (...) {
        // ignore parse errors, leave automatic
      }
    }
  }

  auto parseInt = [&](const char* key, int& out) {
    auto f = hints.find(key);
    if (f != hints.end()) {
      try {
        out = std::stoi(f->second);
      } catch (...) {
        WARN(
            "ConfiguredCollective: invalid int hint %s=%s",
            key,
            f->second.c_str());
      }
    }
  };

  int tmp = NCCL_CONFIG_UNDEF_INT;
  parseInt("maxCTAs", tmp);
  if (tmp != NCCL_CONFIG_UNDEF_INT) {
    cfg.maxCTAs = tmp;
  }
  tmp = NCCL_CONFIG_UNDEF_INT;
  parseInt("minCTAs", tmp);
  if (tmp != NCCL_CONFIG_UNDEF_INT) {
    cfg.minCTAs = tmp;
  }
  tmp = NCCL_CONFIG_UNDEF_INT;
  parseInt("nvlsCTAs", tmp);
  if (tmp != NCCL_CONFIG_UNDEF_INT) {
    cfg.nvlsCTAs = tmp;
  }
  tmp = NCCL_CONFIG_UNDEF_INT;
  parseInt("cgaClusterSize", tmp);
  if (tmp != NCCL_CONFIG_UNDEF_INT) {
    cfg.cgaClusterSize = tmp;
  }
  tmp = NCCL_CONFIG_UNDEF_INT;
  parseInt("CTAPolicy", tmp);
  if (tmp != NCCL_CONFIG_UNDEF_INT) {
    cfg.CTAPolicy = tmp;
  }

  auto iu = hints.find("userProfilerTag");
  if (iu != hints.end()) {
    try {
      cfg.userProfilerTag = std::stoull(iu->second);
    } catch (...) {
      WARN(
          "ConfiguredCollective: invalid userProfilerTag=%s",
          iu->second.c_str());
    }
  }

  if (hints.find("nWarps") != hints.end() ||
      hints.find("warps") != hints.end()) {
    INFO(
        NCCL_TUNING,
        "ConfiguredCollective: nWarps hint dropped (upstream derives warps)");
  }

  return cfg;
}

ncclCollConfig_t maybeCollConfig(
    struct ncclComm* /*comm*/,
    struct ncclTaskColl* /*task*/,
    const std::unordered_map<std::string, std::string>& hints) {
  // A default-initialized config is how NCCL spells "no override": every
  // field is a sentinel, so ncclParseCollConfig applies nothing.
  if (hints.empty()) {
    return NCCL_COLLCONFIG_INITIALIZER;
  }
  return makeCollConfigFromHints(hints);
}

#endif // NCCL_VERSION_CODE >= 2.31

} // namespace ncclx::algoconf
