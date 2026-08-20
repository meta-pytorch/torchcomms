// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Versioned overrides plan: torchcomms hints -> ncclCollConfig_t for NCCL 2.31.
// For 2.29/2.30, overrides remain via InfoExt (handled in NCCLX).
// For 2.31, translate hints to a configured collective. This header is
// version-gated via NCCL_VERSION_CODE probing, not via is_backend_built.
//
// Usage in TorchCommNCCL collective entry points (e.g. allReduce):
//
//   #if NCCL_VERSION_CODE >= NCCL_VERSION(2,31,0)
//     ncclCollConfig_t collConfig = makeCollConfigFromHints(options.hints);
//     // pass collConfig to nccl*Config variant
//     result = nccl_api_->allReduceConfig(..., &collConfig);
//   #else
//     result = nccl_api_->allReduce(...);
//   #endif

#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include <glog/logging.h>
#include <nccl.h>

namespace torch::comms {

namespace configured_collective_detail {

// NCCL's NCCL_ALGO_*/NCCL_PROTO_* macros live in its internal plugin headers
// (src/include/plugin/nccl_tuner.h), which the OSS torchcomms export cannot
// include. The integer encoding is part of the stable tuner plugin ABI, so it
// is safe to mirror here under distinct names (the NCCL macros would otherwise
// textually replace these identifiers wherever both headers are visible).
constexpr int kAlgoTree = 0;
constexpr int kAlgoRing = 1;
constexpr int kAlgoCollnetDirect = 2;
constexpr int kAlgoCollnetChain = 3;
constexpr int kAlgoNvls = 4;
constexpr int kAlgoNvlsTree = 5;
constexpr int kAlgoPat = 6;

constexpr int kProtoLL = 0;
constexpr int kProtoLL128 = 1;
constexpr int kProtoSimple = 2;

} // namespace configured_collective_detail

// Map integer algo/protocol pair to 2.31 algSelection string.
// Duplicates the registry in comms/ncclx/meta/algoconf/ConfiguredCollective.cc
// on purpose: this header ships to OSS (meta-pytorch/torchcomms) and so cannot
// depend on internal NCCLX headers. Both tables must be updated together.
inline std::string algoProtoToString(int algo, int proto) {
  namespace d = configured_collective_detail;
  if (algo == d::kAlgoRing) {
    if (proto == d::kProtoSimple) {
      return "RING_SIMPLE";
    }
    if (proto == d::kProtoLL) {
      return "RING_LL";
    }
    if (proto == d::kProtoLL128) {
      return "RING_LL128";
    }
  } else if (algo == d::kAlgoTree) {
    if (proto == d::kProtoSimple) {
      return "TREE_SIMPLE";
    }
    if (proto == d::kProtoLL) {
      return "TREE_LL";
    }
    if (proto == d::kProtoLL128) {
      return "TREE_LL128";
    }
  } else if (algo == d::kAlgoCollnetDirect) {
    if (proto == d::kProtoSimple) {
      return "COLLNET_DIRECT_SIMPLE";
    }
  } else if (algo == d::kAlgoCollnetChain) {
    if (proto == d::kProtoSimple) {
      return "COLLNET_CHAIN_SIMPLE";
    }
  } else if (algo == d::kAlgoNvls) {
    if (proto == d::kProtoSimple) {
      return "NVLS_SIMPLE";
    }
  } else if (algo == d::kAlgoNvlsTree) {
    if (proto == d::kProtoSimple) {
      return "NVLSTREE_SIMPLE";
    }
  } else if (algo == d::kAlgoPat) {
    if (proto == d::kProtoSimple) {
      return "PAT_SIMPLE";
    }
  }
  return "";
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)

// Build a ncclCollConfig_t from torchcomms hints map. All fields are
// optional; unset fields remain NCCL_CONFIG_UNDEF. The returned config owns
// any algSelection storage and must outlive the nccl*Config call; release it
// afterwards with freeCollConfig().
inline ncclCollConfig_t makeCollConfigFromHints(
    const std::unordered_map<std::string, std::string>& hints) {
  ncclCollConfig_t cfg = NCCL_COLLCONFIG_INITIALIZER;

  auto it = hints.find("algSelection");
  if (it != hints.end() && !it->second.empty()) {
    // NCCL borrows the pointer rather than copying the string, so the storage
    // has to outlive the collective call.
    cfg.algSelection = strdup(it->second.c_str());
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
          cfg.algSelection = strdup(sel.c_str());
        }
      } catch (...) {
        LOG(WARNING) << "[TC] ConfiguredCollective: invalid algo/protocol hint "
                     << "algo=" << ia->second << " protocol=" << ip->second;
      }
    }
  }

  auto parseInt = [&](const char* key, int& out) -> bool {
    auto f = hints.find(key);
    if (f != hints.end()) {
      try {
        out = std::stoi(f->second);
        return true;
      } catch (...) {
        LOG(WARNING) << "[TC] ConfiguredCollective: invalid int hint " << key
                     << "=" << f->second;
      }
    }
    return false;
  };

  int tmp = NCCL_CONFIG_UNDEF_INT;
  if (parseInt("maxCTAs", tmp) || parseInt("max_ctas", tmp)) {
    cfg.maxCTAs = tmp;
  }
  tmp = NCCL_CONFIG_UNDEF_INT;
  if (parseInt("minCTAs", tmp) || parseInt("min_ctas", tmp)) {
    cfg.minCTAs = tmp;
  }
  tmp = NCCL_CONFIG_UNDEF_INT;
  if (parseInt("nvlsCTAs", tmp) || parseInt("nvls_ctas", tmp)) {
    cfg.nvlsCTAs = tmp;
  }
  tmp = NCCL_CONFIG_UNDEF_INT;
  if (parseInt("cgaClusterSize", tmp) || parseInt("cga_cluster_size", tmp)) {
    cfg.cgaClusterSize = tmp;
  }
  tmp = NCCL_CONFIG_UNDEF_INT;
  if (parseInt("CTAPolicy", tmp) || parseInt("cta_policy", tmp)) {
    cfg.CTAPolicy = tmp;
  }

  auto iu = hints.find("userProfilerTag");
  if (iu == hints.end()) {
    iu = hints.find("user_profiler_tag");
  }
  if (iu != hints.end()) {
    try {
      cfg.userProfilerTag = std::stoull(iu->second);
    } catch (...) {
      LOG(WARNING) << "[TC] ConfiguredCollective: invalid userProfilerTag="
                   << iu->second;
    }
  }

  // nWarps is intentionally dropped on 2.31 (upstream derives warps).
  return cfg;
}

// Helper to free a collConfig's owned algSelection string if set.
inline void freeCollConfig(ncclCollConfig_t& cfg) {
  if (cfg.algSelection != nullptr &&
      cfg.algSelection != (const char*)NCCL_CONFIG_UNDEF_PTR) {
    free((void*)cfg.algSelection);
    cfg.algSelection = (const char*)NCCL_CONFIG_UNDEF_PTR;
  }
}

#endif // NCCL_VERSION_CODE >= 2.31

} // namespace torch::comms
