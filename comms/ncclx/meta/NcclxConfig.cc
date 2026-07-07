// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/NcclxConfig.h"

#include "comm.h" // NOLINT
#include "debug.h"
#include "group.h" // NOLINT
#include "nccl.h" // @manual
#include "param.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/algoconf/AlgoStrConv.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using ncclx::algoconf::algoStrToVal;
using ncclx::algoconf::algoValToStr;

namespace ncclx {

Config::Config(const ncclConfig_t* config) {
  initEnv();

  useCtran = NCCL_CTRAN_ENABLE;
  usePatAvg = NCCL_REDUCESCATTER_PAT_AVG_ENABLE;
  noLocal = false;
  sendrecvAlgo = NCCL_SENDRECV_ALGO;
  allgatherAlgo = NCCL_ALLGATHER_ALGO;
  allreduceAlgo = NCCL_ALLREDUCE_ALGO;
  alltoallvAlgo = NCCL_ALLTOALLV_ALGO;
  rmaAlgo = NCCL_RMA_ALGO;

  if (!config) {
    WARN("ncclx::Config: config is null");
    throw std::invalid_argument("config is null");
  }

  if (config->ncclxConfig != NCCL_CONFIG_UNDEF_PTR) {
    WARN("ncclx::Config: ncclxConfig already parsed");
    throw std::invalid_argument("ncclxConfig already parsed");
  }

  // Read hints (if any)
  ncclx::Hints* hints = nullptr;
  if (config->hints != NCCL_CONFIG_UNDEF_PTR && config->hints != nullptr) {
    hints = static_cast<ncclx::Hints*>(config->hints);
  }

  // Check if a hint key is present
  auto hasHint = [&](const char* key) -> bool {
    if (!hints) {
      return false;
    }
    std::string val;
    return hints->get(key, val) == ncclSuccess;
  };

  // Detect conflicts: a field must not be set in both the flat
  // ncclConfig_t (old format) and hints (new format).
  bool conflict = false;
  auto checkPtrConflict = [&](const char* key, const void* flatVal) {
    if (flatVal != nullptr && hasHint(key)) {
      WARN(
          "NCCLX config field '%s' set in both ncclConfig_t and "
          "hints; use one or the other, not both",
          key);
      conflict = true;
    }
  };
  auto checkIntConflict = [&](const char* key, int flatVal) {
    if (flatVal != NCCL_CONFIG_UNDEF_INT && hasHint(key)) {
      WARN(
          "NCCLX config field '%s' set in both ncclConfig_t and "
          "hints; use one or the other, not both",
          key);
      conflict = true;
    }
  };

  checkPtrConflict("commDesc", config->commDesc);
  checkPtrConflict("splitGroupRanks", config->splitGroupRanks);
  checkIntConflict("fastInitMode", config->fastInitMode);

  if (conflict) {
    throw std::invalid_argument("field set in both ncclConfig_t and hints");
  }

  // Helper: read a string value from hints.
  auto getHintStr = [&](const char* key) -> std::string {
    std::string val;
    if (hints && hints->get(key, val) == ncclSuccess) {
      return val;
    }
    return "";
  };

  // Helper: parse a bool from a hint string value.  Accepts 0/1,
  // yes/no, true/false, y/n, t/f (case insensitive).
  auto parseHintBool = [&](const char* key, bool envDef) -> bool {
    std::string val = getHintStr(key);
    if (val.empty()) {
      return envDef;
    }
    std::string lower(val.size(), '\0');
    std::transform(val.begin(), val.end(), lower.begin(), ::tolower);
    if (lower == "1" || lower == "yes" || lower == "true" || lower == "y" ||
        lower == "t") {
      return true;
    }
    if (lower == "0" || lower == "no" || lower == "false" || lower == "n" ||
        lower == "f") {
      return false;
    }
    try {
      return std::stoi(val) != 0;
    } catch (const std::exception&) {
      WARN("NCCLX hint '%s': invalid integer value '%s'", key, val.c_str());
      return envDef;
    }
  };

  // commDesc
  if (config->commDesc) {
    commDesc = config->commDesc;
  } else {
    auto val = getHintStr("commDesc");
    if (!val.empty()) {
      commDesc = val;
    }
  }

  // splitGroupRanks
  if (config->splitGroupRanks) {
    int size = config->splitGroupSize != NCCL_CONFIG_UNDEF_INT
        ? config->splitGroupSize
        : 0;
    splitGroupRanks = std::vector<int>(
        config->splitGroupRanks, config->splitGroupRanks + size);
  } else {
    auto val = getHintStr("splitGroupRanks");
    if (!val.empty()) {
      std::vector<int> elems;
      std::istringstream ss(val);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        try {
          elems.push_back(std::stoi(tok));
        } catch (const std::exception&) {
          WARN(
              "NCCLX hint 'splitGroupRanks': invalid integer '%s'",
              tok.c_str());
          throw std::invalid_argument("splitGroupRanks: invalid integer");
        }
      }
      splitGroupRanks = elems;
    }
  }

  if (config->fastInitMode != NCCL_CONFIG_UNDEF_INT) {
    fastInitMode = config->fastInitMode != 0;
  } else {
    // NCCL_FASTINIT_MODE is a enum, we could not directly convert it to bool
    fastInitMode = parseHintBool(
        "fastInitMode", NCCL_FASTINIT_MODE == NCCL_FASTINIT_MODE::ring_hybrid);
  }
  // Per-communicator pipes NVL transport config overrides
  {
    std::string val = getHintStr("pipesNvlChunkSize");
    if (!val.empty()) {
      try {
        pipesNvlChunkSize = std::stoull(val);
      } catch (const std::exception&) {
        WARN("NCCLX hint 'pipesNvlChunkSize': invalid value '%s'", val.c_str());
      }
    }
  }
  {
    std::string val = getHintStr("pipesIbgdaDataBufferSize");
    if (!val.empty()) {
      try {
        auto parsed = std::stoull(val);
        if (parsed > 0) {
          pipesIbgdaDataBufferSize = parsed;
        } else {
          WARN("NCCLX hint 'pipesIbgdaDataBufferSize': value must be positive");
        }
      } catch (const std::exception&) {
        WARN(
            "NCCLX hint 'pipesIbgdaDataBufferSize': invalid value '%s'",
            val.c_str());
      }
    }
  }

  ibLazyConnect = parseHintBool("ibLazyConnect", NCCL_CTRAN_IBGDA_LAZY_CONNECT);

  // vCliqueSize: hint only (no flat ncclConfig_t field)
  {
    auto val = getHintStr("vCliqueSize");
    if (!val.empty()) {
      try {
        vCliqueSize = std::stoi(val);
      } catch (const std::exception&) {
        WARN(
            "NCCLX hint 'vCliqueSize': invalid integer value '%s'",
            val.c_str());
      }
    }
  }

  // ncclBuffSize: hint only (no flat ncclConfig_t field)
  {
    std::string val = getHintStr("ncclBuffSize");
    if (!val.empty()) {
      try {
        int parsed = std::stoi(val);
        if (parsed <= 0) {
          WARN("NCCLX hint 'ncclBuffSize': value %d must be positive", parsed);
        } else {
          ncclBuffSize = parsed;
        }
      } catch (const std::exception&) {
        WARN("NCCLX hint 'ncclBuffSize': invalid value '%s'", val.c_str());
      }
    }
  }

  // ibSplitDataOnQps: hint only, 0 or 1
  {
    std::string val = getHintStr("ibSplitDataOnQps");
    if (!val.empty()) {
      try {
        int parsed = std::stoi(val);
        if (parsed != 0 && parsed != 1) {
          WARN(
              "NCCLX hint 'ibSplitDataOnQps': value %d must be 0 or 1", parsed);
        } else {
          ibSplitDataOnQps = parsed;
        }
      } catch (const std::exception&) {
        WARN("NCCLX hint 'ibSplitDataOnQps': invalid value '%s'", val.c_str());
      }
    }
  }

  // ibQpsPerConnection: hint only, positive integer
  {
    std::string val = getHintStr("ibQpsPerConnection");
    if (!val.empty()) {
      try {
        int parsed = std::stoi(val);
        if (parsed <= 0) {
          WARN(
              "NCCLX hint 'ibQpsPerConnection': value %d must be positive",
              parsed);
        } else {
          ibQpsPerConnection = parsed;
        }
      } catch (const std::exception&) {
        WARN(
            "NCCLX hint 'ibQpsPerConnection': invalid value '%s'", val.c_str());
      }
    }
  }

  useCtran = parseHintBool("useCtran", NCCL_CTRAN_ENABLE);
  usePatAvg = parseHintBool("usePatAvg", NCCL_REDUCESCATTER_PAT_AVG_ENABLE);
  noLocal = parseHintBool("noLocal", false);

  auto parseAlgoHint = [&](const char* key, auto& field) {
    auto val = getHintStr(key);
    if (!val.empty()) {
      algoStrToVal(val, field);
    }
  };
  parseAlgoHint("sendrecvAlgo", sendrecvAlgo);
  parseAlgoHint("allgatherAlgo", allgatherAlgo);
  parseAlgoHint("allreduceAlgo", allreduceAlgo);
  parseAlgoHint("alltoallvAlgo", alltoallvAlgo);
  parseAlgoHint("rmaAlgo", rmaAlgo);
}

ncclResult_t Config::update(const ncclx::Hints* hints) {
  if (hints == nullptr) {
    return ncclSuccess;
  }

  const auto& mutableKeys = ncclx::mutableHintKeys();

  for (const auto& key : ncclx::knownHintKeys()) {
    if (std::find(mutableKeys.begin(), mutableKeys.end(), key) !=
        mutableKeys.end()) {
      continue;
    }
    std::string val;
    if (hints->get(key, val) == ncclSuccess) {
      WARN(
          "ncclx::commSetConfig: hint key '%s' is not mutable post-init",
          key.c_str());
      return ncclInvalidUsage;
    }
  }

  auto parseAlgoHint = [&](const char* key, auto& field) {
    std::string val;
    if (hints->get(key, val) == ncclSuccess) {
      algoStrToVal(val, field);
    }
  };
  parseAlgoHint("sendrecvAlgo", sendrecvAlgo);
  parseAlgoHint("allgatherAlgo", allgatherAlgo);
  parseAlgoHint("allreduceAlgo", allreduceAlgo);
  parseAlgoHint("alltoallvAlgo", alltoallvAlgo);
  parseAlgoHint("rmaAlgo", rmaAlgo);

  return ncclSuccess;
}

} // namespace ncclx

void ncclxLogCommConfig(ncclComm_t comm) {
  if (comm == nullptr) {
    return;
  }

  const auto& cfg = comm->config;

  // Log non-UNDEF ncclConfig_t and ncclx::Config fields
  std::string fields;
  auto append = [&](const std::string& kv) {
    if (!fields.empty()) {
      fields += ' ';
    }
    fields += kv;
  };
  auto appendInt = [&](const char* name, int val) {
    if (val != NCCL_CONFIG_UNDEF_INT) {
      append(fmt::format("{}={}", name, val));
    }
  };
  auto appendStr = [&](const char* name, const char* val) {
    if (val != nullptr && val != NCCL_CONFIG_UNDEF_PTR) {
      append(fmt::format("{}={}", name, val));
    }
  };

  // ncclConfig_t fields
  appendInt("blocking", cfg.blocking);
  appendInt("cgaClusterSize", cfg.cgaClusterSize);
  appendInt("minCTAs", cfg.minCTAs);
  appendInt("maxCTAs", cfg.maxCTAs);
  appendStr("netName", cfg.netName);
  appendInt("splitShare", cfg.splitShare);
  appendInt("trafficClass", cfg.trafficClass);
  appendStr("commName", cfg.commName);
  appendInt("collnetEnable", cfg.collnetEnable);
  appendInt("CTAPolicy", cfg.CTAPolicy);
  appendInt("shrinkShare", cfg.shrinkShare);
  appendInt("nvlsCTAs", cfg.nvlsCTAs);
  appendInt("nChannelsPerNetPeer", cfg.nChannelsPerNetPeer);
  appendInt("nvlinkCentricSched", cfg.nvlinkCentricSched);
  appendInt("graphUsageMode", cfg.graphUsageMode);
  appendInt("numRmaCtx", cfg.numRmaCtx);
  appendStr("commDesc", cfg.commDesc);
  appendInt("fastInitMode", cfg.fastInitMode);

  // ncclx::Config fields
  if (cfg.ncclxConfig != NCCL_CONFIG_UNDEF_PTR && cfg.ncclxConfig != nullptr) {
    const auto* xCfg = static_cast<const ncclx::Config*>(cfg.ncclxConfig);
    auto appendAlgo = [&](const char* name, const auto& field) {
      append(fmt::format("{}={}", name, algoValToStr(field)));
    };
    append(fmt::format("useCtran={}", xCfg->useCtran));
    append(fmt::format("usePatAvg={}", xCfg->usePatAvg));
    append(fmt::format("noLocal={}", xCfg->noLocal));
    append(fmt::format("ibLazyConnect={}", xCfg->ibLazyConnect));
    appendAlgo("sendrecvAlgo", xCfg->sendrecvAlgo);
    appendAlgo("allgatherAlgo", xCfg->allgatherAlgo);
    appendAlgo("allreduceAlgo", xCfg->allreduceAlgo);
    appendAlgo("alltoallvAlgo", xCfg->alltoallvAlgo);
    appendAlgo("rmaAlgo", xCfg->rmaAlgo);
    auto appendIfSet = [&](const char* name, const auto& opt) {
      if (opt.has_value()) {
        append(fmt::format("{}={}", name, *opt));
      }
    };
    if (xCfg->vCliqueSize != 0) {
      append(fmt::format("vCliqueSize={}", xCfg->vCliqueSize));
    }
    appendIfSet("pipesNvlChunkSize", xCfg->pipesNvlChunkSize);
    appendIfSet("pipesIbgdaDataBufferSize", xCfg->pipesIbgdaDataBufferSize);
    appendIfSet("ncclBuffSize", xCfg->ncclBuffSize);
    appendIfSet("ibSplitDataOnQps", xCfg->ibSplitDataOnQps);
    appendIfSet("ibQpsPerConnection", xCfg->ibQpsPerConnection);
  }

  INFO(
      NCCL_INIT,
      "NCCLX CONFIG: [commHash=%lx commDesc=%s rank=%d nRanks=%d] init %s",
      comm->commHash,
      NCCLX_CONFIG_FIELD(comm->config, commDesc).c_str(),
      comm->rank,
      comm->nRanks,
      fields.empty() ? "(defaults)" : fields.c_str());
}

// C-style wrapper around the ncclx::Config parsing constructor.
// Most NCCL code is C-based, so this function translates C++
// exceptions into ncclResult_t error codes for the C callers.
ncclResult_t ncclxParseCommConfig(ncclConfig_t* config) {
  try {
    config->ncclxConfig = new ncclx::Config(config);
    return ncclSuccess;
  } catch (const std::exception&) {
    return ncclInvalidArgument;
  }
}

__attribute__((visibility("default"))) ncclResult_t
ncclx::commSetConfig(ncclComm_t comm, const ncclConfig_t* config) {
  if (comm == nullptr) {
    WARN("ncclx::commSetConfig: comm is null");
    return ncclInvalidArgument;
  }

  NCCLCHECK(ncclCommEnsureReady(comm));

  if (ncclGroupDepth > 0) {
    WARN("ncclx::commSetConfig: cannot update config inside a group call");
    return ncclInvalidUsage;
  }

  if (config == nullptr) {
    WARN("ncclx::commSetConfig: config is null");
    return ncclInvalidArgument;
  }

  if (config->magic != NCCL_API_MAGIC) {
    WARN(
        "ncclx::commSetConfig: ncclConfig_t not initialized with "
        "NCCL_CONFIG_INITIALIZER");
    return ncclInvalidArgument;
  }

  if (config->blocking != NCCL_CONFIG_UNDEF_INT ||
      config->cgaClusterSize != NCCL_CONFIG_UNDEF_INT ||
      config->minCTAs != NCCL_CONFIG_UNDEF_INT ||
      config->maxCTAs != NCCL_CONFIG_UNDEF_INT ||
      config->splitShare != NCCL_CONFIG_UNDEF_INT ||
      config->trafficClass != NCCL_CONFIG_UNDEF_INT ||
      config->collnetEnable != NCCL_CONFIG_UNDEF_INT ||
      config->CTAPolicy != NCCL_CONFIG_UNDEF_INT ||
      config->shrinkShare != NCCL_CONFIG_UNDEF_INT ||
      config->nvlsCTAs != NCCL_CONFIG_UNDEF_INT ||
      config->nChannelsPerNetPeer != NCCL_CONFIG_UNDEF_INT ||
      config->nvlinkCentricSched != NCCL_CONFIG_UNDEF_INT ||
      config->graphUsageMode != NCCL_CONFIG_UNDEF_INT ||
      config->numRmaCtx != NCCL_CONFIG_UNDEF_INT ||
      config->netName != NCCL_CONFIG_UNDEF_PTR ||
      config->commName != NCCL_CONFIG_UNDEF_PTR ||
      config->commDesc != nullptr ||
      config->splitGroupRanks != NCCL_CONFIG_UNDEF_PTR ||
      config->splitGroupSize != NCCL_CONFIG_UNDEF_INT ||
      config->fastInitMode != NCCL_CONFIG_UNDEF_INT ||
      config->ncclxConfig != NCCL_CONFIG_UNDEF_PTR) {
    WARN(
        "ncclx::commSetConfig: ncclConfig_t fields are not mutable; "
        "only algo hints are allowed");
    return ncclInvalidUsage;
  }

  if (config->hints == NCCL_CONFIG_UNDEF_PTR || config->hints == nullptr) {
    return ncclSuccess;
  }

  if (comm->config.ncclxConfig == NCCL_CONFIG_UNDEF_PTR ||
      comm->config.ncclxConfig == nullptr) {
    WARN("ncclx::commSetConfig: comm has no parsed ncclx::Config");
    return ncclInvalidArgument;
  }

  auto* cfg = static_cast<ncclx::Config*>(comm->config.ncclxConfig);
  const auto* hints = static_cast<const ncclx::Hints*>(config->hints);
  NCCLCHECK(cfg->update(hints));

  std::string updated;
  auto appendIfSet = [&](const char* key, const auto& field) {
    std::string val;
    if (hints->get(key, val) == ncclSuccess) {
      if (!updated.empty()) {
        updated += ' ';
      }
      updated += fmt::format("{}={}", key, algoValToStr(field));
    }
  };
  appendIfSet("sendrecvAlgo", cfg->sendrecvAlgo);
  appendIfSet("allgatherAlgo", cfg->allgatherAlgo);
  appendIfSet("allreduceAlgo", cfg->allreduceAlgo);
  appendIfSet("alltoallvAlgo", cfg->alltoallvAlgo);
  appendIfSet("rmaAlgo", cfg->rmaAlgo);

  if (!updated.empty()) {
    INFO(
        NCCL_INIT,
        "NCCLX CONFIG: [commHash=%lx commDesc=%s rank=%d nRanks=%d] update %s",
        comm->commHash,
        cfg->commDesc.c_str(),
        comm->rank,
        comm->nRanks,
        updated.c_str());
  }
  return ncclSuccess;
}
