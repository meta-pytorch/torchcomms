// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/NcclxConfig.h"

#include "debug.h"
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
  checkPtrConflict("ncclAllGatherAlgo", config->ncclAllGatherAlgo);
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

  // ncclAllGatherAlgo — deprecated string field, kept for MetaFactory compat.
  // The canonical field is allgatherAlgo (enum), parsed below with other algos.
  if (config->ncclAllGatherAlgo) {
    WARN(
        "ncclConfig_t.ncclAllGatherAlgo is deprecated; use ncclConfig_t.hints "
        "with key 'allgatherAlgo' instead");
    ncclAllGatherAlgo = config->ncclAllGatherAlgo;
  } else {
    auto val = getHintStr("ncclAllGatherAlgo");
    if (!val.empty()) {
      ncclAllGatherAlgo = val;
    }
  }
  if (ncclAllGatherAlgo != "undefined") {
    algoStrToVal(ncclAllGatherAlgo, allgatherAlgo);
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
    std::string val = getHintStr("pipesUseDualStateBuffer");
    if (!val.empty()) {
      pipesUseDualStateBuffer = parseHintBool(
          "pipesUseDualStateBuffer", NCCL_CTRAN_PIPES_USE_DUAL_STATE_BUFFER);
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

} // namespace ncclx

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
