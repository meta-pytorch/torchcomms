// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/NcclxConfig.h"

#include "debug.h"
#include "nccl.h" // @manual
#include "param.h"

#include "comms/utils/cvars/nccl_cvars.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncclx {

Config::Config(const ncclConfig_t* config) {
  initEnv();

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
  checkIntConflict("lazyConnect", config->lazyConnect);
  checkIntConflict("lazySetupChannels", config->lazySetupChannels);
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

  // ncclAllGatherAlgo
  if (config->ncclAllGatherAlgo) {
    ncclAllGatherAlgo = config->ncclAllGatherAlgo;
  } else {
    auto val = getHintStr("ncclAllGatherAlgo");
    if (!val.empty()) {
      ncclAllGatherAlgo = val;
    }
  }

  // booleans: flat field > hint > env default
  if (config->lazyConnect != NCCL_CONFIG_UNDEF_INT) {
    lazyConnect = config->lazyConnect != 0;
  } else {
    lazyConnect = parseHintBool("lazyConnect", NCCL_RUNTIME_CONNECT);
  }

  if (config->lazySetupChannels != NCCL_CONFIG_UNDEF_INT) {
    lazySetupChannels = config->lazySetupChannels != 0;
  } else {
    lazySetupChannels =
        parseHintBool("lazySetupChannels", NCCL_LAZY_SETUP_CHANNELS);
  }

  if (config->fastInitMode != NCCL_CONFIG_UNDEF_INT) {
    fastInitMode = config->fastInitMode != 0;
  } else {
    // NCCL_FASTINIT_MODE is a enum, we could not directly convert it to bool
    fastInitMode = parseHintBool(
        "fastInitMode", NCCL_FASTINIT_MODE == NCCL_FASTINIT_MODE::ring_hybrid);
  }
}

} // namespace ncclx
