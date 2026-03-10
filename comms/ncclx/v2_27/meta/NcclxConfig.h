// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>

#include "nccl.h" // @manual

namespace ncclx {

struct Config {
  // NCCLX-specific config fields (canonical storage).
  // New fields should be added here, NOT to ncclConfig_t.
  // When adding a new field, also add its key to knownHintKeys below.
  std::string commDesc;
  std::vector<int> splitGroupRanks;
  std::string ncclAllGatherAlgo;
  bool lazyConnect = false;
  bool lazySetupChannels = false;
  bool fastInitMode = false;
};

// Hint keys corresponding to Config fields above.  Used by
// Hints::set() to warn on unrecognized keys (typo detection).
inline const std::vector<std::string>& knownHintKeys() {
  static const std::vector<std::string> keys = {
      "commDesc",
      "splitGroupRanks",
      "ncclAllGatherAlgo",
      "lazyConnect",
      "lazySetupChannels",
      "fastInitMode",
  };
  return keys;
}

} // namespace ncclx

// Convenience macro: access an NCCLX-specific field from the canonical
// ncclx::Config stored inside an ncclConfig_t.
// Usage: NCCLX_CONFIG_FIELD(comm->config, commDesc)
#define NCCLX_CONFIG_FIELD(cfg, field) \
  (static_cast<ncclx::Config*>((cfg).ncclxConfig)->field)

// Create ncclx::Config from config->hints (new format) or flat
// ncclConfig_t fields (old format) with env-based defaults.  Returns
// ncclInvalidArgument if a field is set in both formats.  Stores the
// result in config->ncclxConfig.  Idempotent: does nothing if
// ncclxConfig is already set.
ncclResult_t ncclxParseCommConfig(ncclConfig_t* config);
