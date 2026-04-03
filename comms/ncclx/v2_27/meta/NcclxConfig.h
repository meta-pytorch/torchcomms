// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nccl.h" // @manual

namespace ncclx {

class Config {
 public:
  // Default constructor — uses in-class initializers.
  Config() = default;

  // Parsing constructor — populates fields from an ncclConfig_t using
  // flat fields (old format), hints (new format), and env defaults.
  // Throws std::invalid_argument on conflict or validation error.
  explicit Config(const ncclConfig_t* config);

  // NCCLX-specific config fields (canonical storage).
  // New fields should be added here, NOT to ncclConfig_t.
  // When adding a new field, also add its key to knownHintKeys below.
  std::string commDesc = "undefined";
  std::vector<int> splitGroupRanks;
  std::string ncclAllGatherAlgo = "undefined";
  bool lazyConnect = false;
  bool lazySetupChannels = false;
  bool fastInitMode = false;

  // Per-communicator MultiPeerTransport (pipes) NVL config overrides.
  // When set, override the corresponding CVARs for this communicator.
  std::optional<size_t> pipesNvlChunkSize;
  std::optional<bool> pipesUseDualStateBuffer;
  int vCliqueSize = 0;
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
      "pipesNvlChunkSize",
      "pipesUseDualStateBuffer",
      "vCliqueSize",
  };
  return keys;
}

} // namespace ncclx

// Convenience macro: access an NCCLX-specific field from the canonical
// ncclx::Config stored inside an ncclConfig_t.
// Usage: NCCLX_CONFIG_FIELD(comm->config, commDesc)
#define NCCLX_CONFIG_FIELD(cfg, field) \
  (static_cast<ncclx::Config*>((cfg).ncclxConfig)->field)

// C-style wrapper around the ncclx::Config parsing constructor.
// Most NCCL code is C-based, so this function translates C++
// exceptions into ncclResult_t error codes for the C callers.
// Stores the result in config->ncclxConfig.  Must be called
// exactly once per config.
ncclResult_t ncclxParseCommConfig(ncclConfig_t* config);
