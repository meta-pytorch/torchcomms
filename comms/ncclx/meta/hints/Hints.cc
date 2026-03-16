// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "argcheck.h" // NOLINT
#include "checks.h" // NOLINT
#include "comm.h" // NOLINT
#include "comms/ctran/algos/AllToAll/AllToAllPHintUtils.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicHintUtils.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/WinHintUtils.h"
#include "meta/NcclxConfig.h" // @manual
#include "meta/wrapper/MetaFactory.h"

#include <algorithm>

namespace ncclx {

using meta::comms::hints::AllToAllPHintUtils;
using meta::comms::hints::AllToAllvDynamicHintUtils;
using meta::comms::hints::WinHintUtils;

__attribute__((visibility("default"))) Hints::Hints() {
  AllToAllvDynamicHintUtils::init(this->kv);
  AllToAllPHintUtils::init(this->kv);
  WinHintUtils::init(this->kv);
}

__attribute__((visibility("default"))) Hints::Hints(
    std::initializer_list<std::pair<std::string, std::string>> init)
    : Hints() {
  for (const auto& [key, val] : init) {
    set(key, val);
  }
}

// Strip the "ncclx::" prefix from a key if present, so callers can use
// either "fastInitMode" or "ncclx::fastInitMode" interchangeably.
static std::string stripNcclxPrefix(const std::string& key) {
  constexpr std::string_view kPrefix = "ncclx::";
  if (key.compare(0, kPrefix.size(), kPrefix) == 0) {
    return key.substr(kPrefix.size());
  }
  return key;
}

__attribute__((visibility("default"))) ncclResult_t
Hints::set(const std::string& key, const std::string& val) {
  auto bareKey = stripNcclxPrefix(key);
  if (bareKey.starts_with("ncclx_alltoallv_dynamic")) {
    NCCLCHECK(
        metaCommToNccl(AllToAllvDynamicHintUtils::set(bareKey, val, this->kv)));
    return ncclSuccess;
  } else if (bareKey.starts_with("ncclx_alltoallp")) {
    NCCLCHECK(metaCommToNccl(AllToAllPHintUtils::set(bareKey, val, this->kv)));
    return ncclSuccess;
  } else if (bareKey.starts_with(("window"))) {
    NCCLCHECK(metaCommToNccl(WinHintUtils::set(bareKey, val, this->kv)));
    return ncclSuccess;
  } else {
    const auto& knownKeys = ncclx::knownHintKeys();
    if (std::find(knownKeys.begin(), knownKeys.end(), bareKey) ==
        knownKeys.end()) {
      WARN("NCCLX Hints: unknown key '%s'; check spelling", bareKey.c_str());
    }
    this->kv[bareKey] = val;
    return ncclSuccess;
  }
}

__attribute__((visibility("default"))) ncclResult_t
Hints::get(const std::string& key, std::string& val) const {
  auto iter = this->kv.find(stripNcclxPrefix(key));
  if (iter != this->kv.end()) {
    val = iter->second;
    return ncclSuccess;
  } else {
    return ncclInvalidArgument;
  }
}

} // namespace ncclx
