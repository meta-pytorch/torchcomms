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

__attribute__((visibility("default"))) ncclResult_t
Hints::set(const std::string& key, const std::string& val) {
  if (key.starts_with("ncclx_alltoallv_dynamic")) {
    NCCLCHECK(
        metaCommToNccl(AllToAllvDynamicHintUtils::set(key, val, this->kv)));
    return ncclSuccess;
  } else if (key.starts_with("ncclx_alltoallp")) {
    NCCLCHECK(metaCommToNccl(AllToAllPHintUtils::set(key, val, this->kv)));
    return ncclSuccess;
  } else if (key.starts_with(("window"))) {
    NCCLCHECK(metaCommToNccl(WinHintUtils::set(key, val, this->kv)));
    return ncclSuccess;
  } else {
    const auto& knownKeys = ncclx::knownHintKeys();
    if (std::find(knownKeys.begin(), knownKeys.end(), key) == knownKeys.end()) {
      WARN("NCCLX Hints: unknown key '%s'; check spelling", key.c_str());
    }
    this->kv[key] = val;
    return ncclSuccess;
  }
}

__attribute__((visibility("default"))) ncclResult_t
Hints::get(const std::string& key, std::string& val) const {
  auto iter = this->kv.find(key);
  if (iter != this->kv.end()) {
    val = iter->second;
    return ncclSuccess;
  } else {
    return ncclInvalidArgument;
  }
}

} // namespace ncclx
