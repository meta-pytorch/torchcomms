// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/transport/NcclxNetPluginHelper.h"

static const ncclConfig_t* s_ncclxCurrentCommConfig = nullptr;

namespace ncclx {

NcclxCommConfigScope::NcclxCommConfigScope(const ncclConfig_t* config) {
  s_ncclxCurrentCommConfig = config;
}

NcclxCommConfigScope::~NcclxCommConfigScope() {
  s_ncclxCurrentCommConfig = nullptr;
}

} // namespace ncclx

const ncclConfig_t* ncclxGetCurrentCommConfig() {
  return s_ncclxCurrentCommConfig;
}
