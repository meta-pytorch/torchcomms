// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "nccl.h" // @manual

namespace ncclx {

// RAII guard that makes the current comm's ncclConfig_t available to
// net-plugin init functions (e.g. ncclIbInit) via ncclxGetCurrentCommConfig().
// Must be held under netPluginMutex — see net.cc.
class NcclxCommConfigScope {
 public:
  explicit NcclxCommConfigScope(const ncclConfig_t* config);
  ~NcclxCommConfigScope();
  NcclxCommConfigScope(const NcclxCommConfigScope&) = delete;
  NcclxCommConfigScope& operator=(const NcclxCommConfigScope&) = delete;
};

} // namespace ncclx

const ncclConfig_t* ncclxGetCurrentCommConfig();
