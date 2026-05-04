// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "debug.h"
#include "meta/NcclxConfig.h" // @manual
#include "nccl.h" // @manual

// Validate per-comm config overrides against the communicator's splitShare
// setting. Must be called after ncclxConfig is populated and splitShare is
// resolved. Returns ncclInvalidArgument if any per-comm override is set
// while splitShare=1 (shared transport buffers can't have different config).
inline ncclResult_t ncclxValidatePerCommConfig(const ncclConfig_t& config) {
  if (!config.ncclxConfig || !config.splitShare)
    return ncclSuccess;

  auto* ncclxCfg = static_cast<ncclx::Config*>(config.ncclxConfig);

  if (ncclxCfg->ncclBuffSize.has_value()) {
    ERR("Per-comm ncclBuffSize override is not supported with splitShare=1");
    return ncclInvalidArgument;
  }
  if (ncclxCfg->ibSplitDataOnQps.has_value()) {
    ERR("Per-comm ibSplitDataOnQps override is not supported with splitShare=1");
    return ncclInvalidArgument;
  }
  if (ncclxCfg->ibQpsPerConnection.has_value()) {
    ERR("Per-comm ibQpsPerConnection override is not supported with splitShare=1");
    return ncclInvalidArgument;
  }

  return ncclSuccess;
}
