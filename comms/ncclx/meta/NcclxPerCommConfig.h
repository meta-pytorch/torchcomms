// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <utility>

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
    ERR(ncclInvalidArgument,
        "Per-comm ncclBuffSize override is not supported with splitShare=1");
    return ncclInvalidArgument;
  }
  if (ncclxCfg->ibSplitDataOnQps.has_value()) {
    ERR(ncclInvalidArgument,
        "Per-comm ibSplitDataOnQps override is not supported with splitShare=1");
    return ncclInvalidArgument;
  }
  if (ncclxCfg->ibQpsPerConnection.has_value()) {
    ERR(ncclInvalidArgument,
        "Per-comm ibQpsPerConnection override is not supported with splitShare=1");
    return ncclInvalidArgument;
  }

  // A split-share child shares the parent's transport resources, so divergent
  // per-comm CTran IB config there would silently have no effect.
  // Adding a field means touching four lists: knownHintKeys() and this guard in
  // NcclxConfig.h / NcclxPerCommConfig.h, the parse block in NcclxConfig.cc,
  // and the struct assembly in MetaFactory.cc.
  const std::pair<bool, const char*> ctranIbOverrides[] = {
      {ncclxCfg->ctranIbNumQps.has_value(), "ctranIbNumQps"},
      {ncclxCfg->ctranIbQpScalingTh.has_value(), "ctranIbQpScalingTh"},
      {ncclxCfg->ctranIbQpMsgs.has_value(), "ctranIbQpMsgs"},
      {ncclxCfg->ctranIbVcMode.has_value(), "ctranIbVcMode"},
      {ncclxCfg->ctranIbMaxNumCqe.has_value(), "ctranIbMaxNumCqe"},
      {ncclxCfg->ctranIbMaxNumNic.has_value(), "ctranIbMaxNumNic"},
      {ncclxCfg->ctranIbEnableLocalFlush.has_value(),
       "ctranIbEnableLocalFlush"},
      {ncclxCfg->ctranIbTrafficClass.has_value(), "ctranIbTrafficClass"},
  };
  for (const auto& [isSet, name] : ctranIbOverrides) {
    if (isSet) {
      ERR(ncclInvalidArgument,
          "Per-comm %s override is not supported with splitShare=1",
          name);
      return ncclInvalidArgument;
    }
  }

  return ncclSuccess;
}
