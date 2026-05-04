// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdexcept>

#include "comm.h"
#include "comms/ctran/algos/AllToAll/AllToAllPHintUtils.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicHintUtils.h"
#include "comms/ctran/window/WinHintUtils.h"
#include "comms/utils/checks.h"
#include "comms/utils/commSpecs.h"
#include "meta/NcclxConfig.h" // @manual
#include "meta/wrapper/MetaFactory.h"

using namespace ctran;

meta::comms::Hints ncclToMetaComm(const ncclx::Hints& hints) {
  meta::comms::Hints ret;
  std::string v;
  for (const auto& k : meta::comms::hints::AllToAllvDynamicHintUtils::keys()) {
    FB_COMMCHECKTHROW(ncclToMetaComm(hints.get(k, v)));
    FB_COMMCHECKTHROW(ret.set(k, v));
  }
  for (const auto& k : meta::comms::hints::AllToAllPHintUtils::keys()) {
    FB_COMMCHECKTHROW(ncclToMetaComm(hints.get(k, v)));
    FB_COMMCHECKTHROW(ret.set(k, v));
  }
  for (const auto& k : meta::comms::hints::WinHintUtils::keys()) {
    FB_COMMCHECKTHROW(ncclToMetaComm(hints.get(k, v)));
    FB_COMMCHECKTHROW(ret.set(k, v));
  }
  return ret;
}

ctranConfig makeCtranConfigFrom(ncclComm* comm) {
  struct ctranConfig tconfig = {
      .blocking = comm->config.blocking,
      .commDesc = NCCLX_CONFIG_FIELD(comm->config, commDesc),
      .ncclAllGatherAlgo =
          NCCLX_CONFIG_FIELD(comm->config, ncclAllGatherAlgo).c_str(),
  };
  // Wire per-comm pipes NVL transport config from ncclx::Config hints
  if (comm->config.ncclxConfig != nullptr) {
    const auto* ncclxCfg =
        static_cast<ncclx::Config*>(comm->config.ncclxConfig);
    if (ncclxCfg->pipesNvlChunkSize.has_value()) {
      tconfig.pipesConfig.nvlChunkSize =
          static_cast<int64_t>(ncclxCfg->pipesNvlChunkSize.value());
    }
    if (ncclxCfg->pipesUseDualStateBuffer.has_value()) {
      tconfig.pipesConfig.useDualStateBuffer =
          ncclxCfg->pipesUseDualStateBuffer.value() ? 1 : 0;
    }
  }

  return tconfig;
}

// TODO: remove this factory method once we have proper CtranComm initialization
// Initialize all fields except Ctran. Since Ctran/Bootstra/Colltrace requires
// stateX and other fields to be initialized beforehand, we split its
// initialization into two parts:
// 1. Pre-initialization to enable Ctran/Bootstra/Colltrace initialization.
// 2. Final initialization (final-init) to set up the remaining fields.
commResult_t setCtranCommBase(ncclComm* ncclCommVal) {
  if (!ncclCommVal) {
    return commInvalidArgument;
  }

  // can not call make_unique with a private constructor
  // CtranComm has provate constructor for sagety reasons for now
  // no one should use CtranComm constructor until refactoring is finished
  // TODO: move to make_unique after finish refactoring and defining a proper
  // constructor
  ncclCommVal->ctranComm_ =
      std::unique_ptr<CtranComm>(std::move(new CtranComm()));

  const auto tconfig = makeCtranConfigFrom(ncclCommVal);
  ncclCommVal->ctranComm_->config_ = tconfig;
  ncclCommVal->ctranComm_->opCount_ = &ncclCommVal->opCount;
  ncclCommVal->ctranComm_->logMetaData_ = ncclCommVal->logMetaData;
  ncclCommVal->ctranComm_->runtimeConn_ = ncclCommVal->runtimeConn;

  return commSuccess;
}

CtranComm* getCtranCommFromNcclComm(ncclComm* ncclComm) {
  if (ncclComm && ncclComm->ctranComm_) {
    return ncclComm->ctranComm_.get();
  }
  return nullptr;
}
