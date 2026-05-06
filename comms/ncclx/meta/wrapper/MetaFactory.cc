// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdexcept>

#include "comm.h"
#include "comms/ctran/algos/AllToAll/AllToAllPHintUtils.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicHintUtils.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/ctran/window/WinHintUtils.h"
#include "comms/utils/checks.h"
#include "comms/utils/commSpecs.h"
#include "meta/NcclxConfig.h" // @manual
#include "meta/commstate/FactoryCommStateX.h"
#include "meta/ctran-integration/BaselineBootstrap.h"
#include "meta/wrapper/MetaFactory.h"

using namespace ctran;

#define NCCLCHECK_COMM(call) NCCLCHECK(metaCommToNccl(call))

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

namespace {

ctranConfig makeCtranConfigFrom(ncclComm* comm) {
  struct ctranConfig tconfig = {
      .blocking = comm->config.blocking,
      .commDesc = NCCLX_CONFIG_FIELD(comm->config, commDesc),
      .ncclAllGatherAlgo =
          NCCLX_CONFIG_FIELD(comm->config, ncclAllGatherAlgo).c_str(),
  };
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

commResult_t setCtranCommBase(ncclComm* ncclCommVal) {
  if (!ncclCommVal) {
    return commInvalidArgument;
  }
  ncclCommVal->ctranComm_ = std::make_unique<CtranComm>();

  const auto tconfig = makeCtranConfigFrom(ncclCommVal);
  ncclCommVal->ctranComm_->config_ = tconfig;
  ncclCommVal->ctranComm_->opCount_ = &ncclCommVal->opCount;
  ncclCommVal->ctranComm_->logMetaData_ = ncclCommVal->logMetaData;
  ncclCommVal->ctranComm_->runtimeConn_ = ncclCommVal->runtimeConn;

  return commSuccess;
}

} // namespace

ncclResult_t createCtranComm(ncclComm* comm) {
  NCCLCHECK_COMM(setCtranCommBase(comm));

  if (NCCL_USE_MEM_CACHE) {
    comm->ctranComm_->memCache_ =
        ncclx::memory::memCacheAllocator::getInstance();
  }

  comm->ctranComm_->bootstrap_ =
      std::make_unique<ncclx::BaselineBootstrap>(comm);

  NCCLCHECK(ncclx::createCommStateXFromNcclComm(comm, comm->ctranComm_.get()));

  comm->ctranComm_->colltraceNew_ = comm->newCollTrace;

  NCCLCHECK_COMM(ctranInit(comm->ctranComm_.get()));

  return ncclSuccess;
}

ncclResult_t destroyCtranComm(ncclComm* comm) {
  if (!comm || !comm->ctranComm_) {
    return ncclSuccess;
  }
  NCCLCHECK_COMM(ctranFinalize(comm->ctranComm_.get()));
  try {
    comm->ctranComm_->destroy();
    comm->ctranComm_.reset();
  } catch (const std::exception& e) {
    CLOGF(ERR, "CtranComm destruction failed: {}", e.what());
    return ncclInternalError;
  }
  return ncclSuccess;
}
