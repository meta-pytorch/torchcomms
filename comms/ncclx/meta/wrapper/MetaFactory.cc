// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdexcept>

#include "comm.h"
#include "comms/ctran/algos/AllToAll/AllToAllPHintUtils.h"
#include "comms/ctran/interfaces/ICtran.h"
#include "comms/ctran/memory/memCacheAllocator.h"
#include "comms/ctran/window/WinHintUtils.h"
#include "comms/utils/commSpecs.h"
#include "meta/NcclxChecks.h"
#include "meta/NcclxConfig.h" // @manual
#include "meta/commstate/FactoryCommStateX.h"
#include "meta/ctran-integration/BaselineBootstrap.h"
#include "meta/wrapper/MetaFactory.h"

using namespace ctran;

#define NCCLCHECK_COMM(call) NCCLCHECK(metaCommToNccl(call))

meta::comms::Hints ncclToMetaComm(const ncclx::Hints& hints) {
  meta::comms::Hints ret;
  std::string v;
  for (const auto& k : meta::comms::hints::AllToAllPHintUtils::keys()) {
    NCCLX_COMMCHECKTHROW(ncclToMetaComm(hints.get(k, v)));
    NCCLX_COMMCHECKTHROW(ret.set(k, v));
  }
  for (const auto& k : meta::comms::hints::WinHintUtils::keys()) {
    NCCLX_COMMCHECKTHROW(ncclToMetaComm(hints.get(k, v)));
    NCCLX_COMMCHECKTHROW(ret.set(k, v));
  }
  return ret;
}

namespace {

ctranConfig makeCtranConfigFrom(ncclComm* comm) {
  struct ctranConfig tconfig = {
      .blocking = comm->config.blocking,
      .commDesc = NCCLX_CONFIG_FIELD(comm->config, commDesc),
      .enableProfiler = NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT > 0,
  };
  if (comm->config.ncclxConfig != nullptr) {
    const auto* ncclxCfg =
        static_cast<ncclx::Config*>(comm->config.ncclxConfig);
    if (ncclxCfg->pipesNvlChunkSize.has_value()) {
      tconfig.primsConfig.nvlChunkSize =
          static_cast<int64_t>(ncclxCfg->pipesNvlChunkSize.value());
    }
    tconfig.primsConfig.ibLazyConnect = ncclxCfg->deviceIbLazyConnect;
    if (ncclxCfg->enablePrims.has_value()) {
      tconfig.primsConfig.enablePrims = ncclxCfg->enablePrims.value();
    }
    if (ncclxCfg->primsChannelBufferSize.has_value()) {
      tconfig.primsConfig.channelBufferSize =
          static_cast<int64_t>(ncclxCfg->primsChannelBufferSize.value());
    }
    if (ncclxCfg->primsChannelPipelineDepth.has_value()) {
      tconfig.primsConfig.channelPipelineDepth =
          ncclxCfg->primsChannelPipelineDepth.value();
    }
    if (ncclxCfg->primsMaxChannels.has_value()) {
      tconfig.primsConfig.maxChannels = ncclxCfg->primsMaxChannels.value();
    }
    if (ncclxCfg->primsMaxBlocks.has_value()) {
      tconfig.primsConfig.maxBlocks = ncclxCfg->primsMaxBlocks.value();
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
  if (ncclCommVal->config.ncclxConfig != nullptr) {
    const auto* ncclxCfg =
        static_cast<ncclx::Config*>(ncclCommVal->config.ncclxConfig);
    ncclCommVal->ctranComm_->tmpbufEagerAlloc_ = ncclxCfg->tmpbufEagerAlloc;
  }

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

  NCCLCHECK(ncclx::initCommStateXFromNcclComm(comm, comm->ctranComm_.get()));

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
    NCCLX_LOG(ERR, "CtranComm destruction failed: {}", e.what());
    return ncclInternalError;
  }
  return ncclSuccess;
}

CtranComm* getCtranCommFromNcclComm(ncclComm* ncclComm) {
  if (ncclComm && ncclComm->ctranComm_) {
    return ncclComm->ctranComm_.get();
  }
  return nullptr;
}
