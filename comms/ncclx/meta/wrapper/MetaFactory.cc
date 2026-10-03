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

// A config-sourced traffic class outside the DSCP range degrades to the env
// settings rather than aborting, so a misconfigured job still starts. Values
// that do reach CtranIbConfig are hard-checked there.
constexpr int kMaxConfigTrafficClass = 255;

CtranIbConfig makeCtranIbConfigFrom(const ncclComm* comm) {
  if (comm->config.ncclxConfig == nullptr ||
      comm->config.ncclxConfig == NCCL_CONFIG_UNDEF_PTR) {
    return {};
  }
  const auto* x = static_cast<const ncclx::Config*>(comm->config.ncclxConfig);
  // The CTran-specific hint beats the general NcclConfig.traffic_class, which
  // also feeds the plain net_ib transport and therefore stays on ncclConfig_t.
  std::optional<int64_t> trafficClass = x->ctranIbTrafficClass;
  if (!trafficClass.has_value() &&
      comm->config.trafficClass != NCCL_CONFIG_UNDEF_INT &&
      comm->config.trafficClass >= 0) {
    if (comm->config.trafficClass <= kMaxConfigTrafficClass) {
      trafficClass = comm->config.trafficClass;
    } else {
      WARN(
          "Ignoring out-of-range trafficClass %d (valid range [0, %d]); falling back to env settings.",
          comm->config.trafficClass,
          kMaxConfigTrafficClass);
    }
  }
  return CtranIbConfig{
      .numQps = x->ctranIbNumQps,
      .qpScalingTh = x->ctranIbQpScalingTh,
      .vcMode = x->ctranIbVcMode,
      .qpMsgs = x->ctranIbQpMsgs,
      .enableLocalFlush = x->ctranIbEnableLocalFlush,
      .maxNumCqe = x->ctranIbMaxNumCqe,
      .maxNumNic = x->ctranIbMaxNumNic,
      .trafficClass = trafficClass,
  };
}

ctranConfig makeCtranConfigFrom(ncclComm* comm) {
  struct ctranConfig tconfig = {
      .blocking = comm->config.blocking,
      .commDesc = NCCLX_CONFIG_FIELD(comm->config, commDesc),
      .enableProfiler = NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT > 0,
      .ibConfig = makeCtranIbConfigFrom(comm),
  };
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
